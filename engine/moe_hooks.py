"""Fused-MoE hooks for HF Qwen3-style expert modules.

Backends (``SGLANG_LITE_MOE_BACKEND``):
  - ``cutlass`` — FlashInfer ``cutlass_fused_moe`` (default; works on SM120)
  - ``trtllm``  — FlashInfer ``trtllm_bf16_routed_moe`` (needs weight shuffle;
                  currently fails GEMM on SM120 with sm100f cubins)
  - ``sgl``     — sgl-kernel (needs torch ABI match; host torch 2.11 often fails)
  - ``auto``    — try sgl → trtllm → cutlass

Weight layout (Qwen3):
  HF ``gate_up_proj`` is ``[E, 2I, H]`` with **gate then up**.
  cutlass Swiglu expects **up then gate** → pack once at attach.

Enable fused MoE: ``SGLANG_LITE_FUSED_MOE=1`` (or auto on radix+CG).
Disable: ``SGLANG_LITE_FUSED_MOE=0``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn


def fused_moe_env() -> Optional[bool]:
    """None = auto, True/False = force."""
    raw = os.environ.get("SGLANG_LITE_FUSED_MOE", "").strip().lower()
    if raw in ("", "auto"):
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def moe_backend_env() -> str:
    raw = os.environ.get("SGLANG_LITE_MOE_BACKEND", "auto").strip().lower()
    if raw in ("", "auto", "default"):
        return "auto"
    if raw in ("cutlass", "fi", "flashinfer"):
        return "cutlass"
    if raw in ("trtllm", "trt", "trt-llm", "tensorrt"):
        return "trtllm"
    if raw in ("sgl", "sgl-kernel", "sgl_kernel", "sglang"):
        return "sgl"
    return raw


def cutlass_fused_moe_available() -> bool:
    try:
        import flashinfer.fused_moe as fm  # noqa: F401

        return hasattr(fm, "cutlass_fused_moe") and hasattr(fm, "ActivationType")
    except Exception:
        return False


def trtllm_bf16_moe_available() -> Tuple[bool, str]:
    """Return (ok, reason).

    FlashInfer advertises SM≥10, but bf16 TRT-LLM cubins are often sm100f-only;
    attach-time smoke catches runtime GEMM failures on SM120.
    """
    try:
        import flashinfer.fused_moe as fm
        import torch

        if not hasattr(fm, "trtllm_bf16_routed_moe"):
            return False, "trtllm_bf16_routed_moe missing"
        if not torch.cuda.is_available():
            return False, "no cuda"
        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            return False, f"arch {major}.{minor} < 10"
        # FI is_trtllm_moe_supported() currently breaks when passed device index
        # (expects torch.device). Treat API presence + SM≥10 as soft-ok.
        return True, f"soft-ok arch={major}.{minor} (smoke at attach)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sgl_kernel_moe_available() -> Tuple[bool, str]:
    try:
        import sgl_kernel  # noqa: F401

        # Import side-effect loads common_ops; fails on torch ABI mismatch.
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _pack_gate_up_for_cutlass(gate_up: torch.Tensor) -> torch.Tensor:
    """HF [E, 2I, H] gate||up → cutlass [E, 2I, H] up||gate."""
    g, u = gate_up.chunk(2, dim=1)
    return torch.cat([u, g], dim=1).contiguous()


def _convert_bf16_to_trtllm_block_layout(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    is_gated: bool = True,
    epilogue_tile_m: int = 128,
    block_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Port of vLLM convert_moe_weights_to_flashinfer_trtllm_block_layout (bf16).

    w13: [E, 2I, H], w2: [E, H, I] → BlockMajorK shuffled uint8-as-bf16 tensors.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict = {}
    E = w13.shape[0]
    w13_u8_0 = w13[0].contiguous().view(torch.uint8)
    w2_u8_0 = w2[0].contiguous().view(torch.uint8)
    w13_rows, w13_cols = w13_u8_0.shape
    w2_rows, w2_cols = w2_u8_0.shape
    if w13_cols % block_k != 0 or w2_cols % block_k != 0:
        raise ValueError(
            f"weight cols not divisible by block_k={block_k}: "
            f"w13_cols={w13_cols}, w2_cols={w2_cols}"
        )
    out13 = torch.empty(
        (E, w13_cols // block_k, w13_rows, block_k),
        dtype=torch.uint8,
        device=w13.device,
    )
    out2 = torch.empty(
        (E, w2_cols // block_k, w2_rows, block_k),
        dtype=torch.uint8,
        device=w2.device,
    )
    for i in range(E):
        w13e = w13[i].contiguous().view(torch.uint8)
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13e, epilogue_tile_m, is_gated_act_gemm=is_gated
        )
        if is_gated:
            rows = w13e.shape[0]
            perm = (perm + rows // 2) % rows
        blocks = w13e.view(w13e.shape[0], out13.shape[1], block_k).permute(1, 0, 2)
        torch.index_select(blocks, 1, perm.to(w13e.device), out=out13[i])

        w2e = w2[i].contiguous().view(torch.uint8)
        perm2 = get_w2_permute_indices_with_cache(cache, w2e, epilogue_tile_m)
        blocks2 = w2e.view(w2e.shape[0], out2.shape[1], block_k).permute(1, 0, 2)
        torch.index_select(blocks2, 1, perm2.to(w2e.device), out=out2[i])
    return out13.view(torch.bfloat16), out2.view(torch.bfloat16)


def _pack_topk_ids(topk_index: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Pack (expert_id << 16) | bf16_weight_bits for trtllm_bf16_routed_moe."""
    # topk_index: [T, K] int, topk_weights: [T, K] float
    eid = topk_index.to(dtype=torch.int32)
    wb = topk_weights.to(dtype=torch.bfloat16).view(torch.int16).to(torch.int32)
    wb = wb & 0xFFFF
    return (eid << 16) | wb


class CutlassExpertsFn(nn.Module):
    """FlashInfer cutlass_fused_moe (SM120-proven thruput path)."""

    name = "cutlass"

    def __init__(self, experts_module: nn.Module):
        super().__init__()
        gu = experts_module.gate_up_proj
        dn = experts_module.down_proj
        gu_data = gu.data if hasattr(gu, "data") else gu
        dn_data = dn.data if hasattr(dn, "data") else dn
        packed = _pack_gate_up_for_cutlass(gu_data)
        gu_data.copy_(packed)
        del packed
        self.fc1 = gu_data
        self.fc2 = dn_data if dn_data.is_contiguous() else dn_data.contiguous()
        if self.fc2.data_ptr() != dn_data.data_ptr() and hasattr(dn, "data"):
            if self.fc2.shape == dn_data.shape:
                dn_data.copy_(self.fc2)
                self.fc2 = dn_data
        import flashinfer.fused_moe as fm

        self._fm = fm
        self._act = fm.ActivationType.Swiglu
        # Decode-friendly: small max tokens for autotune.
        self._tune_max = int(os.environ.get("SGLANG_LITE_MOE_TUNE_MAX_TOKENS", "1"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        idx = top_k_index
        if idx.dtype != torch.int32 or not idx.is_contiguous():
            idx = idx.to(dtype=torch.int32).contiguous()
        scales = top_k_weights
        if scales.dtype != torch.float32 or not scales.is_contiguous():
            scales = scales.to(dtype=torch.float32).contiguous()
        kwargs = dict(
            quant_scales=[],
            activation_type=self._act,
            tune_max_num_tokens=max(1, self._tune_max),
        )
        out = self._fm.cutlass_fused_moe(
            h, idx, scales, self.fc1, self.fc2, hidden_states.dtype, **kwargs
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


class TrtllmExpertsFn(nn.Module):
    """FlashInfer TRT-LLM bf16 routed MoE (experimental on SM120)."""

    name = "trtllm"

    def __init__(self, experts_module: nn.Module):
        super().__init__()
        import flashinfer.fused_moe as fm

        gu = experts_module.gate_up_proj
        dn = experts_module.down_proj
        gu_data = (gu.data if hasattr(gu, "data") else gu).contiguous()
        dn_data = (dn.data if hasattr(dn, "data") else dn).contiguous()
        # Keep HF gate||up for TRT layout conversion (vLLM path assumes gated GEMM).
        self.fc1, self.fc2 = _convert_bf16_to_trtllm_block_layout(
            gu_data, dn_data, is_gated=True
        )
        self.num_experts = int(gu_data.shape[0])
        self.intermediate_size = int(dn_data.shape[2])
        self.top_k = int(
            getattr(
                getattr(experts_module, "config", None),
                "num_experts_per_tok",
                0,
            )
            or 0
        )
        self._fm = fm
        self._act = int(fm.ActivationType.Swiglu)
        self._layout = int(fm.WeightLayout.BlockMajorK)
        self._verified = False

    def _ensure_top_k(self, top_k_index: torch.Tensor) -> int:
        if self.top_k <= 0:
            self.top_k = int(top_k_index.shape[-1])
        return self.top_k

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states
        if h.dtype != torch.bfloat16:
            h = h.to(dtype=torch.bfloat16)
        if not h.is_contiguous():
            h = h.contiguous()
        # flatten to [T, H]
        orig_shape = h.shape
        h2 = h.view(-1, h.shape[-1])
        idx = top_k_index.view(h2.shape[0], -1)
        w = top_k_weights.view(h2.shape[0], -1)
        packed = _pack_topk_ids(idx, w).contiguous()
        top_k = self._ensure_top_k(idx)
        out = self._fm.trtllm_bf16_routed_moe(
            packed,
            h2,
            self.fc1,
            self.fc2,
            num_experts=self.num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            use_shuffled_weight=True,
            weight_layout=self._layout,
            activation_type=self._act,
            do_finalize=True,
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.view(*orig_shape)


class SglExpertsFn(nn.Module):
    """Placeholder: sgl-kernel BF16 fused MoE is not a drop-in for Qwen3 experts.

    sgl-kernel 0.3.x exposes routing helpers + quantized GEMMs (w4a8/marlin), not a
    simple BF16 cutlass_fused_moe equivalent. When common_ops loads, we still fall
    back to FlashInfer cutlass for the expert compute.
    """

    name = "sgl+cutlass"

    def __init__(self, experts_module: nn.Module):
        super().__init__()
        import sgl_kernel  # noqa: F401 — proves ABI load

        self._inner = CutlassExpertsFn(experts_module)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._inner(hidden_states, top_k_index, top_k_weights)


def _is_qwen3_experts(module: nn.Module) -> bool:
    has_gu = hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")
    if not has_gu:
        return False
    gu = module.gate_up_proj
    dn = module.down_proj
    if not (torch.is_tensor(gu) or hasattr(gu, "data")):
        gu = getattr(gu, "data", gu)
        dn = getattr(dn, "data", dn)
    if not torch.is_tensor(gu) or not torch.is_tensor(dn):
        return False
    if gu.dim() != 3 or dn.dim() != 3:
        return False
    if gu.shape[0] != dn.shape[0]:
        return False
    if gu.shape[1] != 2 * dn.shape[2]:
        return False
    if gu.shape[2] != dn.shape[1]:
        return False
    return True


def _make_factory(backend: str) -> Callable[[nn.Module], nn.Module]:
    if backend == "cutlass":
        return CutlassExpertsFn
    if backend == "trtllm":
        return TrtllmExpertsFn
    if backend == "sgl":
        return SglExpertsFn
    raise ValueError(f"unknown moe backend {backend}")


def resolve_moe_backend(preferred: str = "auto") -> Tuple[str, str]:
    """Return (backend_name, note).

    ``auto`` prefers **cutlass** on current PRO6000 evidence (trtllm GEMM fails on
    SM120 sm100f cubins; sgl-kernel often fails torch ABI). Explicit
    ``SGLANG_LITE_MOE_BACKEND=sgl|trtllm`` still forces that path.
    """
    pref = preferred if preferred != "auto" else moe_backend_env()
    if pref == "auto":
        order = ["cutlass", "sgl", "trtllm"]
    else:
        order = [pref]

    notes: List[str] = []
    for b in order:
        if b == "sgl":
            ok, why = sgl_kernel_moe_available()
            notes.append(f"sgl:{why}")
            if ok:
                return "sgl", "; ".join(notes)
        elif b == "trtllm":
            ok, why = trtllm_bf16_moe_available()
            notes.append(f"trtllm:{why}")
            if ok:
                return "trtllm", "; ".join(notes)
        elif b == "cutlass":
            if cutlass_fused_moe_available():
                notes.append("cutlass:ok")
                return "cutlass", "; ".join(notes)
            notes.append("cutlass:missing")
    return "none", "; ".join(notes)


def attach_fused_moe(model: Any, backend: Optional[str] = None) -> int:
    """Attach fused MoE hooks. Returns number of expert modules hooked."""
    backend = backend or moe_backend_env()
    chosen, note = resolve_moe_backend(backend)

    # Prefer exact requested backend when not auto; fall back to cutlass.
    if backend not in ("auto", "", "default") and chosen in ("none", backend):
        if backend == "sgl":
            ok, why = sgl_kernel_moe_available()
            if ok:
                chosen = "sgl"
            else:
                print(f"[sglang-lite] SGLANG_LITE_MOE_BACKEND=sgl unavailable: {why}")
                if cutlass_fused_moe_available():
                    print("[sglang-lite] falling back to cutlass")
                    chosen = "cutlass"
        elif backend == "trtllm":
            ok, why = trtllm_bf16_moe_available()
            if ok:
                chosen = "trtllm"
            else:
                print(f"[sglang-lite] SGLANG_LITE_MOE_BACKEND=trtllm unavailable: {why}")
                if cutlass_fused_moe_available():
                    print("[sglang-lite] falling back to cutlass")
                    chosen = "cutlass"
        elif backend == "cutlass":
            chosen = "cutlass" if cutlass_fused_moe_available() else "none"

    if chosen == "none":
        print(f"[sglang-lite] no MoE backend available ({note})")
        return 0

    factory = _make_factory(chosen)
    n = 0
    last_err: Optional[str] = None
    for name, module in model.named_modules():
        if not _is_qwen3_experts(module):
            continue
        try:
            fused = factory(module)
            # Smoke one forward if trtllm (catch SM120 GEMM early).
            if chosen == "trtllm" and n == 0 and torch.cuda.is_available():
                dn = module.down_proj
                dn_t = dn.data if hasattr(dn, "data") else dn
                H = int(dn_t.shape[1])
                K = int(getattr(model.config, "num_experts_per_tok", 8) or 8)
                dev = next(model.parameters()).device
                h = torch.zeros(1, H, device=dev, dtype=torch.bfloat16)
                idx = torch.zeros(1, K, dtype=torch.int32, device=dev)
                w = torch.full((1, K), 1.0 / max(K, 1), dtype=torch.float32, device=dev)
                with torch.inference_mode():
                    _ = fused(h, idx, w)
            module.add_module(f"_sglang_lite_moe_{chosen}", fused)

            def _make_fwd(fmod: nn.Module):
                def _fwd(hidden_states, top_k_index, top_k_weights, *args, **kwargs):
                    return fmod(hidden_states, top_k_index, top_k_weights)

                return _fwd

            module.forward = _make_fwd(fused)  # type: ignore[method-assign]
            n += 1
        except Exception as e:
            last_err = f"{name}: {e}"
            # If trtllm smoke fails, fall back to cutlass for remaining.
            if chosen == "trtllm" and cutlass_fused_moe_available():
                print(
                    f"[sglang-lite] trtllm MoE failed ({e}); falling back to cutlass"
                )
                return attach_fused_moe(model, backend="cutlass")
            print(f"[sglang-lite] MoE hook failed on {name}: {e}")
            continue
    if n:
        print(
            f"[sglang-lite] fused MoE backend={chosen} attached: {n} expert modules "
            f"({note})"
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        print(
            f"[sglang-lite] fused MoE backend={chosen}: no modules hooked"
            + (f" last_err={last_err}" if last_err else "")
        )
    return n


# Back-compat alias
def attach_cutlass_moe(model: Any) -> int:
    return attach_fused_moe(model, backend="cutlass")


def maybe_attach_fused_moe(model: Any, *, force_hf_cache: bool = False) -> int:
    """Attach fused MoE when enabled or auto-selected.

    Policy:
      - ``SGLANG_LITE_FUSED_MOE=1/0`` forces on/off.
      - Auto-on when **not** FORCE_HF and ``SGLANG_LITE_CUDA_GRAPH_DECODE=1``.
      - Backend via ``SGLANG_LITE_MOE_BACKEND`` (auto|cutlass|trtllm|sgl).
    """
    flag = fused_moe_env()
    if flag is False:
        return 0
    if flag is True:
        return attach_fused_moe(model)
    if force_hf_cache:
        return 0
    from .cuda_graph import cuda_graph_decode_enabled

    if cuda_graph_decode_enabled(default="0"):
        print(
            "[sglang-lite] auto fused MoE "
            f"(paged+CG, backend={moe_backend_env()})"
        )
        return attach_fused_moe(model)
    return 0
