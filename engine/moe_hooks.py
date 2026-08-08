"""FlashInfer fused-MoE hooks for HF Qwen3-style expert modules.

Replaces per-layer ``experts.forward`` with ``flashinfer.fused_moe.cutlass_fused_moe``
when the leaf is available (sm_120 / cutlass path). Gate/router stays in HF.

Weight layout note (Qwen3):
  HF ``gate_up_proj`` is ``[E, 2I, H]`` with **gate then up** halves.
  cutlass Swiglu expects **up then gate** → we pack once at attach.

Enable: ``SGLANG_LITE_FUSED_MOE=1`` (or auto when FORCE_HF thruput + FI present).
Disable: ``SGLANG_LITE_FUSED_MOE=0``.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

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


def cutlass_fused_moe_available() -> bool:
    try:
        import flashinfer.fused_moe as fm  # noqa: F401

        return hasattr(fm, "cutlass_fused_moe") and hasattr(fm, "ActivationType")
    except Exception:
        return False


def _pack_gate_up_for_cutlass(gate_up: torch.Tensor) -> torch.Tensor:
    """HF [E, 2I, H] gate||up → cutlass [E, 2I, H] up||gate."""
    g, u = gate_up.chunk(2, dim=1)
    return torch.cat([u, g], dim=1).contiguous()


class CutlassExpertsFn(nn.Module):
    """Drop-in experts forward using FlashInfer cutlass_fused_moe.

    Packs gate/up **in-place** on the HF Parameter storage (no 2× weight memory).
    After attach, HF eager experts.forward is no longer valid for that module.
    """

    def __init__(self, experts_module: nn.Module):
        super().__init__()
        gu = experts_module.gate_up_proj
        dn = experts_module.down_proj
        gu_data = gu.data if hasattr(gu, "data") else gu
        dn_data = dn.data if hasattr(dn, "data") else dn
        packed = _pack_gate_up_for_cutlass(gu_data)
        # In-place: free the temporary by writing into existing storage.
        gu_data.copy_(packed)
        del packed
        self.fc1 = gu_data
        self.fc2 = dn_data if dn_data.is_contiguous() else dn_data.contiguous()
        if self.fc2.data_ptr() != dn_data.data_ptr() and hasattr(dn, "data"):
            # Rare non-contig: write back contiguous view into param if same numel.
            if self.fc2.shape == dn_data.shape:
                dn_data.copy_(self.fc2)
                self.fc2 = dn_data
        import flashinfer.fused_moe as fm

        self._fm = fm
        self._act = fm.ActivationType.Swiglu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = self._fm.cutlass_fused_moe(
            hidden_states.contiguous(),
            top_k_index.to(dtype=torch.int32).contiguous(),
            top_k_weights.to(dtype=torch.float32).contiguous(),
            self.fc1,
            self.fc2,
            hidden_states.dtype,
            quant_scales=[],
            activation_type=self._act,
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def _is_qwen3_experts(module: nn.Module) -> bool:
    name = type(module).__name__
    if "Experts" not in name and "experts" not in name.lower():
        # Container of experts
        pass
    has_gu = hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")
    if not has_gu:
        return False
    gu = module.gate_up_proj
    dn = module.down_proj
    if not (torch.is_tensor(gu) or hasattr(gu, "data")):
        # Parameter
        gu = getattr(gu, "data", gu)
        dn = getattr(dn, "data", dn)
    if not torch.is_tensor(gu) or not torch.is_tensor(dn):
        return False
    # [E, 2I, H] and [E, H, I]
    if gu.dim() != 3 or dn.dim() != 3:
        return False
    if gu.shape[0] != dn.shape[0]:
        return False
    if gu.shape[1] != 2 * dn.shape[2]:
        return False
    if gu.shape[2] != dn.shape[1]:
        return False
    return True


def attach_cutlass_moe(model: Any) -> int:
    """Replace matching expert modules' forward with cutlass fused MoE.

    Returns number of expert modules hooked.
    """
    if not cutlass_fused_moe_available():
        print("[sglang-lite] cutlass_fused_moe unavailable — skip MoE hooks")
        return 0

    n = 0
    for name, module in model.named_modules():
        if not _is_qwen3_experts(module):
            continue
        try:
            fused = CutlassExpertsFn(module)
            module.add_module("_sglang_lite_cutlass", fused)

            def _make_fwd(fmod: CutlassExpertsFn):
                def _fwd(hidden_states, top_k_index, top_k_weights, *args, **kwargs):
                    return fmod(hidden_states, top_k_index, top_k_weights)

                return _fwd

            module.forward = _make_fwd(fused)  # type: ignore[method-assign]
            n += 1
        except Exception as e:
            print(f"[sglang-lite] MoE hook failed on {name}: {e}")
            continue
    if n:
        print(f"[sglang-lite] cutlass fused MoE hooks attached: {n} expert modules")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        print("[sglang-lite] cutlass fused MoE: no matching expert modules found")
    return n


def maybe_attach_fused_moe(model: Any, *, force_hf_cache: bool = False) -> int:
    """Attach fused MoE when enabled or auto-selected.

    PRO6000 Qwen3-30B (2026-08-08):
      - fused alone ~44; batched_mm ~47; torch.compile ~84 (FORCE_HF default).
      - **paged + CUDA graph + fused + native** ~**101** tok/s warm.

    Policy:
      - ``SGLANG_LITE_FUSED_MOE=1/0`` forces on/off.
      - Auto-on when **not** FORCE_HF and ``SGLANG_LITE_CUDA_GRAPH_DECODE=1``
        (radix-native CG stack). Never auto on FORCE_HF thruput path.
    """
    flag = fused_moe_env()
    if flag is False:
        return 0
    if not cutlass_fused_moe_available():
        if flag is True:
            print("[sglang-lite] SGLANG_LITE_FUSED_MOE=1 but cutlass_fused_moe missing")
        return 0
    if flag is True:
        return attach_cutlass_moe(model)
    # Auto: radix-native / paged CG path only.
    if force_hf_cache:
        return 0
    from .cuda_graph import cuda_graph_decode_enabled

    if cuda_graph_decode_enabled(default="0"):
        print(
            "[sglang-lite] auto cutlass fused MoE "
            "(paged + SGLANG_LITE_CUDA_GRAPH_DECODE)"
        )
        return attach_cutlass_moe(model)
    return 0
