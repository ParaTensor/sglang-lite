"""Conservative decode acceleration helpers (Phase-A thruput).

Full CUDA-graph capture of paged/FI decode is brittle (plan metadata changes
each step). Phase-A defaults:

1. Optional ``torch.compile`` — ``mode=default`` for HF DynamicCache;
   ``reduce-overhead`` is unsafe with DynamicCache / FI paged.
2. Reusable ``[1, 1]`` token buffer to avoid per-step host→device alloc.
3. **HF StaticCache + CUDA graph** when experts are CUDA-graph safe
   (``experts_implementation=batched_mm``). Qwen3-MoE PRO6000: ~77 tok/s.

Enable compile with env ``SGLANG_LITE_TORCH_COMPILE=1`` (or ``true``/``yes``).
Enable HF graph decode with ``SGLANG_LITE_CUDA_GRAPH_DECODE=1`` (default on
when experts_impl is batched_mm — see runner).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch


def compile_enabled() -> bool:
    return os.environ.get("SGLANG_LITE_TORCH_COMPILE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def experts_implementation_from_env() -> Optional[str]:
    """HF MoE expert backend. Empty → caller default.

    ``batched_mm``: ~2× faster than default ``grouped_mm`` on Qwen3-MoE and
    CUDA-graph safe. ``grouped_mm`` (TF default) does CPU↔CUDA in capture.
    """
    raw = os.environ.get("SGLANG_LITE_EXPERTS_IMPL", "").strip().lower()
    if not raw or raw in ("auto", "default", "0", "none"):
        return None
    return raw


def maybe_compile_model(
    model: Any,
    device: str,
    *,
    paged_hooks: bool = False,
    prefer_compile: bool = False,
) -> Any:
    """Wrap model with torch.compile when requested and CUDA is available.

    FlashInfer paged hooks are **not** CUDA-graph safe under
    ``mode=reduce-overhead`` (overwritten workspace tensors). When paged hooks
    are attached we either skip compile or use plain ``default`` mode.

    ``prefer_compile``: treat as enabled when env is unset (used for HF
    ``batched_mm`` thruput path — ~80 tok/s vs ~47 eager on Qwen3-MoE PRO6000).
    Explicit ``SGLANG_LITE_TORCH_COMPILE=0`` still disables.
    """
    env = os.environ.get("SGLANG_LITE_TORCH_COMPILE", "").lower()
    if env in ("0", "false", "no", "off"):
        return model
    if env not in ("1", "true", "yes") and not prefer_compile:
        return model
    if not str(device).startswith("cuda"):
        print("[sglang-lite] torch.compile skipped (device is not cuda)")
        return model
    if not hasattr(torch, "compile"):
        print("[sglang-lite] torch.compile unavailable")
        return model
    # FI + reduce-overhead CUDAGraphs overwrite internal buffers → hard fail.
    # DynamicCache + reduce-overhead also fails (cudagraph tree overwrite on RoPE).
    # Default to mode=default for HF-cache path.
    if paged_hooks:
        mode = os.environ.get("SGLANG_LITE_TORCH_COMPILE_MODE", "skip")
        if mode in ("", "skip", "none", "0"):
            print(
                "[sglang-lite] torch.compile skipped with FlashInfer paged hooks "
                "(set SGLANG_LITE_TORCH_COMPILE_MODE=default to force)"
            )
            return model
    else:
        mode = os.environ.get("SGLANG_LITE_TORCH_COMPILE_MODE", "default")
        if mode in ("skip", "none", "0"):
            return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        print(f"[sglang-lite] torch.compile(mode={mode}) enabled")
        return compiled
    except Exception as e:
        print(f"[sglang-lite] torch.compile failed, using eager model: {e}")
        return model


class DecodeInputBuffer:
    """Reuse a single-token decode input tensor (batch=1, q_len=1)."""

    def __init__(self, device: str):
        self.device = device
        self._buf: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None

    def set_token(self, token_id: int) -> torch.Tensor:
        if self._buf is None or self._buf.device.type != torch.device(self.device).type:
            self._buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._buf[0, 0] = int(token_id)
        return self._buf

    def set_position(self, pos: int) -> torch.Tensor:
        if self._pos is None or self._pos.device.type != torch.device(self.device).type:
            self._pos = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._pos[0, 0] = int(pos)
        return self._pos


def cuda_graph_decode_enabled(default: str = "0") -> bool:
    # default arg lets callers opt-in (e.g. paged radix-native default="1").
    return os.environ.get("SGLANG_LITE_CUDA_GRAPH_DECODE", default).lower() in (
        "1",
        "true",
        "yes",
    )


def radix_native_enabled() -> bool:
    """Prefer FI paged + CUDA-graph decode over FORCE_HF thruput path."""
    return os.environ.get("SGLANG_LITE_RADIX_NATIVE", "").lower() in (
        "1",
        "true",
        "yes",
    )


class HfStaticDecodeCudaGraph:
    """Batch=1 HF decode under CUDA graph with StaticCache + static buffers.

    Requires a CUDA-graph-safe MoE path (``experts_implementation=batched_mm``).
    Protocol:
      1. Prefill into StaticCache (eager).
      2. Warm a few decode steps; capture one step with fixed tok/pos buffers.
      3. Replay: write next token + position into buffers, ``graph.replay()``,
         read logits from the static output tensor.
    """

    def __init__(self, device: str, max_cache_len: int = 4096):
        self.device = device
        self.max_cache_len = int(max_cache_len)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self._disabled = False
        self._warmup_done = 0
        self._warmup_need = 2
        self._static_out: Any = None
        self._tok_buf: Optional[torch.Tensor] = None
        self._pos_buf: Optional[torch.Tensor] = None
        self._cache: Any = None
        self._model: Any = None
        self._logits_to_keep = True

    def reset(self) -> None:
        self.graph = None
        self.captured = False
        self._static_out = None
        self._warmup_done = 0
        self._cache = None
        # keep _disabled and buffers

    def ensure_buffers(self) -> None:
        dev = torch.device(self.device)
        if self._tok_buf is None or self._tok_buf.device != dev:
            self._tok_buf = torch.zeros((1, 1), dtype=torch.long, device=dev)
            self._pos_buf = torch.zeros((1, 1), dtype=torch.long, device=dev)

    def alloc_cache(self, model: Any) -> Any:
        """Allocate a fresh StaticCache bound to model config."""
        from transformers import StaticCache

        self.ensure_buffers()
        self._model = model
        self._cache = StaticCache(
            config=model.config, max_cache_len=self.max_cache_len
        )
        self.reset_capture_only()
        return self._cache

    def reset_capture_only(self) -> None:
        self.graph = None
        self.captured = False
        self._static_out = None
        self._warmup_done = 0

    @property
    def cache(self) -> Any:
        return self._cache

    def _forward_decode(self, past: Any) -> Any:
        assert self._tok_buf is not None and self._pos_buf is not None
        assert self._model is not None
        kwargs = {
            "input_ids": self._tok_buf,
            "past_key_values": past,
            "use_cache": True,
            "position_ids": self._pos_buf,
            "cache_position": self._pos_buf.reshape(-1),
        }
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        return self._model(**kwargs)

    def run_decode_step(
        self,
        *,
        token_id: int,
        pos: int,
        past: Any,
        force_eager: bool = False,
    ) -> Any:
        """One decode step; may capture or replay CUDA graph.

        Returns model outputs (logits on ``outputs.logits``).
        """
        if self._disabled or not str(self.device).startswith("cuda"):
            return self._eager(token_id, pos, past)
        if not torch.cuda.is_available():
            return self._eager(token_id, pos, past)
        if force_eager or not cuda_graph_decode_enabled(default="1"):
            return self._eager(token_id, pos, past)

        self.ensure_buffers()
        assert self._tok_buf is not None and self._pos_buf is not None
        self._tok_buf[0, 0] = int(token_id)
        self._pos_buf[0, 0] = int(pos)

        if not self.captured:
            if self._warmup_done < self._warmup_need:
                self._warmup_done += 1
                out = self._forward_decode(past)
                return out
            # Capture: re-run same token/pos (StaticCache overwrites slot).
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g):
                    self._static_out = self._forward_decode(past)
                self.graph = g
                self.captured = True
                print(
                    f"[sglang-lite] HF StaticCache decode CUDA graph captured "
                    f"(pos={pos}, max_cache_len={self.max_cache_len})"
                )
                return self._static_out
            except Exception as e:
                print(
                    f"[sglang-lite] HF CUDA graph capture failed, "
                    f"disable for session: {e}"
                )
                self._disabled = True
                self.reset_capture_only()
                return self._forward_decode(past)

        assert self.graph is not None
        self.graph.replay()
        return self._static_out

    def _eager(self, token_id: int, pos: int, past: Any) -> Any:
        self.ensure_buffers()
        assert self._tok_buf is not None and self._pos_buf is not None
        self._tok_buf[0, 0] = int(token_id)
        self._pos_buf[0, 0] = int(pos)
        return self._forward_decode(past)


class PagedDecodeCudaGraph:
    """Capture batch=1 paged decode ``model()`` body (FI plan stays outside).

    Protocol (after prefill):
    1. First real step(s) run eager until we decide to capture.
    2. Capture re-runs the **same** step (same token, same start) so KV overwrites
       the same slot — safe because append is position-addressed.
    3. Later steps: ``begin_forward`` (plan + static meta) then ``graph.replay()``.

    Re-capture when page count grows beyond the captured plan size.
    """

    def __init__(self, device: str):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self.capture_npages = -1
        self._static_out: Any = None
        self._warmup_done = 0
        self._warmup_need = 2
        self._disabled = False  # permanent after hard capture failure

    def reset(self) -> None:
        self.graph = None
        self.captured = False
        self.capture_npages = -1
        self._static_out = None
        self._warmup_done = 0
        # keep _disabled

    def maybe_run(
        self,
        *,
        model_fn,
        npages: int,
        force_eager: bool = False,
        enabled: Optional[bool] = None,
    ) -> Optional[Any]:
        """Run model_fn under CUDA graph when ready; else return None (caller eager).

        ``enabled``: override env. When None, uses ``SGLANG_LITE_CUDA_GRAPH_DECODE``.
        Re-captures when page count grows (plan footprint changed).
        """
        if force_eager or self._disabled:
            return None
        if enabled is None:
            enabled = cuda_graph_decode_enabled(default="0")
        if not enabled:
            return None
        if not str(self.device).startswith("cuda"):
            return None
        if not torch.cuda.is_available():
            return None

        # Need re-capture if page footprint grew.
        if self.captured and npages != self.capture_npages:
            self.reset()

        if not self.captured:
            # Warmup identical step (overwrites same KV slot).
            if self._warmup_done < self._warmup_need:
                self._warmup_done += 1
                return None  # caller runs eager once
            # Capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g):
                    self._static_out = model_fn()
                self.graph = g
                self.captured = True
                self.capture_npages = int(npages)
                print(
                    f"[sglang-lite] paged decode CUDA graph captured "
                    f"(npages={npages})"
                )
                return self._static_out
            except Exception as e:
                print(
                    f"[sglang-lite] CUDA graph capture failed, disable for session: {e}"
                )
                self._disabled = True
                self.reset()
                return None

        assert self.graph is not None
        self.graph.replay()
        return self._static_out
