"""Conservative decode acceleration helpers (Phase-A thruput).

Full CUDA-graph capture of paged/FI decode is brittle (plan metadata changes
each step). Phase-A defaults:

1. Optional ``torch.compile(mode=\"reduce-overhead\")`` — uses CUDA graphs under
   the hood for static decode shapes when the model cooperates.
2. Reusable ``[1, 1]`` token buffer to avoid per-step host→device alloc.

Enable compile with env ``SGLANG_LITE_TORCH_COMPILE=1`` (or ``true``/``yes``).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch


def compile_enabled() -> bool:
    return os.environ.get("SGLANG_LITE_TORCH_COMPILE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def maybe_compile_model(
    model: Any,
    device: str,
    *,
    paged_hooks: bool = False,
) -> Any:
    """Wrap model with torch.compile when requested and CUDA is available.

    FlashInfer paged hooks are **not** CUDA-graph safe under
    ``mode=reduce-overhead`` (overwritten workspace tensors). When paged hooks
    are attached we either skip compile or use plain ``default`` mode.
    """
    if not compile_enabled():
        return model
    if not str(device).startswith("cuda"):
        print("[sglang-lite] torch.compile skipped (device is not cuda)")
        return model
    if not hasattr(torch, "compile"):
        print("[sglang-lite] torch.compile unavailable")
        return model
    # FI + reduce-overhead CUDAGraphs overwrite internal buffers → hard fail.
    # DynamicCache + reduce-overhead also fails (cudagraph tree overwrite on RoPE).
    # Default to mode=default for HF-cache path (~1.5–2× decode on Qwen3-MoE).
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


def cuda_graph_decode_enabled() -> bool:
    # Default off until validated on each model family; enable with =1.
    return os.environ.get("SGLANG_LITE_CUDA_GRAPH_DECODE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


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
    ) -> Optional[Any]:
        """Run model_fn under CUDA graph when ready; else return None (caller eager)."""
        if force_eager or self._disabled or not cuda_graph_decode_enabled():
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
