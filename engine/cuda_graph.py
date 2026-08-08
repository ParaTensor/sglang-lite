"""Conservative decode acceleration helpers (Phase-A thruput).

Full CUDA-graph capture of paged/FI decode keeps plan *outside* the graph;
fixed-capacity page metadata (``SGLANG_LITE_PAGED_MAX_PAGES``) keeps tensor
addresses stable across page growth. Phase-A defaults:

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

    def _ensure(self) -> None:
        dev = torch.device(self.device)
        if self._buf is None or self._buf.device != dev:
            self._buf = torch.zeros((1, 1), dtype=torch.long, device=dev)
            self._pos = torch.zeros((1, 1), dtype=torch.long, device=dev)

    @property
    def token_buf(self) -> torch.Tensor:
        self._ensure()
        assert self._buf is not None
        return self._buf

    @property
    def pos_buf(self) -> torch.Tensor:
        self._ensure()
        assert self._pos is not None
        return self._pos

    def set_token(self, token_id: int) -> torch.Tensor:
        self._ensure()
        assert self._buf is not None
        self._buf[0, 0] = int(token_id)
        return self._buf

    def set_token_tensor(self, token: torch.Tensor) -> torch.Tensor:
        """Device-side write (no host int) for burst thruput paths."""
        self._ensure()
        assert self._buf is not None
        self._buf[0, 0] = token.reshape(())
        return self._buf

    def set_position(self, pos: int) -> torch.Tensor:
        self._ensure()
        assert self._pos is not None
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
    3. Later steps: ``begin_forward`` (plan + fixed-capacity static meta) then
       ``graph.replay()``.

    Fixed max-pages buffers (see ``FlashInferBackend._ensure_decode_static``) keep
    plan tensor addresses stable, so page count growth within capacity does **not**
    force re-capture. Invalidate only when static buffer generation changes or
    active pages exceed the capacity recorded at capture.
    """

    def __init__(self, device: str):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self.capture_max_pages = -1  # capacity at capture (not active npages)
        self.capture_buf_gen = -1
        self._static_out: Any = None
        self._warmup_done = 0
        self._warmup_need = 2
        self._disabled = False  # permanent after hard capture failure

    def reset(self) -> None:
        self.graph = None
        self.captured = False
        self.capture_max_pages = -1
        self.capture_buf_gen = -1
        self._static_out = None
        self._warmup_done = 0
        # keep _disabled

    def maybe_run(
        self,
        *,
        model_fn,
        npages: int,
        max_pages: int = -1,
        buf_gen: int = -1,
        force_eager: bool = False,
        enabled: Optional[bool] = None,
    ) -> Optional[Any]:
        """Run model_fn under CUDA graph when ready; else return None (caller eager).

        ``enabled``: override env. When None, uses ``SGLANG_LITE_CUDA_GRAPH_DECODE``.
        ``max_pages`` / ``buf_gen``: fixed static buffer capacity & generation from
        the kernel backend. Growth of *active* ``npages`` within capacity is fine;
        re-capture only when capacity/gen changes or active pages exceed capacity.
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

        cap = int(max_pages) if max_pages > 0 else int(npages)
        gen = int(buf_gen)

        # Invalidate only when buffer addresses/capacity changed, not on every
        # page-boundary (npages += 1) within a fixed-capacity plan.
        if self.captured:
            overflow = npages > self.capture_max_pages > 0
            gen_mismatch = (
                gen >= 0
                and self.capture_buf_gen >= 0
                and gen != self.capture_buf_gen
            )
            cap_grew = (
                max_pages > 0
                and self.capture_max_pages > 0
                and max_pages != self.capture_max_pages
            )
            if overflow or gen_mismatch or cap_grew:
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
                self.capture_max_pages = cap
                self.capture_buf_gen = gen
                print(
                    f"[sglang-lite] paged decode CUDA graph captured "
                    f"(active_pages={npages}, max_pages={cap}, buf_gen={gen})"
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
