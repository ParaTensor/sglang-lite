"""CUDA graph helpers for V4 official Transformer decode.

Official ``Transformer.forward(input_ids, start_pos)`` takes a **Python**
``start_pos`` used for RoPE / compressor indexing. Full capture across all
positions needs either:

- SGLang-style paged metadata (see ``vendor/sglang_v4/reference/...``), or
- dual-pool page restore + fixed plan tensors (Phase V2).

This module provides:

1. Static ``[B, 1]`` token buffers (always useful; cuts host alloc).
2. Optional CUDA graph capture for a **fixed** ``(batch, start_pos)`` pair —
   used for microbench / smoke, not multi-step free generation by default.
3. Env gate ``SGLANG_LITE_V4_CUDA_GRAPH=1``.

Does not import sglang or vllm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from .forward import extract_logits, sample_token


def cuda_graph_enabled() -> bool:
    return os.environ.get("SGLANG_LITE_V4_CUDA_GRAPH", "").lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass
class V4StaticDecodeBuffers:
    """Reusable host→device token slots for decode batch sizes."""

    device: torch.device
    max_batch: int = 4
    _tokens: Dict[int, torch.Tensor] = field(default_factory=dict)

    def tokens(self, batch: int) -> torch.Tensor:
        if batch not in self._tokens:
            self._tokens[batch] = torch.zeros(
                (batch, 1), dtype=torch.long, device=self.device
            )
        return self._tokens[batch]

    def fill(self, batch: int, token_ids: torch.Tensor) -> torch.Tensor:
        buf = self.tokens(batch)
        buf.copy_(token_ids.view(batch, 1))
        return buf


@dataclass
class V4FixedPosCudaGraph:
    """Captured graph for one ``(batch_size, start_pos)`` pair."""

    batch_size: int
    start_pos: int
    graph: torch.cuda.CUDAGraph
    static_tokens: torch.Tensor
    static_logits: torch.Tensor

    def replay(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.static_tokens.copy_(token_ids.view(self.batch_size, 1))
        self.graph.replay()
        return self.static_logits


class V4DecodeAccelerator:
    """Static buffers + optional fixed-pos CUDA graphs for official V4."""

    def __init__(
        self,
        model: Any,
        device: Optional[str] = None,
        max_batch: int = 4,
    ) -> None:
        if device is None:
            p = next(model.parameters())
            device = str(p.device)
        self.model = model
        self.device = torch.device(device)
        self.buffers = V4StaticDecodeBuffers(self.device, max_batch=max_batch)
        self._graphs: Dict[Tuple[int, int], V4FixedPosCudaGraph] = {}
        self.enabled = (
            cuda_graph_enabled()
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def capture_fixed(
        self,
        batch_size: int,
        start_pos: int,
        *,
        warmup: int = 3,
    ) -> bool:
        """Capture decode at a fixed start_pos. Returns False if skipped/failed."""
        if not self.enabled:
            return False
        key = (batch_size, start_pos)
        if key in self._graphs:
            return True
        try:
            static_tokens = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=self.device
            )
            # Warmup (allocator / tilelang JIT).
            for _ in range(warmup):
                _ = self.model(static_tokens, start_pos=start_pos)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.model(static_tokens, start_pos=start_pos)
            logits = extract_logits(out)
            self._graphs[key] = V4FixedPosCudaGraph(
                batch_size=batch_size,
                start_pos=start_pos,
                graph=g,
                static_tokens=static_tokens,
                static_logits=logits,
            )
            print(
                f"[sglang-lite] v4 CUDA graph captured batch={batch_size} "
                f"start_pos={start_pos}"
            )
            return True
        except Exception as e:
            print(f"[sglang-lite] v4 CUDA graph capture failed: {e}")
            return False

    @torch.inference_mode()
    def decode_logits(
        self,
        token_ids: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        """Decode step logits; uses graph when ``(B, start_pos)`` was captured."""
        if token_ids.dim() == 1:
            token_ids = token_ids.view(1, -1)
        bsz = int(token_ids.shape[0])
        token_ids = token_ids.to(self.device, non_blocking=True)
        key = (bsz, int(start_pos))
        g = self._graphs.get(key)
        if g is not None:
            return g.replay(token_ids)
        buf = self.buffers.fill(bsz, token_ids)
        return extract_logits(self.model(buf, start_pos=int(start_pos)))

    @torch.inference_mode()
    def decode_token(
        self,
        token_ids: torch.Tensor,
        start_pos: int,
        *,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        logits = self.decode_logits(token_ids, start_pos)
        return sample_token(logits, temperature=temperature)
