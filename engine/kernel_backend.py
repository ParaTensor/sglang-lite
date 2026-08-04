"""KernelBackend: the single hardware abstraction point for kernel-level ops.

Everything hardware-specific (FlashInfer, CUDA-only kernels, future Ascend/CANN
backends) lives behind this interface. The rest of the engine (RadixCache,
Scheduler, ModelRunner composition) stays device-agnostic and must not import
kernel libraries or branch on device types directly.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .kv_cache import PastKV, RadixCache


class KernelBackend:
    """Device-agnostic fallback backend (pure torch tensor copies)."""

    name = "torch"

    def append_paged_kv(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        write_kv: PastKV,
    ) -> None:
        """Write per-layer K/V for tokens [start, start+n) into paged cache."""
        radix.write_kv(block_table, start, write_kv)


class FlashInferBackend(KernelBackend):
    """CUDA backend using FlashInfer paged-KV kernels."""

    name = "flashinfer"

    def __init__(self, device: str):
        import flashinfer

        self._flashinfer = flashinfer
        self.device = device
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )

    def append_paged_kv(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        write_kv: PastKV,
    ) -> None:
        first_k = write_kv[0][0]
        append_len = first_k.shape[-2] if first_k.dim() == 4 else first_k.shape[0]
        pages_after = (start + append_len + radix.block_size - 1) // radix.block_size
        if len(block_table) < pages_after:
            raise RuntimeError(
                f"append_paged_kv: need {pages_after} pages, have {len(block_table)}"
            )
        device = self.device
        for layer_idx, (k, v) in enumerate(write_kv):
            k_tok, v_tok = radix.normalize_kv(k, v)
            k_new = k_tok.to(device=device, dtype=radix.dtype)
            v_new = v_tok.to(device=device, dtype=radix.dtype)
            batch_indices = torch.zeros(append_len, dtype=torch.int32, device=device)
            positions = torch.arange(
                start, start + append_len, dtype=torch.int32, device=device
            )
            active = block_table[:pages_after]
            kv_indices = torch.tensor(active, dtype=torch.int32, device=device)
            kv_indptr = torch.tensor(
                [0, kv_indices.numel()], dtype=torch.int32, device=device
            )
            last_len = start % radix.block_size if start > 0 else 0
            kv_last = torch.tensor([last_len], dtype=torch.int32, device=device)
            self._flashinfer.append_paged_kv_cache(
                k_new,
                v_new,
                batch_indices,
                positions,
                (radix.k_cache[layer_idx], radix.v_cache[layer_idx]),
                kv_indices,
                kv_indptr,
                kv_last,
                kv_layout="NHD",
            )


def create_kernel_backend(device: str) -> KernelBackend:
    """Pick the best backend the current hardware supports."""
    if device != "cpu":
        try:
            backend: Optional[KernelBackend] = FlashInferBackend(device)
            return backend
        except ImportError:
            pass
    return KernelBackend()
