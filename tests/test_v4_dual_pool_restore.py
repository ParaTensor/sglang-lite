"""Phase 0c-3: bf16 page restore into official kv_cache buffers."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang_lite.kv_cache import KvLayout, RadixCache
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence
from sglang_lite.v4_dual_pool import (
    dual_write_from_bf16,
    dual_write_from_model,
    release_dual_pool_pages,
    restore_dual_pool_to_model,
)
from sglang_lite.v4_prefix_cache import V4PrefixCache, snapshot_v4_kv


class _Attn(nn.Module):
    def __init__(self, t: int = 8):
        super().__init__()
        self.kv_cache = torch.randn(2, t, 512, dtype=torch.bfloat16)
        self.kv_state = torch.arange(8, dtype=torch.float32).view(2, 4)
        self.score_state = torch.ones(2, 1)


class _FakeV4(nn.Module):
    def __init__(self, vocab: int = 32, t: int = 8):
        super().__init__()
        self.vocab = vocab
        self.attn = _Attn(t=t)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        b = input_ids.shape[0]
        logits = torch.zeros(b, self.vocab)
        logits[:, 3] = 10.0
        return logits


def _dual_radix(layers: int = 1, block_size: int = 8, max_tokens: int = 64) -> RadixCache:
    packed = KvLayout.dsv4_packed(584)
    return RadixCache(
        max_tokens=max_tokens,
        block_size=block_size,
        num_layers=layers,
        num_kv_heads=1,
        head_dim=512,
        dtype=torch.bfloat16,
        device="cpu",
        layout=packed,
        swa_layout=packed,
        compressed_layout=packed,
    )


def test_restore_bf16_pool_roundtrip():
    radix = _dual_radix()
    assert radix.restore_bf16_cache is not None
    src = torch.randn(10, 512, dtype=torch.bfloat16)
    handle = dual_write_from_bf16(
        radix, swa_layers=[src], layer_keys=["attn.kv_cache"]
    )
    got = radix.read_restore_bf16(handle.swa_blocks, 10, layer_idx=0)
    assert torch.allclose(got.float(), src.float(), atol=1e-3)
    release_dual_pool_pages(radix, handle)


def test_restore_dual_pool_to_model_writes_kv_row():
    model = _FakeV4(t=8)
    gold = model.attn.kv_cache[0, :8].clone()
    radix = _dual_radix()
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=8)
    assert handle is not None
    assert "attn.kv_cache" in handle.layer_keys

    # Corrupt model buffer
    model.attn.kv_cache.zero_()
    model.attn.kv_state.zero_()
    n, keys = restore_dual_pool_to_model(model, radix, handle, batch_slot=0, n_tokens=8)
    assert n == 1
    assert "attn.kv_cache" in keys
    assert torch.allclose(model.attn.kv_cache[0, :8].float(), gold.float(), atol=1e-2)
    # state not restored by dual-pool
    assert torch.count_nonzero(model.attn.kv_state) == 0
    release_dual_pool_pages(radix, handle)


def test_hybrid_hit_restores_from_pages_not_only_snapshot():
    """End-to-end: prefill dual-write (slim snapshot) → hit restore via pages."""
    model = _FakeV4(t=8)
    # Distinct pattern
    model.attn.kv_cache[0] = torch.arange(8 * 512, dtype=torch.bfloat16).view(8, 512)
    model.attn.kv_state[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=4)
    runner.model = model
    runner._is_real = True
    runner._v4_hybrid = True
    runner._v4_prefix_cache = V4PrefixCache()
    runner.use_paged_as_source = False
    runner.vocab_size = 32
    runner.eos_token_id = 2
    runner.num_layers = 1
    runner.num_kv_heads = 1
    runner.head_dim = 512
    runner._swa_layout = KvLayout.dsv4_packed(584)
    runner.kernel_backend = MagicMock()
    runner.kernel_backend.supports_paged_attention = True

    from sglang_lite.loop import EngineLoop

    loop = EngineLoop(runner, max_batch_size=2)
    radix = loop.radix
    assert radix.restore_bf16_cache is not None

    seq1 = Sequence(seq_id=1, request_id="w", input_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    results: List[Optional[int]] = [None]
    runner._batch_prefill([seq1], [0], radix, results)
    n, entry = runner.v4_match_prefix(seq1.input_ids)
    assert n == 8 and entry is not None
    assert entry.dual_primary is True
    # Slimmed: no full kv_cache tensor in snapshot (page-backed)
    assert not any(k.endswith("kv_cache") for k in entry.buffers)
    assert any("kv_state" in k for k in entry.buffers)

    gold_kv = model.attn.kv_cache[0, :8].clone()
    gold_state = model.attn.kv_state[0].clone()
    runner.v4_release_seq(seq1, batch_slot=0)

    # Corrupt GPU/CPU slot
    model.attn.kv_cache.zero_()
    model.attn.kv_state.zero_()

    # Hit path
    seq2 = Sequence(seq_id=2, request_id="h", input_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    seq2.cached_len = 8
    seq2.cache_hit_tokens = 8
    seq2.last_logits = entry.last_logits
    seq2._v4_prefix_entry = entry
    seq2._v4_kv_pending_restore = True
    assert runner.v4_attach_dual_pool_from_entry(seq2, entry, radix)
    runner._v4_ensure_restored(seq2, batch_slot=0)

    assert getattr(seq2, "_v4_dual_restored", False) is True
    assert torch.allclose(model.attn.kv_cache[0, :8].float(), gold_kv.float(), atol=1e-2)
    assert torch.allclose(model.attn.kv_state[0], gold_state)
    assert radix.dual_restore_count >= 1

    runner.v4_release_seq(seq2, batch_slot=0)
