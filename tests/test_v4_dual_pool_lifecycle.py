"""Phase 0c-2: dual-pool ownership, hit fork, and Hybrid admit wiring."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang_lite.kv_cache import KvLayout, RadixCache
from sglang_lite.loop import EngineLoop, GenParams
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence
from sglang_lite.v4_dual_pool import (
    dual_append_from_model,
    dual_write_from_model,
    release_dual_pool_pages,
    verify_dual_pool_roundtrip,
)
from sglang_lite.v4_prefix_cache import V4PrefixCache, snapshot_v4_kv


class _Attn(nn.Module):
    def __init__(self, t: int = 8):
        super().__init__()
        self.kv_cache = torch.randn(2, t, 512, dtype=torch.bfloat16)
        self.kv_state = torch.zeros(2, 4)
        self.score_state = torch.zeros(2, 1)


class _FakeV4(nn.Module):
    def __init__(self, vocab: int = 32, t: int = 8):
        super().__init__()
        self.vocab = vocab
        self.attn = _Attn(t=t)
        self.calls: List[tuple] = []

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        self.calls.append((tuple(input_ids.shape), int(start_pos)))
        # Grow "tokens" in buffer for dual-append tests.
        n = int(input_ids.shape[1])
        pos = start_pos + n - 1
        if pos < self.attn.kv_cache.shape[1]:
            self.attn.kv_cache[0, pos] = float(pos + 1)
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


def _hybrid_runner(fake: _FakeV4) -> ModelRunner:
    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=4)
    runner.model = fake
    runner._is_real = True
    runner._v4_hybrid = True
    runner._v4_prefix_cache = V4PrefixCache()
    runner.use_paged_as_source = False
    runner.vocab_size = fake.vocab
    runner.eos_token_id = 2
    runner.num_layers = 1
    runner.num_kv_heads = 1
    runner.head_dim = 512
    runner._swa_layout = KvLayout.dsv4_packed(584)
    runner.kernel_backend = MagicMock()
    runner.kernel_backend.supports_paged_attention = True
    return runner


def test_cache_owns_pages_after_seq_release():
    """Prefix entry keeps dual pages after the writing sequence finishes."""
    radix = _dual_radix()
    model = _FakeV4(t=8)
    cache = V4PrefixCache(radix=radix)
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=8)
    assert handle is not None
    blocks = list(handle.swa_blocks)
    # refcount after allocate
    assert radix._allocated_blocks[blocks[0]].ref_count == 1

    cache.insert(
        [1, 2, 3, 4, 5, 6, 7, 8],
        last_logits=torch.zeros(4),
        buffers=snapshot_v4_kv(model, batch_slot=0),
        swa_block_ids=blocks,
        comp_block_ids=blocks,
        dual_pool_tokens=8,
        dual_pool_layers=1,
    )
    # fork by cache → ref 2
    assert radix._allocated_blocks[blocks[0]].ref_count == 2

    # Sequence release (allocate-ref)
    release_dual_pool_pages(radix, handle)
    assert radix._allocated_blocks[blocks[0]].ref_count == 1
    # Entry still has pages
    n, e = cache.match([1, 2, 3, 4, 5, 6, 7, 8])
    assert n == 8 and e is not None and e.swa_block_ids == blocks
    from sglang_lite.v4_dual_pool import DualPoolHandle

    assert verify_dual_pool_roundtrip(
        radix,
        DualPoolHandle(swa_blocks=list(e.swa_block_ids), n_tokens=8),
        n_tokens=8,
    )


def test_hit_forks_pages_and_release_is_safe():
    radix = _dual_radix()
    model = _FakeV4(t=8)
    cache = V4PrefixCache(radix=radix)
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=8)
    assert handle is not None
    blocks = list(handle.swa_blocks)
    cache.insert(
        list(range(8)),
        last_logits=torch.zeros(4),
        buffers=snapshot_v4_kv(model, batch_slot=0),
        swa_block_ids=blocks,
        comp_block_ids=blocks,
        dual_pool_tokens=8,
        dual_pool_layers=1,
    )
    release_dual_pool_pages(radix, handle)  # writer done

    n, entry = cache.match(list(range(8)))
    assert n == 8 and entry is not None
    hit = cache.fork_dual_pool_for_hit(entry)
    assert hit is not None
    assert cache.dual_hit_count == 1
    assert radix.dual_hit_count == 1
    # cache + hit refs
    assert radix._allocated_blocks[blocks[0]].ref_count == 2

    release_dual_pool_pages(radix, hit)
    assert radix._allocated_blocks[blocks[0]].ref_count == 1

    cache.clear()
    assert blocks[0] not in radix._allocated_blocks or radix._allocated_blocks.get(
        blocks[0]
    ) is None or radix._allocated_blocks[blocks[0]].ref_count <= 0 or blocks[
        0
    ] not in radix._allocated_blocks
    # After clear, pages should be free
    assert blocks[0] not in radix._allocated_blocks


def test_dual_append_extends_pages():
    radix = _dual_radix(block_size=4, max_tokens=32)
    model = _FakeV4(t=12)
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=4)
    assert handle is not None
    assert len(handle.swa_blocks) == 1  # 4 tokens / page 4
    ok = dual_append_from_model(model, radix, handle, batch_slot=0, pos=4)
    assert ok is True
    assert handle.n_tokens >= 5
    assert len(handle.swa_blocks) >= 2  # grew for pos=4
    assert radix.dual_append_count >= 1
    release_dual_pool_pages(radix, handle)


def test_hybrid_admit_attaches_dual_pool_on_hit():
    fake = _FakeV4(t=8)
    runner = _hybrid_runner(fake)
    # EngineLoop builds dual-pool radix for hybrid
    loop = EngineLoop(runner, max_batch_size=2)
    assert loop.radix.packed_comp_cache is not None
    assert runner._v4_prefix_cache.radix is loop.radix

    seq1 = Sequence(seq_id=1, request_id="r1", input_ids=[10, 11, 12, 13])
    results: List[Optional[int]] = [None]
    runner._batch_prefill([seq1], [0], loop.radix, results)
    assert len(runner._v4_prefix_cache) == 1
    n, entry = runner.v4_match_prefix([10, 11, 12, 13])
    assert n == 4 and entry is not None
    # Prefill dual-write should have stored pages on entry (after cache fork)
    # Writer seq still holds allocate-ref until release.
    assert entry.swa_block_ids or seq1.swa_block_table

    # Finish writer (release seq fork)
    runner.v4_release_seq(seq1, batch_slot=0)

    # Admit second request with same prefix via loop
    loop.submit("r2", [10, 11, 12, 13, 14], GenParams(max_tokens=2, temperature=0.0))
    loop._admit_pending()
    seq_b = loop.scheduler.waiting[0]
    assert seq_b.cache_hit_tokens == 4
    assert seq_b._v4_kv_pending_restore is True
    # Dual-pool fork attached on hit
    assert getattr(seq_b, "_v4_dual_handle", None) is not None or seq_b.swa_block_table
    if seq_b.swa_block_table:
        assert runner._v4_prefix_cache.dual_hit_count >= 1


def test_replace_entry_releases_old_dual_pages():
    radix = _dual_radix()
    model = _FakeV4(t=8)
    cache = V4PrefixCache(radix=radix)
    h1 = dual_write_from_model(model, radix, batch_slot=0, n_tokens=4)
    assert h1 is not None
    b1 = list(h1.swa_blocks)
    cache.insert(
        [1, 2, 3, 4],
        last_logits=torch.zeros(2),
        buffers=snapshot_v4_kv(model, batch_slot=0),
        swa_block_ids=b1,
        comp_block_ids=b1,
        dual_pool_tokens=4,
    )
    release_dual_pool_pages(radix, h1)
    assert radix._allocated_blocks[b1[0]].ref_count == 1

    h2 = dual_write_from_model(model, radix, batch_slot=0, n_tokens=4)
    assert h2 is not None
    b2 = list(h2.swa_blocks)
    cache.insert(
        [1, 2, 3, 4],
        last_logits=torch.zeros(2),
        buffers=snapshot_v4_kv(model, batch_slot=0),
        swa_block_ids=b2,
        comp_block_ids=b2,
        dual_pool_tokens=4,
    )
    release_dual_pool_pages(radix, h2)
    # Old pages released; only b2 remains
    assert b1[0] not in radix._allocated_blocks
    assert radix._allocated_blocks[b2[0]].ref_count == 1
