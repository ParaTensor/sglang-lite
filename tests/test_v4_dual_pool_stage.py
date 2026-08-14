"""Phase 0c-4: page-primary staging of official kv_cache before decode."""

from __future__ import annotations

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
    stage_official_kv_from_pages,
)
from sglang_lite.v4_prefix_cache import V4PrefixCache


class _Attn(nn.Module):
    def __init__(self, t: int = 8):
        super().__init__()
        self.kv_cache = torch.randn(2, t, 512, dtype=torch.bfloat16)
        self.kv_state = torch.arange(8, dtype=torch.float32).view(2, 4)


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


def test_stage_increments_stage_count_not_restore():
    radix = _dual_radix()
    src = torch.randn(6, 512, dtype=torch.bfloat16)
    handle = dual_write_from_bf16(
        radix, swa_layers=[src], layer_keys=["attn.kv_cache"]
    )
    model = _FakeV4(t=8)
    model.attn.kv_cache.zero_()

    n1, keys = stage_official_kv_from_pages(
        model, radix, handle, batch_slot=0, n_tokens=6
    )
    assert n1 == 1
    assert "attn.kv_cache" in keys
    assert radix.dual_stage_count == 1
    assert radix.dual_restore_count == 0
    assert torch.allclose(model.attn.kv_cache[0, :6].float(), src.float(), atol=1e-2)

    # Corrupt and stage again
    model.attn.kv_cache.zero_()
    n2, _ = stage_official_kv_from_pages(
        model, radix, handle, batch_slot=0, n_tokens=6
    )
    assert n2 == 1
    assert radix.dual_stage_count == 2
    assert radix.dual_restore_count == 0
    assert torch.allclose(model.attn.kv_cache[0, :6].float(), src.float(), atol=1e-2)
    release_dual_pool_pages(radix, handle)


def test_restore_vs_stage_counters():
    radix = _dual_radix()
    src = torch.randn(4, 512, dtype=torch.bfloat16)
    handle = dual_write_from_bf16(
        radix, swa_layers=[src], layer_keys=["attn.kv_cache"]
    )
    model = _FakeV4(t=8)
    model.attn.kv_cache.zero_()
    restore_dual_pool_to_model(model, radix, handle, batch_slot=0, n_tokens=4)
    assert radix.dual_restore_count == 1
    assert radix.dual_stage_count == 0
    model.attn.kv_cache.zero_()
    stage_official_kv_from_pages(model, radix, handle, batch_slot=0, n_tokens=4)
    assert radix.dual_restore_count == 1
    assert radix.dual_stage_count == 1
    release_dual_pool_pages(radix, handle)


def test_runner_stages_once_after_restore_when_page_primary():
    """_v4_stage_pages_before_forward is gated by _v4_need_stage (set after a
    prefix-hit dual restore) and runs at most once — continuous decode must
    NOT re-stage every step (PRO6000 thruput poison, see v4-flash-only §9)."""
    model = _FakeV4(t=8)
    gold = torch.arange(8 * 512, dtype=torch.bfloat16).view(8, 512)
    model.attn.kv_cache[0] = gold.clone()

    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
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
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=8)
    assert handle is not None

    seq = Sequence(seq_id=1, request_id="s", input_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    seq.cached_len = 8
    seq._v4_dual_handle = handle
    seq._v4_dual_radix = radix
    seq._v4_page_primary = False

    model.attn.kv_cache.zero_()
    assert runner._v4_stage_pages_before_forward(seq, batch_slot=0) is False
    assert radix.dual_stage_count == 0
    assert torch.count_nonzero(model.attn.kv_cache) == 0

    # page_primary alone must NOT trigger staging (no need_stage → no re-stage).
    seq._v4_page_primary = True
    assert runner._v4_stage_pages_before_forward(seq, batch_slot=0) is False
    assert radix.dual_stage_count == 0

    # A prefix-hit dual restore sets _v4_need_stage → exactly one stage.
    seq._v4_need_stage = True
    assert runner._v4_stage_pages_before_forward(seq, batch_slot=0) is True
    assert radix.dual_stage_count == 1
    assert torch.allclose(model.attn.kv_cache[0, :8].float(), gold.float(), atol=1e-2)

    # One-shot: subsequent decode steps do not re-stage.
    model.attn.kv_cache.zero_()
    assert runner._v4_stage_pages_before_forward(seq, batch_slot=0) is False
    assert radix.dual_stage_count == 1
    assert torch.count_nonzero(model.attn.kv_cache) == 0
    release_dual_pool_pages(radix, handle)


def test_page_primary_set_after_dual_write_save():
    model = _FakeV4(t=8)
    model.attn.kv_cache[0] = torch.randn(8, 512, dtype=torch.bfloat16)

    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
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

    seq = Sequence(seq_id=1, request_id="w", input_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    seq.cached_len = 8
    seq.last_logits = torch.zeros(32)
    runner._v4_maybe_save_prefix(seq, batch_slot=0, radix=radix)
    assert getattr(seq, "_v4_page_primary", False) is True
    assert seq._v4_dual_handle is not None
    stats = loop.get_stats()
    assert stats["dual_pool"]["dual_write_count"] >= 1
