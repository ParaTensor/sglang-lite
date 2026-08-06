"""Phase 0c dual-pool packed page write/read and dual-write helpers."""

from __future__ import annotations

import torch

from sglang_lite.dsv4_kv_pack import DSV4_PACKED_BYTES, pack_dsv4_kv_bf16
from sglang_lite.kv_cache import KvLayout, RadixCache
from sglang_lite.v4_dual_pool import (
    dual_write_from_bf16,
    dual_write_from_model,
    release_dual_pool_pages,
    write_dual_pool_layer,
)
from sglang_lite.v4_prefix_cache import V4PrefixCache


def _dual_radix(*, layers: int = 2, block_size: int = 16, max_tokens: int = 128) -> RadixCache:
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


def test_dual_pools_allocated():
    cache = _dual_radix()
    assert cache.packed_swa_cache is not None
    assert cache.packed_comp_cache is not None
    assert cache.packed_kv_cache is cache.packed_swa_cache
    stats = cache.get_cache_stats()
    assert stats["has_packed_swa"] is True
    assert stats["has_packed_comp"] is True


def test_write_read_packed_roundtrip_swa_and_comp():
    cache = _dual_radix(layers=1, block_size=8)
    t = 20
    blocks = cache.allocate_blocks((t + 7) // 8)
    src = torch.randint(0, 256, (t, DSV4_PACKED_BYTES), dtype=torch.uint8)
    cache.write_packed_kv(blocks, 0, src, pool="swa", layer_idx=0)
    cache.write_packed_kv(blocks, 0, src.flip(0), pool="comp", layer_idx=0)
    got_swa = cache.read_packed_kv(blocks, t, pool="swa", layer_idx=0)
    got_comp = cache.read_packed_kv(blocks, t, pool="comp", layer_idx=0)
    assert torch.equal(got_swa, src)
    assert torch.equal(got_comp, src.flip(0))


def test_dual_write_from_bf16_and_release():
    cache = _dual_radix(layers=2, block_size=16)
    swa = [torch.randn(12, 512, dtype=torch.bfloat16) for _ in range(2)]
    handle = dual_write_from_bf16(cache, swa_layers=swa)
    assert handle.n_tokens == 12
    assert handle.n_layers_written == 2
    assert len(handle.swa_blocks) >= 1
    # Round-trip first layer SWA pack
    packed0 = pack_dsv4_kv_bf16(swa[0])
    got = cache.read_packed_kv(handle.swa_blocks, 12, pool="swa", layer_idx=0)
    assert got.shape == packed0.shape
    assert torch.equal(got, packed0.to(device=got.device))
    assert cache.dual_write_count == 1
    assert cache.dual_write_tokens == 12
    release_dual_pool_pages(cache, handle)
    assert handle.swa_blocks == []
    # Page zeroed after release (refcount 0)
    assert cache.get_cache_stats()["blocks_used"] == 0


def test_prefix_entry_stores_dual_pool_ids():
    cache = V4PrefixCache()
    buf = {"attn.kv_cache": torch.ones(1, 2)}
    cache.insert(
        [1, 2, 3],
        last_logits=torch.zeros(4),
        buffers=buf,
        swa_block_ids=[10, 11],
        comp_block_ids=[10, 11],
        dual_pool_tokens=3,
    )
    n, e = cache.match([1, 2, 3, 4])
    assert n == 3
    assert e is not None
    assert e.swa_block_ids == [10, 11]
    assert e.dual_pool_tokens == 3


def test_dual_write_from_model_with_fake_kv():
    """Model with a 512-d kv_cache row dual-writes successfully."""

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # [batch, T, 512]
            self.kv_cache = torch.randn(2, 8, 512, dtype=torch.bfloat16)

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    radix = _dual_radix(layers=1, block_size=8, max_tokens=64)
    model = _M()
    handle = dual_write_from_model(model, radix, batch_slot=0, n_tokens=8)
    assert handle is not None
    assert handle.n_tokens == 8
    got = radix.read_packed_kv(handle.swa_blocks, 8, pool="swa", layer_idx=0)
    assert got.shape == (8, DSV4_PACKED_BYTES)
    release_dual_pool_pages(radix, handle)


def test_cow_copies_packed_pools():
    cache = _dual_radix(layers=1, block_size=8)
    blocks = cache.allocate_blocks(1)
    bid = blocks[0]
    src = torch.arange(DSV4_PACKED_BYTES, dtype=torch.uint8).unsqueeze(0).expand(8, -1).contiguous()
    cache.write_packed_kv(blocks, 0, src, pool="swa", layer_idx=0)
    cache.write_packed_kv(blocks, 0, src + 1, pool="comp", layer_idx=0)
    # Second ref → COW
    cache.fork_blocks([bid])
    new_id = cache.cow_block_if_shared(bid)
    assert new_id != bid
    got = cache.read_packed_kv([new_id], 8, pool="swa", layer_idx=0)
    assert torch.equal(got, src)
