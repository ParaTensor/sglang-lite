"""Unit tests for torch sparse_attn (official contract, no GPU required for shapes)."""

from __future__ import annotations

import torch

from sglang_lite.v4_sparse_torch import sparse_attn_torch


def test_sparse_attn_decode_shapes_and_finite():
    torch.manual_seed(0)
    b, s, h, d, k, t = 2, 1, 4, 32, 8, 16
    q = torch.randn(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, t, d, dtype=torch.bfloat16)
    sink = torch.zeros(h, dtype=torch.float32)
    topk = torch.randint(0, t, (b, s, k), dtype=torch.int32)
    topk[:, :, -2:] = -1  # pad
    out = sparse_attn_torch(q, kv, sink, topk, softmax_scale=d**-0.5)
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()


def test_sparse_attn_mask_all_invalid_uses_sink():
    b, s, h, d, k, t = 1, 1, 2, 16, 4, 8
    q = torch.ones(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.ones(b, t, d, dtype=torch.bfloat16)
    sink = torch.zeros(h, dtype=torch.float32)
    topk = torch.full((b, s, k), -1, dtype=torch.int32)
    out = sparse_attn_torch(q, kv, sink, topk, softmax_scale=1.0)
    # With only sink mass, output is zeros (no value mass from keys).
    assert out.abs().float().max().item() < 1e-3
