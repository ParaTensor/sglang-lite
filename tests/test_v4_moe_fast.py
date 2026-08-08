"""Unit tests for V4 MoE fast dispatch (CPU, synthetic)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_activated_only_logic_matches_full_loop():
    """Toy MoE: activated-expert loop == full local expert loop."""
    torch.manual_seed(0)
    T, D, E, K = 4, 16, 8, 2
    x = torch.randn(T, D)
    # fake gate outputs
    indices = torch.tensor([[0, 3], [1, 1], [7, 2], [0, 5]], dtype=torch.long)
    weights = torch.rand(T, K)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    experts = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(E)])
    start, end = 0, E

    def full():
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=E).tolist()
        for i in range(start, end):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += experts[i](x[idx]) * weights[idx, top, None]
        return y

    def activated():
        y = torch.zeros_like(x)
        for i in torch.unique(indices).tolist():
            if not (start <= int(i) < end):
                continue
            idx, top = torch.where(indices == i)
            y[idx] += experts[int(i)](x[idx]) * weights[idx, top, None]
        return y

    a, b = full(), activated()
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_moe_fast_env_default_on(monkeypatch):
    from sglang_lite.v4_moe_fast import moe_fast_enabled

    monkeypatch.delenv("SGLANG_LITE_V4_MOE_FAST", raising=False)
    assert moe_fast_enabled() is True
    monkeypatch.setenv("SGLANG_LITE_V4_MOE_FAST", "0")
    assert moe_fast_enabled() is False
