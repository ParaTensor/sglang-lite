"""Unit tests for v4_runner forward helpers (no GPU / no full model)."""

from __future__ import annotations

import torch

from sglang_lite.v4_runner import extract_logits, sample_token
from sglang_lite.v4_runner.encode import encode_chat_messages


def test_extract_logits_tensor_and_3d():
    x = torch.randn(2, 16)
    assert extract_logits(x).shape == (2, 16)
    y = torch.randn(2, 4, 16)
    out = extract_logits(y)
    assert out.shape == (2, 16)
    assert torch.equal(out, y[:, -1, :])


def test_extract_logits_tuple():
    tokens = torch.zeros(1, 1, dtype=torch.long)
    logits = torch.randn(1, 32)
    assert extract_logits((tokens, logits)).shape == (1, 32)


def test_sample_greedy_and_temp():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    assert int(sample_token(logits, temperature=0.0).item()) == 1
    tid = int(sample_token(logits, temperature=1.0).item())
    assert 0 <= tid <= 2


def test_encode_chat_messages_from_vendor():
    text = encode_chat_messages([{"role": "user", "content": "hi"}])
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0
