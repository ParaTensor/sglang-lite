"""Shared fixtures for sglang-lite tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_mixtral_path(tmp_path_factory) -> str:
    """Build a tiny Mixtral-style model + tokenizer on disk for CPU tests."""
    path = tmp_path_factory.mktemp("tiny_mixtral")
    _build_tiny_mixtral(Path(path))
    return str(path)


@pytest.fixture(scope="session")
def tiny_mixtral_id(tiny_mixtral_path) -> str:
    return f"fixture:{tiny_mixtral_path}"


def _build_tiny_mixtral(path: Path) -> None:
    import torch
    from transformers import GPT2TokenizerFast, MixtralConfig, MixtralForCausalLM

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    config = MixtralConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        router_aux_loss_coef=0.0,
        bos_token_id=tok.bos_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = MixtralForCausalLM(config)
    model.eval()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "sglang_lite_moe_family").write_text("mixtral\n")
