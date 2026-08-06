#!/usr/bin/env python3
"""Build a tiny on-disk Mixtral-style MoE fixture for soak / regression (CPU).

  python scripts/build_tiny_moe_fixture.py --out /tmp/tiny-mixtral
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_tiny_mixtral(path: Path, vocab_size: int = 256) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from transformers import MixtralConfig, MixtralForCausalLM, PreTrainedTokenizerFast

    special = ["<unk>", "<pad>", "<eos>", "<bos>"]
    tok_model = Tokenizer(BPE(unk_token="<unk>"))
    tok_model.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special)
    corpus = [f"hello world {i} test token cache prefix soak" for i in range(64)]
    tok_model.train_from_iterator(corpus, trainer=trainer)
    tok_model.save(str(path / "tokenizer.json"))
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    config = MixtralConfig(
        vocab_size=len(tok),
        hidden_size=256,
        intermediate_size=512,
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
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "sglang_lite_moe_family").write_text("mixtral\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sglang-lite-tiny-mixtral")
    args = ap.parse_args()
    p = build_tiny_mixtral(Path(args.out))
    print(f"fixture:{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
