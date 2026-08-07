#!/usr/bin/env python3
"""Add MiniMaxM2RotaryEmbedding.compute_default_rope_parameters for TF 5.x init."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "        self.original_inv_freq = self.inv_freq\n\n    @torch.no_grad()\n"

METHOD = '''        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        """Classic RoPE inv_freq for transformers 5.x weight init."""
        base = float(getattr(config, "rope_theta", 10000.0))
        head_dim = int(
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        partial = getattr(config, "partial_rotary_factor", None)
        if partial is None and getattr(config, "rotary_dim", None):
            partial = float(config.rotary_dim) / float(head_dim)
        partial = float(partial or 1.0)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, 1.0

    @torch.no_grad()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args()
    p = (args.model_dir / "modeling_minimax_m2.py").resolve()
    text = p.read_text(encoding="utf-8")
    if "def compute_default_rope_parameters" in text:
        print("already")
        return 0
    if MARKER not in text:
        print("marker_missing")
        return 1
    text = text.replace(MARKER, METHOD, 1)
    text = text.replace(
        "            self.rope_init_fn = _default_rope_parameters",
        "            self.rope_init_fn = MiniMaxM2RotaryEmbedding.compute_default_rope_parameters",
    )
    p.write_text(text, encoding="utf-8")
    print("patched", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
