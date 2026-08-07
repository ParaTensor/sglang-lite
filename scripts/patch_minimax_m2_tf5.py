#!/usr/bin/env python3
"""Patch MiniMax-M2 remote modeling for transformers 5.x.

- OutputRecorder moved out of transformers.utils.generic
- ROPE_INIT_FUNCTIONS no longer has "default"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OUTPUT_OLD = "from transformers.utils.generic import OutputRecorder, check_model_inputs"
OUTPUT_NEW = """try:
    from transformers.utils.generic import OutputRecorder, check_model_inputs
except ImportError:  # transformers 5.x moved OutputRecorder
    from transformers.utils.generic import check_model_inputs
    try:
        from transformers.utils.output_capturing import OutputRecorder
    except ImportError:
        from transformers.modeling_utils import OutputRecorder"""

ROPE_OLD = """        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
"""

ROPE_NEW = """        self.config = config
        # transformers 5.x removed ROPE_INIT_FUNCTIONS["default"] — classic inv_freq.
        if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        else:

            def _default_rope_parameters(cfg, device=None, seq_len=None, **kwargs):
                base = float(getattr(cfg, "rope_theta", 10000.0))
                head_dim = int(
                    getattr(cfg, "head_dim", None)
                    or cfg.hidden_size // cfg.num_attention_heads
                )
                partial = getattr(cfg, "partial_rotary_factor", None)
                if partial is None and getattr(cfg, "rotary_dim", None):
                    partial = float(cfg.rotary_dim) / float(head_dim)
                partial = float(partial or 1.0)
                dim = int(head_dim * partial)
                inv = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64)
                        .to(device=device, dtype=torch.float)
                        / dim
                    )
                )
                return inv, 1.0

            self.rope_init_fn = _default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
"""


def patch_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    actions: list[str] = []
    if "output_capturing import OutputRecorder" not in text and OUTPUT_OLD in text:
        text = text.replace(OUTPUT_OLD, OUTPUT_NEW, 1)
        actions.append("output_recorder")
    elif "output_capturing import OutputRecorder" in text:
        actions.append("output_recorder_already")
    else:
        actions.append("output_recorder_skip")

    if 'removed ROPE_INIT_FUNCTIONS["default"]' not in text and ROPE_OLD in text:
        text = text.replace(ROPE_OLD, ROPE_NEW, 1)
        actions.append("rope_default")
    elif 'removed ROPE_INIT_FUNCTIONS["default"]' in text:
        actions.append("rope_already")
    else:
        actions.append("rope_skip")

    if any(a in ("output_recorder", "rope_default") for a in actions):
        bak = path.with_suffix(path.suffix + ".bak_tf5")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(text, encoding="utf-8")
    return actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args()
    target = (args.model_dir / "modeling_minimax_m2.py").resolve()
    if not target.is_file():
        print(f"missing {target}")
        return 2
    actions = patch_file(target)
    print(f"{target}: {', '.join(actions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
