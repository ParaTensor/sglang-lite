#!/usr/bin/env python3
"""Patch DeepSeek-V2 remote modeling for transformers 5.x DynamicCache API.

Old remote code calls DynamicCache.from_legacy_cache even when past is None;
TF 5 removed from_legacy_cache / get_usable_length. Safe no-op if already patched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OLD = """        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
"""

NEW = """        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                if past_key_values is None:
                    past_key_values = DynamicCache()
                elif hasattr(DynamicCache, "from_legacy_cache"):
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                else:
                    # transformers 5.x removed from_legacy_cache
                    past_key_values = DynamicCache(ddp_cache_data=list(past_key_values))
            if hasattr(past_key_values, "get_usable_length"):
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            else:
                past_key_values_length = int(past_key_values.get_seq_length())
"""


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "transformers 5.x removed from_legacy_cache" in text:
        return "already_patched"
    if OLD not in text:
        if "from_legacy_cache" not in text:
            return "no_legacy_call"
        return "pattern_mismatch"
    bak = path.with_suffix(path.suffix + ".bak_tf5")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="Dir containing modeling_deepseek.py")
    args = ap.parse_args()
    target = args.model_dir / "modeling_deepseek.py"
    if not target.is_file():
        # follow symlink roots
        target = (args.model_dir / "modeling_deepseek.py").resolve()
    if not target.is_file():
        print(f"missing {target}")
        return 2
    status = patch_file(target.resolve())
    print(f"{status}: {target.resolve()}")
    return 0 if status in ("patched", "already_patched", "no_legacy_call") else 1


if __name__ == "__main__":
    raise SystemExit(main())
