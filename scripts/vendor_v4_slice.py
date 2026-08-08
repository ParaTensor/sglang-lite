#!/usr/bin/env python3
"""Copy a pinned V4-related upstream slice into engine/vendor/.

Usage examples::

  python scripts/vendor_v4_slice.py deepseek-infer --src /path/to/inference \\
      --pin "hf:DeepSeek-V4-Flash@rev"

  python scripts/vendor_v4_slice.py sglang-v4 --src /path/to/sglang --pin abc123 \\
      --files python/sglang/srt/models/deepseek_v4.py

Does not run ``pip install sglang`` / ``vllm``. Records pin in docs/vendor/SOURCES.md
when --update-sources is set.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "engine" / "vendor"
SOURCES = ROOT / "docs" / "vendor" / "SOURCES.md"

TARGETS = {
    "deepseek-infer": VENDOR / "deepseek_infer",
    "sglang-v4": VENDOR / "sglang_v4",
    "vllm-v4": VENDOR / "vllm_v4",
}

# Default relative files for deepseek-infer (official tree).
DEFAULT_DEEPSEEK_FILES = (
    "model.py",
    "config.json",
)


def _copy_file(src_root: Path, rel: str, dst_root: Path) -> Path:
    src = src_root / rel
    if not src.is_file():
        raise FileNotFoundError(f"missing source file: {src}")
    dest = dst_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _copy_tree_subset(src_root: Path, dst_root: Path, files: list[str]) -> list[Path]:
    out: list[Path] = []
    for rel in files:
        out.append(_copy_file(src_root, rel, dst_root))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=sorted(TARGETS))
    p.add_argument("--src", required=True, type=Path, help="Upstream directory root")
    p.add_argument("--pin", required=True, help="Commit SHA / tag / hf revision")
    p.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Relative paths to copy (default: deepseek-infer defaults or required list)",
    )
    p.add_argument(
        "--update-sources",
        action="store_true",
        help="Append a pin note under docs/vendor/SOURCES.md",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies only",
    )
    args = p.parse_args(argv)

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"error: --src is not a directory: {src}", file=sys.stderr)
        return 2

    dst = TARGETS[args.target]
    files = list(args.files) if args.files else []
    if not files:
        if args.target == "deepseek-infer":
            files = [f for f in DEFAULT_DEEPSEEK_FILES if (src / f).is_file()]
            if "model.py" not in files and not (src / "model.py").is_file():
                print("error: model.py required under --src for deepseek-infer", file=sys.stderr)
                return 2
            if "model.py" not in files:
                files.insert(0, "model.py")
        else:
            print(
                "error: --files is required for sglang-v4 / vllm-v4 "
                "(list only the V4 hot-path modules)",
                file=sys.stderr,
            )
            return 2

    print(f"vendor target={args.target} pin={args.pin}")
    print(f"  src={src}")
    print(f"  dst={dst}")
    for rel in files:
        print(f"  + {rel}")

    if args.dry_run:
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    copied = _copy_tree_subset(src, dst, files)
    pin_file = dst / "VENDOR_PIN.txt"
    pin_file.write_text(
        f"target={args.target}\npin={args.pin}\nsrc={src}\nfiles:\n"
        + "\n".join(f"  - {f}" for f in files)
        + "\n",
        encoding="utf-8",
    )
    print(f"copied {len(copied)} file(s); wrote {pin_file}")

    if args.update_sources and SOURCES.is_file():
        note = (
            f"\n## Pin log\n\n- `{args.target}` pin=`{args.pin}` "
            f"files={files} (auto {__file__})\n"
        )
        # Avoid duplicating the heading every time.
        text = SOURCES.read_text(encoding="utf-8")
        if "## Pin log" not in text:
            SOURCES.write_text(text.rstrip() + note, encoding="utf-8")
        else:
            SOURCES.write_text(
                text.rstrip()
                + f"\n- `{args.target}` pin=`{args.pin}` files={files}\n",
                encoding="utf-8",
            )
        print(f"updated {SOURCES}")

    print(
        "Next: fix imports to sglang_lite.vendor.*; run V4 smoke; "
        "update docs/vendor/SOURCES.md table row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
