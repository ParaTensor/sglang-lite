#!/usr/bin/env python3
"""Phase 1 kernel capability + SM120 sparse MLA numerical probe.

Reports arch routing and whether FlashInfer SM120 sparse MLA produces
non-zero finite output (the historical blocker on 5090/PRO6000).

  CUDA_VISIBLE_DEVICES=0 \\
  SGLANG_LITE_FI_PREFIX=/tmp/fi1616 \\
  FLASHINFER_DISABLE_VERSION_CHECK=1 \\
    python scripts/phase1_kernel_probe.py --out ~/bench/phase1_kernel_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _maybe_insert_fi_prefix() -> str:
    prefix = os.environ.get("SGLANG_LITE_FI_PREFIX", "")
    if prefix and Path(prefix).is_dir():
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        # Ensure incomplete jit_cache packages don't crash import.
        jc = Path(prefix) / "flashinfer_jit_cache" / "__init__.py"
        if jc.is_file():
            text = jc.read_text(encoding="utf-8", errors="replace")
            if "__version__" not in text:
                jc.write_text(
                    text
                    + (
                        '\n__version__ = "0.6.16.post1+cu130"\n'
                        "def get_jit_cache_dir():\n"
                        "    import pathlib\n"
                        '    return pathlib.Path(__file__).resolve().parent / "jit_cache"\n'
                    ),
                    encoding="utf-8",
                )
            elif "get_jit_cache_dir" not in text:
                jc.write_text(
                    text
                    + (
                        "\ndef get_jit_cache_dir():\n"
                        "    import pathlib\n"
                        '    return pathlib.Path(__file__).resolve().parent / "jit_cache"\n'
                    ),
                    encoding="utf-8",
                )
        if prefix not in sys.path:
            sys.path.insert(0, prefix)
    return prefix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--no-numerical",
        action="store_true",
        help="Skip SM120 sparse numerical probe (symbol-only)",
    )
    args = ap.parse_args()

    prefix = _maybe_insert_fi_prefix()
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch

    from sglang_lite.capability import (
        ArchFamily,
        probe_kernel_capabilities,
        probe_sm120_sparse_numerical,
        select_moe_gemm_backend,
        select_sparse_mla_backend,
    )

    summary: dict = {
        "fi_prefix": prefix or None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "env": {
            "SGLANG_LITE_V4_DISABLE_FI_SPARSE": os.environ.get(
                "SGLANG_LITE_V4_DISABLE_FI_SPARSE", ""
            ),
            "SGLANG_LITE_V4_FORCE_FI_SPARSE": os.environ.get(
                "SGLANG_LITE_V4_FORCE_FI_SPARSE", ""
            ),
            "SGLANG_LITE_V4_FI_SPARSE_NUM_PROBE": os.environ.get(
                "SGLANG_LITE_V4_FI_SPARSE_NUM_PROBE", ""
            ),
        },
    }
    if torch.cuda.is_available():
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["capability"] = list(torch.cuda.get_device_capability(0))

    # Symbol-only first (cheap); numerical via dedicated helper for detail.
    caps = probe_kernel_capabilities(
        "cuda" if torch.cuda.is_available() else "cpu",
        numerical_probe=False,
    )
    summary["arch_family"] = caps.arch_family.value
    summary["flashinfer_version"] = caps.flashinfer_version
    summary["has_sparse_mla_sm120_symbol"] = caps.has_sparse_mla_sm120
    summary["has_b12x_moe"] = caps.has_b12x_moe
    summary["has_sgl_kernel"] = caps.has_sgl_kernel
    summary["has_deep_gemm_sm120"] = caps.has_deep_gemm_sm120
    summary["sparse_mla_backend_selected_symbol_only"] = select_sparse_mla_backend(
        caps
    ).value
    summary["moe_gemm_backend_selected"] = select_moe_gemm_backend(caps).value

    summary["sm120_sparse_numerical"] = {
        "ran": False,
        "ok": False,
        "absmean": None,
        "finite": None,
        "error": None,
    }
    if (
        not args.no_numerical
        and torch.cuda.is_available()
        and caps.arch_family == ArchFamily.SM120
        and caps.has_sparse_mla_sm120
    ):
        num = probe_sm120_sparse_numerical()
        summary["sm120_sparse_numerical"] = {
            "ran": True,
            "ok": bool(num.get("ok")),
            "absmean": num.get("absmean"),
            "finite": num.get("finite"),
            "shape": num.get("shape"),
            "error": num.get("error"),
        }
        # Re-probe with numerical flag so selection reflects gate.
        caps_num = probe_kernel_capabilities("cuda", numerical_probe=True)
        summary["sparse_mla_backend_selected_with_num_probe"] = (
            select_sparse_mla_backend(caps_num).value
        )
        summary["sparse_mla_sm120_numerical_ok"] = (
            caps_num.sparse_mla_sm120_numerical_ok
        )
        summary["sparse_mla_sm120_absmean"] = caps_num.sparse_mla_sm120_absmean
    else:
        summary["sparse_mla_backend_selected_with_num_probe"] = None
        if caps.arch_family != ArchFamily.SM120:
            summary["sm120_sparse_numerical"]["error"] = "not_sm120"
        elif not caps.has_sparse_mla_sm120:
            summary["sm120_sparse_numerical"]["error"] = "symbol_missing"
        elif args.no_numerical:
            summary["sm120_sparse_numerical"]["error"] = "skipped"

    num = summary["sm120_sparse_numerical"]
    recommend_fi = bool(
        caps.arch_family == ArchFamily.SM120
        and caps.has_sparse_mla_sm120
        and num.get("ok")
    )
    summary["phase1_recommendation"] = {
        "enable_fi_sparse_sm120": recommend_fi,
        "keep_SGLANG_LITE_V4_DISABLE_FI_SPARSE": "0" if recommend_fi else "1",
        "reason": (
            "numerical probe absmean>0 — safe to clear DISABLE and rely on auto-select"
            if recommend_fi
            else "symbol missing or numerical absmean≈0 / import error; "
            "keep official sparse_attn (DISABLE=1 or default conservative select)"
        ),
        "moe_backend": select_moe_gemm_backend(caps).value,
        "default_sparse_backend": select_sparse_mla_backend(caps).value,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
        print(f"[phase1-probe] wrote {args.out}")
    # Exit 0 always for data collection; recommendation is in JSON.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
