"""Vendored upstream slices for DeepSeek-V4-Flash (no runtime sglang/vllm).

See ``docs/vendor/SOURCES.md`` and ``NOTICE_VENDOR.md``.
"""

from __future__ import annotations

from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent


def deepseek_infer_root() -> Path:
    return VENDOR_ROOT / "deepseek_infer"


def sglang_v4_root() -> Path:
    return VENDOR_ROOT / "sglang_v4"


def vllm_v4_root() -> Path:
    return VENDOR_ROOT / "vllm_v4"
