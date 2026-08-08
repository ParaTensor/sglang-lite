"""Thin leaf-kernel facade for V4-Flash (no sglang/vllm package import).

Phase V2 target: call ops that match official ``kernel.py`` / SGLang dsv4 leaves
when present. Until then, the live path remains the vendored TileLang
``kernel.py`` loaded next to ``model.py`` via sys.path.

This module only documents entry points and safe capability probes so future
ports land in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class V4KernelCaps:
    tilelang_official: bool
    sgl_kernel: bool
    flashinfer: bool
    note: str = ""


def probe_v4_kernel_caps() -> V4KernelCaps:
    """Detect available V4 leaf backends without loading the full model."""
    tile = False
    sgl = False
    fi = False
    try:
        import tilelang  # noqa: F401

        tile = True
    except Exception:
        pass
    try:
        import sgl_kernel  # noqa: F401

        sgl = True
    except Exception:
        pass
    try:
        import flashinfer  # noqa: F401

        fi = True
    except Exception:
        pass
    note = "live_ops=vendor.deepseek_infer.kernel (tilelang)"
    if not tile:
        note = "tilelang missing — Hybrid load will fail until installed"
    return V4KernelCaps(
        tilelang_official=tile,
        sgl_kernel=sgl,
        flashinfer=fi,
        note=note,
    )


def prefer_official_tilelang() -> bool:
    """Product default: official vendored kernel.py (TileLang sparse_attn/GEMM)."""
    return True


__all__ = ["V4KernelCaps", "probe_v4_kernel_caps", "prefer_official_tilelang"]
