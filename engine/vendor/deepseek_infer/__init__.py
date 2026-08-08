"""Vendored DeepSeek-V4-Flash official inference graph.

Prefer loading via ``sglang_lite.model_loader.import_official_inference``
which inserts this directory on ``sys.path`` so ``import model`` / ``import kernel``
match upstream. Do not rewrite to package-relative imports unless a full
repackage is done carefully (tilelang kernel + relative layout).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

__all__ = ["ROOT"]
