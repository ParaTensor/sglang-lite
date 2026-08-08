"""V4 chat encoding via vendored ``encoding_dsv4`` (no external HF tree required)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model_loader import vendored_deepseek_infer_dir


def _ensure_encoding_on_path() -> Path:
    root = vendored_deepseek_infer_dir()
    key = str(root.resolve())
    if key not in sys.path:
        sys.path.insert(0, key)
    return root


def encode_chat_messages(
    messages: list,
    *,
    thinking_mode: str = "chat",
) -> Optional[str]:
    """Render messages with official encoding_dsv4; None if unavailable."""
    root = _ensure_encoding_on_path()
    if not (root / "encoding_dsv4.py").is_file():
        return None
    try:
        from encoding_dsv4 import encode_messages  # type: ignore
    except Exception:
        return None
    norm: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            norm.append(
                {
                    "role": m.get("role", "user"),
                    "content": m.get("content") or "",
                }
            )
        else:
            norm.append(
                {
                    "role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", "") or "",
                }
            )
    return encode_messages(norm, thinking_mode=thinking_mode)
