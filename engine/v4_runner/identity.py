"""V4-Flash identity helpers (no heavy imports)."""

from __future__ import annotations

from typing import Optional

from ..models import (
    DEEPSEEK_V4,
    MoEFamily,
    assert_moe_supported,
    assert_v4_flash_only,
    v4_only_enabled,
)

__all__ = [
    "is_deepseek_v4_flash_id",
    "require_v4_flash",
    "v4_only_enabled",
]


def is_deepseek_v4_flash_id(
    model_id: str,
    model_type: Optional[str] = None,
) -> bool:
    """True if id/type resolve to the DeepSeek-V4 family (not fixtures)."""
    if not model_id or model_id == "stub":
        return False
    lower = model_id.lower()
    if "ds-v4" in lower or "deepseek-v4" in lower or "deepseek_v4" in lower:
        return True
    if model_type and model_type.lower() == "deepseek_v4":
        return True
    try:
        fam = assert_moe_supported(model_id, model_type)
    except ValueError:
        return False
    return fam.name == DEEPSEEK_V4.name


def require_v4_flash(
    model_id: str,
    model_type: Optional[str] = None,
) -> MoEFamily:
    """Assert product gate and return DEEPSEEK_V4 family."""
    return assert_v4_flash_only(model_id, model_type)
