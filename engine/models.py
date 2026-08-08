"""MoE model registry for sglang-lite.

Only popular MoE families are in scope. Dense models are rejected.
Verified ids are those successfully loaded in this process (plus explicit register).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set


@dataclass(frozen=True)
class MoEFamily:
    name: str
    model_types: frozenset
    example_ids: frozenset


MIXTRAL = MoEFamily(
    name="mixtral",
    model_types=frozenset({"mixtral"}),
    example_ids=frozenset(
        {
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        }
    ),
)

QWEN_MOE = MoEFamily(
    name="qwen_moe",
    model_types=frozenset({"qwen2_moe", "qwen3_moe"}),
    example_ids=frozenset(
        {
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "Qwen/Qwen2-57B-A14B-Instruct",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        }
    ),
)

DEEPSEEK_MOE = MoEFamily(
    name="deepseek_moe",
    model_types=frozenset({"deepseek_v2", "deepseek_v3", "deepseek_moe"}),
    example_ids=frozenset(
        {
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "deepseek-ai/DeepSeek-V2-Chat",
        }
    ),
)

DEEPSEEK_V4 = MoEFamily(
    name="deepseek_v4",
    model_types=frozenset({"deepseek_v4"}),
    example_ids=frozenset(
        {
            "deepseek-ai/DeepSeek-V4-Flash",
            "local:ds-v4-flash",
        }
    ),
)

# MiniMax-M2 / M2.5 ~230B total / ~10B active (≤300B gate). M3 (~428B + multimodal)
# is out of MVP size/modality scope; registry still accepts minimax_m3 for text
# experiments if weights are local and HF remote code loads.
MINIMAX_MOE = MoEFamily(
    name="minimax_moe",
    model_types=frozenset({"minimax_m2", "minimax_m3", "minimax"}),
    example_ids=frozenset(
        {
            "MiniMaxAI/MiniMax-M2",
            "MiniMaxAI/MiniMax-M2.5",
            "MiniMaxAI/MiniMax-M3",
        }
    ),
)

FAMILIES: List[MoEFamily] = [
    MIXTRAL,
    QWEN_MOE,
    DEEPSEEK_MOE,
    DEEPSEEK_V4,
    MINIMAX_MOE,
]

# Only ids that successfully loaded (or were explicitly registered) this process.
_VERIFIED: Set[str] = set()


def is_fixture_model(model_id: str) -> bool:
    return model_id.startswith("fixture:") or model_id.startswith("local:")


def register_verified(model_id: str) -> None:
    """Mark a model id as verified for this process (after load succeeds)."""
    _VERIFIED.add(model_id)


def list_verified_models() -> List[str]:
    return sorted(_VERIFIED)


def known_example_ids() -> List[str]:
    """Documented example hub ids (not automatically advertised)."""
    out: Set[str] = set()
    for fam in FAMILIES:
        out |= set(fam.example_ids)
    return sorted(out)


def family_for_model_type(model_type: Optional[str]) -> Optional[MoEFamily]:
    if not model_type:
        return None
    mt = model_type.lower()
    for fam in FAMILIES:
        if mt in fam.model_types:
            return fam
    return None


def _model_type_from_local_config(model_id: str) -> Optional[str]:
    """Read model_type from a local HF/ModelScope checkpoint directory."""
    root = Path(model_id)
    if is_fixture_model(model_id):
        root = Path(model_id.split(":", 1)[1])
    if not root.is_dir():
        return None
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mt = raw.get("model_type")
    if isinstance(mt, str) and mt:
        return mt
    # Multimodal wrappers (out of scope) may nest text_config.model_type.
    text_cfg = raw.get("text_config")
    if isinstance(text_cfg, dict):
        mt2 = text_cfg.get("model_type")
        if isinstance(mt2, str) and mt2:
            return mt2
    return None


def assert_moe_supported(model_id: str, model_type: Optional[str] = None) -> MoEFamily:
    """Raise ValueError if the model is not an allowed MoE family."""
    if is_fixture_model(model_id) or model_id == "stub":
        return MIXTRAL

    fam = family_for_model_type(model_type)
    if fam is not None:
        return fam

    # Local dirs: resolve model_type before name heuristics (e.g. Qwen3-30B-A3B).
    if model_type is None:
        mt_local = _model_type_from_local_config(model_id)
        fam = family_for_model_type(mt_local)
        if fam is not None:
            return fam

    lower = model_id.lower()
    if "mixtral" in lower:
        return MIXTRAL
    if "qwen" in lower and "moe" in lower:
        return QWEN_MOE
    # Qwen3 MoE naming often uses A3B / A14B without the substring "moe".
    if "qwen" in lower and (
        "a3b" in lower or "a14b" in lower or "a2.7b" in lower or "-moe" in lower
    ):
        return QWEN_MOE
    if "deepseek" in lower and "v4" in lower:
        return DEEPSEEK_V4
    if "ds-v4" in lower or "deepseek-v4" in lower:
        return DEEPSEEK_V4
    if "deepseek" in lower and ("moe" in lower or "v2" in lower or "v3" in lower):
        return DEEPSEEK_MOE
    if "minimax" in lower:
        return MINIMAX_MOE

    raise ValueError(
        f"model '{model_id}' is not a supported MoE family "
        f"(mixtral / qwen-moe / deepseek-moe / deepseek_v4 / minimax). "
        "Dense models are out of scope."
    )


def default_serving_model_list() -> List[str]:
    return list_verified_models()
