"""MoE model registry for sglang-lite.

Only popular MoE families are in scope. Dense models are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class MoEFamily:
    name: str
    # HF model_type values that map to this family
    model_types: frozenset
    # Example hub ids (documented / listed in /v1/models when verified)
    verified_ids: frozenset


# Families we explicitly support (architecture-level).
MIXTRAL = MoEFamily(
    name="mixtral",
    model_types=frozenset({"mixtral"}),
    verified_ids=frozenset(
        {
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        }
    ),
)

QWEN_MOE = MoEFamily(
    name="qwen_moe",
    model_types=frozenset({"qwen2_moe", "qwen3_moe"}),
    verified_ids=frozenset(
        {
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "Qwen/Qwen2-57B-A14B-Instruct",
        }
    ),
)

DEEPSEEK_MOE = MoEFamily(
    name="deepseek_moe",
    model_types=frozenset({"deepseek_v2", "deepseek_v3", "deepseek_moe"}),
    verified_ids=frozenset(
        {
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "deepseek-ai/DeepSeek-V2-Chat",
        }
    ),
)

FAMILIES: List[MoEFamily] = [MIXTRAL, QWEN_MOE, DEEPSEEK_MOE]

# Ids that may appear in /v1/models once verified on this build.
# Tiny local fixtures use the special prefix "fixture:" and bypass hub checks.
_VERIFIED: Set[str] = set()
for fam in FAMILIES:
    _VERIFIED |= set(fam.verified_ids)


def is_fixture_model(model_id: str) -> bool:
    return model_id.startswith("fixture:") or model_id.startswith("local:")


def register_verified(model_id: str) -> None:
    """Mark a model id as verified for this process (e.g. after load succeeds)."""
    _VERIFIED.add(model_id)


def list_verified_models() -> List[str]:
    return sorted(_VERIFIED)


def family_for_model_type(model_type: Optional[str]) -> Optional[MoEFamily]:
    if not model_type:
        return None
    mt = model_type.lower()
    for fam in FAMILIES:
        if mt in fam.model_types:
            return fam
    return None


def assert_moe_supported(model_id: str, model_type: Optional[str] = None) -> MoEFamily:
    """Raise ValueError if the model is not an allowed MoE family."""
    if is_fixture_model(model_id) or model_id == "stub":
        # stub only when explicitly requested; fixture treated as Mixtral-style
        return MIXTRAL

    fam = family_for_model_type(model_type)
    if fam is not None:
        return fam

    # Hub id heuristic for known families before config is loaded
    lower = model_id.lower()
    if "mixtral" in lower:
        return MIXTRAL
    if "qwen" in lower and "moe" in lower:
        return QWEN_MOE
    if "deepseek" in lower and ("moe" in lower or "v2" in lower or "v3" in lower):
        return DEEPSEEK_MOE

    raise ValueError(
        f"model '{model_id}' is not a supported MoE family "
        f"(mixtral / qwen-moe / deepseek-moe). Dense models are out of scope."
    )


def default_serving_model_list() -> List[str]:
    """Models advertised on GET /v1/models (verified MoE only)."""
    return list_verified_models()
