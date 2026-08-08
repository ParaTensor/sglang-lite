"""ModelLoader TP shard plan tests (no GPU / no weights required)."""

from __future__ import annotations

import pytest

from sglang_lite.model_loader import build_tp_shard_plan, default_tp8_plans, find_rank_shard
from sglang_lite.models import (
    DEEPSEEK_V4,
    MINIMAX_MOE,
    assert_moe_supported,
    family_for_model_type,
)


def test_tp8_expert_split_covers_all():
    plans = default_tp8_plans(256)
    assert len(plans) == 8
    covered = []
    for p in plans:
        assert p.n_local_experts == 32
        covered.extend(p.expert_ids())
    assert covered == list(range(256))


def test_tp_plan_rejects_bad_rank():
    with pytest.raises(ValueError):
        build_tp_shard_plan(8, 8, n_experts=256)


def test_tp_plan_requires_divisible_experts():
    with pytest.raises(ValueError):
        build_tp_shard_plan(8, 0, n_experts=255)


def test_deepseek_v4_family_registered():
    assert family_for_model_type("deepseek_v4") is DEEPSEEK_V4
    fam = assert_moe_supported("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4")
    assert fam.name == "deepseek_v4"
    fam2 = assert_moe_supported("/data/ds-v4-flash")
    assert fam2.name == "deepseek_v4"


def test_minimax_family_registered(monkeypatch):
    # Legacy multi-MoE path (product default is V4-only).
    monkeypatch.setenv("SGLANG_LITE_V4_ONLY", "0")
    assert family_for_model_type("minimax_m2") is MINIMAX_MOE
    fam = assert_moe_supported("MiniMaxAI/MiniMax-M2", "minimax_m2")
    assert fam.name == "minimax_moe"
    fam2 = assert_moe_supported("/data/MiniMax-M2.5")
    assert fam2.name == "minimax_moe"
    fam3 = assert_moe_supported("MiniMaxAI/MiniMax-M3", "minimax_m3")
    assert fam3.name == "minimax_moe"


def test_qwen3_a3b_name_and_local_config(tmp_path, monkeypatch):
    from sglang_lite.models import QWEN_MOE

    monkeypatch.setenv("SGLANG_LITE_V4_ONLY", "0")
    fam = assert_moe_supported("Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert fam is QWEN_MOE
    cfg = tmp_path / "config.json"
    cfg.write_text('{"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}')
    fam2 = assert_moe_supported(str(tmp_path))
    assert fam2 is QWEN_MOE


def test_find_rank_shard_naming(tmp_path):
    p = tmp_path / "model3-mp8.safetensors"
    p.write_bytes(b"x")
    assert find_rank_shard(tmp_path, 3, 8) == p
