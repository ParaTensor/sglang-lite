"""Product gate: SGLANG_LITE_V4_ONLY rejects non-V4 models by default."""

from __future__ import annotations

import os

import pytest

from sglang_lite.models import (
    DEEPSEEK_V4,
    assert_moe_supported,
    assert_v4_flash_only,
    v4_only_enabled,
)
from sglang_lite.v4_runner import is_deepseek_v4_flash_id, require_v4_flash


@pytest.fixture
def v4_only_on(monkeypatch):
    monkeypatch.setenv("SGLANG_LITE_V4_ONLY", "1")


@pytest.fixture
def v4_only_off(monkeypatch):
    monkeypatch.setenv("SGLANG_LITE_V4_ONLY", "0")


def test_v4_only_default_is_on(monkeypatch):
    monkeypatch.delenv("SGLANG_LITE_V4_ONLY", raising=False)
    assert v4_only_enabled() is True


def test_accepts_deepseek_v4(v4_only_on):
    fam = assert_moe_supported("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4")
    assert fam is DEEPSEEK_V4
    assert require_v4_flash("/data/ds-v4-flash").name == "deepseek_v4"
    assert is_deepseek_v4_flash_id("/models/DeepSeek-V4-Flash")


def test_rejects_qwen_and_mixtral(v4_only_on):
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash only"):
        assert_moe_supported("Qwen/Qwen3-30B-A3B-Instruct-2507")
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash only"):
        assert_moe_supported("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral")
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash only"):
        assert_v4_flash_only("MiniMaxAI/MiniMax-M2", "minimax_m2")


def test_fixtures_still_allowed(v4_only_on):
    # Unit fixtures keep synthetic Mixtral family for HF tiny graphs.
    fam = assert_moe_supported("fixture:/tmp/tiny")
    assert fam.name == "mixtral"
    fam2 = assert_moe_supported("stub")
    assert fam2.name == "mixtral"


def test_legacy_multi_moe_when_disabled(v4_only_off):
    assert v4_only_enabled() is False
    fam = assert_moe_supported("Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert fam.name == "qwen_moe"
    fam2 = assert_moe_supported("MiniMaxAI/MiniMax-M2", "minimax_m2")
    assert fam2.name == "minimax_moe"


def test_vendor_package_layout():
    from sglang_lite.vendor import deepseek_infer_root, sglang_v4_root, vllm_v4_root
    from sglang_lite.model_loader import resolve_v4_paths, vendored_deepseek_infer_dir

    assert deepseek_infer_root().is_dir()
    assert sglang_v4_root().is_dir()
    assert vllm_v4_root().is_dir()
    assert vendored_deepseek_infer_dir() == deepseek_infer_root()
    # Live official graph must be present (pinned).
    assert (deepseek_infer_root() / "model.py").is_file()
    assert (deepseek_infer_root() / "kernel.py").is_file()
    assert (deepseek_infer_root() / "encoding_dsv4.py").is_file()
    assert "class ModelArgs" in (deepseek_infer_root() / "model.py").read_text(
        encoding="utf-8"
    )
    assert "class Transformer" in (deepseek_infer_root() / "model.py").read_text(
        encoding="utf-8"
    )
    pin = (deepseek_infer_root() / "VENDOR_PIN.txt").read_text(encoding="utf-8")
    assert "60d8d707" in pin
    # Default path resolution prefers vendor when env unset.
    paths = resolve_v4_paths(hf_ckpt="/tmp/does-not-need-to-exist")
    assert paths.inference_dir.resolve() == deepseek_infer_root().resolve()
    # SGLang reference tree present but not required for import.
    ref = sglang_v4_root() / "reference"
    assert ref.is_dir()
    assert (sglang_v4_root() / "VENDOR_PIN.txt").is_file()


def test_config_lite_sets_v4_only(monkeypatch):
    monkeypatch.delenv("SGLANG_LITE_V4_ONLY", raising=False)
    from sglang_lite.config import Config

    Config.from_env(preset="lite")
    assert os.environ.get("SGLANG_LITE_V4_ONLY") == "1"
