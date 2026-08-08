"""V4-Flash load entry: prefer vendor graph, then external inference/.

Does not ``import sglang`` or ``import vllm``. Weight layout still follows
official convert.py MP shards (see model_loader.load_v4_hybrid_model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..model_loader import (
    DeepseekV4Paths,
    TpShardPlan,
    import_official_inference,
    load_v4_hybrid_model,
    resolve_v4_paths,
    vendored_deepseek_infer_dir,
)
from .identity import require_v4_flash


@dataclass(frozen=True)
class V4GraphSource:
    """Where the official Transformer definition was loaded from."""

    path: Path
    kind: str  # "vendor" | "env" | "hf_inference"


def resolve_v4_graph_source(
    paths: Optional[DeepseekV4Paths] = None,
) -> V4GraphSource:
    """Pick graph root without importing the model module yet."""
    vendor = vendored_deepseek_infer_dir()
    if (vendor / "model.py").is_file():
        return V4GraphSource(path=vendor.resolve(), kind="vendor")

    paths = paths or resolve_v4_paths()
    env_infer = paths.inference_dir
    if (env_infer / "model.py").is_file():
        kind = "env" if paths.inference_dir != paths.hf_ckpt / "inference" else "hf_inference"
        return V4GraphSource(path=env_infer.resolve(), kind=kind)

    raise FileNotFoundError(
        "No V4 model graph found. Copy official inference/model.py into "
        f"{vendor}/ (preferred vendor path) or set SGLANG_LITE_DSV4_INFER. "
        "See docs/vendor/SOURCES.md."
    )


def load_v4_flash(
    model_id: Optional[str] = None,
    *,
    paths: Optional[DeepseekV4Paths] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    device: Optional[str] = None,
    max_batch_size: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any], TpShardPlan, V4GraphSource]:
    """Load DeepSeek-V4-Flash weights + graph for one TP rank.

    Returns ``(model, config_dict, shard_plan, graph_source)``.
    """
    paths = paths or resolve_v4_paths(hf_ckpt=model_id)
    require_v4_flash(str(paths.hf_ckpt), "deepseek_v4")
    source = resolve_v4_graph_source(paths)
    # Ensure import uses the resolved tree (vendor preferred).
    import_official_inference(source.path)
    model, cfg, plan = load_v4_hybrid_model(
        paths=paths,
        rank=rank,
        world_size=world_size,
        device=device,
        max_batch_size=max_batch_size,
    )
    return model, cfg, plan, source
