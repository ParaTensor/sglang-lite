"""DeepSeek-V4-Flash dedicated hot path (product sole runner).

Live graph: ``vendor/deepseek_infer`` (official). SGLang slices under
``vendor/sglang_v4/reference`` are REFERENCE ONLY (no runtime import sglang).
"""

from __future__ import annotations

from .cuda_graph import V4DecodeAccelerator, cuda_graph_enabled
from .encode import encode_chat_messages
from .forward import decode_step, extract_logits, model_forward_logits, sample_token
from .identity import (
    is_deepseek_v4_flash_id,
    require_v4_flash,
    v4_only_enabled,
)
from .load import load_v4_flash, resolve_v4_graph_source

__all__ = [
    "is_deepseek_v4_flash_id",
    "require_v4_flash",
    "v4_only_enabled",
    "load_v4_flash",
    "resolve_v4_graph_source",
    "extract_logits",
    "model_forward_logits",
    "sample_token",
    "decode_step",
    "encode_chat_messages",
    "V4DecodeAccelerator",
    "cuda_graph_enabled",
]
