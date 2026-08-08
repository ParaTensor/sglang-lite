"""Lean MoE decode using HF *weights* without HF CausalLM.forward.

Bypasses ``create_causal_mask``, DynamicCache, and CausalLM output packaging —
the main host/Python tax on the paged decode path. Attention still uses the
FlashInfer paged hooks already attached to ``layer.self_attn``; MoE stays on
the HF ``layer.mlp`` (batched_mm or cutlass fused when hooked).

Scope: B=1, q_len=1 decode for Qwen3-MoE-shaped modules only. Prefill and
non-matching architectures keep the full HF forward.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


def native_decode_enabled(default: str = "1") -> bool:
    """Default on for radix-native paged path; set ``SGLANG_LITE_NATIVE_DECODE=0`` to disable."""
    return os.environ.get("SGLANG_LITE_NATIVE_DECODE", default).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_qwen3_moe_shaped(model: Any) -> bool:
    inner = getattr(model, "model", None)
    if inner is None:
        return False
    need = ("embed_tokens", "layers", "norm", "rotary_emb")
    if any(not hasattr(inner, a) for a in need):
        return False
    if not hasattr(model, "lm_head"):
        return False
    layers = getattr(inner, "layers", None)
    if not layers or len(layers) == 0:
        return False
    layer0 = layers[0]
    for a in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
        if not hasattr(layer0, a):
            return False
    return True


class PagedNativeDecode:
    """B=1 paged decode body: embed → layers → norm → lm_head (logits only)."""

    def __init__(self, model: Any):
        if not _is_qwen3_moe_shaped(model):
            raise ValueError("model is not Qwen3-MoE-shaped for native decode")
        self.model = model
        self.inner = model.model
        self.embed_tokens: nn.Module = self.inner.embed_tokens
        self.layers: Sequence[nn.Module] = self.inner.layers
        self.norm: nn.Module = self.inner.norm
        self.rotary_emb: nn.Module = self.inner.rotary_emb
        self.lm_head: nn.Module = model.lm_head
        # Optional tied embeddings
        self._n_layers = int(
            getattr(model.config, "num_hidden_layers", len(self.layers))
        )

    @torch.inference_mode()
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits ``[B, S, V]`` for the given tokens (decode: B=1,S=1).

        Requires FlashInfer ``begin_forward`` already set so paged attn hooks run.
        """
        hidden = self.embed_tokens(input_ids)
        # RoPE: same API as HF Qwen3MoeRotaryEmbedding (x, position_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)

        for layer in self.layers[: self._n_layers]:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            # Monkeypatched self_attn uses FI paged KV when KernelBackend ctx is set.
            attn_out = layer.self_attn(
                hidden_states=hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
            )
            if isinstance(attn_out, (tuple, list)):
                hidden = attn_out[0]
            else:
                hidden = attn_out
            hidden = residual + hidden

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def forward_outputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> SimpleNamespace:
        """HF-output-shaped wrapper (``.logits`` only)."""
        return SimpleNamespace(logits=self.forward_logits(input_ids, position_ids))


def try_build_paged_native(model: Any) -> Optional[PagedNativeDecode]:
    if not _is_qwen3_moe_shaped(model):
        return None
    try:
        return PagedNativeDecode(model)
    except Exception as e:
        print(f"[sglang-lite] native decode unavailable: {e}")
        return None
