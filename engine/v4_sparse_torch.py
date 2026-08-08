"""SM120-friendly sparse attention without TileLang / FlashInfer pack tax.

Mirrors official ``kernel.sparse_attn`` semantics (gather top-k KV + online
softmax + learnable sink). Decode (q_len==1) uses a tight torch path; prefill
falls back to the caller-supplied official implementation.

This is intentional product code for PRO6000: FI SM120 sparse works numerically
but e2e was slower than TileLang due to per-step DSV4 pack. Torch gather+GEMM
on bf16 matches the official contract without cubin packing.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


@torch.inference_mode()
def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention (official TileLang contract).

    Parameters
    ----------
    q:
        ``[B, S, H, D]`` bf16/fp16
    kv:
        ``[B, T, D]`` bf16/fp16 (shared across heads; MLA latent)
    attn_sink:
        ``[H]`` float32 learnable sink logits
    topk_idxs:
        ``[B, S, K]`` int32; ``-1`` pads
    softmax_scale:
        usually ``1/sqrt(D)`` or model ``softmax_scale``
    """
    if q.dim() != 4 or kv.dim() != 3 or topk_idxs.dim() != 3:
        raise ValueError(
            f"bad ranks q={tuple(q.shape)} kv={tuple(kv.shape)} "
            f"topk={tuple(topk_idxs.shape)}"
        )
    b, s, h, d = q.shape
    if kv.shape[0] != b or kv.shape[2] != d:
        raise ValueError(f"kv {tuple(kv.shape)} incompatible with q {tuple(q.shape)}")
    if topk_idxs.shape[0] != b or topk_idxs.shape[1] != s:
        raise ValueError(
            f"topk {tuple(topk_idxs.shape)} incompatible with q {tuple(q.shape)}"
        )

    # Decode fast path: S==1
    if s == 1:
        return _sparse_attn_decode(q, kv, attn_sink, topk_idxs, float(softmax_scale))

    # Prefill: loop over S (small for short prompts) reusing decode kernel body.
    outs = []
    for i in range(s):
        qi = q[:, i : i + 1]
        ti = topk_idxs[:, i : i + 1]
        outs.append(_sparse_attn_decode(qi, kv, attn_sink, ti, float(softmax_scale)))
    return torch.cat(outs, dim=1)


def _sparse_attn_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """q: [B,1,H,D], topk: [B,1,K] → [B,1,H,D]."""
    b, _, h, d = q.shape
    idx = topk_idxs[:, 0]  # [B, K]
    valid = idx >= 0
    idx_safe = idx.clamp(min=0).to(dtype=torch.long)

    # gather KVs: [B, K, D]
    gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, d)
    gathered = torch.gather(kv, 1, gather_idx)

    qh = q[:, 0].float()  # [B, H, D]
    k = gathered.float()  # [B, K, D]
    # scores [B, H, K]
    scores = torch.matmul(qh, k.transpose(1, 2)) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    sink = attn_sink.to(device=q.device, dtype=torch.float32).view(1, h, 1)
    # max over keys and sink for numerical stability
    scores_max = scores.amax(dim=-1, keepdim=True)
    scores_max = torch.maximum(scores_max, sink)
    # all -inf row (empty topk) → scores_max may be -inf; protect
    scores_max = torch.where(
        torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max)
    )

    exp_s = torch.exp(scores - scores_max)
    exp_s = exp_s.masked_fill(~valid.unsqueeze(1), 0.0)
    exp_sink = torch.exp(sink - scores_max)
    denom = exp_s.sum(dim=-1, keepdim=True) + exp_sink
    attn = exp_s / denom.clamp_min(1e-9)
    out = torch.matmul(attn, k)  # [B, H, D]
    return out.unsqueeze(1).to(dtype=q.dtype)


def attach_torch_sparse_attn(
    *,
    window_size: int = 128,
    orig: Optional[Callable] = None,
) -> Callable:
    """Return a ``sparse_attn``-compatible function using torch decode path.

    Prefill (``q_len>1``) uses ``orig`` if provided, else torch prefill loop.
    """

    stats = {"torch": 0, "official": 0}

    def routed(q, kv, attn_sink, topk_idxs, softmax_scale):
        try:
            if q.shape[1] == 1:
                out = sparse_attn_torch(
                    q, kv, attn_sink, topk_idxs, float(softmax_scale)
                )
                stats["torch"] += 1
                return out
            if orig is not None:
                stats["official"] += 1
                return orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            out = sparse_attn_torch(q, kv, attn_sink, topk_idxs, float(softmax_scale))
            stats["torch"] += 1
            return out
        except Exception:
            if orig is not None:
                stats["official"] += 1
                return orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            raise

    routed._sglang_lite_stats = stats  # type: ignore[attr-defined]
    routed._sglang_lite_kind = "torch_sparse"  # type: ignore[attr-defined]
    return routed


def install_torch_sparse_on_official_modules(
    *,
    window_size: int = 128,
) -> bool:
    """Monkey-patch ``model.sparse_attn`` / ``kernel.sparse_attn`` → torch path."""
    try:
        import kernel as kernel_mod  # type: ignore
        import model as model_mod  # type: ignore
    except ImportError:
        return False
    orig = getattr(model_mod, "sparse_attn", None) or getattr(
        kernel_mod, "sparse_attn", None
    )
    if orig is None:
        return False
    routed = attach_torch_sparse_attn(window_size=window_size, orig=orig)
    kernel_mod.sparse_attn = routed
    model_mod.sparse_attn = routed
    return True
