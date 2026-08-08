"""Faster MoE dispatch for official DeepSeek-V4 Hybrid (decode-friendly).

Official ``MoE.forward`` loops **all local experts** with ``bincount`` +
``torch.where`` (host syncs). For decode (few tokens × top-k), almost all
experts are idle — we only walk **activated** expert ids.

Also fuses act_quant for expert SwiGLU gate/up (same input × two FP4 GEMMs).

Does not import sglang/vllm. Optional deep_gemm / B12x hooks can plug into
``linear_fast`` later.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("sglang_lite.v4_moe_fast")


def moe_fast_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_MOE_FAST", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _expert_forward_fused(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
    """Expert SwiGLU with single act_quant for FP4 w1/w3."""
    dtype = x.dtype
    w1 = self.w1.weight
    w3 = self.w3.weight
    # FP4 experts: quantize activation once for both gate and up projections.
    if w1.dtype == torch.float4_e2m1fn_x2:
        import kernel as K  # type: ignore  # vendor on sys.path

        xq, s = K.act_quant(x, K.block_size, K.scale_fmt, K.scale_dtype)
        gate = K.fp4_gemm(xq, s, w1, w1.scale, K.scale_dtype).float()
        up = K.fp4_gemm(xq, s, w3, w3.scale, K.scale_dtype).float()
    elif w1.dtype == torch.float8_e4m3fn:
        import kernel as K  # type: ignore

        xq, s = K.act_quant(x, K.block_size, K.scale_fmt, K.scale_dtype)
        gate = K.fp8_gemm(xq, s, w1, w1.scale, K.scale_dtype).float()
        up = K.fp8_gemm(xq, s, w3, w3.scale, K.scale_dtype).float()
    else:
        gate = self.w1(x).float()
        up = self.w3(x).float()

    lim = float(getattr(self, "swiglu_limit", 0) or 0)
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return self.w2(h.to(dtype))


def _moe_forward_activated_only(self, x: torch.Tensor, input_ids: torch.Tensor):
    """Only run experts that appear in top-k indices (local shard)."""
    import model as M  # type: ignore  # vendor globals: world_size, dist, rank

    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, input_ids.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)

    # Unique activated experts (device → host once). Decode: ≤ topk (e.g. 6).
    uniq = torch.unique(indices)
    # Host list of a few ints — far fewer than n_local_experts (32).
    for i in uniq.tolist():
        ei = int(i)
        if ei < self.experts_start_idx or ei >= self.experts_end_idx:
            continue
        expert = self.experts[ei]
        if expert is None:
            continue
        tok_idx, top = torch.where(indices == ei)
        if tok_idx.numel() == 0:
            continue
        y[tok_idx] += expert(x[tok_idx], weights[tok_idx, top, None])

    if M.world_size > 1:
        M.dist.all_reduce(y)
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


def attach_v4_moe_fast(model: Any) -> dict:
    """Patch MoE/Expert methods on a loaded official Transformer.

    Returns stats dict with counts of patched modules.
    """
    if not moe_fast_enabled():
        return {"enabled": False, "moe": 0, "expert": 0}

    import model as M  # type: ignore

    MoE = M.MoE
    Expert = M.Expert

    n_moe = 0
    n_exp = 0
    # Bind optimized methods on classes (all instances).
    if getattr(MoE.forward, "_sglang_lite_moe_fast", False) is not True:
        MoE.forward = _moe_forward_activated_only
        MoE.forward._sglang_lite_moe_fast = True  # type: ignore[attr-defined]
        n_moe = 1
    if getattr(Expert.forward, "_sglang_lite_moe_fast", False) is not True:
        Expert.forward = _expert_forward_fused
        Expert.forward._sglang_lite_moe_fast = True  # type: ignore[attr-defined]
        n_exp = 1

    # Count modules for logging
    n_moe_mod = sum(1 for m in model.modules() if isinstance(m, MoE))
    n_exp_mod = sum(1 for m in model.modules() if isinstance(m, Expert))
    logger.info(
        "v4 MoE fast: patched MoE.forward + Expert.forward (mods moe=%s expert=%s)",
        n_moe_mod,
        n_exp_mod,
    )
    print(
        f"[sglang-lite] v4 MoE fast armed (activated-expert dispatch + fused act_quant); "
        f"moe_modules={n_moe_mod} expert_modules={n_exp_mod}"
    )
    return {
        "enabled": True,
        "moe_class_patched": n_moe,
        "expert_class_patched": n_exp,
        "moe_modules": n_moe_mod,
        "expert_modules": n_exp_mod,
    }
