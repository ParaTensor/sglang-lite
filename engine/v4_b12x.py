"""FlashInfer B12x MoE / dense GEMM capability for SM120.

Official DeepSeek-V4 Hybrid uses **MXFP4** weights:
  packed e2m1, **e8m0** block scales, **sf_vec=32**.

FlashInfer B12x (0.6.12) on SM120 expects **NVFP4**:
  e2m1 packed, **e4m3** scales, **sf_vec=16**, swizzled scale layout;
  ``B12xMoEWrapper`` also **rejects Expert Parallelism**
  (``num_local_experts != num_experts``).

Our production Hybrid path is **TP + EP** (e.g. TP8 → 32 local experts of 256).
Therefore B12x cannot be the default MoE backend without:
  1. full-expert (non-EP) deployment, and
  2. an offline NVFP4 repack of every expert weight + scale.

DeepGEMM SM120 (``v4_deep_gemm``) matches the official layout and is the
correct high-performance path. This module only probes / documents B12x so
we do not silently weld the wrong format.

Env:
  ``SGLANG_LITE_V4_B12X=0`` default — do not attach
  ``SGLANG_LITE_V4_B12X=1`` attempt experimental attach (will no-op unless
  weights are already NVFP4 and EP is disabled)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("sglang_lite.v4_b12x")


def b12x_env_enabled() -> bool:
    raw = os.environ.get("SGLANG_LITE_V4_B12X", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def probe_b12x() -> dict:
    """Import / construct smoke for B12xMoEWrapper (no weight run)."""
    out: dict = {
        "ok": False,
        "has_wrapper": False,
        "has_dense_gemm": False,
        "ep_supported": False,
        "layout": "nvfp4_e4m3_sf16",
        "official_layout": "mxfp4_e8m0_sf32",
        "compatible_with_hybrid_ep": False,
        "error": None,
        "flashinfer_version": None,
    }
    try:
        import flashinfer

        out["flashinfer_version"] = getattr(flashinfer, "__version__", "?")
        out["has_wrapper"] = hasattr(flashinfer, "B12xMoEWrapper") or hasattr(
            flashinfer, "b12x_fused_moe"
        )
        try:
            from flashinfer.gemm import Sm120B12xBlockScaledDenseGemmKernel  # noqa: F401

            out["has_dense_gemm"] = True
        except Exception:
            out["has_dense_gemm"] = False

        if out["has_wrapper"]:
            from flashinfer.fused_moe import B12xMoEWrapper

            # Construct tiny config; EP with num_local != num_experts raises.
            try:
                _ = B12xMoEWrapper(
                    num_experts=8,
                    top_k=2,
                    hidden_size=256,
                    intermediate_size=256,
                    num_local_experts=8,
                    use_cuda_graph=False,
                    quant_mode="nvfp4",
                )
                out["ok"] = True
            except Exception as e:
                out["error"] = f"construct: {type(e).__name__}: {e}"
            # Document EP limitation without raising.
            out["ep_supported"] = False
            out["compatible_with_hybrid_ep"] = False
            out["note"] = (
                "B12x needs NVFP4 (e4m3 sf16) + no EP; Hybrid uses MXFP4 e8m0 sf32 + EP. "
                "Use DeepGEMM (v4_deep_gemm) for official weights."
            )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def attach_v4_b12x(model: Any = None, *, world_size: int = 1) -> dict:
    """Attach B12x only when explicitly enabled and layout/EP allow it.

    Default: report probe and leave MoE on DeepGEMM / TileLang.
    """
    stats = {
        "enabled": False,
        "attached": False,
        "probe": probe_b12x(),
        "reason": None,
    }
    if not b12x_env_enabled():
        stats["reason"] = "env_off"
        print(
            "[sglang-lite] v4 B12x idle (SGLANG_LITE_V4_B12X=0); "
            "official Hybrid uses DeepGEMM for MXFP4 e8m0 — B12x needs NVFP4+no-EP"
        )
        return stats

    if world_size > 1:
        stats["reason"] = "ep_unsupported"
        print(
            f"[sglang-lite] v4 B12x skipped: world_size={world_size} implies EP; "
            "B12xMoEWrapper requires num_local_experts == num_experts"
        )
        return stats

    # Experimental full-attach path reserved — weights still wrong format.
    stats["reason"] = "layout_mismatch_mxfp4_vs_nvfp4"
    print(
        "[sglang-lite] v4 B12x not attached: Hybrid checkpoint is MXFP4 e8m0/sf32; "
        "B12x expects NVFP4 e4m3/sf16. Convert offline before enabling."
    )
    logger.info("B12x attach refused: %s", stats)
    return stats


__all__ = ["attach_v4_b12x", "b12x_env_enabled", "probe_b12x"]
