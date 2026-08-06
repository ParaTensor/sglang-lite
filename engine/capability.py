"""Device capability probing and kernel routing tables (leaf selection only).

sglang-lite owns the routing decision; FlashInfer / sgl-kernel / DeepGEMM supply
leaf implementations. Never treat SM120 as SM100 — they are different families.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ArchFamily(str, Enum):
    UNKNOWN = "unknown"
    SM80 = "sm80"
    SM90 = "sm90"
    SM100 = "sm100"
    SM120 = "sm120"


class SparseMlaBackend(str, Enum):
    NONE = "none"
    FLASHINFER_MLA = "flashinfer_mla"  # standard BatchMLA (non-sparse)
    FLASHINFER_SPARSE_SM120 = "flashinfer_sparse_sm120"
    FLASHINFER_SPARSE_SM100 = "flashinfer_sparse_sm100"
    OFFICIAL_SPARSE_ATTN = "official_sparse_attn"


class MoeGemmBackend(str, Enum):
    NONE = "none"
    TORCH = "torch"
    FLASHINFER_B12X = "flashinfer_b12x"
    SGL_KERNEL_FP8 = "sgl_kernel_fp8"
    DEEP_GEMM = "deep_gemm"


def capability_to_arch_family(major: int, minor: int) -> ArchFamily:
    """Map CUDA (major, minor) to an architecture family.

    Uses family buckets rather than ``major == 10`` checks so SM100 (10.x)
    and SM120 (12.x) never collapse into each other.
    """
    if major == 8:
        return ArchFamily.SM80
    if major == 9:
        return ArchFamily.SM90
    if major == 10:
        return ArchFamily.SM100
    if major == 12:
        return ArchFamily.SM120
    return ArchFamily.UNKNOWN


def probe_cuda_arch_family(device: Optional[int] = None) -> Tuple[ArchFamily, Tuple[int, int]]:
    """Return (arch_family, (major, minor)) for a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return ArchFamily.UNKNOWN, (0, 0)
    idx = 0 if device is None else int(device)
    major, minor = torch.cuda.get_device_capability(idx)
    return capability_to_arch_family(major, minor), (major, minor)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class KernelCapabilities:
    """Declared leaf capabilities for the current process/device."""

    arch_family: ArchFamily
    cuda_capability: Tuple[int, int]
    flashinfer_version: Optional[str] = None
    has_sparse_mla_sm120: bool = False
    # Phase 1: optional numerical smoke (absmean>0). None = not run.
    sparse_mla_sm120_numerical_ok: Optional[bool] = None
    sparse_mla_sm120_absmean: Optional[float] = None
    has_b12x_moe: bool = False
    has_sgl_kernel: bool = False
    has_deep_gemm_sm120: bool = False

    @property
    def sparse_mla_backend(self) -> SparseMlaBackend:
        return select_sparse_mla_backend(self)

    @property
    def moe_gemm_backend(self) -> MoeGemmBackend:
        return select_moe_gemm_backend(self)


def select_sparse_mla_backend(caps: KernelCapabilities) -> SparseMlaBackend:
    """Pick sparse MLA leaf. SM120 must never fall through to SM100 TRTLLM.

    Product policy (sglang-lite): **official TileLang is the production path**.
    Hybrid loaders set ``SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`` so the FI leaf is
    never armed in normal serve/bench — even if this selector would return FI.
    FORCE is the only supported way to experiment with FI today.

    Selection for SM120 (capability table only; arming is separate):

    * Symbol missing → ``OFFICIAL_SPARSE_ATTN``
    * ``SGLANG_LITE_V4_FORCE_FI_SPARSE=1`` → ``FLASHINFER_SPARSE_SM120``
    * ``sparse_mla_sm120_numerical_ok is True`` → ``FLASHINFER_SPARSE_SM120``
      (eligible for leaf; Hybrid still requires DISABLE=0 + attach)
    * else → ``OFFICIAL_SPARSE_ATTN``
    """
    fam = caps.arch_family
    if fam == ArchFamily.SM120:
        if not caps.has_sparse_mla_sm120:
            return SparseMlaBackend.OFFICIAL_SPARSE_ATTN
        if _env_truthy("SGLANG_LITE_V4_FORCE_FI_SPARSE"):
            return SparseMlaBackend.FLASHINFER_SPARSE_SM120
        if caps.sparse_mla_sm120_numerical_ok is True:
            return SparseMlaBackend.FLASHINFER_SPARSE_SM120
        return SparseMlaBackend.OFFICIAL_SPARSE_ATTN
    if fam == ArchFamily.SM100:
        return SparseMlaBackend.FLASHINFER_SPARSE_SM100
    if fam in (ArchFamily.SM90, ArchFamily.SM80):
        return SparseMlaBackend.FLASHINFER_MLA
    return SparseMlaBackend.NONE


def select_moe_gemm_backend(caps: KernelCapabilities) -> MoeGemmBackend:
    """Prefer B12x / sgl-kernel on SM120; DeepGEMM only when probed OK."""
    if caps.arch_family == ArchFamily.SM120:
        if caps.has_deep_gemm_sm120:
            return MoeGemmBackend.DEEP_GEMM
        if caps.has_b12x_moe:
            return MoeGemmBackend.FLASHINFER_B12X
        if caps.has_sgl_kernel:
            return MoeGemmBackend.SGL_KERNEL_FP8
        return MoeGemmBackend.TORCH
    if caps.has_deep_gemm_sm120:
        return MoeGemmBackend.DEEP_GEMM
    if caps.has_sgl_kernel:
        return MoeGemmBackend.SGL_KERNEL_FP8
    if caps.has_b12x_moe:
        return MoeGemmBackend.FLASHINFER_B12X
    return MoeGemmBackend.TORCH


def probe_sm120_sparse_numerical(
    *,
    device: str = "cuda",
    absmean_eps: float = 1e-6,
) -> Dict[str, Any]:
    """Tiny numerical smoke for FlashInfer SM120 sparse MLA.

    Returns a dict with keys: ok, absmean, finite, error, shape.
    Does **not** import heavy model code — only FI symbol + random tensors.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "absmean": None,
        "finite": None,
        "error": None,
        "shape": None,
    }
    try:
        import torch
        import flashinfer.mla as mla  # type: ignore
    except Exception as e:
        result["error"] = f"import: {type(e).__name__}: {e}"
        return result

    if not torch.cuda.is_available():
        result["error"] = "cuda_unavailable"
        return result

    try:
        from .dsv4_kv_pack import pack_dsv4_kv_bf16, to_paged_hnd

        B, qlen, H = 1, 1, 8
        # Use real bf16 → 584 pack → **footer** paged layout (not random uint8).
        # Random raw bytes historically produced absmean=0 even when the kernel works.
        n_swa_tokens = 64  # one page
        q = torch.randn(B, qlen, H, 512, device=device, dtype=torch.bfloat16)
        kv_bf16 = torch.randn(n_swa_tokens, 512, device=device, dtype=torch.bfloat16)
        packed = pack_dsv4_kv_bf16(kv_bf16)
        swa = to_paged_hnd(packed)  # [1, 1, 64, 584] footer physical
        # Sequential valid indices into the single page; pad to legal topk=128 with -1.
        swa_idx = torch.full((B * qlen, 128), -1, device=device, dtype=torch.int32)
        swa_idx[0, :n_swa_tokens] = torch.arange(
            n_swa_tokens, device=device, dtype=torch.int32
        )
        swa_topk_lens = torch.full(
            (B * qlen,), n_swa_tokens, device=device, dtype=torch.int32
        )
        workspace = torch.empty(256 * 1024 * 1024, device=device, dtype=torch.uint8)
        sm_scale = 512 ** -0.5
        # Prefer kwargs (SM120 path); matches v4_sparse_mla / FI 0.6.16 signature.
        out = mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa,
            workspace_buffer=workspace,
            sparse_indices=swa_idx,
            compressed_kv_cache=None,
            bmm1_scale=float(sm_scale),
            bmm2_scale=1.0,
            kv_layout="HND",
            swa_topk_lens=swa_topk_lens,
        )
        of = out.detach().float()
        absmean = float(of.abs().mean().item())
        finite = bool(torch.isfinite(of).all().item())
        result["absmean"] = absmean
        result["finite"] = finite
        result["shape"] = list(out.shape)
        result["ok"] = finite and absmean > absmean_eps
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def probe_kernel_capabilities(
    device: str = "cuda",
    *,
    numerical_probe: Optional[bool] = None,
) -> KernelCapabilities:
    """Runtime probe of arch family and available leaf modules.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    numerical_probe:
        If True, run :func:`probe_sm120_sparse_numerical` when the SM120 sparse
        symbol exists. If None, honor env ``SGLANG_LITE_V4_FI_SPARSE_NUM_PROBE=1``.
        Default is **off** so import/load stays cheap; Phase 1 scripts opt in.
    """
    arch = ArchFamily.UNKNOWN
    cap = (0, 0)
    fi_ver: Optional[str] = None
    has_sm120_sparse = False
    has_b12x = False
    has_sgl = False
    has_dg = False
    num_ok: Optional[bool] = None
    num_absmean: Optional[float] = None

    if numerical_probe is None:
        numerical_probe = _env_truthy("SGLANG_LITE_V4_FI_SPARSE_NUM_PROBE")

    if device != "cpu":
        try:
            arch, cap = probe_cuda_arch_family()
        except Exception:
            pass

        try:
            import flashinfer

            fi_ver = getattr(flashinfer, "__version__", "?")
            try:
                import flashinfer.mla._sparse_mla_sm120  # noqa: F401

                has_sm120_sparse = True
            except Exception:
                has_sm120_sparse = False
            has_b12x = hasattr(flashinfer, "B12xMoEWrapper") or hasattr(
                flashinfer, "b12x_fused_moe"
            )
        except Exception:
            pass

        try:
            import sgl_kernel  # noqa: F401

            has_sgl = True
        except Exception:
            has_sgl = False

        # DeepGEMM SM120: only true after a successful tiny gemm probe.
        # Default False — deep_gemm 0.1.4 fails on sm_120.
        has_dg = False
        if arch != ArchFamily.SM120:
            try:
                import deep_gemm  # noqa: F401

                has_dg = True
            except Exception:
                has_dg = False

        if numerical_probe and arch == ArchFamily.SM120 and has_sm120_sparse:
            num = probe_sm120_sparse_numerical(
                device=device if device.startswith("cuda") else "cuda",
            )
            num_ok = bool(num.get("ok"))
            if num.get("absmean") is not None:
                num_absmean = float(num["absmean"])

    return KernelCapabilities(
        arch_family=arch,
        cuda_capability=cap,
        flashinfer_version=fi_ver,
        has_sparse_mla_sm120=has_sm120_sparse,
        sparse_mla_sm120_numerical_ok=num_ok,
        sparse_mla_sm120_absmean=num_absmean,
        has_b12x_moe=has_b12x,
        has_sgl_kernel=has_sgl,
        has_deep_gemm_sm120=has_dg,
    )
