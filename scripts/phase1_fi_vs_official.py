#!/usr/bin/env python3
"""Path A: capture real official sparse_attn tensors and compare vs FlashInfer SM120.

Loads Hybrid V4 (torchrun TP), installs a capture+compare hook on the first
*decode* (q_len==1) sparse_attn call, then:

  1. runs official TileLang sparse_attn  → ref
  2. packs / builds FI kwargs (several variants)
  3. runs trtllm_batch_decode_sparse_mla_dsv4
  4. reports absmean / max_abs / max_diff vs ref

Usage (PRO6000 host)::

  source ~/venvs/sglang-lite/bin/activate
  export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
  export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
  export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1
  export SGLANG_LITE_FI_PREFIX=/tmp/fi1616 FLASHINFER_DISABLE_VERSION_CHECK=1
  torchrun --nproc-per-node=8 scripts/phase1_fi_vs_official.py \\
    --max-new 4 --out ~/bench/phase1_fi_vs_official.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _maybe_fi_prefix() -> str:
    prefix = os.environ.get("SGLANG_LITE_FI_PREFIX", "")
    if prefix and Path(prefix).is_dir():
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        if prefix not in sys.path:
            sys.path.insert(0, prefix)
        # Patch incomplete jit_cache package (same as phase1_kernel_probe).
        jc = Path(prefix) / "flashinfer_jit_cache" / "__init__.py"
        if jc.is_file():
            text = jc.read_text(encoding="utf-8", errors="replace")
            if "get_jit_cache_dir" not in text:
                jc.write_text(
                    text
                    + (
                        "\ndef get_jit_cache_dir():\n"
                        "    import pathlib\n"
                        '    return pathlib.Path(__file__).resolve().parent / "jit_cache"\n'
                    ),
                    encoding="utf-8",
                )
            if "__version__" not in text:
                jc.write_text(
                    jc.read_text(encoding="utf-8")
                    + '\n__version__ = "0.6.16.post1+cu130"\n',
                    encoding="utf-8",
                )
    return prefix


def _stats(t: Optional["torch.Tensor"]) -> Dict[str, Any]:
    if t is None:
        return {"none": True}
    import torch

    f = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "absmean": float(f.abs().mean().item()),
        "absmax": float(f.abs().max().item()),
        "finite": bool(torch.isfinite(f).all().item()),
        "numel": int(t.numel()),
    }


def _pad_topk(
    idx: "torch.Tensor", lens: "torch.Tensor", target: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Pad last dim of indices to ``target`` with -1; keep lengths unchanged."""
    import torch

    if idx.shape[-1] == target:
        return idx, lens
    if idx.shape[-1] > target:
        # Truncate (should not happen for SWA=128).
        return idx[..., :target].contiguous(), torch.clamp(lens, max=target)
    pad = target - idx.shape[-1]
    padded = torch.nn.functional.pad(idx, (0, pad), value=-1)
    return padded.contiguous(), lens


def _nearest_legal_topk(k: int) -> int:
    for cand in (128, 512, 1024, 2048):
        if k <= cand:
            return cand
    return 2048


def build_fi_kwargs_variants(
    q, kv, attn_sink, topk_idxs, softmax_scale, *, window_size: int = 128
) -> List[Tuple[str, Dict[str, Any]]]:
    """Build several FI call kwargs from official tensors."""
    import torch

    from sglang_lite.dsv4_kv_pack import (
        DSV4_PAGE_SIZE,
        pack_dsv4_kv_bf16,
        split_swa_compress_indices,
        to_paged_hnd,
    )

    try:
        import kernel as official_kernel  # type: ignore

        act_quant = official_kernel.act_quant
    except Exception:
        act_quant = None

    b, s, h, d = q.shape
    assert s == 1
    t = kv.shape[1]
    win = min(window_size, t)
    swa_tokens = kv[:, :win, :]
    comp_tokens = kv[:, win:, :] if t > win else kv[:, :0, :]

    # Split using *config* window_size columns (official first 128 = SWA).
    swa_k_cols = min(window_size, topk_idxs.shape[-1])
    swa_idx, swa_lens, comp_idx, comp_lens = split_swa_compress_indices(
        topk_idxs, window_size=swa_k_cols
    )

    # Only pack filled SWA ring + non-empty compress prefix for lighter trials;
    # "full_cache" packs entire kv_cache row (matches production hook).
    variants: List[Tuple[str, Dict[str, Any]]] = []

    def _pack_pages(tokens_2d, act_fn):
        if tokens_2d.numel() == 0:
            return None
        flat = tokens_2d.reshape(-1, d)
        packed = pack_dsv4_kv_bf16(flat, act_quant_fn=act_fn)
        return to_paged_hnd(packed, page_size=DSV4_PAGE_SIZE)

    workspace = torch.empty(256 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    sinks = attn_sink
    if sinks is not None and sinks.dim() == 1:
        sinks = sinks.to(dtype=torch.float32)

    # --- variant recipes ---
    packs = [
        ("act_quant", act_quant),
        ("torch_fp8", None),
    ]
    for pack_name, act_fn in packs:
        if pack_name == "act_quant" and act_fn is None:
            continue
        swa_pages = _pack_pages(swa_tokens, act_fn)
        comp_pages = (
            _pack_pages(comp_tokens, act_fn)
            if comp_tokens.numel() > 0 and comp_idx.numel() > 0
            else None
        )

        # A0: current production-style (no topk pad)
        variants.append(
            (
                f"{pack_name}/as_is",
                dict(
                    query=q.contiguous(),
                    swa_kv_cache=swa_pages.contiguous(),
                    workspace_buffer=workspace,
                    sparse_indices=swa_idx.contiguous(),
                    compressed_kv_cache=comp_pages,
                    bmm1_scale=float(softmax_scale),
                    bmm2_scale=1.0,
                    sinks=sinks,
                    kv_layout="HND",
                    swa_topk_lens=swa_lens.contiguous(),
                    extra_sparse_indices=(
                        comp_idx.contiguous() if comp_idx.numel() else None
                    ),
                    extra_sparse_topk_lens=(
                        comp_lens.contiguous() if comp_idx.numel() else None
                    ),
                ),
            )
        )

        # A1: pad SWA→128, compress→legal topk with -1
        swa_pad, swa_lens_p = _pad_topk(swa_idx, swa_lens, 128)
        if comp_idx.numel():
            ck = _nearest_legal_topk(comp_idx.shape[-1])
            comp_pad, comp_lens_p = _pad_topk(comp_idx, comp_lens, ck)
        else:
            comp_pad, comp_lens_p = None, None
        variants.append(
            (
                f"{pack_name}/pad_legal_topk",
                dict(
                    query=q.contiguous(),
                    swa_kv_cache=swa_pages.contiguous(),
                    workspace_buffer=workspace,
                    sparse_indices=swa_pad,
                    compressed_kv_cache=comp_pages if comp_pad is not None else None,
                    bmm1_scale=float(softmax_scale),
                    bmm2_scale=1.0,
                    sinks=sinks,
                    kv_layout="HND",
                    swa_topk_lens=swa_lens_p.contiguous(),
                    extra_sparse_indices=comp_pad,
                    extra_sparse_topk_lens=(
                        comp_lens_p.contiguous() if comp_lens_p is not None else None
                    ),
                ),
            )
        )

        # A2: SWA only (ignore compress)
        variants.append(
            (
                f"{pack_name}/swa_only_pad128",
                dict(
                    query=q.contiguous(),
                    swa_kv_cache=swa_pages.contiguous(),
                    workspace_buffer=workspace,
                    sparse_indices=swa_pad,
                    compressed_kv_cache=None,
                    bmm1_scale=float(softmax_scale),
                    bmm2_scale=1.0,
                    sinks=sinks,
                    kv_layout="HND",
                    swa_topk_lens=swa_lens_p.contiguous(),
                    extra_sparse_indices=None,
                    extra_sparse_topk_lens=None,
                ),
            )
        )

        # A3: no sinks
        variants.append(
            (
                f"{pack_name}/pad_legal_no_sinks",
                dict(
                    query=q.contiguous(),
                    swa_kv_cache=swa_pages.contiguous(),
                    workspace_buffer=workspace,
                    sparse_indices=swa_pad,
                    compressed_kv_cache=comp_pages if comp_pad is not None else None,
                    bmm1_scale=float(softmax_scale),
                    bmm2_scale=1.0,
                    sinks=None,
                    kv_layout="HND",
                    swa_topk_lens=swa_lens_p.contiguous(),
                    extra_sparse_indices=comp_pad,
                    extra_sparse_topk_lens=(
                        comp_lens_p.contiguous() if comp_lens_p is not None else None
                    ),
                ),
            )
        )

    # A4: bf16 512 HND pages (FI allows non-packed bf16 on SM120 path)
    def _bf16_pages(tokens):
        if tokens.numel() == 0:
            return None
        flat = tokens.reshape(-1, d).contiguous()
        tlen = flat.shape[0]
        pad = (DSV4_PAGE_SIZE - tlen % DSV4_PAGE_SIZE) % DSV4_PAGE_SIZE
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad, d)], dim=0)
        n = flat.shape[0] // DSV4_PAGE_SIZE
        return flat.view(n, 1, DSV4_PAGE_SIZE, d)

    swa_bf = _bf16_pages(swa_tokens)
    comp_bf = (
        _bf16_pages(comp_tokens)
        if comp_tokens.numel() > 0 and comp_idx.numel() > 0
        else None
    )
    swa_pad, swa_lens_p = _pad_topk(swa_idx, swa_lens, 128)
    if comp_idx.numel():
        ck = _nearest_legal_topk(comp_idx.shape[-1])
        comp_pad, comp_lens_p = _pad_topk(comp_idx, comp_lens, ck)
    else:
        comp_pad, comp_lens_p = None, None
    variants.append(
        (
            "bf16_512/pad_legal_topk",
            dict(
                query=q.contiguous(),
                swa_kv_cache=swa_bf.contiguous(),
                workspace_buffer=workspace,
                sparse_indices=swa_pad,
                compressed_kv_cache=comp_bf if comp_pad is not None else None,
                bmm1_scale=float(softmax_scale),
                bmm2_scale=1.0,
                sinks=sinks,
                kv_layout="HND",
                swa_topk_lens=swa_lens_p.contiguous(),
                extra_sparse_indices=comp_pad,
                extra_sparse_topk_lens=(
                    comp_lens_p.contiguous() if comp_lens_p is not None else None
                ),
            ),
        )
    )

    return variants


def compare_fi(ref, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import flashinfer.mla as mla  # type: ignore

    out: Dict[str, Any] = {"ok": False}
    try:
        fi_out = mla.trtllm_batch_decode_sparse_mla_dsv4(**kwargs)
        if fi_out.dim() == 3:
            fi_out = fi_out.view_as(ref)
        out["fi"] = _stats(fi_out)
        diff = (fi_out.float() - ref.float()).abs()
        out["max_abs_diff"] = float(diff.max().item())
        out["mean_abs_diff"] = float(diff.mean().item())
        out["ok"] = bool(
            out["fi"]["finite"] and out["fi"]["absmean"] > 1e-6
        )
        out["close_1e2"] = bool(torch.allclose(fi_out.float(), ref.float(), atol=1e-2, rtol=1e-2))
        out["close_5e1"] = bool(torch.allclose(fi_out.float(), ref.float(), atol=0.5, rtol=0.1))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


def install_capture_hook(window_size: int = 128) -> Dict[str, Any]:
    """Replace official sparse_attn with capture+compare on first decode."""
    import kernel as kernel_mod  # type: ignore
    import model as model_mod  # type: ignore
    import torch

    orig = getattr(model_mod, "sparse_attn", None) or getattr(
        kernel_mod, "sparse_attn", None
    )
    if orig is None:
        raise RuntimeError("official sparse_attn not found")

    state: Dict[str, Any] = {
        "captured": False,
        "compare": None,
        "meta": {},
        "calls": {"prefill": 0, "decode": 0},
    }

    def routed(q, kv, attn_sink, topk_idxs, softmax_scale):
        is_decode = q.shape[1] == 1
        if is_decode:
            state["calls"]["decode"] += 1
        else:
            state["calls"]["prefill"] += 1
            return orig(q, kv, attn_sink, topk_idxs, softmax_scale)

        ref = orig(q, kv, attn_sink, topk_idxs, softmax_scale)
        if state["captured"]:
            return ref

        state["captured"] = True
        # Clone for analysis (detach from graph).
        qc = q.detach().contiguous()
        kvc = kv.detach().contiguous()
        sinkc = attn_sink.detach().contiguous() if attn_sink is not None else None
        topkc = topk_idxs.detach().contiguous()
        refc = ref.detach().contiguous()

        from sglang_lite.dsv4_kv_pack import split_swa_compress_indices

        swa_idx, swa_lens, comp_idx, comp_lens = split_swa_compress_indices(
            topkc, window_size=min(window_size, topkc.shape[-1])
        )
        meta = {
            "q": _stats(qc),
            "kv": _stats(kvc),
            "attn_sink": _stats(sinkc),
            "topk_idxs": {
                **_stats(topkc),
                "valid_frac": float((topkc >= 0).float().mean().item()),
                "min": int(topkc.min().item()),
                "max": int(topkc.max().item()),
                "swa_cols": min(window_size, topkc.shape[-1]),
                "comp_cols": max(0, topkc.shape[-1] - window_size),
            },
            "swa_idx": _stats(swa_idx),
            "swa_lens": {
                "min": int(swa_lens.min().item()),
                "max": int(swa_lens.max().item()),
                "mean": float(swa_lens.float().mean().item()),
            },
            "comp_idx": _stats(comp_idx) if comp_idx.numel() else None,
            "comp_lens": (
                {
                    "min": int(comp_lens.min().item()),
                    "max": int(comp_lens.max().item()),
                    "mean": float(comp_lens.float().mean().item()),
                }
                if comp_idx.numel()
                else None
            ),
            "softmax_scale": float(softmax_scale),
            "window_size": window_size,
            "ref": _stats(refc),
            "fi_version": None,
        }
        try:
            import flashinfer

            meta["fi_version"] = getattr(flashinfer, "__version__", "?")
        except Exception as e:
            meta["fi_import_error"] = f"{type(e).__name__}: {e}"

        results = []
        if meta.get("fi_version"):
            for name, kwargs in build_fi_kwargs_variants(
                qc, kvc, sinkc, topkc, float(softmax_scale), window_size=window_size
            ):
                # Record shapes of key inputs
                rec = {
                    "variant": name,
                    "swa_kv_shape": list(kwargs["swa_kv_cache"].shape),
                    "swa_kv_dtype": str(kwargs["swa_kv_cache"].dtype),
                    "sparse_indices_shape": list(kwargs["sparse_indices"].shape),
                    "swa_topk_lens": [
                        int(x) for x in kwargs["swa_topk_lens"].tolist()
                    ],
                    "extra_idx_shape": (
                        list(kwargs["extra_sparse_indices"].shape)
                        if kwargs.get("extra_sparse_indices") is not None
                        else None
                    ),
                    "extra_topk_lens": (
                        [int(x) for x in kwargs["extra_sparse_topk_lens"].tolist()]
                        if kwargs.get("extra_sparse_topk_lens") is not None
                        else None
                    ),
                    "compressed_shape": (
                        list(kwargs["compressed_kv_cache"].shape)
                        if kwargs.get("compressed_kv_cache") is not None
                        else None
                    ),
                }
                rec.update(compare_fi(refc, kwargs))
                results.append(rec)
                # free intermediate if any large — workspace shared
        else:
            results.append({"variant": "none", "error": "flashinfer not importable"})

        state["meta"] = meta
        state["compare"] = results
        # Keep serving correct tokens via official path.
        return ref

    kernel_mod.sparse_attn = routed
    model_mod.sparse_attn = routed
    state["orig"] = orig
    return state


def main() -> int:
    # CVD remap before torch (TileLang device0).
    if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--hf-ckpt",
        default=os.environ.get("SGLANG_LITE_DSV4_HF", ""),
    )
    ap.add_argument(
        "--converted",
        default=os.environ.get("SGLANG_LITE_DSV4_CONVERTED", ""),
    )
    args = ap.parse_args()

    # Prefer system CUDA.
    if Path("/usr/local/cuda/bin").is_dir():
        os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    if Path("/usr/local/cuda/include").is_dir():
        os.environ["CPATH"] = "/usr/local/cuda/include"

    fi_prefix = _maybe_fi_prefix()
    # Always keep official path for generation; we only compare offline in hook.
    os.environ["SGLANG_LITE_V4_DISABLE_FI_SPARSE"] = "1"

    repo = Path(__file__).resolve().parents[1]
    eng = repo / "engine"
    for p in (eng, repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.hf_ckpt:
        os.environ["SGLANG_LITE_DSV4_HF"] = args.hf_ckpt
    if args.converted:
        os.environ["SGLANG_LITE_DSV4_CONVERTED"] = args.converted

    from sglang_lite import LiteEngine

    t0 = time.perf_counter()
    engine = LiteEngine(
        model_name=str(args.hf_ckpt or os.environ.get("SGLANG_LITE_DSV4_HF", "")),
        device="cuda",
        max_batch_size=1,
        start_loop=False,
    )
    assert getattr(engine.runner, "_v4_hybrid", False)

    win = 128
    try:
        cfg = getattr(engine.runner, "model_config", None) or {}
        if isinstance(cfg, dict):
            win = int(cfg.get("window_size", cfg.get("sliding_window", 128)))
    except Exception:
        pass

    # Install after Hybrid load so model/kernel modules exist.
    state = install_capture_hook(window_size=win)
    if rank == 0:
        print(f"[fi-vs-off] hook installed window={win} fi_prefix={fi_prefix or None}")

    # Encode prompt (prefer official chat encoding like v4_lite_engine_gen).
    hf = Path(os.environ.get("SGLANG_LITE_DSV4_HF", args.hf_ckpt))
    prompt_ids: list[int]
    encoding_dir = hf / "encoding"
    infer_dir = hf / "inference"
    if encoding_dir.is_dir():
        for p in (str(encoding_dir), str(infer_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from encoding_dsv4 import encode_messages

            rendered = encode_messages(
                [{"role": "user", "content": args.prompt}], thinking_mode="chat"
            )
            prompt_ids = engine.runner.tokenizer.encode(rendered)
        except Exception as e:
            if rank == 0:
                print(f"[fi-vs-off] encoding_dsv4 fallback: {e}")
            prompt_ids = engine.runner.tokenize(args.prompt)
    else:
        prompt_ids = engine.runner.tokenize(args.prompt)

    if rank == 0:
        print(f"[fi-vs-off] prompt_tokens={len(prompt_ids)} max_new={args.max_new}")

    # All ranks must participate in every forward (start_loop=False).
    reqs = [
        {
            "request_id": "fi-vs-off-0",
            "input_ids": list(prompt_ids),
            "max_tokens": args.max_new,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    ]
    outs = engine.generate_batch(reqs, timeout_s=600.0)
    if rank == 0 and outs:
        print(f"[fi-vs-off] sample_text={outs[0].get('text', '')[:120]!r}")

    if dist.is_initialized():
        dist.barrier()

    summary: Dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "load_s": round(time.perf_counter() - t0, 3),
        "fi_prefix": fi_prefix or None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt": args.prompt,
        "max_new": args.max_new,
        "window_size": win,
        "calls": state["calls"],
        "captured": state["captured"],
        "meta": state.get("meta"),
        "compare": state.get("compare"),
    }

    # Only rank0 writes / prints full JSON
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if rank == 0:
        print(text)
        if args.out:
            Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
            print(f"[fi-vs-off] wrote {args.out}")

        # Verdict
        compares = state.get("compare") or []
        any_ok = any(c.get("ok") for c in compares)
        any_close = any(c.get("close_1e2") or c.get("close_5e1") for c in compares)
        print(
            f"[fi-vs-off] verdict any_nonzero={any_ok} any_close={any_close} "
            f"n_variants={len(compares)}"
        )

    engine.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
