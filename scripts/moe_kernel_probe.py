#!/usr/bin/env python3
"""Probe MoE leaf backends (cutlass / trtllm / sgl) on target GPU.

Reports:
  - import / capability status
  - microbench ms/layer for synthetic Qwen3-30B-A3B shapes
  - optional e2e thruput via moe_thruput_probe env (caller)

  python scripts/moe_kernel_probe.py
  python scripts/moe_kernel_probe.py --e2e --model ~/models/Qwen3-30B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _bench(fn, iters: int = 100, warmup: int = 20) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def probe_status() -> Dict[str, Any]:
    _ensure_path()
    from sglang_lite.moe_hooks import (
        cutlass_fused_moe_available,
        resolve_moe_backend,
        sgl_kernel_moe_available,
        trtllm_bf16_moe_available,
    )
    import torch

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    sgl_ok, sgl_why = sgl_kernel_moe_available()
    trt_ok, trt_why = trtllm_bf16_moe_available()
    chosen, note = resolve_moe_backend("auto")
    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": cap,
        "torch": torch.__version__,
        "cutlass": cutlass_fused_moe_available(),
        "trtllm": {"ok": trt_ok, "why": trt_why},
        "sgl": {"ok": sgl_ok, "why": sgl_why},
        "auto_resolve": {"backend": chosen, "note": note},
    }


def microbench_cutlass(H: int, I: int, E: int, K: int, layers: int = 48) -> Dict[str, Any]:
    import torch
    import flashinfer.fused_moe as fm

    g1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
    g, u = g1.chunk(2, dim=1)
    g1 = torch.cat([u, g], dim=1).contiguous()
    g2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16).contiguous()
    h = torch.randn(1, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, E, (1, K), device="cuda", dtype=torch.int32)
    sc = torch.softmax(torch.randn(1, K, device="cuda", dtype=torch.float32), dim=-1)

    def run():
        return fm.cutlass_fused_moe(
            h,
            idx,
            sc,
            g1,
            g2,
            torch.bfloat16,
            quant_scales=[],
            activation_type=fm.ActivationType.Swiglu,
            tune_max_num_tokens=1,
        )

    ms = _bench(run)
    return {
        "backend": "cutlass",
        "ms_per_layer": round(ms, 4),
        "ms_48_layers": round(ms * layers, 3),
        "moe_only_tok_s_ceiling": round(1000.0 / (ms * layers), 1),
        "shape": {"H": H, "I": I, "E": E, "K": K},
    }


def microbench_trtllm(H: int, I: int, E: int, K: int, layers: int = 48) -> Dict[str, Any]:
    import torch
    import flashinfer.fused_moe as fm
    from sglang_lite.moe_hooks import (
        _convert_bf16_to_trtllm_block_layout,
        _pack_topk_ids,
        trtllm_bf16_moe_available,
    )

    ok, why = trtllm_bf16_moe_available()
    if not ok:
        return {"backend": "trtllm", "ok": False, "error": why}

    g1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
    g2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
    try:
        g1b, g2b = _convert_bf16_to_trtllm_block_layout(g1, g2, is_gated=True)
    except Exception as e:
        return {"backend": "trtllm", "ok": False, "error": f"convert: {e}"}

    h = torch.randn(1, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, E, (1, K), device="cuda", dtype=torch.int32)
    sc = torch.softmax(torch.randn(1, K, device="cuda", dtype=torch.float32), dim=-1)
    packed = _pack_topk_ids(idx, sc).contiguous()

    def run():
        return fm.trtllm_bf16_routed_moe(
            packed,
            h,
            g1b,
            g2b,
            num_experts=E,
            top_k=K,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            use_shuffled_weight=True,
            weight_layout=int(fm.WeightLayout.BlockMajorK),
            activation_type=int(fm.ActivationType.Swiglu),
            do_finalize=True,
        )

    try:
        # one smoke
        run()
        torch.cuda.synchronize()
        ms = _bench(run, iters=50, warmup=10)
        return {
            "backend": "trtllm",
            "ok": True,
            "ms_per_layer": round(ms, 4),
            "ms_48_layers": round(ms * layers, 3),
            "moe_only_tok_s_ceiling": round(1000.0 / (ms * layers), 1),
            "shape": {"H": H, "I": I, "E": E, "K": K},
        }
    except Exception as e:
        return {"backend": "trtllm", "ok": False, "error": str(e)[:400]}


def microbench_sgl() -> Dict[str, Any]:
    from sglang_lite.moe_hooks import sgl_kernel_moe_available

    ok, why = sgl_kernel_moe_available()
    if not ok:
        return {
            "backend": "sgl",
            "ok": False,
            "error": why,
            "note": "BF16 drop-in fused MoE not exported; quant/w4a8/marlin only. "
            "Import also needs torch ABI match (2.9.x in SGLang docker).",
        }
    # If import works, still no BF16 fused MoE leaf — report helpers only.
    import sgl_kernel

    helpers = [
        n
        for n in dir(sgl_kernel)
        if "moe" in n.lower() and not n.startswith("_")
    ]
    return {
        "backend": "sgl",
        "ok": True,
        "bf16_fused_moe_dropin": False,
        "exports_with_moe": helpers[:30],
        "note": "No simple BF16 experts.forward replacement; use cutlass for Qwen3 bf16.",
    }


def run_e2e(model: str, backend: str, out: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["SGLANG_LITE_RADIX_NATIVE"] = "1"
    env["SGLANG_LITE_FUSED_MOE"] = "1"
    env["SGLANG_LITE_MOE_BACKEND"] = backend
    env["SGLANG_LITE_CUDA_GRAPH_DECODE"] = "1"
    env["SGLANG_LITE_NATIVE_DECODE"] = "1"
    env["SGLANG_LITE_TORCH_COMPILE"] = "0"
    env["SGLANG_LITE_EXPERTS_IMPL"] = "batched_mm"
    env["SGLANG_LITE_DECODE_BURST"] = "128"
    env["SGLANG_LITE_V4_DISABLE_FI_SPARSE"] = "1"
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "moe_thruput_probe.py"),
        "--model",
        model,
        "--device",
        "cuda",
        "--cases",
        "1x128",
        "--out",
        out,
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    row: Dict[str, Any] = {
        "backend": backend,
        "returncode": p.returncode,
        "out": out,
    }
    if p.returncode != 0:
        row["stderr_tail"] = (p.stderr or p.stdout)[-800:]
        return row
    try:
        data = json.loads(Path(out).read_text())
        case = data["cases"][0]
        row["tok_s_warm"] = case.get("tok_s_warm")
        row["tok_s_cold"] = case.get("tok_s_cold")
        row["sample_text"] = (case.get("sample_text") or "")[:60]
    except Exception as e:
        row["parse_error"] = str(e)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--I", type=int, default=768)
    ap.add_argument("--E", type=int, default=128)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="/tmp/moe_kernel_probe.json")
    args = ap.parse_args()

    _ensure_path()
    report: Dict[str, Any] = {"status": probe_status(), "micro": {}}
    print(json.dumps(report["status"], indent=2))

    if report["status"]["cuda"]:
        if report["status"]["cutlass"]:
            report["micro"]["cutlass"] = microbench_cutlass(args.H, args.I, args.E, args.K)
            print("cutlass micro", report["micro"]["cutlass"])
        report["micro"]["trtllm"] = microbench_trtllm(args.H, args.I, args.E, args.K)
        print("trtllm micro", report["micro"]["trtllm"])
        report["micro"]["sgl"] = microbench_sgl()
        print("sgl micro", report["micro"]["sgl"])

    if args.e2e:
        if not args.model:
            print("--e2e requires --model", file=sys.stderr)
            return 2
        report["e2e"] = []
        for b in ("cutlass", "trtllm", "sgl"):
            out = f"/tmp/thru_moe_{b}.json"
            print(f"=== e2e backend={b} ===")
            row = run_e2e(args.model, b, out)
            report["e2e"].append(row)
            print(row)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
