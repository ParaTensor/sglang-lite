#!/usr/bin/env python3
"""TP Hybrid pure-decode thruput: DeepGEMM vs TileLang fp4_gemm.

  export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
  export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
  torchrun --nproc-per-node=8 scripts/v4_deep_gemm_thruput.py --max-new 96 --deep-gemm 1
  torchrun --nproc-per-node=8 scripts/v4_deep_gemm_thruput.py --max-new 96 --deep-gemm 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--deep-gemm", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    vendor = root / "engine" / "vendor" / "deepseek_infer"
    if not (vendor / "model.py").is_file():
        print("missing vendor deepseek_infer", file=sys.stderr)
        return 2
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(root / "engine"))
    os.environ["SGLANG_LITE_V4_DEEP_GEMM"] = "1" if args.deep_gemm else "0"
    os.environ.setdefault("SGLANG_LITE_V4_MOE_FAST", "1")

    # CCCL headers for TileLang cold compile on some hosts
    fi_cccl = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "flashinfer"
        / "data"
        / "cccl"
        / "libcudacxx"
        / "include"
    )
    incs = [p for p in (str(fi_cccl), "/usr/local/cuda/include") if Path(p).is_dir()]
    if incs:
        cur = os.environ.get("CPLUS_INCLUDE_PATH", "")
        os.environ["CPLUS_INCLUDE_PATH"] = ":".join(incs + ([cur] if cur else []))
        os.environ["CPATH"] = os.environ["CPLUS_INCLUDE_PATH"]

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from model import ModelArgs, Transformer
    import kernel as K

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    cfg = json.loads((vendor / "config.json").read_text(encoding="utf-8"))
    if isinstance(cfg.get("compress_ratios"), list):
        cfg["compress_ratios"] = tuple(cfg["compress_ratios"])
    names = {f.name for f in fields(ModelArgs)}
    margs = ModelArgs(**{k: v for k, v in cfg.items() if k in names})
    with torch.device("cuda"):
        model = Transformer(margs)
    ckpt = Path(os.environ["SGLANG_LITE_DSV4_CONVERTED"]).expanduser()
    load_model(model, str(ckpt / f"model{rank}-mp{world}.safetensors"), strict=False)
    model.eval()

    from v4_moe_fast import attach_v4_moe_fast

    attach_v4_moe_fast(model)
    if args.deep_gemm:
        from v4_deep_gemm import attach_v4_deep_gemm

        st = attach_v4_deep_gemm(model)
        if rank == 0:
            print("ATTACH", st)

    n_calls = [0]
    _orig = K.fp4_gemm

    def _count(*a, **k):
        n_calls[0] += 1
        return _orig(*a, **k)

    K.fp4_gemm = _count  # type: ignore[assignment]

    hf = Path(
        os.environ.get("SGLANG_LITE_DSV4_HF", os.path.expanduser("~/models/ds-v4-flash"))
    )
    tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
    prompt_ids = tok.encode(
        encode_messages(
            [{"role": "user", "content": args.prompt}], thinking_mode="chat"
        )
    )

    @torch.inference_mode()
    def prefill():
        tokens = list(prompt_ids)
        logits = model(torch.tensor([tokens], dtype=torch.long, device="cuda"))
        return tokens, logits

    @torch.inference_mode()
    def decode_n(tokens, logits, max_new: int) -> int:
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax().item())
            tokens.append(nxt)
            logits = model(torch.tensor([[nxt]], dtype=torch.long, device="cuda"))
        return max_new

    tokens, logits = prefill()
    decode_n(tokens, logits, 8)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()

    n_calls[0] = 0
    tokens, logits = prefill()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    t0 = time.perf_counter()
    n = decode_n(tokens, logits, args.max_new)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    dt = time.perf_counter() - t0
    if rank == 0:
        print(
            json.dumps(
                {
                    "deep_gemm": bool(args.deep_gemm),
                    "max_new": args.max_new,
                    "pure_decode_s": round(dt, 4),
                    "tok_s": round(n / dt, 3),
                    "fp4_gemm_calls": n_calls[0],
                    "calls_per_tok": round(n_calls[0] / max(n, 1), 1),
                    "world": world,
                }
            )
        )
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
