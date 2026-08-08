#!/usr/bin/env python3
"""Official vendor Transformer thruput ceiling (ignore_eos, CVD remap).

Compares pure official generate loop to LiteEngine Hybrid without scheduler tax.

  export SGLANG_LITE_DSV4_HF=... SGLANG_LITE_DSV4_CONVERTED=...
  export SGLANG_LITE_DSV4_INFER=$PWD/engine/vendor/deepseek_infer
  torchrun --nproc-per-node=8 scripts/v4_official_timed_ceiling.py --max-new 128
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
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompt", type=str, default="Hello")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    vendor = root / "engine" / "vendor" / "deepseek_infer"
    infer = Path(
        os.environ.get("SGLANG_LITE_DSV4_INFER")
        or (str(vendor) if (vendor / "model.py").is_file() else "")
    )
    if not infer.is_dir():
        print("set SGLANG_LITE_DSV4_INFER", file=sys.stderr)
        return 2
    sys.path.insert(0, str(infer))

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from model import ModelArgs, Transformer

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(33377335)

    cfg = json.loads((infer / "config.json").read_text(encoding="utf-8"))
    if isinstance(cfg.get("compress_ratios"), list):
        cfg["compress_ratios"] = tuple(cfg["compress_ratios"])
    names = {f.name for f in fields(ModelArgs)}
    margs = ModelArgs(**{k: v for k, v in cfg.items() if k in names})
    with torch.device("cuda"):
        model = Transformer(margs)
    ckpt = Path(os.environ["SGLANG_LITE_DSV4_CONVERTED"]).expanduser()
    load_model(model, str(ckpt / f"model{rank}-mp{world}.safetensors"), strict=False)
    model.eval()

    hf = Path(
        os.environ.get("SGLANG_LITE_DSV4_HF", os.path.expanduser("~/models/ds-v4-flash"))
    )
    tok = AutoTokenizer.from_pretrained(str(hf), trust_remote_code=True)
    prompt_ids = tok.encode(
        encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat")
    )

    @torch.inference_mode()
    def gen_ignore_eos(max_new: int) -> list[int]:
        """Always emit max_new tokens (thruput ceiling; no EOS stop)."""
        pl = len(prompt_ids)
        total = min(model.max_seq_len, pl + max_new)
        tokens = torch.full((1, total), -1, dtype=torch.long, device="cuda")
        tokens[0, :pl] = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        prev = 0
        for cur in range(pl, total):
            logits = model.forward(tokens[:, prev:cur], prev)
            tokens[0, cur] = logits.argmax(dim=-1)
            prev = cur
        return tokens[0, pl:pl + max_new].tolist()

    def rprint(*a, **k):
        if rank == 0:
            print(*a, **k)

    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    t0 = time.perf_counter()
    out = gen_ignore_eos(args.max_new)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    tc = time.perf_counter() - t0

    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    t1 = time.perf_counter()
    out = gen_ignore_eos(args.max_new)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    tw = time.perf_counter() - t1

    n = len(out)
    if rank == 0:
        summary = {
            "path": "official_vendor_ignore_eos",
            "max_new": args.max_new,
            "n_tokens": n,
            "tok_s_cold": n / tc if tc else 0,
            "tok_s_warm": n / tw if tw else 0,
            "cold_s": tc,
            "warm_s": tw,
            "text": tok.decode(out)[:120],
        }
        print(json.dumps(summary, ensure_ascii=False))
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
