#!/usr/bin/env python3
"""Exercise dual_write / dual_hit / dual_restore / dual_stage (0c-4) and print stats.

  source ~/venvs/sglang-lite/bin/activate
  export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
  export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
  export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1 SGLANG_LITE_FI_PREFIX=
  torchrun --nproc-per-node=8 scripts/v4_dual_stats_probe.py \\
    --out ~/bench/v4_dual_stats_0c4.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# CVD remap before torch (TileLang device_id=0)
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
# Prefer empty FI prefix for official path (avoid incomplete /tmp/fi1616).
if "SGLANG_LITE_FI_PREFIX" not in os.environ:
    os.environ["SGLANG_LITE_FI_PREFIX"] = ""


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def rprint(*a, **k):
    if rank() == 0:
        print(*a, **k)


def _dual_field(stats: dict, key: str) -> int:
    dp = stats.get("dual_pool") or {}
    cache = stats.get("cache") or {}
    v4 = stats.get("v4_prefix") or {}
    for src in (dp, cache, v4):
        if src.get(key) is not None:
            return int(src.get(key) or 0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--prompt", default="Hello")
    args = ap.parse_args()

    if Path("/usr/local/cuda/bin").is_dir():
        os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    if Path("/usr/local/cuda/include").is_dir():
        os.environ["CPATH"] = "/usr/local/cuda/include"

    hf = Path(os.environ.get("SGLANG_LITE_DSV4_HF", "")).expanduser()
    if not hf.is_dir():
        hf = Path.home() / "models/DeepSeek-V4-Flash-0731"
    conv = Path(os.environ.get("SGLANG_LITE_DSV4_CONVERTED", "")).expanduser()
    if not conv.is_dir():
        conv = Path.home() / "models/ds-v4-mp8"
    os.environ["SGLANG_LITE_DSV4_HF"] = str(hf)
    os.environ["SGLANG_LITE_DSV4_CONVERTED"] = str(conv)

    repo = Path(__file__).resolve().parents[1]
    for p in (repo, repo / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    import torch.distributed as dist

    from sglang_lite.core import LiteEngine

    rprint(f"[probe] hf={hf} conv={conv}")
    rprint(f"[probe] torch={torch.__version__} cuda={torch.version.cuda}")

    eng = LiteEngine(str(hf), device="cuda", max_batch_size=4, start_loop=False)
    rprint(f"[probe] hybrid={eng.runner._v4_hybrid}")
    rprint(
        f"[probe] radix packed_swa={eng.radix.packed_swa_cache is not None} "
        f"restore_bf16={eng.radix.restore_bf16_cache is not None}"
    )

    ids = None
    for p in (hf / "encoding", hf / "inference"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from encoding_dsv4 import encode_messages

        rendered = encode_messages(
            [{"role": "user", "content": args.prompt}], thinking_mode="chat"
        )
        ids = list(eng.runner.tokenizer.encode(rendered))
    except Exception as e:
        rprint(f"[probe] encoding fallback: {e}")
        ids = list(eng.runner.tokenizer.encode(args.prompt))

    rprint(f"[probe] prompt_ids len={len(ids)} max_new={args.max_new}")

    out1 = eng.generate(
        "dual-cold",
        ids,
        max_tokens=args.max_new,
        temperature=0.0,
    )
    stats1 = eng.get_stats()
    rprint("[probe] cold text=", repr((out1.get("text") or "")[:100]))
    rprint("[probe] cold usage=", out1.get("usage"))
    rprint(
        "[probe] after cold dual_pool=",
        json.dumps(stats1.get("dual_pool", {}), ensure_ascii=False),
    )

    out2 = eng.generate(
        "dual-warm",
        ids,
        max_tokens=args.max_new,
        temperature=0.0,
    )
    stats2 = eng.get_stats()
    rprint("[probe] warm text=", repr((out2.get("text") or "")[:100]))
    rprint("[probe] warm usage=", out2.get("usage"))
    rprint(
        "[probe] after warm dual_pool=",
        json.dumps(stats2.get("dual_pool", {}), ensure_ascii=False),
    )

    cold_stage = _dual_field(stats1, "dual_stage_count")
    warm_stage = _dual_field(stats2, "dual_stage_count")
    summary = {
        "cold_text": out1.get("text"),
        "warm_text": out2.get("text"),
        "cold_cache_hit_tokens": (out1.get("usage") or {}).get("cache_hit_tokens"),
        "warm_cache_hit_tokens": (out2.get("usage") or {}).get("cache_hit_tokens"),
        "dual_pool": stats2.get("dual_pool"),
        "v4_prefix": stats2.get("v4_prefix"),
        "dual_stage_count_after_cold": cold_stage,
        "dual_stage_count_after_warm": warm_stage,
        "gate": {
            "dual_write_ge1": _dual_field(stats2, "dual_write_count") >= 1,
            "dual_hit_ge1": _dual_field(stats2, "dual_hit_count") >= 1
            or int((stats2.get("v4_prefix") or {}).get("dual_hit_count") or 0) >= 1,
            "dual_restore_ge1": _dual_field(stats2, "dual_restore_count") >= 1,
            "dual_stage_ge1": warm_stage >= 1,  # 0c-4: page-primary stage
            "warm_hit_tokens_gt0": int(
                (out2.get("usage") or {}).get("cache_hit_tokens") or 0
            )
            > 0,
            "text_ok": "Hello" in (out1.get("text") or "")
            or "你好" in (out1.get("text") or "")
            or "help" in (out1.get("text") or "").lower(),
        },
    }
    ok = all(summary["gate"].values())
    if rank() == 0:
        text = json.dumps(summary, ensure_ascii=False, indent=2)
        print(text)
        out = Path(args.out).expanduser() if args.out else (
            Path.home() / "bench/v4_dual_stats_0c4.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        rprint("[probe] wrote", out)
        rprint("[probe] SUMMARY", json.dumps(summary["gate"], ensure_ascii=False))
        rprint("[probe] OVERALL", "PASS" if ok else "PARTIAL/FAIL")

    # All ranks: no post-result barrier (NCCL destroy is flaky; results already written).
    try:
        eng.shutdown()
    except Exception:
        pass
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
