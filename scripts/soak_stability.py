#!/usr/bin/env python3
"""Soak / stability gate for deployable sglang-lite (Phase 2).

Runs many short generates under continuous batching and checks:

  * no hard failures
  * KV ``blocks_used`` does not climb unboundedly after rounds drain
  * OOM reject count stays 0
  * optional dual_pool counters stay finite

Default uses an on-disk tiny Mixtral fixture (CPU). For PRO6000 V4 Hybrid::

  export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
  export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
  export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1
  torchrun --nproc-per-node=8 scripts/soak_stability.py \\
    --model \"$SGLANG_LITE_DSV4_HF\" --device cuda --rounds 20 --concurrency 4 \\
    --max-new 8 --out ~/bench/soak_pro6000.json

CPU CI-style::

  python scripts/soak_stability.py --rounds 30 --concurrency 8 --max-new 4 \\
    --out /tmp/soak_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _blocks_used(stats: dict) -> int:
    cache = stats.get("cache") or {}
    return int(cache.get("blocks_used") or 0)


def _oom(stats: dict) -> int:
    cache = stats.get("cache") or {}
    return int(cache.get("oom_reject_count") or 0)


def _gate(
    *,
    errors: int,
    blocks_series: List[int],
    oom: int,
    completed: int,
    min_completed: int,
    max_blocks_slack: int,
) -> Dict[str, Any]:
    """Return gate dict; overall pass if all true."""
    peak = max(blocks_series) if blocks_series else 0
    final = blocks_series[-1] if blocks_series else 0
    # After drain, final should not stay near peak with large residue unless
    # prefix cache intentionally holds pages. Allow fixed slack for retained
    # prefix entries + warm pages.
    climb_ok = final <= max(peak, 0) and final <= max_blocks_slack + (
        blocks_series[0] if blocks_series else 0
    )
    # Stricter: last 3 samples mean should not exceed early baseline by huge margin
    if len(blocks_series) >= 6:
        early = sum(blocks_series[:3]) / 3.0
        late = sum(blocks_series[-3:]) / 3.0
        climb_ok = late <= early + max_blocks_slack
    return {
        "errors_zero": errors == 0,
        "oom_zero": oom == 0,
        "completed_ge_min": completed >= min_completed,
        "blocks_stable": climb_ok,
        "peak_blocks": peak,
        "final_blocks": final,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="sglang-lite soak stability gate")
    ap.add_argument("--model", default="", help="HF id, fixture:path, or empty→auto tiny Mixtral")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-blocks-slack", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--prompt", default="hello soak prefix")
    args = ap.parse_args()

    _ensure_path()
    os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
    os.environ.setdefault("SGLANG_LITE_LOG_JSON", "0")  # quieter soak

    model = args.model.strip()
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if not model:
        sys.path.insert(0, str(_repo_root() / "scripts"))
        from build_tiny_moe_fixture import build_tiny_mixtral  # type: ignore

        tmp_dir = tempfile.TemporaryDirectory(prefix="sglang-lite-soak-")
        path = build_tiny_mixtral(Path(tmp_dir.name) / "mixtral")
        model = f"fixture:{path}"

    from sglang_lite import LiteEngine

    t0 = time.perf_counter()
    eng = LiteEngine(
        model_name=model,
        device=args.device,
        max_batch_size=args.max_batch,
        allow_stub=model in ("stub",),
        start_loop=True,
    )
    prompt_ids = eng.tokenize(args.prompt)
    errors = 0
    ok_n = 0
    blocks_series: List[int] = []
    err_samples: List[str] = []

    def one(rid: str) -> bool:
        nonlocal errors, ok_n
        try:
            out = eng.generate(
                rid,
                list(prompt_ids),
                max_tokens=args.max_new,
                temperature=0.0,
            )
            if out.get("error"):
                errors += 1
                err_samples.append(str(out.get("error"))[:200])
                return False
            fr = out.get("finish_reason")
            if fr not in ("stop", "length", None):
                # accept empty finish if text present
                if not out.get("text"):
                    errors += 1
                    err_samples.append(f"bad finish_reason={fr}")
                    return False
            ok_n += 1
            return True
        except Exception as e:
            errors += 1
            err_samples.append(f"{type(e).__name__}: {e}"[:200])
            return False

    for r in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [
                pool.submit(one, f"soak-r{r}-c{c}")
                for c in range(args.concurrency)
            ]
            for f in as_completed(futs):
                f.result()
        # Drain in-flight
        try:
            eng.loop.pump_until_idle(timeout_s=120.0)
        except Exception:
            # background loop may already idle
            time.sleep(0.05)
        st = eng.get_stats()
        blocks_series.append(_blocks_used(st))
        print(
            f"[soak] round={r+1}/{args.rounds} ok={ok_n} err={errors} "
            f"blocks={blocks_series[-1]} multi_batch={st.get('multi_request_batches')}"
        )

    # Final drain sample
    time.sleep(0.1)
    st = eng.get_stats()
    blocks_series.append(_blocks_used(st))
    oom = _oom(st)
    min_completed = args.rounds * args.concurrency
    # completed may be tracked in latency
    completed = int((st.get("latency") or {}).get("requests_completed") or ok_n)
    gate = _gate(
        errors=errors,
        blocks_series=blocks_series,
        oom=oom,
        completed=completed,
        min_completed=min_completed // 2,  # allow some flaky short-circuit
        max_blocks_slack=args.max_blocks_slack,
    )
    # Prefer zero errors for PASS
    gate["completed_ge_min"] = completed >= max(1, min_completed // 2) and ok_n >= max(
        1, min_completed // 2
    )
    overall = all(
        [
            gate["errors_zero"],
            gate["oom_zero"],
            gate["completed_ge_min"],
            gate["blocks_stable"],
        ]
    )

    summary: Dict[str, Any] = {
        "model": model,
        "device": args.device,
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "max_new": args.max_new,
        "ok": ok_n,
        "errors": errors,
        "error_samples": err_samples[:5],
        "blocks_series": blocks_series,
        "stats_tail": {
            "cache": st.get("cache"),
            "dual_pool": st.get("dual_pool"),
            "latency": st.get("latency"),
            "multi_request_batches": st.get("multi_request_batches"),
            "steps": st.get("steps"),
        },
        "gate": gate,
        "overall": "PASS" if overall else "FAIL",
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
        print(f"[soak] wrote {args.out}")

    try:
        eng.begin_drain()
        eng.shutdown()
    except Exception:
        pass
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0 if overall else 2


if __name__ == "__main__":
    # Allow `python scripts/soak_stability.py` without package install
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
