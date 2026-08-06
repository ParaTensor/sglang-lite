#!/usr/bin/env python3
"""Soak / stability gate for deployable sglang-lite (Phase 2).

Checks under sustained load:

  * no hard failures
  * KV ``blocks_used`` does not climb unboundedly after rounds drain
  * OOM reject count stays 0
  * dual_pool counters stay finite (V4)

Profiles (``--profile``)::

  smoke   ~1–2 min   rounds=10  conc=4   max_new=4
  short   ~5–10 min  rounds=40  conc=8   max_new=8
  medium  ~20–30 min rounds=120 conc=8   max_new=8
  long    ~60+ min   duration_s=3600 conc=4 max_new=8  (time-based)

CPU fixture::

  python scripts/soak_stability.py --profile short --out /tmp/soak.json

PRO6000 V4 Hybrid (torchrun, all ranks)::

  source scripts/env_lite.sh
  export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
  export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
  torchrun --nproc-per-node=8 scripts/soak_stability.py \\
    --model \"$SGLANG_LITE_DSV4_HF\" --device cuda --profile medium \\
    --out ~/bench/soak_v4_medium.json
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
from typing import Any, Dict, List, Optional, Tuple

PROFILES = {
    "smoke": {"rounds": 10, "concurrency": 4, "max_new": 4, "duration_s": 0},
    "short": {"rounds": 40, "concurrency": 8, "max_new": 8, "duration_s": 0},
    "medium": {"rounds": 120, "concurrency": 8, "max_new": 8, "duration_s": 0},
    # long: time-based; rounds used as max ceiling
    "long": {"rounds": 100000, "concurrency": 4, "max_new": 8, "duration_s": 3600},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _rprint(*a, **k):
    if _rank() == 0:
        print(*a, **k)


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
    peak = max(blocks_series) if blocks_series else 0
    final = blocks_series[-1] if blocks_series else 0
    climb_ok = True
    if len(blocks_series) >= 6:
        early = sum(blocks_series[:3]) / 3.0
        late = sum(blocks_series[-3:]) / 3.0
        climb_ok = late <= early + max_blocks_slack
    else:
        climb_ok = final <= max_blocks_slack + (blocks_series[0] if blocks_series else 0)
    return {
        "errors_zero": errors == 0,
        "oom_zero": oom == 0,
        "completed_ge_min": completed >= min_completed,
        "blocks_stable": climb_ok,
        "peak_blocks": peak,
        "final_blocks": final,
    }


def _maybe_cvd_remap() -> None:
    if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]


def main() -> int:
    _maybe_cvd_remap()

    ap = argparse.ArgumentParser(description="sglang-lite soak stability gate")
    ap.add_argument("--model", default="", help="HF id, fixture:path, or empty→auto tiny Mixtral")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--profile",
        default="short",
        choices=sorted(PROFILES.keys()),
        help="Duration/load profile (smoke|short|medium|long)",
    )
    ap.add_argument("--rounds", type=int, default=None, help="Override profile rounds")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument(
        "--duration-s",
        type=int,
        default=None,
        help="If >0, run until this many seconds (overrides pure round limit)",
    )
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-blocks-slack", type=int, default=128)
    ap.add_argument("--out", default="")
    ap.add_argument("--prompt", default="hello soak prefix")
    args = ap.parse_args()

    prof = dict(PROFILES[args.profile])
    rounds = int(args.rounds if args.rounds is not None else prof["rounds"])
    concurrency = int(
        args.concurrency if args.concurrency is not None else prof["concurrency"]
    )
    max_new = int(args.max_new if args.max_new is not None else prof["max_new"])
    duration_s = int(
        args.duration_s if args.duration_s is not None else prof["duration_s"]
    )

    _ensure_path()
    if Path("/usr/local/cuda/bin").is_dir():
        os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
    os.environ.setdefault("SGLANG_LITE_LOG_JSON", "0")

    model = args.model.strip()
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if not model:
        sys.path.insert(0, str(_repo_root() / "scripts"))
        from build_tiny_moe_fixture import build_tiny_mixtral  # type: ignore

        tmp_dir = tempfile.TemporaryDirectory(prefix="sglang-lite-soak-")
        path = build_tiny_mixtral(Path(tmp_dir.name) / "mixtral")
        model = f"fixture:{path}"

    import torch
    import torch.distributed as dist

    from sglang_lite import LiteEngine

    # V4 / TP: all ranks must share schedule → no background loop
    is_tp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    start_loop = not is_tp

    t0 = time.perf_counter()
    eng = LiteEngine(
        model_name=model,
        device=args.device,
        max_batch_size=max(args.max_batch, concurrency),
        allow_stub=model in ("stub",),
        start_loop=start_loop,
    )
    is_v4 = bool(getattr(eng.runner, "_v4_hybrid", False))
    if is_v4 and start_loop:
        # Prefer sync pump for Hybrid even single-process
        eng.loop._ready = True

    # Prefer chat encoding for V4
    prompt_ids: List[int]
    if is_v4:
        try:
            prompt_ids = eng.runner.apply_chat_template(
                [{"role": "user", "content": args.prompt}]
            )
        except Exception:
            prompt_ids = eng.tokenize(args.prompt)
    else:
        prompt_ids = eng.tokenize(args.prompt)

    _rprint(
        f"[soak] model={model} device={args.device} profile={args.profile} "
        f"rounds={rounds} conc={concurrency} max_new={max_new} duration_s={duration_s} "
        f"v4={is_v4} tp={is_tp}"
    )

    errors = 0
    ok_n = 0
    blocks_series: List[int] = []
    dual_stage_series: List[int] = []
    err_samples: List[str] = []
    round_i = 0
    deadline = t0 + duration_s if duration_s > 0 else None

    def one_local(rid: str) -> bool:
        nonlocal errors, ok_n
        try:
            out = eng.generate(
                rid,
                list(prompt_ids),
                max_tokens=max_new,
                temperature=0.0,
            )
            if out.get("error"):
                errors += 1
                err_samples.append(str(out.get("error"))[:200])
                return False
            fr = out.get("finish_reason")
            if fr not in ("stop", "length", None) and not out.get("text"):
                errors += 1
                err_samples.append(f"bad finish_reason={fr}")
                return False
            ok_n += 1
            return True
        except Exception as e:
            errors += 1
            err_samples.append(f"{type(e).__name__}: {e}"[:200])
            return False

    def one_tp_batch(r: int) -> Tuple[int, int]:
        """All ranks: identical generate_batch; return (ok, err) on rank0 stats."""
        nonlocal errors, ok_n
        reqs = [
            {
                "request_id": f"soak-r{r}-c{c}",
                "input_ids": list(prompt_ids),
                "max_tokens": max_new,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            for c in range(concurrency)
        ]
        try:
            outs = eng.generate_batch(reqs, timeout_s=600.0)
            local_ok = 0
            local_err = 0
            for o in outs:
                if o.get("error"):
                    local_err += 1
                    err_samples.append(str(o.get("error"))[:200])
                else:
                    local_ok += 1
            ok_n += local_ok
            errors += local_err
            return local_ok, local_err
        except Exception as e:
            errors += concurrency
            err_samples.append(f"{type(e).__name__}: {e}"[:200])
            return 0, concurrency

    while round_i < rounds:
        if deadline is not None and time.perf_counter() >= deadline:
            _rprint(f"[soak] duration_s={duration_s} reached after {round_i} rounds")
            break

        if is_tp or is_v4:
            # Hybrid TP (and V4 single): use batch API so all ranks stay aligned
            if not start_loop:
                one_tp_batch(round_i)
            else:
                # single process V4 with background loop — still use threads
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = [
                        pool.submit(one_local, f"soak-r{round_i}-c{c}")
                        for c in range(concurrency)
                    ]
                    for f in as_completed(futs):
                        f.result()
                try:
                    eng.loop.pump_until_idle(timeout_s=300.0)
                except Exception:
                    time.sleep(0.05)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = [
                    pool.submit(one_local, f"soak-r{round_i}-c{c}")
                    for c in range(concurrency)
                ]
                for f in as_completed(futs):
                    f.result()
            try:
                eng.loop.pump_until_idle(timeout_s=120.0)
            except Exception:
                time.sleep(0.05)

        st = eng.get_stats()
        blocks_series.append(_blocks_used(st))
        dual_stage_series.append(
            int((st.get("dual_pool") or {}).get("dual_stage_count") or 0)
        )
        round_i += 1
        if _rank() == 0 and (round_i <= 3 or round_i % 10 == 0 or round_i == rounds):
            _rprint(
                f"[soak] round={round_i}/{rounds} ok={ok_n} err={errors} "
                f"blocks={blocks_series[-1]} stage={dual_stage_series[-1]} "
                f"elapsed={time.perf_counter()-t0:.1f}s"
            )

        if dist.is_initialized():
            # keep ranks aligned between rounds
            try:
                dist.barrier()
            except Exception:
                pass

    time.sleep(0.1)
    st = eng.get_stats()
    blocks_series.append(_blocks_used(st))
    dual_stage_series.append(
        int((st.get("dual_pool") or {}).get("dual_stage_count") or 0)
    )
    oom = _oom(st)
    min_completed = max(1, (round_i * concurrency) // 2)
    completed = int((st.get("latency") or {}).get("requests_completed") or ok_n)
    gate = _gate(
        errors=errors,
        blocks_series=blocks_series,
        oom=oom,
        completed=completed,
        min_completed=min_completed,
        max_blocks_slack=args.max_blocks_slack,
    )
    gate["completed_ge_min"] = completed >= min_completed or ok_n >= min_completed
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
        "profile": args.profile,
        "rounds_planned": rounds,
        "rounds_done": round_i,
        "concurrency": concurrency,
        "max_new": max_new,
        "duration_s_limit": duration_s,
        "v4_hybrid": is_v4,
        "tp": is_tp,
        "ok": ok_n,
        "errors": errors,
        "error_samples": err_samples[:8],
        "blocks_series": blocks_series,
        "dual_stage_series": dual_stage_series,
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

    if _rank() == 0:
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        print(text)
        if args.out:
            outp = Path(args.out).expanduser()
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(text + "\n", encoding="utf-8")
            print(f"[soak] wrote {outp}")

    try:
        eng.begin_drain()
        eng.shutdown()
    except Exception:
        pass
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0 if overall else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
