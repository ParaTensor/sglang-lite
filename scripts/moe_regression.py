#!/usr/bin/env python3
"""Minimal multi-MoE regression (Phase 2 deploy gate).

Loads one or more MoE model ids, runs a short generate, checks finish_reason
and basic stats. Default: tiny on-disk Mixtral fixture (CPU, no network).

  # CPU fixture (CI / laptop)
  python scripts/moe_regression.py --out /tmp/moe_reg.json

  # Real hub / local checkpoints (optional)
  python scripts/moe_regression.py \\
    --model fixture:/tmp/tiny-mixtral \\
    --model /path/to/Qwen-MoE \\
    --device cuda --max-new 8

Env ``SGLANG_LITE_MOE_REG_MODELS`` colon-separated list overrides --model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _run_one(model: str, device: str, max_new: int, max_batch: int) -> Dict[str, Any]:
    from sglang_lite import LiteEngine
    from sglang_lite.models import assert_moe_supported

    row: Dict[str, Any] = {"model": model, "device": device, "ok": False}
    t0 = time.perf_counter()
    try:
        # fixture: / local path skip hub name checks when loading
        if not (model.startswith("fixture:") or model.startswith("local:")):
            try:
                assert_moe_supported(model)
            except Exception as e:
                # allow path-like dirs
                if not Path(model).is_dir():
                    row["error"] = f"moe_check: {e}"
                    return row

        eng = LiteEngine(
            model_name=model,
            device=device,
            max_batch_size=max_batch,
            allow_stub=model == "stub",
            start_loop=True,
        )
        try:
            ids = eng.tokenize("Hello MoE regression")
            out = eng.generate(
                f"reg-{Path(model).name[:32]}",
                ids,
                max_tokens=max_new,
                temperature=0.0,
            )
            stats = eng.get_stats()
            fr = out.get("finish_reason")
            text = (out.get("text") or "")[:120]
            row.update(
                {
                    "ok": fr in ("stop", "length") or bool(text),
                    "finish_reason": fr,
                    "text_prefix": text,
                    "prompt_tokens": len(ids),
                    "usage": out.get("usage"),
                    "steps": stats.get("steps"),
                    "cache_hit_count": (stats.get("cache") or {}).get("hit_count"),
                    "requests_completed": (stats.get("latency") or {}).get(
                        "requests_completed"
                    ),
                    "load_generate_s": round(time.perf_counter() - t0, 3),
                }
            )
            if out.get("error"):
                row["ok"] = False
                row["error"] = str(out["error"])[:300]
        finally:
            try:
                eng.begin_drain()
                eng.shutdown()
            except Exception:
                pass
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-1500:]
        row["load_generate_s"] = round(time.perf_counter() - t0, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeatable. Empty → auto tiny Mixtral fixture",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _ensure_path()
    os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
    os.environ.setdefault("SGLANG_LITE_LOG_JSON", "0")

    models: List[str] = list(args.model)
    env_models = os.environ.get("SGLANG_LITE_MOE_REG_MODELS", "").strip()
    if env_models:
        models = [m for m in env_models.split(":") if m]

    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if not models:
        sys.path.insert(0, str(_repo_root() / "scripts"))
        from build_tiny_moe_fixture import build_tiny_mixtral  # type: ignore

        tmp_dir = tempfile.TemporaryDirectory(prefix="sglang-lite-moe-reg-")
        path = build_tiny_mixtral(Path(tmp_dir.name) / "mixtral")
        models = [f"fixture:{path}"]

    results = []
    for m in models:
        print(f"[moe-reg] running model={m} device={args.device}")
        row = _run_one(m, args.device, args.max_new, args.max_batch)
        results.append(row)
        print(
            f"[moe-reg] ok={row.get('ok')} finish={row.get('finish_reason')} "
            f"err={row.get('error', '')[:80]}"
        )

    overall = all(r.get("ok") for r in results) and len(results) > 0
    summary = {
        "overall": "PASS" if overall else "FAIL",
        "device": args.device,
        "n_models": len(results),
        "results": results,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
        print(f"[moe-reg] wrote {args.out}")

    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
