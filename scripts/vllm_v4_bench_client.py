#!/usr/bin/env python3
"""Client thruput probe against a running OpenAI-compatible V4 server (vLLM).

Example:
  python scripts/vllm_v4_bench_client.py --url http://127.0.0.1:8000 --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def one(url: str, model: str, max_tokens: int, ignore_eos: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": ignore_eos,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read().decode())
    dt = time.perf_counter() - t0
    usage = obj.get("usage") or {}
    n = int(usage.get("completion_tokens") or 0)
    text = (obj.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {
        "completion_tokens": n,
        "wall_s": dt,
        "tok_s": (n / dt) if dt > 0 else 0.0,
        "text_prefix": text[:100],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="/model")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    for i in range(args.warmup):
        one(args.url, args.model, min(16, args.max_tokens), ignore_eos=False)

    if args.batch <= 1:
        cold = one(args.url, args.model, args.max_tokens, True)
        warm = one(args.url, args.model, args.max_tokens, True)
        print(json.dumps({"cold": cold, "warm": warm}, indent=2, ensure_ascii=False))
        return 0

    def batch_once(tag: str) -> dict:
        t0 = time.perf_counter()
        outs = []
        with ThreadPoolExecutor(max_workers=args.batch) as ex:
            futs = [
                ex.submit(one, args.url, args.model, args.max_tokens, True)
                for _ in range(args.batch)
            ]
            for f in as_completed(futs):
                outs.append(f.result())
        wall = time.perf_counter() - t0
        tot = sum(o["completion_tokens"] for o in outs)
        return {
            "tag": tag,
            "wall_s": wall,
            "total_tokens": tot,
            "tok_s_aggregate": tot / wall if wall else 0.0,
        }

    print(json.dumps({"cold": batch_once("cold"), "warm": batch_once("warm")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
