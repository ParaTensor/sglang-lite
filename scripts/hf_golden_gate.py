#!/usr/bin/env python3
"""HF golden gate for sglang-lite (PegaInfer-style accuracy gate).

Subcommands
  dump-hf   — HuggingFace greedy generate → JSON (oracle)
  run-lite  — LiteEngine greedy generate → JSON
  compare   — classify HF vs lite (exact / first_diff / error)

Example (PRO6000 / local checkpoint)::

  python scripts/hf_golden_gate.py dump-hf \\
    --model ~/models/Qwen3-30B-A3B-Instruct \\
    --cases test_data/hf_golden_cases.json --out /tmp/golden/hf.json

  python scripts/hf_golden_gate.py run-lite \\
    --model ~/models/Qwen3-30B-A3B-Instruct \\
    --cases test_data/hf_golden_cases.json --path force_hf \\
    --out /tmp/golden/lite_hf.json

  python scripts/hf_golden_gate.py compare \\
    --hf /tmp/golden/hf.json --lite /tmp/golden/lite_hf.json \\
    --out /tmp/golden/compare.json --require-exact

Design notes (scope-safe):
  - No new engine features; only reads Generation API + HF generate.
  - FORCE_HF path is the exactness target; radix_native reports drift.
  - Case set is versioned JSON under test_data/ (committed prompts only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _repo_root()
    for p in (root, root / "engine"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _sha256_tokens(ids: Sequence[int]) -> str:
    raw = ",".join(str(int(x)) for x in ids).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"no cases in {path}")
    out: List[Dict[str, Any]] = []
    for c in cases:
        out.append(
            {
                "id": str(c["id"]),
                "prompt": str(c["prompt"]),
                "max_new": int(c.get("max_new", 16)),
                "ignore_eos": bool(c.get("ignore_eos", True)),
                # When True, --require-exact fails the gate if this case mismatches.
                "require_exact": bool(c.get("require_exact", True)),
            }
        )
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[golden] wrote {path}")


def cmd_dump_hf(args: argparse.Namespace) -> int:
    """HF oracle must match lite's *load* choices (single device + experts_impl).

    Using ``device_map=auto`` on multi-GPU hosts can shift greedy tokens vs the
    single-GPU FORCE_HF path (seen: hello_32 first_diff@18 while lite bit-matches
    ``model.generate`` on the same weights).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = _load_cases(Path(args.cases))
    model_path = str(Path(args.model).expanduser())
    device = args.device
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    experts_impl = os.environ.get("SGLANG_LITE_EXPERTS_IMPL", "batched_mm").strip()
    if experts_impl in ("", "auto", "default", "0", "none"):
        experts_impl = "batched_mm"

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    # Align with ModelRunner FORCE_HF thruput path (single device, not device_map=auto).
    if device != "cpu":
        load_kwargs["device_map"] = None
    try:
        load_kwargs["experts_implementation"] = experts_impl
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("experts_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        experts_impl = "(default)"
    except Exception:
        # Older TF may reject experts_implementation via other errors.
        load_kwargs.pop("experts_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        experts_impl = "(default)"
    model.eval()
    if device == "cpu":
        model = model.to("cpu")
    else:
        model = model.to(device)
    load_s = round(time.perf_counter() - t0, 3)
    print(
        f"[golden:hf] load device={device} experts_implementation={experts_impl} "
        f"(single-device; not device_map=auto)"
    )

    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for c in cases:
            ids = tok(c["prompt"], return_tensors="pt")
            dev = next(model.parameters()).device
            ids = {k: v.to(dev) for k, v in ids.items()}
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": c["max_new"],
                "do_sample": False,
                "use_cache": True,
            }
            if c["ignore_eos"]:
                gen_kwargs["eos_token_id"] = None
                gen_kwargs["pad_token_id"] = tok.pad_token_id or tok.eos_token_id
            t1 = time.perf_counter()
            out = model.generate(**ids, **gen_kwargs)
            gen_s = round(time.perf_counter() - t1, 3)
            prompt_len = int(ids["input_ids"].shape[-1])
            full = out[0].tolist()
            gen_ids = [int(x) for x in full[prompt_len:]]
            # Cap to max_new (some stacks append EOS)
            if c["ignore_eos"] and len(gen_ids) > c["max_new"]:
                gen_ids = gen_ids[: c["max_new"]]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(
                {
                    "id": c["id"],
                    "prompt": c["prompt"],
                    "max_new": c["max_new"],
                    "ignore_eos": c["ignore_eos"],
                    "require_exact": bool(c.get("require_exact", True)),
                    "prompt_tokens": prompt_len,
                    "output_ids": gen_ids,
                    "text": text,
                    "token_sha256": _sha256_tokens(gen_ids),
                    "text_sha256": _sha256_text(text),
                    "generate_s": gen_s,
                }
            )
            print(
                f"[golden:hf] {c['id']}: n={len(gen_ids)} "
                f"sha={results[-1]['token_sha256'][:12]}… text={text[:48]!r}"
            )

    payload = {
        "schema": 2,
        "source": "huggingface",
        "model": model_path,
        "device": device,
        "load_s": load_s,
        "load": {
            "device_map": None,
            "experts_implementation": experts_impl,
            "single_device": True,
        },
        "torch": getattr(torch, "__version__", ""),
        "cases": results,
    }
    _write_json(Path(args.out), payload)
    return 0


def _apply_lite_path(path: str) -> None:
    """Set env for a named thruput/accuracy path before LiteEngine import side effects."""
    path = path.strip().lower()
    os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
    os.environ.setdefault("SGLANG_LITE_LOG_JSON", "0")
    os.environ.setdefault("SGLANG_LITE_EXPERTS_IMPL", "batched_mm")
    if path in ("force_hf", "hf", "compile"):
        os.environ["SGLANG_LITE_FORCE_HF_CACHE"] = "1"
        os.environ.setdefault("SGLANG_LITE_TORCH_COMPILE", "0")  # exactness > thruput
        os.environ.setdefault("SGLANG_LITE_FUSED_MOE", "0")
        os.environ.setdefault("SGLANG_LITE_CUDA_GRAPH_DECODE", "0")
        os.environ.setdefault("SGLANG_LITE_NATIVE_DECODE", "0")
        # Burst does not change greedy tokens (verified), but step path is simpler for gates.
        os.environ.setdefault("SGLANG_LITE_DECODE_BURST", "1")
    elif path in ("radix_native", "radix", "paged"):
        os.environ["SGLANG_LITE_FORCE_HF_CACHE"] = "0"
        os.environ.setdefault("SGLANG_LITE_TORCH_COMPILE", "0")
        os.environ.setdefault("SGLANG_LITE_CUDA_GRAPH_DECODE", "1")
        os.environ.setdefault("SGLANG_LITE_FUSED_MOE", "1")
        os.environ.setdefault("SGLANG_LITE_NATIVE_DECODE", "1")
        os.environ.setdefault("SGLANG_LITE_DECODE_BURST", "128")
    elif path in ("radix_eager", "paged_eager"):
        os.environ["SGLANG_LITE_FORCE_HF_CACHE"] = "0"
        os.environ.setdefault("SGLANG_LITE_TORCH_COMPILE", "0")
        os.environ.setdefault("SGLANG_LITE_CUDA_GRAPH_DECODE", "0")
        os.environ.setdefault("SGLANG_LITE_FUSED_MOE", "0")
        os.environ.setdefault("SGLANG_LITE_NATIVE_DECODE", "1")
        os.environ.setdefault("SGLANG_LITE_DECODE_BURST", "1")
    else:
        raise ValueError(
            f"unknown --path {path!r}; want force_hf | radix_native | radix_eager"
        )


def cmd_run_lite(args: argparse.Namespace) -> int:
    _ensure_path()
    _apply_lite_path(args.path)
    cases = _load_cases(Path(args.cases))
    model_path = str(Path(args.model).expanduser())

    from sglang_lite import LiteEngine

    t0 = time.perf_counter()
    eng = LiteEngine(
        model_name=model_path,
        device=args.device,
        max_batch_size=2,
        allow_stub=False,
        start_loop=False,
    )
    eng._gen_ignore_eos = True
    # Keep streaming text for human-readable dumps; tokens come from output_ids.
    eng._gen_skip_streaming_text = False
    load_s = round(time.perf_counter() - t0, 3)

    # Radix/FI paths are report-only unless caller forces require_exact on compare.
    is_radix = args.path in (
        "radix_native",
        "radix",
        "paged",
        "radix_eager",
        "paged_eager",
    )
    results: List[Dict[str, Any]] = []
    try:
        for c in cases:
            # Per-case ignore_eos: engine flag is global; golden cases default true.
            eng._gen_ignore_eos = bool(c["ignore_eos"])
            ids = eng.tokenize(c["prompt"])
            t1 = time.perf_counter()
            out = eng.generate(
                f"golden-{c['id']}",
                ids,
                max_tokens=c["max_new"],
                temperature=0.0,
            )
            gen_s = round(time.perf_counter() - t1, 3)
            gen_ids = [int(x) for x in (out.get("output_ids") or [])]
            if c["ignore_eos"] and len(gen_ids) > c["max_new"]:
                gen_ids = gen_ids[: c["max_new"]]
            text = out.get("text") or ""
            # Case file may say require_exact; radix dumps mark report-only.
            req_exact = bool(c.get("require_exact", True)) and not is_radix
            results.append(
                {
                    "id": c["id"],
                    "prompt": c["prompt"],
                    "max_new": c["max_new"],
                    "ignore_eos": c["ignore_eos"],
                    "require_exact": req_exact,
                    "prompt_tokens": len(ids),
                    "output_ids": gen_ids,
                    "text": text,
                    "token_sha256": _sha256_tokens(gen_ids),
                    "text_sha256": _sha256_text(text),
                    "generate_s": gen_s,
                    "finish_reason": out.get("finish_reason"),
                    "usage": out.get("usage"),
                }
            )
            print(
                f"[golden:lite/{args.path}] {c['id']}: n={len(gen_ids)} "
                f"sha={results[-1]['token_sha256'][:12]}… text={text[:48]!r}"
            )
    finally:
        try:
            eng.begin_drain()
            eng.shutdown()
        except Exception:
            pass

    payload = {
        "schema": 1,
        "source": "sglang_lite",
        "path": args.path,
        "model": model_path,
        "device": args.device,
        "load_s": load_s,
        "env": {
            "FORCE_HF_CACHE": os.environ.get("SGLANG_LITE_FORCE_HF_CACHE"),
            "TORCH_COMPILE": os.environ.get("SGLANG_LITE_TORCH_COMPILE"),
            "CUDA_GRAPH_DECODE": os.environ.get("SGLANG_LITE_CUDA_GRAPH_DECODE"),
            "FUSED_MOE": os.environ.get("SGLANG_LITE_FUSED_MOE"),
            "NATIVE_DECODE": os.environ.get("SGLANG_LITE_NATIVE_DECODE"),
            "EXPERTS_IMPL": os.environ.get("SGLANG_LITE_EXPERTS_IMPL"),
        },
        "cases": results,
    }
    _write_json(Path(args.out), payload)
    return 0


def _first_diff(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    if len(a) != len(b):
        return n
    return None


def cmd_compare(args: argparse.Namespace) -> int:
    hf = json.loads(Path(args.hf).read_text(encoding="utf-8"))
    lite = json.loads(Path(args.lite).read_text(encoding="utf-8"))
    hf_by = {c["id"]: c for c in hf.get("cases", [])}
    lite_by = {c["id"]: c for c in lite.get("cases", [])}
    ids = sorted(set(hf_by) | set(lite_by))

    case_rows: List[Dict[str, Any]] = []
    all_exact = True
    required_fail = False
    for cid in ids:
        h = hf_by.get(cid)
        l = lite_by.get(cid)
        if h is None or l is None:
            row = {
                "id": cid,
                "classification": "error",
                "error": "missing_side",
                "hf_present": h is not None,
                "lite_present": l is not None,
                "require_exact": True,
            }
            all_exact = False
            required_fail = True
            case_rows.append(row)
            continue
        h_ids = [int(x) for x in h.get("output_ids") or []]
        l_ids = [int(x) for x in l.get("output_ids") or []]
        fd = _first_diff(h_ids, l_ids)
        token_exact = fd is None
        text_exact = (h.get("text") or "") == (l.get("text") or "")
        if token_exact and text_exact:
            cls = "all_token_text_exact"
        elif token_exact:
            cls = "token_exact_text_diff"
            all_exact = False
        else:
            cls = "first_diff"
            all_exact = False
        # Case-level gate: default True for FORCE_HF; radix runs may mark False.
        req_exact = bool(h.get("require_exact", True))
        if l.get("require_exact") is not None:
            req_exact = bool(l.get("require_exact"))
        if req_exact and cls != "all_token_text_exact":
            required_fail = True
        case_rows.append(
            {
                "id": cid,
                "classification": cls,
                "first_diff_index": fd,
                "require_exact": req_exact,
                "hf_n": len(h_ids),
                "lite_n": len(l_ids),
                "hf_token_sha256": h.get("token_sha256"),
                "lite_token_sha256": l.get("token_sha256"),
                "hf_text_prefix": (h.get("text") or "")[:80],
                "lite_text_prefix": (l.get("text") or "")[:80],
            }
        )
        print(
            f"[golden:compare] {cid}: {cls}"
            + (f" @ {fd}" if fd is not None else "")
            + ("" if req_exact else " (report-only)")
        )

    overall = "all_token_text_exact" if all_exact else "mismatch"
    report = {
        "schema": 2,
        "overall": overall,
        "all_exact": all_exact,
        "required_cases_ok": not required_fail,
        "hf_model": hf.get("model"),
        "lite_model": lite.get("model"),
        "lite_path": lite.get("path"),
        "case_count": len(case_rows),
        "cases": case_rows,
    }
    _write_json(Path(args.out), report)
    if args.require_exact and required_fail:
        print(
            "[golden] FAIL: --require-exact and a require_exact case mismatched",
            file=sys.stderr,
        )
        return 1
    print(f"[golden] overall={overall} required_cases_ok={not required_fail}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """One-shot: dump-hf → run-lite → compare.

    FORCE_HF: ``--require-exact`` (default on).
    radix_*: report-only unless ``--require-exact`` is passed.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_path = out_dir / "hf.json"
    lite_path = out_dir / f"lite_{args.path}.json"
    cmp_path = out_dir / f"compare_{args.path}.json"

    ns_hf = argparse.Namespace(
        model=args.model,
        cases=args.cases,
        device=args.device,
        out=str(hf_path),
    )
    rc = cmd_dump_hf(ns_hf)
    if rc != 0:
        return rc

    ns_lite = argparse.Namespace(
        model=args.model,
        cases=args.cases,
        device=args.device,
        path=args.path,
        out=str(lite_path),
    )
    rc = cmd_run_lite(ns_lite)
    if rc != 0:
        return rc

    is_radix = args.path in (
        "radix_native",
        "radix",
        "paged",
        "radix_eager",
        "paged_eager",
    )
    require = bool(args.require_exact) if not is_radix else bool(args.require_exact)
    # force_hf defaults require_exact=True via argparse; radix defaults False.
    ns_cmp = argparse.Namespace(
        hf=str(hf_path),
        lite=str(lite_path),
        out=str(cmp_path),
        require_exact=require,
    )
    return cmd_compare(ns_cmp)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_hf = sub.add_parser("dump-hf", help="HF greedy oracle dump")
    p_hf.add_argument("--model", required=True)
    p_hf.add_argument(
        "--cases",
        default=str(_repo_root() / "test_data" / "hf_golden_cases.json"),
    )
    p_hf.add_argument("--device", default="cuda")
    p_hf.add_argument("--out", required=True)
    p_hf.set_defaults(func=cmd_dump_hf)

    p_lite = sub.add_parser("run-lite", help="LiteEngine greedy dump")
    p_lite.add_argument("--model", required=True)
    p_lite.add_argument(
        "--cases",
        default=str(_repo_root() / "test_data" / "hf_golden_cases.json"),
    )
    p_lite.add_argument("--device", default="cuda")
    p_lite.add_argument(
        "--path",
        default="force_hf",
        help="force_hf | radix_native | radix_eager",
    )
    p_lite.add_argument("--out", required=True)
    p_lite.set_defaults(func=cmd_run_lite)

    p_cmp = sub.add_parser("compare", help="Compare HF vs lite JSON")
    p_cmp.add_argument("--hf", required=True)
    p_cmp.add_argument("--lite", required=True)
    p_cmp.add_argument("--out", required=True)
    p_cmp.add_argument(
        "--require-exact",
        action="store_true",
        help="exit 1 if any case with require_exact=true mismatches",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_gate = sub.add_parser(
        "gate",
        help="dump-hf + run-lite + compare in one shot",
    )
    p_gate.add_argument("--model", required=True)
    p_gate.add_argument(
        "--cases",
        default=str(_repo_root() / "test_data" / "hf_golden_cases.json"),
    )
    p_gate.add_argument("--device", default="cuda")
    p_gate.add_argument(
        "--path",
        default="force_hf",
        help="force_hf | radix_native | radix_eager",
    )
    p_gate.add_argument(
        "--out-dir",
        default="/tmp/golden",
        help="directory for hf.json / lite_*.json / compare_*.json",
    )
    p_gate.add_argument(
        "--require-exact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="exit 1 on require_exact case mismatch "
        "(default: on for force_hf, off for radix_*)",
    )
    p_gate.set_defaults(func=cmd_gate)

    args = ap.parse_args(argv)
    if args.cmd == "gate" and args.require_exact is None:
        args.require_exact = args.path in ("force_hf", "hf", "compile")
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"[golden] error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
