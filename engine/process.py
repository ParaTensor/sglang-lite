"""Internal engine process HTTP server (GenerationRequest / TokenDelta).

Not an OpenAI surface — that lives in Rust control/serving.
Run: python -m sglang_lite.process --model <moe> --port 9001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .loop import EngineLoop, GenParams
from .models import list_verified_models
from .runner import ModelRunner

logger = logging.getLogger("sglang_lite.process")

app = FastAPI(title="sglang-lite engine process")
LOOP: Optional[EngineLoop] = None
READY = False
MODEL_NAME = "stub"


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class GenerationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    messages: Optional[List[Dict[str, Any]]] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = True
    timeout_s: float = 300.0


class CancelRequest(BaseModel):
    request_id: str


def _input_ids_from_req(req: GenerationRequest) -> List[int]:
    assert LOOP is not None
    if req.input_ids:
        return list(req.input_ids)
    messages = req.messages or []
    return LOOP.runner.apply_chat_template(messages)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "sglang-lite-engine"}


@app.get("/readyz")
async def readyz():
    if not READY or LOOP is None or not LOOP.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {"status": "ready", "model": MODEL_NAME}


@app.get("/metrics")
async def metrics():
    if LOOP is None:
        return "# no engine\n"
    stats = LOOP.get_stats()
    lines = [
        "# sglang-lite engine metrics",
        f"sglang_lite_up 1",
        f"sglang_lite_ready {1 if READY else 0}",
        f"sglang_lite_waiting_requests {stats['waiting']}",
        f"sglang_lite_running_requests {stats['running']}",
        f"sglang_lite_engine_steps {stats['steps']}",
        f"sglang_lite_multi_request_batches {stats['multi_request_batches']}",
        f"sglang_lite_cache_hit_count {stats['cache'].get('hit_count', 0)}",
        f"sglang_lite_cache_miss_count {stats['cache'].get('miss_count', 0)}",
        f"sglang_lite_kv_blocks_used {stats['cache'].get('blocks_used', 0)}",
        f"sglang_lite_oom_reject_count {stats['cache'].get('oom_reject_count', 0)}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in list_verified_models()]}


@app.get("/stats")
async def stats():
    if LOOP is None:
        return {}
    return LOOP.get_stats()


@app.post("/v1/cancel")
async def cancel(req: CancelRequest):
    if LOOP is None:
        return JSONResponse({"ok": False}, status_code=503)
    ok = LOOP.cancel(req.request_id)
    return {"ok": ok, "request_id": req.request_id}


@app.post("/v1/generate")
async def generate(req: GenerationRequest, request: Request):
    if LOOP is None or not READY:
        return JSONResponse(
            {"error": "engine not ready"},
            status_code=503,
        )
    try:
        input_ids = _input_ids_from_req(req)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    params = GenParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        stop=req.stop,
        timeout_s=req.timeout_s,
    )
    try:
        submitted = LOOP.submit(req.request_id, input_ids, params)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=429)

    async def ndjson_stream():
        dq = submitted.delta_queue
        while True:
            if await request.is_disconnected():
                LOOP.cancel(req.request_id)
                break
            try:
                item = await asyncio.get_event_loop().run_in_executor(None, dq.get, True, 0.5)
            except Exception:
                continue
            line = json.dumps(item, ensure_ascii=False) + "\n"
            yield line
            if item.get("finish_reason") is not None or item.get("error"):
                break

        if not req.stream:
            return

    if req.stream:
        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    # Non-stream: aggregate
    text_parts: List[str] = []
    finish = "stop"
    usage = None
    error = None
    async for chunk in ndjson_stream():
        data = json.loads(chunk)
        if data.get("text"):
            text_parts.append(data["text"])
        if data.get("finish_reason"):
            finish = data["finish_reason"]
        if data.get("usage"):
            usage = data["usage"]
        if data.get("error"):
            error = data["error"]
    if error:
        return JSONResponse({"error": error}, status_code=500)
    return {
        "text": "".join(text_parts),
        "finish_reason": finish,
        "usage": usage
        or {
            "prompt_tokens": len(input_ids),
            "completion_tokens": 0,
            "total_tokens": len(input_ids),
            "cache_hit_tokens": 0,
        },
    }


def build_loop(model: str, device: str, allow_stub: bool, max_batch_size: int) -> EngineLoop:
    runner = ModelRunner(model, device=device, max_batch=max_batch_size, allow_stub=allow_stub)
    loop = EngineLoop(runner, max_batch_size=max_batch_size)
    loop.start()
    # Warmup: one tiny forward if real model
    if runner._is_real and runner.tokenizer is not None:
        try:
            ids = runner.tokenize("hi")[:4] or [1, 2]
            sub = loop.submit(
                "warmup",
                ids,
                GenParams(max_tokens=1, temperature=0.0, timeout_s=60.0),
            )
            # drain
            while True:
                item = sub.delta_queue.get(timeout=60.0)
                if item.get("finish_reason") is not None:
                    break
        except Exception as e:
            logger.warning("warmup skipped: %s", e)
    return loop


def main(argv: Optional[List[str]] = None) -> None:
    global LOOP, READY, MODEL_NAME
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="sglang-lite engine process")
    p.add_argument("--model", required=True, help="MoE model id or fixture:<path>")
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--allow-stub", action="store_true")
    args = p.parse_args(argv)

    MODEL_NAME = args.model
    LOOP = build_loop(args.model, args.device, args.allow_stub, args.max_batch_size)
    READY = True
    logger.info("engine ready model=%s device=%s port=%s", args.model, args.device, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
