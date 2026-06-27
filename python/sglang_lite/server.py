"""
Internal HTTP server for the real sglang-lite Python execution core.

Uses the full LiteEngine (Radix + Scheduler + real Runner).

Run:
    PYTHONPATH=python python -m sglang_lite.server --port 9001 --model stub
"""

from __future__ import annotations

import argparse
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine.engine import LiteEngine

app = FastAPI(title="sglang-lite Python Core (real engine)")

# Global engine
ENGINE: Optional[LiteEngine] = None


class GenRequest(BaseModel):
    request_id: Optional[str] = None
    model: str = "stub"
    messages: Optional[List[dict]] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


class GenResult(BaseModel):
    text: str
    finish_reason: Optional[str] = "stop"
    usage: dict


def _extract_input_ids(req: GenRequest) -> List[int]:
    if req.input_ids:
        return req.input_ids

    # crude extraction from messages (last user message)
    last_user = ""
    if req.messages:
        for m in reversed(req.messages):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_user = content
                break
    if not last_user:
        last_user = "Hello"

    if ENGINE and ENGINE.runner.tokenizer:
        return ENGINE.runner.tokenize(last_user)
    return [hash(c) % 32000 for c in last_user[:80]]


@app.post("/generate", response_model=GenResult)
async def generate(req: GenRequest):
    global ENGINE
    rid = req.request_id or f"req-{uuid.uuid4().hex[:12]}"
    input_ids = _extract_input_ids(req)

    result = ENGINE.generate(
        rid,
        input_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return GenResult(
        text=result["text"],
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage", {}),
    )


@app.post("/generate_stream")
async def generate_stream(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:12]}"
    input_ids = _extract_input_ids(req)

    async def event_generator():
        for delta in ENGINE.generate_stream(
            rid, input_ids, max_tokens=req.max_tokens, temperature=req.temperature
        ):
            import json
            yield f"data: {json.dumps(delta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    stats = ENGINE.get_stats() if ENGINE else {}
    return {
        "status": "ok",
        "phase": "0",
        "core": "sglang-lite-python",
        "stats": stats,
    }


@app.get("/stats")
async def stats():
    return ENGINE.get_stats() if ENGINE else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model", type=str, default="stub")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    global ENGINE
    ENGINE = LiteEngine(
        model_name=args.model,
        device=args.device,
        max_batch_size=4,
    )

    print(f"[sglang-lite-core] Real engine starting on :{args.port}")
    print(f"  model={args.model}  device={args.device}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
