"""LiteEngine facade — thin wrapper over EngineLoop for simple/scripts use."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Dict, Generator, List, Optional

import torch

from .kv_cache import RadixCache
from .loop import EngineLoop, GenParams
from .runner import ModelRunner
from .scheduler import Sequence

logger = logging.getLogger("sglang_lite")


def _log_structured(level: str, msg: str, **kwargs):
    log_data = {"level": level, "msg": msg, "ts": time.time(), **kwargs}
    if level == "INFO":
        logger.info(json.dumps(log_data))
    elif level == "WARNING":
        logger.warning(json.dumps(log_data))
    else:
        logger.error(json.dumps(log_data))


class LiteEngine:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
        max_batch_size: int = 4,
        allow_stub: bool = False,
        start_loop: bool = True,
    ):
        self.model_name = model_name
        self.runner = ModelRunner(
            model_name, device=device, max_batch=max_batch_size, allow_stub=allow_stub
        )
        self.radix = RadixCache(
            max_tokens=65536,
            block_size=16,
            num_layers=self.runner.num_layers,
            num_kv_heads=self.runner.num_kv_heads,
            head_dim=self.runner.head_dim,
            dtype=torch.float32 if device == "cpu" else torch.float16,
            device=device if device != "cpu" else "cpu",
        )
        self.loop = EngineLoop(
            self.runner,
            radix=self.radix,
            max_batch_size=max_batch_size,
        )
        self.scheduler = self.loop.scheduler
        if start_loop:
            self.loop.start()

    def tokenize(self, text: str) -> List[int]:
        return self.runner.tokenize(text)

    def detokenize(self, ids: List[int]) -> str:
        return self.runner.detokenize(ids)

    def add_request(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Sequence:
        params = GenParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
        )
        self.loop.submit(request_id, input_ids, params)
        # Wait briefly for admission
        for _ in range(1000):
            seq = self.scheduler._by_request.get(request_id)
            if seq is not None:
                _log_structured(
                    "INFO",
                    "request_added",
                    request_id=request_id,
                    prompt_len=len(input_ids),
                    cached_len=seq.cache_hit_tokens,
                )
                return seq
            time.sleep(0.001)
        raise RuntimeError(f"failed to admit request {request_id}")

    def step(self, max_steps: int = 1) -> List[Dict]:
        # Loop runs in background; step is a no-op observability hook
        return []

    def generate(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict:
        params = GenParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
        )
        sub = self.loop.submit(request_id, input_ids, params)
        text_parts: List[str] = []
        finish = "stop"
        usage = None
        while True:
            item = sub.delta_queue.get(timeout=300.0)
            if item.get("error"):
                raise RuntimeError(item["error"])
            if item.get("text"):
                text_parts.append(item["text"])
            if item.get("finish_reason") is not None:
                finish = item["finish_reason"]
                usage = item.get("usage")
                break
        return {
            "request_id": request_id,
            "text": "".join(text_parts),
            "finish_reason": finish,
            "finished": True,
            "usage": usage
            or {
                "prompt_tokens": len(input_ids),
                "completion_tokens": 0,
                "total_tokens": len(input_ids),
                "cache_hit_tokens": 0,
            },
        }

    def generate_stream(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Generator[Dict, None, None]:
        params = GenParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
        )
        sub = self.loop.submit(request_id, input_ids, params)
        while True:
            item = sub.delta_queue.get(timeout=300.0)
            yield {
                "request_id": request_id,
                "token": item.get("token"),
                "text": item.get("text") or "",
                "finished": item.get("finish_reason") is not None,
                "finish_reason": item.get("finish_reason"),
                "usage": item.get("usage"),
                "error": item.get("error"),
            }
            if item.get("finish_reason") is not None or item.get("error"):
                break

    def cancel(self, request_id: str) -> bool:
        return self.loop.cancel(request_id)

    def get_stats(self) -> Dict:
        return self.loop.get_stats()

    def shutdown(self) -> None:
        self.loop.stop()
