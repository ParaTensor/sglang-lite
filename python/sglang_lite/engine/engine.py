"""
LiteEngine - the real orchestrator for Phase 0+.

It wires together:
- RadixCache
- Scheduler
- ModelRunner

And provides a clean generate() and generate_stream() API
that the internal server (and later Rust) can call.
"""

from __future__ import annotations

import time
from typing import Dict, Generator, List, Optional

from .kv_cache import RadixCache
from .runner import ModelRunner
from .scheduler import Scheduler, Sequence


class LiteEngine:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
        max_batch_size: int = 4,
        max_tokens: int = 2048,
    ):
        self.radix = RadixCache(max_tokens=max_tokens)
        self.scheduler = Scheduler(self.radix, max_batch_size=max_batch_size)
        self.runner = ModelRunner(model_name, device=device, max_batch=max_batch_size)

        self._seq_map: Dict[str, Sequence] = {}   # request_id -> Sequence

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
    ) -> Sequence:
        seq = self.scheduler.add_request(request_id, input_ids)
        seq.max_tokens = max_tokens          # attach generation params
        seq.temperature = temperature
        self._seq_map[request_id] = seq
        return seq

    def step(self, max_steps: int = 1) -> List[Dict]:
        """
        Run the engine for up to max_steps scheduler steps.
        Returns list of finished or intermediate results.
        """
        results = []
        for _ in range(max_steps):
            batch, is_prefill = self.scheduler.step()
            if not batch:
                break

            next_tokens = self.runner.run_batch(batch, self.radix, is_prefill)

            for seq, tok, pre in zip(batch, next_tokens, is_prefill):
                if seq.finished or tok is None:
                    continue

                if pre:
                    # After prefill we usually emit the first real token
                    # and move to decode phase
                    self.scheduler.update_after_prefill(seq, [], seq.kv_state or [])
                    # We already got the first token from prefill in many cases
                    # For simplicity we treat the returned tok as first generated
                    self.scheduler.update_after_decode(seq, tok, seq.kv_state or [])
                else:
                    self.scheduler.update_after_decode(seq, tok, seq.kv_state or [])

                # Check stopping conditions
                if len(seq.output_ids) >= getattr(seq, "max_tokens", 128):
                    self.scheduler.mark_finished(seq, "length")
                    results.append(self._make_result(seq, finished=True))
                elif tok == self._get_eos_token():
                    self.scheduler.mark_finished(seq, "stop")
                    results.append(self._make_result(seq, finished=True))
                else:
                    # still running
                    results.append(self._make_result(seq, finished=False))

            # remove finished from running list
            self.scheduler.running = [s for s in self.scheduler.running if not s.finished]

        return results

    def generate(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict:
        """Blocking generation (convenient for /generate)."""
        seq = self.add_request(request_id, input_ids, max_tokens, temperature)

        # Run until this sequence finishes
        while not seq.finished:
            self.step(max_steps=1)
            if len(seq.output_ids) > max_tokens + 10:
                break

        if not seq.finished:
            self.scheduler.mark_finished(seq, "length")

        return self._make_result(seq, finished=True)

    def generate_stream(
        self,
        request_id: str,
        input_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[Dict, None, None]:
        """Yields token deltas one by one. Good for SSE."""
        seq = self.add_request(request_id, input_ids, max_tokens, temperature)

        last_len = 0
        while not seq.finished:
            self.step(max_steps=1)

            current_out = seq.output_ids
            new_tokens = current_out[last_len:]
            last_len = len(current_out)

            for t in new_tokens:
                text = self.detokenize([t])
                yield {
                    "request_id": request_id,
                    "token": t,
                    "text": text,
                    "finished": seq.finished,
                    "finish_reason": seq.finish_reason or None,
                }

            if seq.finished:
                break

        # final
        yield {
            "request_id": request_id,
            "token": None,
            "text": "",
            "finished": True,
            "finish_reason": seq.finish_reason or "stop",
            "usage": self._usage(seq),
        }

    def get_stats(self) -> Dict:
        cache_stats = self.radix.get_cache_stats()
        return {
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "cache": cache_stats,
            "model": self.runner.model_name,
            "device": self.runner.device,
        }

    # --- helpers ---

    def _make_result(self, seq: Sequence, finished: bool) -> Dict:
        text = self.detokenize(seq.output_ids)
        return {
            "request_id": seq.request_id,
            "text": text,
            "output_ids": list(seq.output_ids),
            "finish_reason": seq.finish_reason if finished else None,
            "finished": finished,
            "usage": self._usage(seq),
        }

    def _usage(self, seq: Sequence) -> Dict:
        prompt = len(seq.input_ids)
        completion = len(seq.output_ids)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _get_eos_token(self) -> int:
        if self.runner.tokenizer is not None and hasattr(self.runner.tokenizer, "eos_token_id"):
            return self.runner.tokenizer.eos_token_id or 2
        return 2
