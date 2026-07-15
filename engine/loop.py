"""Central continuous-batching engine loop.

HTTP handlers only submit work and consume deltas; this loop owns scheduling.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .kv_cache import RadixCache
from .runner import ModelRunner
from .scheduler import Scheduler, Sequence


@dataclass
class GenParams:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    timeout_s: float = 300.0


@dataclass
class SubmitResult:
    request_id: str
    delta_queue: "queue.Queue[Dict[str, Any]]"
    seq: Sequence


@dataclass
class _Pending:
    request_id: str
    input_ids: List[int]
    params: GenParams
    delta_queue: "queue.Queue[Dict[str, Any]]"
    enqueued_at: float = field(default_factory=time.time)


class EngineLoop:
    """Long-running loop composing RadixCache + Scheduler + ModelRunner."""

    def __init__(
        self,
        runner: ModelRunner,
        radix: Optional[RadixCache] = None,
        max_batch_size: int = 8,
        max_tokens_per_batch: int = 512,
        max_waiting: int = 128,
        max_prompt_tokens: int = 8192,
        idle_sleep_s: float = 0.001,
    ):
        self.runner = runner
        if radix is None:
            radix = RadixCache(
                max_tokens=65536,
                block_size=16,
                num_layers=runner.num_layers,
                num_kv_heads=runner.num_kv_heads,
                head_dim=runner.head_dim,
                dtype=torch.float32 if runner.device == "cpu" else torch.float16,
                device=runner.device if runner.device != "cpu" else "cpu",
            )
        self.radix = radix
        self.scheduler = Scheduler(
            self.radix,
            max_batch_size=max_batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            max_waiting=max_waiting,
            max_prompt_tokens=max_prompt_tokens,
        )
        self.idle_sleep_s = idle_sleep_s

        self._submit_q: queue.Queue[Optional[_Pending]] = queue.Queue()
        self._delta_qs: Dict[str, queue.Queue] = {}
        self._prev_text: Dict[str, str] = {}
        self._deadlines: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._ready = False
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self.steps = 0
        self.multi_request_batches = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="sglang-lite-engine-loop", daemon=True)
        self._thread.start()
        self._ready = True

    def stop(self, drain: bool = True) -> None:
        self._stopping = True
        self._submit_q.put(None)
        if self._thread:
            self._thread.join(timeout=30.0)
        self._ready = False

    def submit(self, request_id: str, input_ids: List[int], params: Optional[GenParams] = None) -> SubmitResult:
        if self._stopping:
            raise RuntimeError("engine is draining; not accepting new requests")
        params = params or GenParams()
        dq: queue.Queue = queue.Queue(maxsize=256)
        pending = _Pending(
            request_id=request_id,
            input_ids=input_ids,
            params=params,
            delta_queue=dq,
        )
        with self._lock:
            self._delta_qs[request_id] = dq
            self._prev_text[request_id] = ""
            self._deadlines[request_id] = time.time() + params.timeout_s
        self._submit_q.put(pending)
        # seq filled after admission; return placeholder — caller uses delta_queue
        return SubmitResult(request_id=request_id, delta_queue=dq, seq=None)  # type: ignore[arg-type]

    def cancel(self, request_id: str) -> bool:
        ok = self.scheduler.cancel(request_id)
        self._emit(
            request_id,
            {
                "text": "",
                "finish_reason": "cancelled",
                "usage": None,
                "error": None,
            },
            final=True,
        )
        return ok

    def _emit(self, request_id: str, payload: Dict[str, Any], final: bool = False) -> None:
        dq = self._delta_qs.get(request_id)
        if dq is None:
            return
        try:
            dq.put(payload, timeout=1.0)
        except queue.Full:
            # backpressure: drop slow client by cancelling
            self.scheduler.cancel(request_id)
        if final:
            with self._lock:
                self._delta_qs.pop(request_id, None)
                self._prev_text.pop(request_id, None)
                self._deadlines.pop(request_id, None)

    def _admit_pending(self) -> None:
        while True:
            try:
                item = self._submit_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._stopping = True
                break
            try:
                seq = self.scheduler.add_request(
                    item.request_id,
                    item.input_ids,
                    max_tokens=item.params.max_tokens,
                    temperature=item.params.temperature,
                    top_p=item.params.top_p,
                    top_k=item.params.top_k,
                    seed=item.params.seed,
                    stop_strings=item.params.stop,
                )
                eos = self.runner.eos_token_id
                if eos is not None:
                    seq.stop_token_ids = [eos]
            except MemoryError as e:
                self._emit(
                    item.request_id,
                    {
                        "text": "",
                        "finish_reason": "error",
                        "usage": None,
                        "error": f"oom: {e}",
                    },
                    final=True,
                )
            except Exception as e:
                self._emit(
                    item.request_id,
                    {
                        "text": "",
                        "finish_reason": "error",
                        "usage": None,
                        "error": str(e),
                    },
                    final=True,
                )

    def _check_timeouts(self) -> None:
        now = time.time()
        expired = [rid for rid, dl in list(self._deadlines.items()) if now > dl]
        for rid in expired:
            self.scheduler.cancel(rid)
            self._emit(
                rid,
                {"text": "", "finish_reason": "timeout", "usage": None, "error": None},
                final=True,
            )

    def _usage(self, seq: Sequence) -> Dict[str, int]:
        prompt = len(seq.input_ids)
        completion = len(seq.output_ids)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cache_hit_tokens": int(getattr(seq, "cache_hit_tokens", 0) or 0),
        }

    def _finish_if_needed(self, seq: Sequence, tok: int) -> bool:
        if seq.cancelled:
            return True
        if len(seq.output_ids) >= seq.max_tokens:
            self.scheduler.mark_finished(seq, "length")
            return True
        if tok in (seq.stop_token_ids or []):
            self.scheduler.mark_finished(seq, "stop")
            return True
        if seq.stop_strings:
            text = self.runner.detokenize(seq.output_ids)
            for s in seq.stop_strings:
                if s and s in text:
                    self.scheduler.mark_finished(seq, "stop")
                    return True
        return seq.finished

    def _run(self) -> None:
        while not self._stopping or self.scheduler.waiting or self.scheduler.running:
            self._admit_pending()
            self._check_timeouts()
            batch, is_prefill = self.scheduler.step()
            if not batch:
                if self._stopping:
                    break
                time.sleep(self.idle_sleep_s)
                continue

            self.steps += 1
            if len(batch) > 1:
                self.multi_request_batches += 1

            try:
                next_tokens = self.runner.run_batch(batch, self.radix, is_prefill)
            except Exception as e:
                for seq in batch:
                    self.scheduler.mark_finished(seq, "error")
                    self._emit(
                        seq.request_id,
                        {
                            "text": "",
                            "finish_reason": "error",
                            "usage": self._usage(seq),
                            "error": str(e),
                        },
                        final=True,
                    )
                continue

            for seq, tok, pre in zip(batch, next_tokens, is_prefill):
                if seq.finished or tok is None:
                    continue
                if pre:
                    self.scheduler.update_after_prefill(seq, [], seq.kv_state)
                self.scheduler.update_after_decode(seq, tok, seq.kv_state)

                prev = self._prev_text.get(seq.request_id, "")
                delta_text = self.runner.detokenize_delta(seq.output_ids, prev)
                self._prev_text[seq.request_id] = self.runner.detokenize(seq.output_ids)

                finished = self._finish_if_needed(seq, tok)
                payload = {
                    "text": delta_text,
                    "token": tok,
                    "finish_reason": seq.finish_reason if finished else None,
                    "usage": self._usage(seq) if finished else None,
                    "error": None,
                }
                self._emit(seq.request_id, payload, final=finished)

            self.scheduler.running = [s for s in self.scheduler.running if not s.finished]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "steps": self.steps,
            "multi_request_batches": self.multi_request_batches,
            "last_batch_trace": list(self.scheduler.last_batch_trace),
            "cache": self.radix.get_cache_stats(),
            "model": self.runner.model_name,
            "device": self.runner.device,
        }
