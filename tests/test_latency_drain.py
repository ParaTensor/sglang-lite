"""Phase 2: TTFT / tok-s aggregates and graceful drain."""

from __future__ import annotations

import time

import torch

from sglang_lite.core import LiteEngine
from sglang_lite.loop import EngineLoop, GenParams
from sglang_lite.metrics_prom import metrics_dict_from_stats, render_prometheus
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence


def test_record_ttft_and_tok_s():
    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
    loop = EngineLoop(runner, max_batch_size=2)
    loop.mark_ready()
    seq = Sequence(seq_id=1, request_id="r1", input_ids=[1, 2, 3])
    seq.created_ts = time.time() - 0.2
    seq.output_ids = [4]
    loop._record_first_token(seq)
    assert loop.latency["ttft_count"] == 1.0
    assert loop.latency["last_ttft_s"] >= 0.15
    assert loop.latency["ttft_le_1"] == 1.0
    time.sleep(0.05)
    seq.output_ids = [4, 5, 6, 7]
    loop._record_request_finished(seq)
    assert loop.latency["requests_completed"] == 1.0
    assert loop.latency["completion_tokens_total"] == 4.0
    assert loop.latency["last_tok_s"] > 0
    stats = loop.get_stats()
    assert stats["latency"]["ttft_avg_s"] > 0
    assert stats["latency"]["tok_s_avg"] > 0
    body = render_prometheus(stats)
    assert "sglang_lite_ttft_seconds_count 1" in body
    assert "sglang_lite_tok_s_avg" in body
    flat = metrics_dict_from_stats(stats)
    assert flat["sglang_lite_requests_completed_total"] == 1.0


def test_drain_rejects_submit():
    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
    loop = EngineLoop(runner, max_batch_size=2)
    loop.mark_ready()
    assert loop.ready is True
    snap = loop.begin_drain()
    assert snap["draining"] is True
    assert loop.ready is False
    try:
        loop.submit("x", [1, 2, 3], GenParams(max_tokens=4))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "drain" in str(e).lower()
    st = loop.drain_status()
    assert st["idle"] is True


def test_lite_engine_drain_api():
    eng = LiteEngine(model_name="stub", device="cpu", allow_stub=True, max_batch_size=2)
    # Inject one finished-request sample without full generate (stub shapes vary).
    seq = Sequence(seq_id=9, request_id="inject", input_ids=[1])
    seq.created_ts = time.time() - 0.05
    seq.output_ids = [2, 3]
    eng.loop._record_first_token(seq)
    eng.loop._record_request_finished(seq)
    stats = eng.get_stats()
    assert stats["latency"]["requests_completed"] >= 1
    assert "ttft_count" in stats["latency"]
    eng.begin_drain()
    assert eng.drain_status()["draining"] is True
    assert eng.get_stats()["draining"] is True
    eng.shutdown()
