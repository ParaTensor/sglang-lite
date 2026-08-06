"""Phase 2: structured request-id JSON logs."""

from __future__ import annotations

import json
import logging
import time

from sglang_lite.loop import EngineLoop
from sglang_lite.reqlog import log_event
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def test_log_event_json_line():
    h = _ListHandler()
    log = logging.getLogger("sglang_lite.req")
    log.addHandler(h)
    log.setLevel(logging.INFO)
    try:
        log_event(
            "request_submit",
            request_id="rid-1",
            prompt_tokens=12,
            max_tokens=32,
        )
        assert h.records, "expected a log line"
        obj = json.loads(h.records[-1])
        assert obj["event"] == "request_submit"
        assert obj["request_id"] == "rid-1"
        assert obj["prompt_tokens"] == 12
        assert "ts" in obj
    finally:
        log.removeHandler(h)


def test_loop_emit_submit_first_finish():
    h = _ListHandler()
    log = logging.getLogger("sglang_lite.req")
    log.addHandler(h)
    log.setLevel(logging.INFO)
    try:
        runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
        loop = EngineLoop(runner, max_batch_size=2)
        loop.mark_ready()
        loop.submit("rid-loop", [1, 2, 3, 4], None)
        events = [json.loads(r)["event"] for r in h.records if r.startswith("{")]
        assert "request_submit" in events

        seq = Sequence(seq_id=1, request_id="rid-loop", input_ids=[1, 2, 3, 4])
        seq.created_ts = time.time() - 0.05
        seq.output_ids = [9]
        loop._record_first_token(seq)
        seq.output_ids = [9, 10, 11]
        seq.finish_reason = "length"
        loop._record_request_finished(seq)
        events = [json.loads(r) for r in h.records if r.startswith("{")]
        kinds = [e["event"] for e in events]
        assert "request_first_token" in kinds
        assert "request_finish" in kinds
        fin = [e for e in events if e["event"] == "request_finish"][-1]
        assert fin["request_id"] == "rid-loop"
        assert fin["completion_tokens"] == 3
        assert fin["finish_reason"] == "length"
    finally:
        log.removeHandler(h)
