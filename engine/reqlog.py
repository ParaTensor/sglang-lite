"""Structured request-id logging for sglang-lite (Phase 2).

Emits one JSON object per line on logger ``sglang_lite.req`` so scrapers and
`jq` can filter by ``request_id`` across process → loop → finish.

Enable verbose body fields with env ``SGLANG_LITE_LOG_JSON=1`` (default on for
structured events; set to ``0`` to silence).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

_logger = logging.getLogger("sglang_lite.req")


def _enabled() -> bool:
    v = os.environ.get("SGLANG_LITE_LOG_JSON", "1").lower()
    return v not in ("0", "false", "no", "off")


def log_event(
    event: str,
    *,
    request_id: Optional[str] = None,
    **fields: Any,
) -> None:
    """Log a structured event. Always includes ``ts`` and ``event``."""
    if not _enabled():
        return
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "event": event,
    }
    if request_id is not None:
        payload["request_id"] = str(request_id)
    for k, v in fields.items():
        if v is None:
            continue
        # Keep JSON serializable / compact.
        if isinstance(v, float):
            payload[k] = round(v, 6)
        elif isinstance(v, (str, int, bool)):
            payload[k] = v
        else:
            payload[k] = str(v)
    try:
        _logger.info("%s", json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    except Exception:
        _logger.info("event=%s request_id=%s", event, request_id)
