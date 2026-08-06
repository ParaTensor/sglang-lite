"""
Configuration system for sglang-lite Phase 1.

Supports:
- Environment variables
- "lite" preset (sensible defaults for simple deployments)
- Basic validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    model: str = "stub"
    device: str = "cpu"
    port: int = 9001
    max_batch_size: int = 4
    max_concurrent: int = 32
    max_tokens_default: int = 128
    request_timeout: float = 300.0  # seconds
    queue_timeout: float = 60.0
    log_level: str = "INFO"
    # Future: quant, etc.

    @classmethod
    def from_env(cls, preset: str = "lite") -> "Config":
        """Load config from env. ``preset=lite`` applies stable deploy defaults.

        Also applies production safety defaults when missing:
        ``SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`` (official sparse main path).
        """
        if preset == "lite":
            # Lite preset: minimal, stable, low resource, official attention path.
            base = {
                "max_batch_size": 4,
                "max_concurrent": 32,
                "request_timeout": 300.0,
                "queue_timeout": 60.0,
                "max_tokens_default": 128,
                "device": "cuda",
                "port": 9001,
                "log_level": "INFO",
            }
            # Official main path unless operator explicitly set FI envs.
            os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")
            os.environ.setdefault("SGLANG_LITE_LOG_JSON", "1")
        else:
            base = {}

        return cls(
            model=os.getenv("SGLANG_LITE_MODEL", base.get("model", "stub")),
            device=os.getenv("SGLANG_LITE_DEVICE", base.get("device", "cpu")),
            port=int(os.getenv("SGLANG_LITE_PORT", base.get("port", 9001))),
            max_batch_size=int(
                os.getenv("SGLANG_LITE_MAX_BATCH_SIZE", base.get("max_batch_size", 4))
            ),
            max_concurrent=int(
                os.getenv("SGLANG_LITE_MAX_CONCURRENT", base.get("max_concurrent", 32))
            ),
            max_tokens_default=int(
                os.getenv("SGLANG_LITE_MAX_TOKENS", base.get("max_tokens_default", 128))
            ),
            request_timeout=float(
                os.getenv("SGLANG_LITE_REQUEST_TIMEOUT", base.get("request_timeout", 300.0))
            ),
            queue_timeout=float(
                os.getenv("SGLANG_LITE_QUEUE_TIMEOUT", base.get("queue_timeout", 60.0))
            ),
            log_level=os.getenv("SGLANG_LITE_LOG_LEVEL", base.get("log_level", "INFO")),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
