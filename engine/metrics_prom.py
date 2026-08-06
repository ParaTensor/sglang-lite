"""Render EngineLoop stats as Prometheus text exposition (0.0.4).

Used by ``sglang_lite.process`` ``GET /metrics``. Keep flat numeric gauges only;
no labels beyond fixed names so scrape stays trivial for lite deployments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _num(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def render_prometheus(
    stats: Optional[Mapping[str, Any]],
    *,
    ready: bool = True,
    tp_world_size: int = 1,
) -> str:
    """Return Prometheus exposition body for engine stats."""
    stats = dict(stats or {})
    cache = dict(stats.get("cache") or {})
    dual = dict(stats.get("dual_pool") or {})
    v4p = dict(stats.get("v4_prefix") or {})

    lines: List[str] = [
        "# HELP sglang_lite_up Engine process up",
        "# TYPE sglang_lite_up gauge",
        "sglang_lite_up 1",
        "# HELP sglang_lite_ready 1 if engine accepted load",
        "# TYPE sglang_lite_ready gauge",
        f"sglang_lite_ready {_num(ready or stats.get('ready')):.0f}",
        "# HELP sglang_lite_waiting_requests Scheduler waiting queue length",
        "# TYPE sglang_lite_waiting_requests gauge",
        f"sglang_lite_waiting_requests {_num(stats.get('waiting')):.0f}",
        "# HELP sglang_lite_running_requests Scheduler running set size",
        "# TYPE sglang_lite_running_requests gauge",
        f"sglang_lite_running_requests {_num(stats.get('running')):.0f}",
        "# HELP sglang_lite_engine_steps Continuous batching steps",
        "# TYPE sglang_lite_engine_steps counter",
        f"sglang_lite_engine_steps {_num(stats.get('steps')):.0f}",
        "# HELP sglang_lite_multi_request_batches Batches with >1 request",
        "# TYPE sglang_lite_multi_request_batches counter",
        f"sglang_lite_multi_request_batches {_num(stats.get('multi_request_batches')):.0f}",
        "# HELP sglang_lite_model_forward_count ModelRunner forward calls",
        "# TYPE sglang_lite_model_forward_count counter",
        f"sglang_lite_model_forward_count {_num(stats.get('model_forward_count')):.0f}",
        "# HELP sglang_lite_cache_hit_count Radix prefix hit count",
        "# TYPE sglang_lite_cache_hit_count counter",
        f"sglang_lite_cache_hit_count {_num(cache.get('hit_count')):.0f}",
        "# HELP sglang_lite_cache_miss_count Radix prefix miss count",
        "# TYPE sglang_lite_cache_miss_count counter",
        f"sglang_lite_cache_miss_count {_num(cache.get('miss_count')):.0f}",
        "# HELP sglang_lite_kv_blocks_used KV page blocks in use",
        "# TYPE sglang_lite_kv_blocks_used gauge",
        f"sglang_lite_kv_blocks_used {_num(cache.get('blocks_used')):.0f}",
        "# HELP sglang_lite_oom_reject_count Requests rejected for KV OOM",
        "# TYPE sglang_lite_oom_reject_count counter",
        f"sglang_lite_oom_reject_count {_num(cache.get('oom_reject_count')):.0f}",
        "# HELP sglang_lite_tp_world_size Tensor parallel world size",
        "# TYPE sglang_lite_tp_world_size gauge",
        f"sglang_lite_tp_world_size {int(tp_world_size)}",
        "# HELP sglang_lite_v4_hybrid 1 if DeepSeek-V4 Hybrid path",
        "# TYPE sglang_lite_v4_hybrid gauge",
        f"sglang_lite_v4_hybrid {_num(stats.get('v4_hybrid')):.0f}",
        # Dual-pool / 0c metrics
        "# HELP sglang_lite_dual_write_count Dual-pool write events",
        "# TYPE sglang_lite_dual_write_count counter",
        f"sglang_lite_dual_write_count {_num(dual.get('dual_write_count')):.0f}",
        "# HELP sglang_lite_dual_write_tokens Tokens dual-written to pages",
        "# TYPE sglang_lite_dual_write_tokens counter",
        f"sglang_lite_dual_write_tokens {_num(dual.get('dual_write_tokens')):.0f}",
        "# HELP sglang_lite_dual_hit_count Prefix hits that forked dual-pool pages",
        "# TYPE sglang_lite_dual_hit_count counter",
        f"sglang_lite_dual_hit_count {_num(dual.get('dual_hit_count', v4p.get('dual_hit_count'))):.0f}",
        "# HELP sglang_lite_dual_append_count Decode appends into dual-pool",
        "# TYPE sglang_lite_dual_append_count counter",
        f"sglang_lite_dual_append_count {_num(dual.get('dual_append_count')):.0f}",
        "# HELP sglang_lite_dual_restore_count Prefix-hit restores from pages",
        "# TYPE sglang_lite_dual_restore_count counter",
        f"sglang_lite_dual_restore_count {_num(dual.get('dual_restore_count')):.0f}",
        "# HELP sglang_lite_dual_stage_count Page-primary stages before decode (0c-4)",
        "# TYPE sglang_lite_dual_stage_count counter",
        f"sglang_lite_dual_stage_count {_num(dual.get('dual_stage_count')):.0f}",
        "# HELP sglang_lite_dual_has_restore_bf16 1 if bf16 restore pool allocated",
        "# TYPE sglang_lite_dual_has_restore_bf16 gauge",
        f"sglang_lite_dual_has_restore_bf16 {_num(dual.get('has_restore_bf16')):.0f}",
        "# HELP sglang_lite_prefix_dual_primary Entries stored as dual_primary slim",
        "# TYPE sglang_lite_prefix_dual_primary gauge",
        f"sglang_lite_prefix_dual_primary {_num(v4p.get('prefix_dual_primary')):.0f}",
    ]
    # Backend names as info-like gauges (1) with fixed metric names only — encode
    # backend id in the metric suffix would explode cardinality; expose length-0
    # text via comments instead for humans scraping raw /metrics.
    sparse = str(stats.get("sparse_mla_backend") or "none")
    moe = str(stats.get("moe_gemm_backend") or "none")
    arch = str(stats.get("arch_family") or "unknown")
    lines.append(f"# sparse_mla_backend={sparse} moe_gemm_backend={moe} arch={arch}")
    lines.append("")
    return "\n".join(lines)


def metrics_dict_from_stats(stats: Mapping[str, Any]) -> Dict[str, float]:
    """Flat name→value map (for tests / JSON tooling)."""
    body = render_prometheus(stats, ready=bool(stats.get("ready", True)))
    out: Dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out
