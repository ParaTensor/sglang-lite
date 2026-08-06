"""Phase 2: Prometheus metrics include dual_pool / 0c counters."""

from __future__ import annotations

from sglang_lite.metrics_prom import metrics_dict_from_stats, render_prometheus


def test_render_includes_dual_stage_and_core_gauges():
    stats = {
        "ready": True,
        "waiting": 1,
        "running": 2,
        "steps": 10,
        "multi_request_batches": 3,
        "model_forward_count": 20,
        "v4_hybrid": True,
        "sparse_mla_backend": "official_sparse_attn",
        "moe_gemm_backend": "flashinfer_b12x",
        "arch_family": "sm120",
        "cache": {
            "hit_count": 4,
            "miss_count": 5,
            "blocks_used": 12,
            "oom_reject_count": 0,
        },
        "dual_pool": {
            "dual_write_count": 1,
            "dual_write_tokens": 5,
            "dual_hit_count": 1,
            "dual_append_count": 14,
            "dual_restore_count": 1,
            "dual_stage_count": 14,
            "has_restore_bf16": True,
        },
        "v4_prefix": {
            "prefix_dual_primary": 1,
            "dual_hit_count": 1,
        },
        "draining": False,
        "latency": {
            "requests_completed": 3,
            "completion_tokens_total": 30,
            "ttft_sum_s": 0.6,
            "ttft_count": 3,
            "ttft_avg_s": 0.2,
            "last_ttft_s": 0.15,
            "tok_s_avg": 12.0,
            "last_tok_s": 11.0,
            "ttft_le_0_1": 0,
            "ttft_le_0_5": 3,
            "ttft_le_1": 3,
            "ttft_le_5": 3,
            "ttft_le_inf": 3,
        },
    }
    body = render_prometheus(stats, ready=True, tp_world_size=8)
    assert "sglang_lite_up 1" in body
    assert "sglang_lite_dual_stage_count 14" in body
    assert "sglang_lite_dual_write_count 1" in body
    assert "sglang_lite_dual_restore_count 1" in body
    assert "sglang_lite_tp_world_size 8" in body
    assert "sglang_lite_v4_hybrid 1" in body
    assert "official_sparse_attn" in body
    assert "sglang_lite_ttft_seconds_avg 0.2" in body
    assert "sglang_lite_tok_s_avg 12" in body
    assert "sglang_lite_requests_completed_total 3" in body

    flat = metrics_dict_from_stats(stats)
    assert flat["sglang_lite_dual_stage_count"] == 14.0
    assert flat["sglang_lite_cache_hit_count"] == 4.0
    assert flat["sglang_lite_waiting_requests"] == 1.0
    assert flat["sglang_lite_ttft_seconds_count"] == 3.0


def test_render_empty_stats_safe():
    body = render_prometheus(None, ready=False, tp_world_size=1)
    assert "sglang_lite_up 1" in body
    assert "sglang_lite_ready 0" in body
    assert "sglang_lite_dual_stage_count 0" in body
