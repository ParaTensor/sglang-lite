# sglang-lite Scope & Feature Classification

This is the authoritative reference for what belongs in core vs. what gets pushed out.

## One-sentence Mission

> An extremely cohesive production token generator for dense models. Everything not required to reliably turn requests into token streams at high throughput stays outside.

## Classification Rules

- **重构 / Must Control** — Re-implement or own the logic. This is where complexity lives and where we gain long-term maintainability + differentiation.
- **Hybrid (过渡)** — Reuse proven pieces (loaders, specific kernel wrappers) but wrap and gradually own the path.
- **直接引用** — Safe to import directly (tokenizer, HF config registry, stable model definitions).
- **不做 (MVP)** — Explicitly out of scope. Implementations that try to add them will be rejected.

## Detailed Table

| Module / Area                  | Specific Feature                        | Classification | Rationale (cohesion / ops / perf)                                                                 | Migration / Alternative                  | Priority |
|--------------------------------|-----------------------------------------|----------------|---------------------------------------------------------------------------------------------------|------------------------------------------|----------|
| **API Layer (Rust)**           | POST /v1/chat/completions + streaming   | **重构**      | The contract. All external behavior decided here. Early scope enforcement.                        | -                                        | P0       |
| **API Layer (Rust)**           | GET /v1/models + /healthz               | **重构**      | Minimal surface.                                                                                  | -                                        | P0       |
| **API Layer (Rust)**           | Request validation & internal mapping   | **重构**      | Define clean GenerationRequest here.                                                              | -                                        | P0       |
| **API Layer (Rust)**           | Tool calls (function calling)           | **重构**      | Only placeholder shape + clear error. Execution belongs in harness.                               | Gateway layer                            | P1       |
| **API Layer**                  | Structured / JSON mode                  | **不做**      | Requires FSM / constrained decoding. Breaks token-factory cohesion.                               | outlines / xgrammar in gateway           | -        |
| **KV & Memory**                | RadixAttention (prefix tree reuse)      | **重构**      | LLM-specific memory problem + scheduler are inseparable. SGLang's biggest strength for agent chat. | - (default)                              | P0       |
| **KV & Memory**                | PagedAttention (block table)            | **重构**      | Pluggable alternative for general workloads. Reuse memory pool infra from Radix impl.             | config switch                            | P1       |
| **KV & Memory**                | Eviction, fragmentation, OOM handling   | **重构**      | Must live with scheduler decisions.                                                               | -                                        | P0       |
| **Scheduling**                 | Continuous batching                     | **重构**      | Core throughput mechanism. Coupled to cache state.                                                | -                                        | P0       |
| **Scheduling**                 | Waiting queue + admission + timeout     | **重构**      | Production robustness is scheduler's job.                                                         | -                                        | P0       |
| **Scheduling**                 | Simple priority (FCFS + basic)          | **重构**      | Fairness without over-engineering.                                                                | -                                        | P0       |
| **Execution**                  | Heavy CUDA graph for decode             | **重构**      | Biggest single lever for low CPU overhead & high decode throughput.                               | -                                        | P0       |
| **Execution**                  | Prefill handling                        | **重构**      | Must coordinate with KV allocation.                                                               | -                                        | P0       |
| **Execution**                  | Basic quantization (BF16/FP8/AWQ)       | Hybrid         | Loader can be reused; the memory accounting + forward path must be owned.                         | SGLang loader pieces + thin wrapper      | P0       |
| **Execution**                  | Attention kernel (flash, triton)        | Hybrid         | Call specific kernels. Do **not** take their high-level schedulers.                               | flashinfer / sgl-kernel / custom triton  | P0       |
| **Model Support**              | Llama-family, Qwen2.5/3 dense, Mistral  | 直接引用      | HF + proven loading paths. Limit to dense only.                                                   | Register only approved model families    | P0       |
| **Model Support**              | Tokenizer (HF)                          | 直接引用      | Mature, no point reimplementing.                                                                  | -                                        | P0       |
| **Model Support**              | New dense model quick add               | **重构**      | Registry + loader hook only. No deep per-model hacks in core.                                     | Simple config + extension point          | P1       |
| **Observability**              | Prometheus (t/s, cache_hit, batch, q)   | **重构**      | Only the metrics that matter for this lite scope.                                                 | -                                        | P0       |
| **Observability**              | Structured logs + request id            | **重构**      | Correlate across unigateway / engine.                                                             | -                                        | P0       |
| **Observability**              | Graceful shutdown + health              | **重构**      | 3am stability.                                                                                        | -                                        | P0       |
| **Advanced**                   | Speculative decoding                    | **不做**      | Complex, variable gain.                                                                           | Optional plugin later                    | -        |
| **Advanced**                   | Prefill / Decode disaggregation         | **不做**      | Distributed systems concern, not single-node token factory.                                       | Future "advanced" mode                   | -        |
| **Advanced**                   | Dynamic multi-LoRA / hot swap           | **不做**      | Huge complexity for narrow win.                                                                   | -                                        | -        |
| **Advanced**                   | Multimodal encoders                     | **不做**      | Completely different data and execution path.                                                     | Separate multimodal service              | -        |
| **Advanced**                   | MoE expert parallelism                  | **不做** (MVP)| Different memory + scheduling model.                                                              | Extension only                           | -        |

## Summary Counts (MVP)

- **重构** (own completely): 12+
- **Hybrid** (reuse pieces, own path): 3
- **直接引用** (infrastructure): ~4 (tokenizers, HF loaders, basic quant loaders, kernel functions)
- **明确不做**: 7+

This ratio is intentional. The value of lite is disciplined reduction of accidental complexity.

## What "Lite" Means in Practice

- Very small number of startup flags (sensible presets).
- Predictable behavior under load.
- Easy to reason about one request's journey through the system.
- Codebase should stay small enough that a single engineer can hold the mental model.

## Enforcement

Any PR that adds a feature from the "不做" column or significantly increases coupling across the API/KV/Scheduler boundary without strong justification will be closed with reference to this document.

See also [architecture.md](./architecture.md).
