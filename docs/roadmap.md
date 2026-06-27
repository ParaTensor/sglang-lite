# sglang-lite Roadmap

## Phase 0 — Core Verification (Current Focus)

Goal: A runnable system that proves the architecture and can serve real (small) models with basic continuous batching + Radix skeleton.

**Deliverables**
- [x] Project skeleton + docs (README, architecture, scope, roadmap)
- [ ] Rust axum OpenAI server
  - POST /v1/chat/completions (non-stream + stream via SSE)
  - GET /v1/models
  - GET /healthz
  - Strict minimal request model + early reject
  - Clean internal GenerationRequest sent to backend
- [ ] Internal protocol (GenerationRequest / deltas) defined in both Rust and Python (shared or duplicated with tests)
- [ ] Python execution package with clear interfaces:
  - `KVCacheManager` (Radix skeleton + allocate/evict)
  - `Scheduler` (waiting + running, step())
  - `ModelRunner` stub + simple real path (HF generate for tiny model on CPU/CUDA when available)
- [ ] Integration: Rust can talk to Python (HTTP first)
- [ ] Basic metrics emitted (even if stubbed): queue depth, batch size, fake t/s
- [ ] Can run a full chat roundtrip for a small open model (e.g. Qwen2.5-0.5B or TinyLlama) and observe logs
- [ ] CI skeleton (cargo check + fmt + clippy, python lint + basic test)

**Success criteria**
- `curl .../v1/chat/completions` returns valid OpenAI shaped response (streaming works)
- Code for core pieces is readable and < a few thousand LOC total for engine
- Easy to point at a real model dir and get tokens

**Non goals in Phase 0**
- 70B perf
- Real Radix tree with GPU pages working end to end
- Production metrics
- Any multimodal or structured output

## Phase 1 — Production Shell + Real Core

- Full RadixAttention implementation with block / page management, prefix matching, eviction policy
- Scheduler that actually does continuous batching (add prefill, promote decode sequences, manage seq lens)
- CUDA graph capture around decode forward (the big win)
- Real quant support (at least AWQ + FP8 path with correct memory accounting)
- Good model support matrix for Llama-3.1/3.2/4 + Qwen2.5/3 dense (up to 72B single node + TP)
- Prometheus + /metrics, structured logging, graceful drain
- Configuration system with "lite" preset (very few flags)
- Request timeout, max concurrent, queue limits
- Error normalization that matches OpenAI behavior closely
- Basic benchmark script (share prefix chat workload)

**Stretch**
- Optional Paged strategy selectable at startup
- Simple model registry extension point

## Phase 2 — Polish + Integration + Hardening

- Deep integration tests with unigateway (semantic routing + KV affinity hints)
- Rust control plane enhancements (more early filters, intent metadata passthrough)
- Optional: move hot scheduler or cache management bits to Rust via PyO3
- Performance parity target vs SGLang on prefix-heavy chat workloads (within 10-15% on same hardware)
- Packaging: docker images, helm chart skeleton
- Documentation: operator guide, "when to choose sglang-lite vs full SGLang"
- Formal support policy for model families

## Later / Optional

- gRPC control plane (alternative to HTTP)
- Disaggregated mode (advanced)
- Speculative as plugin
- Rust-only hot path experiment (full control)

## Non-Roadmap (永远不做 in core)

See scope.md "不做" list. If a feature is desired by many users, the correct answer is "put it in the gateway/harness layer or as a separate sidecar".

## How to Track Progress

- Issues / milestones will mirror the phases.
- Every merged PR should update this file or link the changed scope.

Current status: **Phase 0 largely complete** (framework + control plane solid, real Radix + scheduler + runner working on CPU, full-stack integration verified, prefix sharing demo passes).
