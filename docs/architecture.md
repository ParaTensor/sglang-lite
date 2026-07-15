# sglang-lite Architecture

## Guiding Principle: High Cohesion + Composability

sglang-lite remains a **pure library** whose job is to provide the three core capabilities for a MoE "Token Factory". These capabilities are not SGLang-specific; they are the same engine-level responsibilities found in vLLM:

| Engine-neutral capability | sglang-lite / SGLang-oriented implementation | vLLM implementation |
| --- | --- | --- |
| KV lifecycle + prefix reuse | RadixKVCache / RadixAttention | KVCacheManager + APC + PagedAttention blocks |
| Continuous token scheduling | BatchingScheduler | vLLM V1 Scheduler |
| Model execution | MoEModelRunner + CUDA graph/kernel backend | GPUModelRunner / Worker + CUDA graph/compile |

The shared capability model does **not** imply implementation or internal API compatibility. Each backend owns its scheduler state, cache indexing, block lifecycle, and execution path. UniGateway should depend on capability and protocol contracts, not these internal classes.

It also does not imply complete product equivalence. FlashInfer can provide attention, KV, sampling, and related GPU kernels, but it does not provide the scheduler, cache policy, model registry/loaders, distributed executor, broad feature surface, or production compatibility matrix that make up vLLM as a complete engine. `UniGateway + sglang-lite + FlashInfer` can replace vLLM only inside sglang-lite's deliberately narrow supported envelope.

sglang-lite realizes the three capabilities with these deeply coupled pieces:

1. **RadixKVCache** (was KVCacheManager)
   - Composed of: RadixTree + KVAllocator + MemoryBudget + EvictionPolicy
2. **BatchingScheduler** (was Scheduler)
   - Composed of: SequenceTable + BatchFormer + (AdmissionController can be peeled to unigateway)
3. **MoEModelRunner** (was ModelRunner)
   - Composed of: ModelLoader + MoERouter + PrefillExecutor + DecodeExecutor + KernelBackend

unigateway (or any host) owns:
- The main orchestration loop (what used to be LiteEngine)
- Admission control, request queuing, timeouts
- Higher-level batching policy selection
- Metrics collection, logging context, lifecycle

sglang-lite only exports the fine-grained building blocks + a small default composition helper.

## Layered Architecture (MVP) — sglang-lite as pure engine, unigateway as driver

**sglang-lite is now a pure library** (the "Token Factory").

All serving, routing, auth, rate-limiting, configuration, advanced observability, and driver integration are peeled to **unigateway** (or thin dedicated layers).

unigateway acts as the **backend driver** for sglang-lite (the actual driver code lives in the unigateway repository):
- It loads and manages the sglang-lite engine via standard protocols (HTTP or gRPC). Direct embedding is avoided to preserve unigateway's generality as an SDK.
- It handles the OpenAI surface, streaming, validation, metrics, routing, auth, etc.
- Any "sglang-lite backend" registration and connection logic moves to unigateway.
- sglang-lite only exposes the minimal engine API.

**Important note for the UniGateway team**:
- Please keep UniGateway's core abstractions (`ProviderDriver`, registry, routing, etc.) completely general.
- Do **not** introduce sglang-lite specific concepts into the core engine or protocol layers.
- All MoE/Radix-specific logic must stay inside the sglang-lite driver implementation.
- Treat sglang-lite the same way you would treat any other local or remote LLM backend, including vLLM.
- Prefer generic local-inference concepts (`PrefixCache`, `BlockKVCache`, `ContinuousScheduler`, `ModelExecutor`, `BackendCapabilities`) over SGLang/Radix-specific terms in UniGateway core.
- **Critical boundary**: PyO3, direct Python embedding, or any in-process library calls are explicitly not used. All communication must go through HTTP or gRPC. This boundary is to keep UniGateway as a general-purpose embeddable SDK.
- Detailed requirements document: `docs/unigateway-sglang-lite-requirements.md`
- vLLM positioning document: `docs/vllm-positioning-compatibility.md`

## vLLM-compatible positioning

sglang-lite's external position should be compatible with vLLM as another `local-inference` backend, while its internal implementation remains MoE-only and Radix-first.

The primary conclusion is that the original three sglang-lite core capabilities also apply to vLLM. Compatibility therefore starts from a shared capability model:

1. KV allocation, eviction, block lifecycle, and optional prefix reuse.
2. Continuous scheduling across prefill/decode token work.
3. Model execution, kernel selection, CUDA graph/compile, and MoE routing where supported.

RadixAttention is only one implementation of the first capability; it must not define the generic backend contract.

Compatibility target:

- **Protocol**: keep the minimal OpenAI-compatible chat completions surface aligned with vLLM's local server shape where in scope: `/v1/chat/completions`, streaming chunks, `/v1/models`, health, request-id passthrough, and OpenAI-shaped errors.
- **Capabilities**: expose generic backend capabilities (`chat_completions`, `streaming`, `prefix_cache`, `prefix_cache_metric`, `moe_optimized`) rather than backend-name-specific checks.
- **Metrics**: treat `usage.cache_hit_tokens` as a generic prefix-cache metric. It should be usable for both sglang-lite RadixKVCache and vLLM Automatic Prefix Caching.
- **KV abstraction**: keep RadixKVCache as the default internal structure, but make block/page terminology compatible with vLLM-style KVCacheManager/PagedAttention concepts.

Non-goals:

- Do not implement vLLM's broad API surface in sglang-lite core (`/v1/responses`, audio, multimodal, embeddings for non-core models).
- Do not expose vLLM-specific request parameters as stable sglang-lite contract unless they are part of the shared minimal sampling surface.
- Do not import vLLM internals or copy its scheduler/KV manager wholesale.
- Do not add dense, multimodal, LoRA, speculative decoding, or disaggregated serving to core for the sake of vLLM compatibility.

```
Clients / unigateway (full gateway + driver)
  • Owns: HTTP server, routing, auth, metrics, admission control,
    request lifecycle, higher-level policies
  • Composes sglang-lite building blocks
        ↓ uses
sglang-lite (pure library — only the three building blocks)

  RadixKVCache
    ├── RadixTree
    ├── KVAllocator / MemoryManager
    └── EvictionPolicy (can be swapped)

  BatchingScheduler
    ├── SequenceTable
    └── BatchFormer (unigateway can supply custom policy)

  MoEModelRunner
    ├── ModelLoader
    ├── MoERouter
    ├── PrefillExecutor / DecodeExecutor
    └── KernelBackend

Everything else (serving, config, observability export, driver glue)
is in unigateway or host application.
```

**Key peelings to unigateway:**
- All HTTP serving and OpenAI surface
- Auth, rate limit, routing, semantic routing
- Advanced metrics, structured logging, tracing
- Configuration and presets
- Graceful shutdown, health, timeouts coordination
- Driver glue (how to load/call sglang-lite engine)

sglang-lite only owns the three high-cohesion pieces inside the MoE engine.

## Internal Protocol (to be defined precisely in code)

**GenerationRequest** (from Rust → Python)

- request_id: string (for streaming correlation)
- model: string
- input_ids: Vec<u32>   // already tokenized (or raw prompt + do tokenization in python)
- sampling_params: { temperature, top_p, top_k, max_tokens, stop, ... }
- priority?
- created_ts

**GenerationResponse** (stream of)

- request_id
- token_id / text delta
- finish_reason?
- usage (on last)

## Key Data Structures (Python core sketch)

```python
# radix cache
class RadixCache:
    root: Node
    # each node: token seq slice, child map, refcount, kv pages

class Sequence:
    seq_id: int
    input_ids: list[int]
    output_ids: list[int]
    # cache hit length, allocated blocks, etc.

class Scheduler:
    waiting: deque
    running: list[Sequence]
    # radix cache ref

    def add_request(...)
    def step() -> Batch   # returns batch of tokens to run
```

## Communication Options (evolution)

| Stage   | Mechanism          | Pros                          | Cons                     | When                |
|---------|--------------------|-------------------------------|--------------------------|---------------------|
| MVP     | Local HTTP (FastAPI from Python) | Easy debug, separate process | Serialization + latency  | Now                 |
| Phase1  | gRPC (tonic + py)  | Typed, efficient, streaming  | More boilerplate         | When core stabilizes|
| Later (not planned) | Direct embedding (e.g. PyO3) | Avoided | Conflicts with unigateway generality | Not considered for unigateway driver |

## Model Support Strategy

- Only register popular **MoE** models (dense models are out of scope).
- Use HF `AutoModelForCausalLM` + `AutoTokenizer` initially for loading.
- Later: direct safetensors weight loading + custom modeling files for speed (like nano-vLLM style).
- Extension point: small model registry + loader trait.
- vLLM compatibility is handled through protocol/capability mapping, not by expanding sglang-lite model scope to match vLLM.

## Observability

Rust layer owns:
- Request lifecycle events
- External API metrics (latency, error rate)
- High level engine metrics forwarded from python (tokens/s, batch size, cache hit rate, queue depth)

Python emits structured counters via prometheus client or JSON over the control channel.

## Failure Philosophy

- Engine never "hides" errors. Clear 4xx for bad requests at Rust layer.
- 5xx only for true internal failures.
- Graceful degradation on OOM (evict, reject new).
- Always return OpenAI-compatible error shape.

See [scope.md](./scope.md) and [roadmap.md](./roadmap.md).
