# vLLM 定位与兼容评估

本文回答一个边界问题：sglang-lite 目前明显参考了 SGLang 的 RadixAttention 路线，但是否也应该考虑 vLLM 的定位与兼容性？

结论：**需要兼容 vLLM 的“后端定位”和协议生态，但不追求兼容 vLLM 的完整功能面或内部实现。**

sglang-lite 应该被定义为 `local-inference backend` 家族中的一个 MoE 优化后端；vLLM 是同一层级的通用后端。UniGateway 面向两者暴露统一的 provider/driver/capability 抽象，而不是把 UniGateway 的核心抽象绑定到 SGLang 或 sglang-lite。

## 1. vLLM 的相关定位

vLLM V1 的公开设计重点包括：

- OpenAI-compatible HTTP server：`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings` 等更宽的 OpenAI-like 表面。
- EngineCore：将 tokenization、multimodal preprocessing、detokenization、streaming 等 CPU-heavy 工作与核心 scheduler/model executor loop 解耦。
- Scheduler：统一调度 prompt tokens 和 output tokens，用 `{request_id: num_tokens}` 表示每步 token budget；支持 chunked prefill、prefix caching、speculative decoding 等。
- KVCacheManager / PagedAttention：以 block table 管理 KV cache。
- Automatic Prefix Caching：使用 block hash（parent hash + block tokens + extra hashes）缓存完整 KV blocks，面向多轮对话、重复长上下文。
- GPUModelRunner / Worker：负责 model forward、sampler、KV cache buffer、CUDA graph/compile 等执行细节。

参考：

- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/
- vLLM automatic prefix caching: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
- vLLM V1 architecture blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html

## 2. sglang-lite 与 vLLM 的定位差异

| 维度 | vLLM | sglang-lite |
| --- | --- | --- |
| 产品定位 | 通用 LLM/VLM serving engine + OpenAI-compatible server | 极简 MoE Token Factory / pure library |
| 模型范围 | Dense、MoE、multimodal、pooling、audio 等广泛模型 | 只支持主流 MoE；dense/multimodal 不进 core |
| API 面 | 宽 OpenAI-like 表面 + vLLM extra params | 最小 chat completions / models / health；完整 OpenAI 表面由 UniGateway 承担 |
| Prefix cache | Hash-based APC over full blocks | RadixKVCache 默认；可暴露 generic prefix-cache/block-KV capability |
| KV cache | PagedAttention / KVCacheManager | RadixKVCache + allocator；可兼容 block table / paged KV 语义 |
| Scheduler | 统一 token budget，支持多功能组合 | MoE-aware continuous batching；优先保持可理解与低复杂度 |
| 高级功能 | structured output、LoRA、spec decode、disagg、multimodal 等 | core 明确不做；上移 gateway 或独立 sidecar |

因此，sglang-lite 不应该成为 “mini vLLM”，也不应该只做 “mini SGLang”。更准确的定位是：

> 一个 MoE-only、prefix-heavy workloads 优先的 local inference backend；通过 UniGateway 与 vLLM 共享 OpenAI-compatible 协议、provider capability 和观测指标抽象。

## 3. 兼容层级

### P0：协议兼容

sglang-lite 应保持与 vLLM local serving 最小交集兼容：

- `POST /v1/chat/completions`
- streaming SSE chunk shape
- `GET /v1/models`
- health endpoint
- OpenAI-compatible error shape
- request id 透传（例如 `X-Request-Id` / response metadata）
- `usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens`
- 可选扩展：`usage.cache_hit_tokens`

对于 vLLM 的 extra parameters，sglang-lite 应采用稳定策略：

- 已明确支持的 sampling 字段正常接受。
- scope 外字段要么显式 4xx reject，要么 warning-ignore；不要无声改变语义。
- `structured_outputs`、multimodal content、LoRA、spec decode、disagg 等不进入 core。

### P1：Provider capability 兼容

UniGateway 不应只认识 `sglang-lite`，而应抽象出 generic local inference capability：

```toml
[[providers]]
name = "local-moe"
provider_type = "local-inference"
backend_kind = "sglang-lite"
base_url = "http://localhost:8000/v1"
api_key = ""

[providers.capabilities]
chat_completions = true
streaming = true
prefix_cache = true
prefix_cache_metric = "usage.cache_hit_tokens"
moe_optimized = true
multimodal = false
structured_output = false
```

vLLM 可以使用同一族抽象：

```toml
[[providers]]
name = "local-vllm"
provider_type = "local-inference"
backend_kind = "vllm"
base_url = "http://localhost:8000/v1"
api_key = "token-abc123"

[providers.capabilities]
chat_completions = true
responses_api = true
streaming = true
prefix_cache = true
vllm_extra_params = true
multimodal = true
```

`provider_type = "sglang-lite"` 可以作为兼容别名保留，但新设计不应把核心路由、pool、protocol 抽象写死为 sglang-lite 专用。

### P2：内部概念映射兼容

sglang-lite 内部仍可以使用 RadixKVCache，但对外不要暴露 RadixTree 作为 driver 的硬协议。建议对外使用通用术语：

| 通用概念 | sglang-lite 内部 | vLLM 对应 |
| --- | --- | --- |
| `PrefixCache` | RadixKVCache | Automatic Prefix Caching |
| `BlockKVCache` | KVAllocator / block table | KVCacheManager / PagedAttention blocks |
| `ContinuousScheduler` | BatchingScheduler | vLLM V1 Scheduler |
| `ModelExecutor` | MoEModelRunner | GPUModelRunner / Worker |
| `BackendCapabilities` | MoE-only + Radix metrics | broad model/API/hardware capability |

这允许 UniGateway 的 KV-affinity、prefix-cache-aware routing、metrics aggregation 同时兼容 sglang-lite 与 vLLM。

## 4. 可借鉴但不照搬的 vLLM 设计

### 值得吸收

- **统一 token budget 表达**：可以作为 `BatchFormer` 的策略输入，帮助混合 prefill/decode/chunked prefill，但不必立刻完整实现 vLLM 的 scheduler。
- **hash-based full-block 索引**：可作为 RadixKVCache 的可选二级索引，用于 block-level lookup、tenant salt、LoRA/multimodal hash 扩展；默认仍保留 Radix prefix tree。
- **persistent batch / diff update 思路**：适合未来降低 Python CPU overhead。
- **request id / metadata 透传**：便于 UniGateway 统一追踪 sglang-lite 与 vLLM。
- **capability 声明**：让 routing 不依赖 backend 名称。

### 不应吸收进 core

- vLLM 的完整 OpenAI server 表面。
- Responses/audio/embeddings/multimodal 等宽 API。
- vLLM-specific extra params 作为 sglang-lite 公共契约。
- Structured output、dynamic LoRA、spec decode、PD disaggregation。
- 直接复用 vLLM scheduler/KVCacheManager 内部实现，导致 lite 失去可理解性。

## 5. 对现有文档/实现的调整要求

1. `README.md`、`architecture.md`、`scope.md` 必须把外部抽象从 “SGLang/Radix 特化” 提升到 “local inference backend + generic prefix cache capability”。
2. UniGateway 需求中应明确：`SglangLiteDriver` 是 `LocalInferenceDriver` 家族的一个实现；vLLM 应作为同层级 backend 兼容。
3. `cache_hit_tokens` 是 generic prefix-cache metric，不应命名为 Radix-only metric。
4. sglang-lite 的 Paged/block KV 设计要兼容 vLLM-style block table 语义，但实现可继续保持 Radix-first。
5. 不要为追求 vLLM 兼容而扩大 core scope；不在 core 支持 dense/multimodal/structured output。

## 6. 选择建议

| 场景 | 优先选择 |
| --- | --- |
| 需要最广模型/硬件/API 兼容 | vLLM |
| 需要 VLM、多 API、多 LoRA、spec decode、disagg | vLLM 或 full SGLang |
| MoE-only、prefix-heavy chat/agent、希望极低运维复杂度 | sglang-lite |
| 需要 SGLang frontend language / structured generation runtime | full SGLang |
| Gateway 层统一多后端路由、auth、metrics、policy | UniGateway |

sglang-lite 的核心价值不是替代 vLLM，而是在更窄的 MoE + prefix-heavy 生产场景里保持更小、更稳定、更容易调试；同时通过 UniGateway 与 vLLM 在协议、capability、metrics 层保持兼容。
