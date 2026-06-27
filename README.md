# sglang-lite

**A minimal, production-grade LLM inference engine.** High cohesion "Token Factory" focused on:

- RadixAttention (prefix sharing for chat/agent workloads)
- Continuous batching scheduler
- Low-overhead decode with CUDA graph
- OpenAI-compatible API (the critical control point)

> "Engine 只负责「可靠地、高效地吐 token」。" — 把 agent 逻辑、structured output、多模态全部上移到 gateway / harness 层（unigateway + IntentLoop/Zene）。

## Mission (一句话)

一个极简、生产可用的 LLM inference engine，只为热门 **dense** 模型提供稳定、高吞吐、低延迟的 token 生成能力。核心围绕 **continuous batching + 高效 KV cache (Radix 默认) + CUDA graph decode**，暴露极简稳定的 OpenAI 兼容接口。

**它不做**：agent runtime、multimodal、内置 constrained decoding、speculative、disagg serving、dynamic multi-LoRA。

## Why sglang-lite?

SGLang / vLLM 功能强大，但**偶然复杂性**过重（参考 Eric Zhang 帖）。真实生产中大量 workload（多轮 chat、RAG、工具调用）不需要全部特性，却为它们付出配置地狱、调试困难、每月变更带来的不稳定成本。

sglang-lite 采用**高内聚**设计：
- KV Cache 完整生命周期 + Scheduler + 执行 必须深度耦合，组成核心。
- 其他一切（业务逻辑）上移。

目标：在目标场景（prefix-heavy chat/agent）做到稳定 80%+ 理论上限 + 极低运维负担。

参考：nano-vLLM (~1.2k LOC 教学版)、mini-sglang。

## Target Workloads (优先级)

1. 多轮对话 / Agent 流量（RadixAttention 优势最大，prefix sharing）
2. RAG + Tool Use（structured 逻辑上移 gateway 用 outlines/xgrammar）
3. 高并发 batch 推理（评测、合成数据）

## Non-Goals (MVP 阶段坚决不做)

- 多模态 (Vision/Audio/Video)
- 内置 Structured Output / JSON mode / Grammar (上移)
- Speculative Decoding
- Prefill-Decode Disaggregation
- 动态 Multi-LoRA
- MoE Expert Parallelism (先只 dense)
- Diffusion / 非 Transformer
- 内置 SGLang 式 frontend language

## Feature 取舍表（高内聚版）

| 类别         | Feature                          | 分类          | 理由 / 策略                              | 替代 / 上移位置          |
|--------------|----------------------------------|---------------|------------------------------------------|--------------------------|
| API         | OpenAI /v1/chat/completions + Streaming | **重构**     | 关键控制点，完全自己掌控                 | -                        |
| API         | /v1/models + Health              | **重构**     | 极简实现                                 | -                        |
| KV Cache    | RadixAttention (前缀树)          | **重构**     | 核心，必须自己实现以保持内聚             | - (默认)                 |
| KV Cache    | PagedAttention                   | **重构**     | 可插拔备选策略                           | 配置开关                 |
| Scheduling  | Continuous Batching + 动态批     | **重构**     | 与 KV Cache 深度耦合                     | -                        |
| Scheduling  | 队列、优先级、Timeout            | **重构**     | 生产鲁棒性                               | -                        |
| Execution   | CUDA Graph Decode (重度)         | **重构**     | 降低 CPU overhead 的关键                 | -                        |
| Execution   | 基础 Quant (BF16/FP8/AWQ)        | Hybrid       | Loader 复用，路径自控                    | SGLang loader 片段       |
| Generation  | 基础 Sampling (temp/top_p/... )  | 复用/Hybrid  | 标准逻辑                                 | -                        |
| Model       | 主流 Dense 模型加载 (Llama/Qwen/Mistral) | 直接引用 | day-0 支持成本最低                       | HF + SGLang registry 片段 |
| Model       | Tokenizer                        | 直接引用     | transformers 成熟                        | -                        |
| Ops         | Prometheus metrics + /healthz + graceful | **重构** | 生产必需，只暴露 lite 关心指标           | -                        |
| Structured  | JSON mode / Grammar              | **不做**     | 破坏内聚                                 | Gateway (xgrammar)       |
| Multimodal  | Vision 等                        | **不做**     | 内聚度极低                               | 独立服务                 |
| Advanced    | Speculative / Disagg / Multi-LoRA| **不做** (MVP) | 复杂且非核心                             | 后期 plugin              |

**分类统计**：重构/自控 ~12+，直接引用（基础设施）少量，Hybrid 过渡，不做若干。

## 推荐架构（Rust + Python 混合）

```
Client / unigateway
        ↓ (OpenAI)
[Rust API Layer]  (axum + tokio)   <--- 控制点（必须自己写）
  - 严格验证 + early reject
  - 干净 internal GenerationRequest
  - Streaming 控制 + 错误归一化
  - Metrics, auth hook, rate limit hook
        ↓ (HTTP/gRPC / PyO3 / channel)
[Python Core Engine] (Triton / FlashInfer / torch)
  - KVCacheManager (Radix first)
  - ContinuousBatchingScheduler
  - ModelRunner + CUDA Graph capture
        ↑ token stream + usage
```

**为什么这个组合？**

- Rust：类型安全、低开销 streaming、完美和 unigateway 融合、完全掌控入口契约。
- Python + Triton：当前生态里 kernel/CUDA graph/KV 实现最成熟、迭代快。参考 nano-vLLM 路径。
- 渐进：先用 Python 核心跑通真实 workload，再考虑热点路径用 PyO3 迁移到 Rust。

**通信（MVP）**：先用本地 HTTP（松耦合，易 debug），或 gRPC。后续 PyO3 收紧。

## 技术选型

- **Rust API**：axum, tokio, serde, serde_json, tracing, reqwest (to python), uuid
- **Python Core**：torch, transformers (for loading/tokenizer initially), triton (kernels), flashinfer (可选)
- **接口**：极简 OpenAI 兼容 + 内部清晰 request/response
- **部署**：单机 + 简单 TP，先不做复杂分布式

## MVP 路线图（Phase）

**Phase 0 (验证核心，当前)**

- Rust：完整的 /v1/chat/completions (stream + 非 stream) + /v1/models + /healthz
- 定义 GenerationRequest / Token 内部协议
- Python 侧：stub engine（返回可预测 token），可切换到真实 HF 小模型
- 能跑通 Llama-3.1-8B / Qwen2.5-7B 级别（或更小用于 dev）
- 基本 continuous batching + 简单 Radix skeleton（即使不完美先跑通）
- 度量：tokens/s、TTFT、cache hit（即使 mock）

**Phase 1 (生产可用)**

- 真实 Radix KV Cache + Scheduler 深度集成
- CUDA graph decode 路径
- Prometheus metrics、结构化日志、graceful shutdown、请求超时
- 主流模型支持（70B 级别验证 throughput）
- 配置 preset（lite 默认）

**Phase 2**

- Paged 备选策略
- 有限模型扩展点
- 可选：把 scheduler/kv 部分迁移 Rust
- 与 unigateway 深度集成验证（KV affinity 等）

## 快速开始（当前 stub 状态）

```bash
# Rust API (stub)
cargo run -p sglang-lite-api -- serve --port 8000

# 或 Python stub
python -m sglang_lite.server
```

然后：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 128,
    "stream": true
  }'
```

See [docs/roadmap.md](docs/roadmap.md) and [docs/architecture.md](docs/architecture.md)。

## 与你的栈协同

- **unigateway**：L7 路由、semantic routing、KV cache affinity、auth/rate-limit
- **IntentLoop / Zene**：agent loop、memory
- **Engine**：只做高性能稳定的 token 工厂

这样每一层内聚度都高，整体复杂度下降。

## 贡献与范围纪律

请严格遵守 scope。新增 feature 前先问：它是否属于「Token Factory」高内聚核心？如果是业务逻辑，请上移。

## License

Apache-2.0 (to be confirmed)

## Status

Early stage. Phase 0 in progress. Not for production yet.

See git history and docs for detailed design discussions.
