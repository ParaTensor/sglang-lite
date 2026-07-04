# sglang-lite 架构边界与进一步演进方向

## 1. 总体定位

sglang-lite 是**极简的纯 MoE 引擎库**（Token Factory）。

**核心原则**：
- 只保留三个高内聚构建块：RadixKVCache、BatchingScheduler、MoEModelRunner。
- 内部允许进一步拆分以提升可组合性。
- 编排（orchestration）、admission control、serving 等全部上移到驱动层（unigateway）。
- 保持代码量极小，专注 MoE 下的 Radix prefix sharing 和连续批处理。

## 2. 三个核心构建块的进一步内部拆分

为方便 unigateway 作为 driver 进行灵活组合和策略替换，我们对三个组件进行内部模块化拆分：

### RadixKVCache
- RadixTree（纯 token 前缀树逻辑）
- KVAllocator + MemoryManager（block 分配、引用计数、内存预算）
- EvictionPolicy（可插拔的驱逐策略）

### BatchingScheduler
- SequenceTable（序列生命周期管理）
- BatchFormer（批次形成策略，unigateway 可提供自定义实现）
- （AdmissionController / 超时 / 队列逻辑建议剥离到 unigateway）

### MoEModelRunner
- ModelLoader
- MoERouter
- PrefillExecutor
- DecodeExecutor（CUDA graph 可选、保守实现）
- KernelBackend（attention 执行委托给 FlashInfer / Triton 等外部库）

**LiteEngine** 将退化为极薄的 facade 或示例，主要用于简单 standalone 使用。真正的编排、主循环、请求生命周期由 unigateway driver 负责。

## 3. 与 unigateway 的关系（推荐分工）

- **sglang-lite**：只提供可组合的 building blocks + 最小 factory。
- **unigateway**（driver 代码放在 unigateway 仓库）：
  - 直接使用/组合上述三个构建块。
  - 负责主循环、admission control、request queuing、超时。
  - 提供完整的 OpenAI 表面、routing、auth、rate-limit、metrics 聚合等。
  - 所有 driver glue（如何加载、调用 sglang-lite）都放在 unigateway。

**目标**：让 sglang-lite 变得更小、更专注，unigateway 作为通用驱动层灵活集成。

## 4. 进一步可剥离的内容

- Tokenizer：彻底外部化，由调用方负责。
- 详细 metrics 采集逻辑：只保留 hook，导出和聚合由 unigateway 负责。
- 配置策略（lite preset）：由 unigateway 控制。
- 高级 batching policy：可由 unigateway 动态选择。
- Attention backend 选择：由 sglang-lite 内部根据 hints 决定，unigateway 仅传递高层偏好。

## 5. 演进建议

1. 优先把 LiteEngine 编排逻辑剥离，unigateway 直接操作三个构建块。
2. 在 unigateway 侧实现 SglangLiteDriver，使用上面拆分后的组件。
3. sglang-lite 保持极简，只做 MoE 友好的 KV + 调度 + 执行。
4. 持续内部拆分，提升可测试性和可替换性，但不要过度解耦影响性能。

此文档用于 sglang-lite 团队内部对齐演进方向。
