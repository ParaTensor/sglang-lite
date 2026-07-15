# sglang-lite Serving Module (serving/)

**重要**：这个目录**只做极薄的包装和组合**。

## 定位（严格边界）

- sglang-lite 的 `engine/` 是**纯引擎库**（Token Factory）。
- sglang-lite 的 Rust 控制面库在 `control/`，包含 OpenAI 协议最小适配、内部协议、engine 客户端。
- sglang-lite 必须能够独立运行；Unigateway 是可选的高级网关。
- `serving/` 只提供：
  - 组合 `control/` 形成的官方 standalone wrapper
  - 与 Unigateway 组合的薄胶水（import / HTTP / gRPC）
  - 单引擎启动配置、健康/就绪、部署入口

**不允许**在这里实现模型执行、跨后端 routing、auth、业务逻辑或宽 OpenAI 表面。

## 目标实现方式与当前状态

- 本目录（`serving/`）**完全用 Rust 实现**，作为正式可执行 wrapper：
  - 依赖 `sglang-lite-control`（`../control`）获得薄控制面。
  - 组合路由并启动 axum 服务。
- 完成后，单独运行时提供最小、真实、可流式的推理服务。
- 需要多后端 routing、auth、全局限流和策略时，可在上游部署 Unigateway。
- 不再有任何 Python 代码或重型 serving 逻辑。

当前仍默认使用 `StubEngineClient`，Python 转发也不是真实 token-level streaming，
因此尚不属于生产可用服务。补齐顺序和退出标准见
`docs/standalone-inference-service-roadmap.md`。

## 目录结构

- `Cargo.toml` + `src/main.rs`：薄 Rust serving 入口（仅组合，实际 serving 由 Unigateway 提供）。
- 严禁在 `serving/` 实现完整 OpenAI 逻辑或控制面细节；这些属于 `control/`。

## 推荐用法

用户可以直接启动，也可以在 Unigateway 中把它注册为一个 local-inference backend。

本地测试（从 workspace 根目录）：

```bash
cargo run -p sglang-lite-serving
```

或指定端口/Python core：

```bash
PORT=8000 SGLANG_LITE_PYTHON_CORE=http://localhost:9001 cargo run -p sglang-lite-serving
```

## 开发规则（必须遵守）

- 这里**只用 Rust**。
- 保持极薄：只做组合，不复制 Unigateway 核心 serving 逻辑。
- standalone 必需的 chat/stream/models/health/readiness/metrics 和生命周期能力保留。
- 多后端网关与业务能力必须上移到 Unigateway（Rust）或其他 gateway。
- 新代码必须符合 `AGENTS.md` “模块边界定义”。

参考 `AGENTS.md` 中的“模块边界定义”。
