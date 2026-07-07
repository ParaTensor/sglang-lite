# sglang-lite Serving Module (serving/)

**重要**：这个目录**只做极薄的包装和组合**。

## 定位（严格边界）

- sglang-lite 本身是**纯引擎库**（Token Factory），见 `engine/`。
- sglang-lite 的 Rust 控制面库在 `control/`，包含 OpenAI 协议最小适配、内部协议、engine 客户端。
- **全部真实 serving 逻辑由 Unigateway 实现**（Rust）。
- `serving/` 只提供：
  - 组合 `control/` 形成的可执行 standalone wrapper
  - 与 Unigateway 组合的薄胶水（import / HTTP / gRPC）
  - 配置、部署、示例

**不允许**在这里实现完整的 HTTP 服务、OpenAI 协议细节、FastAPI 等。

## 当前实现方式（Rust 实现）

- **真实完整的 serving 全部由 Unigateway（Rust）实现**。
- 本目录（`serving/`）**完全用 Rust 实现**，仅作为可执行 wrapper：
  - 依赖 `sglang-lite-control`（`../control`）获得薄控制面。
  - 组合路由并启动 axum 服务。
- 通过依赖 Unigateway（外部） + sglang-lite 后端实现生产级 serving。
- 不再有任何 Python 代码或重型 serving 逻辑。

## 目录结构

- `Cargo.toml` + `src/main.rs`：薄 Rust serving 入口（仅组合，实际 serving 由 Unigateway 提供）。
- 严禁在 `serving/` 实现完整 OpenAI 逻辑或控制面细节；这些属于 `control/`。

## 推荐用法

在 Unigateway 项目中通过 feature 或 driver 使用 `sglang-lite-serving` 作为后端支持。

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
- 所有完整 serving 能力必须上移到 Unigateway（Rust）。
- 新代码必须符合 `AGENTS.md` “模块边界定义”。

参考 `AGENTS.md` 中的“模块边界定义”。
