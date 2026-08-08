# PegaInfer 借鉴笔记（sglang-lite）

对照项目：[pegainfer-project/pegainfer](https://github.com/pegainfer-project/pegainfer)  
记录日期：2026-08-08  

**一句话**：把 PegaInfer 当「热路径与门禁的标杆」，不要当「用 Rust 重写 engine」的理由。

---

## 1. 他们在验证什么

PegaInfer 是 **纯 Rust + CUDA** 的 LLM 推理引擎：运行时无 PyTorch / ONNX / Python 框架；
默认只编 Qwen3 线，其它模型 feature-gated。架构是 **胶水层**：

- 自持：scheduler、KV、decode 主路径、内存安全
- 叶子：FlashInfer、Triton AOT、TileLang、Dynamo `kvbm-logical` 等
- 契约：`GenerateRequest` / `TokenEvent`（frontend → core → per-model crate）

产品叙事重点：**冷启动、idle RSS、warm-cache TTFT 随长度走平**，吞吐与 vLLM 同阶竞争。

---

## 2. 与 sglang-lite 的边界对齐

| 维度 | PegaInfer | sglang-lite（已定） | 态度 |
|------|-----------|---------------------|------|
| 控制面 | Rust server + frontend | `control/` + `serving/` | 同构，保持 |
| 执行面 | 纯 Rust/CUDA 模型 crate | Python `engine/` 三件套 | **异构，不跟** |
| 内部契约 | GenerateRequest / TokenEvent | GenerationRequest / TokenDelta | 钉死、禁 OpenAI 渗入 |
| 内核 | 自持 + FlashInfer 叶子 | KernelBackend + FI / fused MoE | 同构 |
| 模型策略 | 默认 Qwen3；MoE feature-gated | MoE-only Token Factory | 学「默认一条线」，不学 per-model 拆引擎 |

已定分层见 [architecture.md](./architecture.md)、[scope.md](./scope.md)、[AGENTS.md](../AGENTS.md)：
`control(Rust) + engine(Python)`，禁止 PyO3 同进程嵌入。

---

## 3. 该抄（按优先级）

### P0 — 热路径与门禁

1. **薄契约**  
   OpenAI 只在 Rust；引擎只认 `GenerationRequest` / `TokenDelta`。  
   用测试锁形状，禁止 chat template / 宽协议字段进 `engine/`。

2. **Decode CUDA Graph 纪律**  
   - 中间 buffer 预分配（无 capture 期 `cudaMalloc`）  
   - plan / 可变元数据图外；body 图内  
   - position / page 元数据走固定 device buffer  
   你们已有：固定 `SGLANG_LITE_PAGED_MAX_PAGES` + FI `use_cuda_graph` + native body。  
   继续审计图内分配，而不是先上 Green Contexts。

3. **HF golden gate**（形态对齐 PegaInfer `hf_golden_gate`）  
   - 固定 case JSON（prompt × max_new × greedy）  
   - 同权重快照：HF dump vs lite 输出  
   - 分类：`all_token_text_exact` / `first_diff` / `error`  
   - FORCE_HF 路径目标 **token-exact**；radix/CG/fused 可先报告 first-diff  
   脚本：[`scripts/hf_golden_gate.py`](../scripts/hf_golden_gate.py)  
   Case：[`test_data/hf_golden_cases.json`](../test_data/hf_golden_cases.json)

4. **Scheduler 单所有者**  
   调度循环独占 model + KV + FI wrapper；空闲 blocking 等待；prefill 优先再 batch decode。  
   收紧 `loop.py` / `scheduler.py`，不加架构。

### P1 — 产品叙事（Radix 真生效后）

- idle RSS / 冷启动（有空再记，不阻塞正确性）  
- **warm prefix TTFT vs context length** + `cache_hit_tokens`  
- 不要先卖 HBM→DRAM offload 故事

### P2 — 明确延后

- Green Contexts SM 静态切分 / PD 共置  
- Pegaflow 式分层 offload  
- per-model 各写一套 scheduler/executor  
- 宽模型矩阵、EP8、通用量化面

---

## 4. 不该抄

| 不要 | 原因 |
|------|------|
| 整引擎纯 Rust、无 Python 运行时 | 推翻已定产品赌注与节奏 |
| per-model 各一套调度/执行 | 拆掉 Radix + Scheduler + Runner 高内聚 |
| 为 parity 扩 dense / 多模态 / 宽 EP | [scope.md](./scope.md) 禁止 |
| 自持 paged Radix 前做 offload | 扩 scope |
| 把 V4 拆成独立引擎 crate | 学「默认一条成熟线」，不是一模型一引擎 |

---

## 5. 映射到当前进度（2026-08-08）

PRO6000 Qwen3-30B-A3B 1×128 warm（见 [deepseek-v4-flash-plan.md](./deepseek-v4-flash-plan.md) § thruput）：

| 栈 | tok/s |
|----|------:|
| paged eager | ~45 |
| paged + CG + batched_mm | ~79 |
| **paged + native + CG + fused MoE** | **~101** |
| FORCE_HF + compile | ~84 |
| SGLang | ~155 |

结论：PegaInfer 式「热路径工程」已在进行（固定 page plan、CG、native loop、fused under graph）。
距 SGLang ~1.5× 余量在 **kernel/融合**，不在再剥 HF Python，也不在换语言。

---

## 6. 建议动作清单

| # | 动作 | 状态 |
|---|------|------|
| 1 | 文档：本文件 + runbook / optimization-priorities 交叉引用 | 本 PR |
| 2 | `scripts/hf_golden_gate.py` + `test_data/hf_golden_cases.json` | 本 PR |
| 3 | generate 最终帧带 `output_ids`（token 级对比） | 本 PR |
| 4 | Decode graph 图内零分配审计 | 后续 |
| 5 | loop 单所有者 / 空闲阻塞收紧 | 后续 |
| 6 | warm-prefix TTFT 曲线（Radix hit 真通后） | 后续 |

运行 golden gate：

```bash
# 一键（FORCE_HF：默认 --require-exact）
python scripts/hf_golden_gate.py gate \
  --model ~/models/Qwen3-30B-A3B-Instruct \
  --path force_hf --out-dir /tmp/golden

# 分步
python scripts/hf_golden_gate.py dump-hf \
  --model ~/models/Qwen3-30B-A3B-Instruct \
  --out /tmp/golden/hf.json
python scripts/hf_golden_gate.py run-lite \
  --model ~/models/Qwen3-30B-A3B-Instruct \
  --path force_hf --out /tmp/golden/lite_force_hf.json
python scripts/hf_golden_gate.py compare \
  --hf /tmp/golden/hf.json --lite /tmp/golden/lite_force_hf.json \
  --out /tmp/golden/compare_force_hf.json --require-exact

# radix-native：报告 first-diff，默认不 require-exact
python scripts/hf_golden_gate.py gate \
  --model ~/models/Qwen3-30B-A3B-Instruct \
  --path radix_native --out-dir /tmp/golden --no-require-exact
```

**门禁纪律（2026-08-08 PRO6000 实测）**

- Oracle：`dump-hf` 必须 **单卡** + `experts_implementation=batched_mm`，
  **禁止** `device_map=auto`（多卡分片曾导致 hello_32 @18 假分叉；lite 与同权重
  `model.generate` 始终 exact）。
- **FORCE_HF**：`hello_16` / `hello_32` / `capital_16` → **all_token_text_exact** PASS。
- **radix_native**（CG+fused+native，报告-only）：first_diff
  `hello_16@3`、`hello_32@11`、`capital_16@1`；不作为发版 exact 门槛。

---

## 7. 若再深挖源码

优先：

1. PegaInfer decode graph 捕获条件与预分配 buffer 布局  
2. `KvPool` page-first 布局与 prefix 命中  

延后：Green Contexts 博客、Pegaflow offload。
