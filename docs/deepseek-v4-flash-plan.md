# DeepSeek V4-Flash 支持路线评审：先底座、后家族、最大化复用

状态：设计评审稿（不改变现有 scope 结论，细化执行顺序与复用边界）。
关联文档：[scope.md](scope.md)、[standalone-inference-service-roadmap.md](standalone-inference-service-roadmap.md)、[architecture.md](architecture.md)。

## 1. 结论

**不把 V4-Flash 当成"补齐模型列表"的入口，而是当成逼出真实执行底座的压力测试目标。**

顺序：先补完通用 MoE 执行底座（真 paged attention、中央 engine loop、KV layout
抽象），再以 DeepSeek-V2-Lite 作为 MLA/非标准 KV layout 的先导模型，最后才把
`deepseek_v4` 作为一个受控家族注册进 `engine/models.py`。

同时明确**自研 vs 复用**边界：sglang-lite 只自持三个核心（RadixKVCache、
BatchingScheduler、MoEModelRunner 的组合与 engine loop），其余尽可能以小模块方式
复用 FlashInfer、sgl-kernel、Triton kernel、HF/官方模型定义等成熟组件。

## 2. 现状差距（基于当前代码）

V4 难的不是在 registry 加一行 `deepseek_v4`，而是它把「权重如何切到 8 卡」
「FP4/FP8 如何计算」「CSA/HCA 的 KV 形态」「Radix 如何复用这些 page」绑在一起。
当前执行面离这个要求还有距离：

1. **执行面仍是 HF 前向包装**。`engine/runner.py` 用
   `AutoModelForCausalLM + device_map=auto` 单进程加载；paged KV 是"影子存储"：
   每步 `radix.build_cache()` 把 page 重建成 HF `DynamicCache` 再喂给模型
   （`_past_for_seq`），真相仍是 HF `past_key_values`。284B/13B-active 级别下，
   重建 cache 的显存与拷贝开销本身就不可行。
2. **FlashInfer 未进入 attention hot path**。prefill/decode wrapper 已创建但
   forward 仍走 HF SDPA；FlashInfer 只用于 paged append。
3. **KV 写入是逐 token Python 循环**（`kv_cache.py: write_kv`），且 page 布局
   硬编码为全层同构 `[layers, blocks, block_size, kv_heads, head_dim]`。
   MLA compressed KV（DeepSeek-V2/V3 起）已不满足该假设——这笔 layout 债在
   V2-Lite 阶段就会被逼出，不必等 V4。
4. **P0 golden path 未闭环**：prefill 分组长度不一致时逐条串行回退；
   roadmap 中 M2/M3（真 per-layer KV + 中央 async loop）尚未完成。

因此第一步不是"接 V4"，而是把 M2/M3 真正做完。

## 3. 自研 vs 复用边界（本文档的核心补充）

原则：**核心组合权自持，叶子实现最大化复用**。sglang-lite 的差异化在于
「Radix 前缀复用 × continuous batching × MoE 执行」这条 hot path 的组合方式，
不在于任何单个 kernel 或 loader。凡与这条 hot path 的"驱动/调度核心"无关的内容，
一律优先引用外部小模块，不自研。

| 层 | 策略 | 复用来源 | 说明 |
| --- | --- | --- | --- |
| Attention kernel（paged prefill/decode、MLA） | **直接引用** | FlashInfer（含 MLA/paged wrapper）、sgl-kernel | 收口到 KernelBackend 接口后整体替换，不自写 kernel |
| MoE kernel（grouped GEMM、fused expert、topk routing） | **直接引用** | FlashInfer / sgl-kernel / Triton 现成 kernel | 同上；expert 路由计算留在模型图内 |
| 量化（FP8/FP4 GEMM、KV 量化） | **直接引用** | sgl-kernel / DeepSeek 官方 kernel（DeepGEMM 类） | 量化布局归 loader，量化计算归 KernelBackend，不散落 |
| 权重加载与分片 | Hybrid | HF safetensors 加载器 + 自持的分片映射 | 只自写 "哪个 shard 放哪张卡" 的映射，不自写文件格式 |
| Tokenizer / chat template | **直接引用** | HF tokenizers | 已是现状 |
| 模型图定义（V2-Lite / V4） | Hybrid | transformers remote code 或 DeepSeek 官方 `inference/` | 可整体引用模型定义，只要求其 attention/MoE 调用点可被 KernelBackend 接管 |
| TP 通信原语 | **直接引用** | torch.distributed / NCCL | 只自持 "TP=8 单机" 这一种受控拓扑的初始化与 shape 约定 |
| RadixKVCache（page 生命周期、COW、eviction） | **重构（自持）** | — | 核心差异化，唯一需要扩展的是 per-layer KV layout 描述符 |
| BatchingScheduler / engine loop | **重构（自持）** | — | 核心差异化 |
| Runner 的 batch 组装与采样 | **重构（自持）** | 采样可复用 flashinfer sampling kernel | 组合逻辑自持，叶子 kernel 复用 |

**不做的自研**：不自写 attention/MoE/量化 kernel，不自写分布式框架，不复制
SGLang 的完整 ModelRunner/EPLB/spec-decode 体系。若需要从 SGLang 借用，只借
**叶子级小模块**（如 sgl-kernel 的单个算子），不引入其调度或内存管理框架——
否则就失去了 lite 的意义。

对应到 AGENTS.md 的分层纪律，执行面重构建议只拆两层，避免 Phase 0 出现四层抽象：

- **KernelBackend**：attention + MoE + 量化计算的唯一收口，内部由外部 kernel 组成；
- **ModelLoader**：分片与量化布局；
- Prefill/Decode 执行逻辑暂留在 `ModelRunner` 内（保持 2 跳可追踪）。

### 3.0 可复用叶子组件清单（GPU 实测已回填，2026-08-04 / 8×5090）

以下均为独立可 pip 安装 / vendor 的叶子件，不携带外部调度或全局状态
（接入时在 KernelBackend 内固定版本并做 capability 探测）。实测环境：
torch `2.11.0+cu130`、CUDA 13.0、GPU RTX 5090（sm_120 / capability 12.0）。
测法见 3.0.1，逐项记录见 3.0.2；复现脚本 `scripts/leaf_component_probe.py`。

| 组件 | 实测版本 | 提供能力 | 接入点 | sm_120 结论 |
| --- | --- | --- | --- | --- |
| `flashinfer-python` | **0.6.12** | paged prefill/decode、`BatchMLAPagedAttentionWrapper`、`append_paged_kv_cache`；另有 `trtllm_batch_decode_sparse_mla_dsv4`（V4 sparse MLA） | KernelBackend（paged 已接入；MLA 可接 S2） | **标准 MLA/paged：可用**；**V4 sparse MLA（CSA 路径）：API 有、sm_120 跑不通**（`Unsupported architecture`） |
| `sgl-kernel` | **0.4.4** | `topk_softmax` / MoE 辅助、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_decode`、norm/rope | KernelBackend（MoE/量化，S2+） | **可安装可 import**；topk 路由 id 与 torch 一致；**无 CSA/HCA 公开符号**，有 dsv4 融合算子 |
| `deep-gemm` | **0.1.4** | DeepSeek FP8/FP4 grouped GEMM API 面齐全 | **暂不接入**（见下） | **5090 不支持**：`bf16_gemm_nt` 即报 `Unsupported architecture` |
| DeepSeek 官方 `inference/` / transformers remote code | 本机有 `ds-v4-flash`；**无 V2-Lite 权重** | V4 模型图含 Indexer/Compressor/`sparse_attn`/HC(Sinkhorn) | ModelLoader / vendor（Hybrid） | V4 图可对照；**V2-Lite greedy/paged 验收 BLOCKED（缺权重）** |

**待实测确认（已实测）**：

- flashinfer / sgl-kernel 对 V4-Flash 新 attention（官方代码表现为
  **Indexer + compressed KV + `sparse_attn`**，以及 **HC / `hc_split_sinkhorn`**；
  文档简称 CSA/HCA）：flashinfer 提供 DSV4 sparse MLA API，但 **TRTLLM-GEN
  路径在 sm_120 上报 Unsupported architecture**；sgl-kernel 提供 `dsv4_fused_*`
  与 MLA 辅助，**无以 CSA/HCA 命名的完整 attention 入口**。S5 前需以官方
  `inference/kernel.py:sparse_attn` 或上游修复作为回退。
- deep-gemm 与 5090（sm_120）：**不兼容**（0.1.4）。回退：`sgl-kernel` 的
  `fp8_*_mm` / `fp8_blockwise_scaled_grouped_mm` 或 cuBLASLt。
- **因此暂不把 `sgl-kernel` / `deep-gemm` 写入 `pyproject.toml` 可选依赖**
  （sgl-kernel 可用但 fused_moe 对 HF Mixtral 的完整数值对齐尚未做完；
  deep-gemm 明确不达标）。

#### 3.0.1 叶子组件实测计划（GPU 环境，逐项验收）

统一前提：8×5090（sm_120）、CUDA/torch 版本与仓库固定的依赖矩阵一致；
每项先做"能装能跑"再做"数值对齐"；任何失败都记录具体版本与报错，
不带病接入 KernelBackend。

**T1 flashinfer-python（attention 主路径，已部分接入）**

- 测法：现有 `tests/test_gpu_paged_attention.py` 为基线；补充
  MLA wrapper 冒烟（`BatchMLAPagedAttentionWrapper` 或等价接口），用
  DeepSeek-V2-Lite 的 head 配置构造随机 KV，对比 torch 朴素实现的输出。
- 验收：paged prefill/decode 数值与参考实现 atol≤1e-2（bf16）；MLA wrapper
  在 sm_120 上可编译可运行。
- 回退：MLA 不可用则暂时用"解压回标准 KV + 普通 paged attention"的慢路径，
  并记 issue 跟踪上游。

**T2 sgl-kernel（MoE 与量化算子，S2+ 接入）**

- 测法：`pip install sgl-kernel` 后跑最小算子冒烟：`fused_moe`/topk 用
  Mixtral 小配置对比 HF 参考 MoE 层输出；FP8 GEMM 对比 bf16 GEMM 的
  相对误差。
- 验收：sm_120 上可安装（有预编译 wheel 或可源码编译）；fused_moe 输出与
  HF 参考 rel-err≤1e-2；单算子调用不依赖 SGLang runtime（import 仅
  sgl_kernel）。
- 回退：装不上或算子缺失时，MoE 先走 HF 模型图内的朴素实现，仅损失性能。

**T3 deep-gemm（FP8 grouped GEMM，V4 expert 候选）**

- 测法：安装后跑其自带 benchmark/测试；重点确认 sm_120 支持（该库最初面向
  Hopper sm_90，Blackwell 支持需实测）；FP8 grouped GEMM 数值对比 bf16 参考。
- 验收：sm_120 可运行且数值达标；不达标则明确记录"5090 不支持"。
- 回退：用 sgl-kernel 的 FP8 GEMM 或 cuBLASLt 路径替代。

**T4 官方模型图（V2-Lite 先导 → V4）**

- 测法：transformers remote code 加载 DeepSeek-V2-Lite，短 prompt greedy
  输出与 HF 参考一致（现有 reference correctness 测试的 GPU 版）；确认其
  attention/MoE 调用点可被 KernelBackend monkeypatch 接管（同 PR #4 手法）。
- 验收：V2-Lite 走 paged 路径 `paged_rebuild_count==0` 且逐 token 对齐；
  MLA 的 KV 进 RadixCache 需要的 layout 描述符明确（这是 S2 的输入）。
- 回退：remote code 不可控时 vendor 模型定义进仓库（标注来源与 license）。

产出要求：每项一份简短记录（版本、命令、结果、结论），汇总回填本文档
3.0 表格的"待实测确认"栏；全部通过后才把 sgl-kernel / deep-gemm 写进
`pyproject.toml` 可选依赖。

#### 3.0.2 实测记录（2026-08-04，5090 / sm_120）

**T1 flashinfer-python — PASS（标准路径）/ FAIL（V4 sparse）**

- 版本：flashinfer `0.6.12`；torch `2.11.0+cu130`
- 命令：`pytest tests/test_gpu_paged_attention.py`；
  `PYTHONPATH=. python scripts/leaf_component_probe.py`；
  另手测 `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`
- 结果：
  - 既有 GPU paged 基线 **3 passed**（`paged_rebuild_count==0`）
  - `BatchMLAPagedAttentionWrapper`（fa2）在 sm_120 可跑；vs torch 参考
    `max_abs≈0.0078`（bf16，≤1e-2）
  - `is_sm12x_supported(cuda)=True`
  - `trtllm_batch_decode_sparse_mla_dsv4`（64 heads / dim 512）→
    `TllmGenFmhaRunner ... Unsupported architecture`
- 结论：S2 可接标准 MLA wrapper；V4 CSA 类 sparse MLA **不能**依赖当前
  flashinfer TRTLLM-GEN 路径上 5090。

**T2 sgl-kernel — PARTIAL PASS（可装可跑，MoE 数值对齐未闭环）**

- 版本：sgl-kernel `0.4.4`（import 仅 `sgl_kernel`，无 SGLang runtime）
- 命令：probe 脚本 + `topk_softmax` / `fused_add_rmsnorm` 手测
- 结果：
  - 安装/import OK；暴露 `moe.*`、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_*`
  - `topk_softmax`：top-k **id 集合 8/8 与 torch.topk(softmax) 一致**
    （renormalize 后权重尺度不同，属预期）
  - `fused_add_rmsnorm` 在 sm_120 正常
  - 公开 API **无 CSA/HCA 命名**；有 V4 相关 `dsv4_fused_q_norm_rope` 等
- 结论：可作为 S2+ MoE/量化叶子候选；完整 `fused_moe`↔HF Mixtral rel-err
  验收仍待补；**暂不写入 pyproject**。

**T3 deep-gemm — FAIL（5090 不支持）**

- 版本：deep-gemm / `deep_gemm` `0.1.4`；`get_cuda_arch()==12.0`
- 命令：`deep_gemm.bf16_gemm_nt(a,b,d)`（256² bf16）
- 结果：`Assertion error (.../gemm.hpp:436): Unsupported architecture`
  （FP8 API 面存在但未再测——bf16 入口已失败）
- 结论：**明确记录「5090 / sm_120 不支持 deep-gemm 0.1.4」**；V4 expert
  GEMM 回退 sgl-kernel FP8 或 cuBLASLt；**不写入 pyproject**。

**T4 官方模型图 — BLOCKED（V2-Lite）/ 可对照（V4-Flash）**

- 本机：`~/models/ds-v4-flash`（含 `inference/model.py`）；
  `deepseek-ai/DeepSeek-V4-Flash`；**无 DeepSeek-V2-Lite 权重/缓存**
- V4 图要点（S2/S5 输入）：`Attention` = MLA + sliding window +
  可选 `Compressor`/`Indexer` + `sparse_attn`；Block 侧 HC
 （`hc_mult` + `hc_split_sinkhorn`）；expert `fp4` + attention 侧 FP8 量化配置
- S2 layout 描述符需求（在缺 V2-Lite 权重时仍成立）：
  - 标准：`[layers, blocks, block_size, kv_heads, head_dim]`
  - MLA compressed：`ckv [blocks, page_size, kv_lora_rank]` +
    `kpe [blocks, page_size, qk_rope_head_dim]`（常见 `page_size=1`）
  - V4 另需 SWA 窗口池 + compressed 池 + sparse index（与当前单一
    isomorphic page 布局不兼容）
- 结论：V2-Lite greedy / `paged_rebuild_count==0` **未跑**（缺权重）；
  取得 V2-Lite 后再补 T4 闭环。

### 3.1 硬件后端隔离（非 NVIDIA 卡的前置设计）

未来需要支持华为昇腾（Ascend/CANN）及其他非 NVIDIA GPU，因此硬件隔离边界必须在
KernelBackend 设计时一次划清，避免事后剥离 CUDA 假设：

- **KernelBackend 是唯一的硬件抽象点**。attention、MoE、量化、采样 kernel 与
  TP 通信原语全部收口在此；CUDA + FlashInfer/sgl-kernel 只是它的第一个实现
  （`CudaKernelBackend`），昇腾对应 `AscendKernelBackend`（torch_npu + CANN 算子/
  MindIE kernel），其他卡同理各自成模块。
- **KernelBackend 之外禁止硬件特有假设**：RadixKVCache、scheduler、loader、
  engine loop 只依赖 torch device 语义与 layout 描述符，不 import flashinfer、
  不写 `torch.cuda.*` 直调、不假定 NCCL（通信后端由 KernelBackend 声明，
  如 NCCL/HCCL）。现状中 runner 顶层 `import flashinfer` 与 CUDA-only 分支
  是要在 S1 重构中清掉的典型例子。
- **能力声明而非 if-else**：每个后端声明自己支持的 dtype（FP4/FP8 是否可用）、
  KV layout、CUDA graph 等价物等 capability；上层按声明降级（如昇腾初期
  BF16-only），不在核心路径散落设备判断。
- **优先级**：Phase 0/1 只实现和验证 CUDA 后端；昇腾等以"接口预留 + 不阻塞"的
  方式对待——即接口设计时用上述规则审查，但不提前实现，避免抽象空转。

## 4. 实施顺序

```text
S1  M2/M3 补完：paged KV 成为唯一真相（去掉 DynamicCache 重建），
    FlashInfer prefill/decode kernel 进入 hot path，中央 async engine loop。
    验证模型：fixture / Mixtral 小模型。
S2  KV layout 抽象 + DeepSeek-V2-Lite 先导：
    RadixKVCache 增加 per-layer layout 描述符（标准 KV / MLA compressed），
    KernelBackend 收口 MLA attention（FlashInfer MLA wrapper）。
S3  单机 TP=8（受控拓扑，5090×8）：loader 分片 + KernelBackend TP shape；
    EP 只做最小可用，不做 expert load-balancing 调度。
S4  数值路径：BF16 → FP8；量化计算全部走 KernelBackend。
S5  注册 deepseek_v4 家族：tokenizer 直接引用；模型图 Hybrid
    （官方 inference/ 或 remote code）；FP4 expert + FP8 attention 接入；
    CSA/HCA 的 KV 形态以 S2 的 layout 描述符承接。
    mHC、hash routing 等留在模型图内部，scheduler 不感知。
```

V2-Lite 比 Mixtral 更适合当先导，因为它同时踩中 MoE + MLA 两个 V4 前置依赖，
且单卡可跑，能在 CI/开发机验证正确性。

## 5. 明确不做或后置

与 scope.md 一致：

- DSpark/MTP 投机解码：scope 外，主模型稳定 decode 即可；
- 1M 上下文：后置，先保证常规长度正确性与吞吐；
- tool/reasoning parser：上移 gateway；
- 完整 EP、disagg、多节点：非目标。

## 6. 验收标准（窄口径）

8×5090 上：

1. 加载本地 `ds-v4-flash` 权重成功（TP=8）；
2. 短 prompt 生成正确，数值与官方参考实现 / vLLM 短输出对齐；
3. prefix 复用时 `cache_hit_tokens` 上升，且对应真实 prefill 计算减少；
4. standalone serving 能真实 SSE 流式输出。

任何一条达不到，说明底座还没好，不应继续堆 V4 特例。

## 7. 风险与开放问题

- V4-Flash 的 CSA/HCA、FP4 expert、mHC 细节以官方 `inference/` 参考实现为准，
  本文对其的描述属于需求输入，接入前需逐项核对；
- FlashInfer 对 V4 新 attention 形态的支持进度是外部依赖，KernelBackend 接口
  设计需允许临时回退到官方 kernel；
- MLA 的 Radix page 复用语义（compressed KV 的 COW/fork）需要在 S2 单独验证。
