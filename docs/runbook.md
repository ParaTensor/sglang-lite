# sglang-lite 运维 Runbook（稳部署）

面向 **DeepSeek-V4-Flash 专用** standalone Token Factory。主路径为 **vendor 官方图**
（`engine/vendor/deepseek_infer`）+ 官方 attention 核；FlashInfer SM120 sparse **不默认开启**。
权威产品宪章：[v4-flash-only.md](./v4-flash-only.md)。

## 1. 环境预设

```bash
cd /path/to/sglang-lite
source scripts/env_lite.sh

# DeepSeek-V4-Flash（唯一一等公民）
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
export SGLANG_LITE_MODEL="$SGLANG_LITE_DSV4_HF"
# 图默认走仓内 vendor（env_lite 已 setdefault）
# export SGLANG_LITE_DSV4_INFER=$PWD/engine/vendor/deepseek_infer
```

关键变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `SGLANG_LITE_V4_ONLY` | `1` | 拒绝非 V4-Flash 模型 |
| `SGLANG_LITE_DSV4_INFER` | `engine/vendor/deepseek_infer` | 官方/vendor 图路径 |
| `SGLANG_LITE_V4_DISABLE_FI_SPARSE` | `1` | 官方 `sparse_attn` 主路径 |
| `SGLANG_LITE_V4_CUDA_GRAPH` | 关 | 固定 start_pos 微基准 CUDA graph |
| `SGLANG_LITE_DECODE_BURST` | `64` | 单请求 decode 连打步数（thruput 可 `128`） |
| `SGLANG_LITE_V4_DUAL_APPEND` | `1` | decode 写 dual-pool；thruput 可 `0` |
| `SGLANG_LITE_LOG_JSON` | `1` | `sglang_lite.req` JSON 行日志 |
| `SGLANG_LITE_MAX_BATCH_SIZE` | `4` | continuous batch 上限 |
| `SGLANG_LITE_REQUEST_TIMEOUT` | `300` | 单请求超时秒 |

## 2. 启动

**DeepSeek-V4-Flash TP=8（PRO6000 宿主）— 唯一生产路径**

```bash
source scripts/env_lite.sh
export PYTHONPATH=$PWD/engine:$PWD
torchrun --nproc-per-node=8 -m sglang_lite.process \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda --port 9001
```

**权重转换**（首次，官方 convert 已 vendor）：

```bash
export EXPERTS=256 MP=8
python engine/vendor/deepseek_infer/convert.py \
  --hf-ckpt-path "$SGLANG_LITE_DSV4_HF" \
  --save-path "$SGLANG_LITE_DSV4_CONVERTED" \
  --n-experts $EXPERTS --model-parallel $MP
```

**Legacy 多 MoE**（非 KPI，需显式）：`SGLANG_LITE_V4_ONLY=0`

可选控制面（Rust）：

```bash
# 另开终端
sglang-lite-serving --engine-url http://127.0.0.1:9001 --port 8000
```

## 2.1 KPI：对打 SGLang

同机同权重 warm decode tok/s（见 v4-flash-only §5）：

```bash
bash scripts/v4_vs_sglang_bench.sh --mp 8 --max-new 128
```

记录 lite 与 SGLang 的 warm tok/s；主 KPI 要求 **lite > SGLang**。

## 3. 健康与指标

```bash
curl -s localhost:9001/healthz
curl -s localhost:9001/readyz
curl -s localhost:9001/metrics | head -40
curl -s localhost:9001/stats | jq '.latency, .dual_pool, .cache'
```

关注：

- `sglang_lite_ready` / `sglang_lite_draining`
- `sglang_lite_kv_blocks_used`（soak 中不应无界爬升）
- `sglang_lite_oom_reject_count`
- `sglang_lite_dual_stage_count`（V4 page-primary）
- `sglang_lite_ttft_seconds_avg` / `sglang_lite_tok_s_avg`

结构化日志（request_id）：

```bash
# 进程 stdout / 日志里过滤
# {"event":"request_finish","request_id":"...","ttft_s":...,"tok_s":...}
```

## 4. 优雅排空

```bash
curl -s -X POST localhost:9001/v1/drain
# 轮询直到 idle
curl -s localhost:9001/v1/drain
# readyz 在 drain 中为 503
curl -s -o /dev/null -w '%{http_code}\n' localhost:9001/readyz
# 再停进程（Ctrl+C / kill）
```

## 5. 稳定性门禁（上线前）

**Soak 时长策略（`--profile`）**

| profile | 大致时长 | 默认 rounds | concurrency | max_new | 用途 |
|---------|----------|-------------|-------------|---------|------|
| `smoke` | 1–2 min | 10 | 4 | 4 | PR / 冒烟 |
| `short` | 5–10 min | 40 | 8 | 8 | 日常门禁 |
| `medium` | 20–30 min | 120 | 8 | 8 | 发版前 |
| `long` | **墙钟 1h**（`--duration-s 3600`） | 上限很大 | 4 | 8 | 稳部署 / 过夜前 |

也可用 `--duration-s N` 按秒截断（与 rounds 取先到者）。

```bash
# CPU fixture 短 soak
python scripts/soak_stability.py --profile short --out /tmp/soak.json
# overall PASS：errors=0、oom=0、blocks 稳定
```

**PRO6000 SSH**

```bash
ssh -p 2208 bodesi@39.183.171.3   # hostname=pro6000
# 代码：~/src/sglang-lite（git main）
# venv：source ~/venvs/sglang-lite/bin/activate
# 权重：~/models/DeepSeek-V4-Flash-0731  shards：~/models/ds-v4-mp8
```

**PRO6000 V4 Hybrid（真机）**

```bash
source scripts/env_lite.sh
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
# 15–30 min 稳部署
torchrun --nproc-per-node=8 scripts/soak_stability.py \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda \
  --profile long --duration-s 1800 --concurrency 2 --max-new 8 \
  --max-blocks-slack 256 \
  --out ~/bench/soak_v4_30min.json
```

判据：`errors=0`、`oom=0`、`blocks_used` 全程平坦（V4 实测恒为 2）、`dual_stage` 单调增。
**多 MoE 最小回归**

```bash
# CPU tiny Mixtral fixture
python scripts/moe_regression.py --out /tmp/moe_reg.json

# PRO6000 多 MoE 真机回归（≤300B；2026-08-07 已验 PASS×3）
# 矩阵：Qwen1.5-MoE-A2.7B + DeepSeek-V2-Lite + MiniMax-M2（~230B）
# MiniMax-M3 ~428B+多模态 → SKIP
source ~/venvs/sglang-lite/bin/activate
cd ~/project/sglang-lite
export PYTHONPATH=engine:.
export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1

# 小：Qwen1.5-MoE（FI paged）
CUDA_VISIBLE_DEVICES=0 python scripts/moe_regression.py \
  --model ~/models/Qwen1.5-MoE-A2.7B-Chat --device cuda --max-new 16 \
  --out ~/bench/moe_reg_qwen.json

# Phase-A 吞吐探针（加载后 tok/s；非 vLLM/SGLang 对照）
# 注意：Qwen3.5-27B 是 Dense+多模态，**不在 scope**；用 Qwen3-30B-A3B（文本 MoE）。
# 探针默认：FORCE_HF_CACHE=1、EXPERTS_IMPL=batched_mm、TORCH_COMPILE=1。
# PRO6000 warm ≈ 83–85 tok/s（SGLang ≈155 → ~1.85×）。load 含 compile warmup ~110s。
CUDA_VISIBLE_DEVICES=0 python scripts/moe_thruput_probe.py \
  --model ~/models/Qwen3-30B-A3B-Instruct --device cuda \
  --cases 1x64,1x128 --out ~/bench/thru_qwen3_30b_a3b.json
# bit-exact eager（~47 tok/s）：SGLANG_LITE_TORCH_COMPILE=0 ...
# 实验 cutlass fused MoE（e2e ~44，默认关）：SGLANG_LITE_FUSED_MOE=1 SGLANG_LITE_TORCH_COMPILE=0 ...
# Radix-native（PRO6000 Qwen3-30B ~103 tok/s warm；距 SGLang~155 仍~1.5×）：
#   SGLANG_LITE_RADIX_NATIVE=1 python scripts/moe_thruput_probe.py ...
#   默认：CUDA_GRAPH=1 + FUSED_MOE=1 + NATIVE_DECODE=1 + FUSE_QKV=1
#   MoE 后端：SGLANG_LITE_MOE_BACKEND=cutlass|trtllm|sgl|auto（PRO6000 仅 cutlass 可用）
#   叶核探测：python scripts/moe_kernel_probe.py --e2e --model ~/models/Qwen3-30B-A3B-Instruct
#
# HF golden gate（见 docs/pega-lessons.md）：
#   # FORCE_HF exact（默认 require-exact；dump 单卡+batched_mm）
#   python scripts/hf_golden_gate.py gate \
#     --model ~/models/Qwen3-30B-A3B-Instruct --path force_hf --out-dir /tmp/golden
#   # radix-native 只报告 first-diff
#   python scripts/hf_golden_gate.py gate \
#     --model ~/models/Qwen3-30B-A3B-Instruct --path radix_native \
#     --out-dir /tmp/golden --no-require-exact

# 小：DeepSeek-V2-Lite（MLA → HF cache；首次需 TF5 patch）
python scripts/patch_deepseek_v2_tf5.py ~/models/DeepSeek-V2-Lite-Chat
CUDA_VISIBLE_DEVICES=0 python scripts/moe_regression.py \
  --model ~/models/DeepSeek-V2-Lite-Chat --device cuda --max-new 16 \
  --out ~/bench/moe_reg_ds_v2lite.json

# 中大：MiniMax-M2（GQA=6 跳过 FI paged；FP8 dequantize；多卡 device_map=auto）
python scripts/patch_minimax_m2_tf5.py ~/models/MiniMax-M2
python scripts/patch_minimax_m2_rope_init.py ~/models/MiniMax-M2
# config.quantization_config.dequantize=true（权重目录侧）
python scripts/moe_regression.py \
  --model ~/models/MiniMax-M2 --device cuda --max-new 8 \
  --out ~/bench/moe_reg_minimax_m2.json
```

下载权重（HF SSL 不稳时用 ModelScope）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen1.5-MoE-A2.7B-Chat', cache_dir='$HOME/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-V2-Lite-Chat', cache_dir='$HOME/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('MiniMax/MiniMax-M2', cache_dir='$HOME/models/ms_cache'))"
# 软链到 ~/models/<Name>
```

**V4 dual + 吞吐（已有）**

```bash
torchrun --nproc-per-node=8 scripts/v4_dual_stats_probe.py --out ~/bench/v4_dual.json
torchrun --nproc-per-node=8 scripts/v4_lite_engine_gen.py --case 1x128
# 对照 docs/deepseek-v4-flash-plan.md §6.4.1 / §6.4.2
```

## 6. 常见故障

| 现象 | 处理 |
|------|------|
| soft gate 乱码 / top5 怪异 | 确认宿主 torch+cu130，勿用坏 Docker cu129 数值栈 |
| TileLang device mismatch | CVD=`LOCAL_RANK`，进程内 `cuda:0` |
| FI 更慢或全 0 | 保持 `DISABLE_FI=1`；FI 仅 FORCE 实验 |
| NCCL destroy 退出 SIGABRT | 已知；门禁看 JSON 结果，探针已避免结果后 barrier |
| blocks_used 只增不减 | soak FAIL；查 cancel/finish 是否 `v4_release_seq` / Radix release |
| `/readyz` 503 | 未 READY 或 **draining** |

## 7. 明确不做（部署范围）

structured output / tool 执行、投机解码、PD disagg、多模态、完整 EP、默认 FI 换核。  
业务与宽网关能力在 UniGateway，不进 `engine/`。
