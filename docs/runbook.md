# sglang-lite 运维 Runbook（稳部署）

面向 **standalone Token Factory** 最小可部署路径。主路径为官方 attention 核；
FlashInfer SM120 sparse **不默认开启**。

## 1. 环境预设

```bash
cd /path/to/sglang-lite
source scripts/env_lite.sh

# DeepSeek-V4-Flash（可选）
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
export SGLANG_LITE_MODEL="$SGLANG_LITE_DSV4_HF"
```

关键变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `SGLANG_LITE_V4_DISABLE_FI_SPARSE` | `1` | 官方 `sparse_attn` 主路径 |
| `SGLANG_LITE_LOG_JSON` | `1` | `sglang_lite.req` JSON 行日志 |
| `SGLANG_LITE_MAX_BATCH_SIZE` | `4` | continuous batch 上限 |
| `SGLANG_LITE_REQUEST_TIMEOUT` | `300` | 单请求超时秒 |

## 2. 启动

**CPU / 单卡 Mixtral 类（HF id）**

```bash
python -m sglang_lite.process \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --device cuda --port 9001
```

**DeepSeek-V4-Flash TP=8（PRO6000 宿主）**

```bash
source scripts/env_lite.sh
export PYTHONPATH=$PWD/engine:$PWD
torchrun --nproc-per-node=8 -m sglang_lite.process \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda --port 9001
```

可选控制面（Rust）：

```bash
# 另开终端
sglang-lite-serving --engine-url http://127.0.0.1:9001 --port 8000
```

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

**Soak（默认 tiny Mixtral fixture，CPU）**

```bash
python scripts/soak_stability.py --rounds 30 --concurrency 8 --max-new 4 \
  --out /tmp/soak.json
# overall PASS：errors=0、oom=0、blocks 稳定
```

**PRO6000 V4（可选）**

```bash
source scripts/env_lite.sh
torchrun --nproc-per-node=8 scripts/soak_stability.py \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda \
  --rounds 10 --concurrency 2 --max-new 8 \
  --out ~/bench/soak_pro6000.json
```

**多 MoE 最小回归**

```bash
python scripts/moe_regression.py --out /tmp/moe_reg.json
# 额外模型：
# python scripts/moe_regression.py --model fixture:/path --model /path/to/qwen-moe
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
