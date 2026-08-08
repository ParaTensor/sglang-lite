#!/usr/bin/env bash
# KPI gate: DeepSeek-V4-Flash warm decode tok/s — sglang-lite vs SGLang (same weights).
#
# Prerequisites (weight host, typically 8×GPU / PRO6000):
#   export SGLANG_LITE_DSV4_HF=/path/to/DeepSeek-V4-Flash
#   export SGLANG_LITE_DSV4_CONVERTED=/path/to/mp-shards   # convert.py output
#   SGLang installed separately for baseline (not imported by lite)
#
# Usage:
#   bash scripts/v4_vs_sglang_bench.sh
#   bash scripts/v4_vs_sglang_bench.sh --mp 8 --max-new 128 --skip-sglang

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env_lite.sh

MP="${MP:-8}"
MAX_NEW="${MAX_NEW:-128}"
SKIP_SGLANG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mp) MP="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --skip-sglang) SKIP_SGLANG=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

export SGLANG_LITE_V4_ONLY=1
HF="${SGLANG_LITE_DSV4_HF:?set SGLANG_LITE_DSV4_HF}"
CONV="${SGLANG_LITE_DSV4_CONVERTED:?set SGLANG_LITE_DSV4_CONVERTED}"
# Prefer in-tree vendor graph
export SGLANG_LITE_DSV4_INFER="${SGLANG_LITE_DSV4_INFER:-$ROOT/engine/vendor/deepseek_infer}"
OUT_DIR="${OUT_DIR:-/tmp/sglang-lite-v4-kpi}"
mkdir -p "$OUT_DIR"

echo "=== sglang-lite V4 KPI ==="
echo "HF=$HF CONV=$CONV INFER=$SGLANG_LITE_DSV4_INFER MP=$MP max_new=$MAX_NEW"

# 1) Official-path wall clock via lite vendor generate (same as upstream generate.py)
python scripts/v4_official_bench.py \
  --mp "$MP" \
  --max-new-tokens "$MAX_NEW" \
  --num-prompts 4 \
  2>&1 | tee "$OUT_DIR/lite_official_bench.log" || {
    echo "[warn] v4_official_bench failed — ensure convert shards + GPUs"
  }

# 2) LiteEngine short gen (continuous batch path)
if [[ -f scripts/v4_lite_short_gen.py ]]; then
  echo "=== LiteEngine short gen ==="
  torchrun --nproc-per-node="$MP" scripts/v4_lite_short_gen.py \
    --max-new-tokens "$MAX_NEW" \
    2>&1 | tee "$OUT_DIR/lite_short_gen.log" || true
fi

if [[ "$SKIP_SGLANG" -eq 1 ]]; then
  echo "skip SGLang baseline"
  exit 0
fi

if ! python -c "import sglang" 2>/dev/null; then
  echo "[warn] sglang not installed — skip baseline. Install on host for KPI."
  exit 0
fi

echo "=== SGLang baseline (operator-owned launch) ==="
# Documented recipe; exact flags may differ by SGLang version — pin in runbook.
SGLANG_LOG="$OUT_DIR/sglang_baseline.log"
if [[ -f scripts/sglang_offline_bench.py ]]; then
  python scripts/sglang_offline_bench.py \
    --model-path "$HF" \
    --max-new-tokens "$MAX_NEW" \
    2>&1 | tee "$SGLANG_LOG" || true
else
  cat <<EOF | tee "$SGLANG_LOG"
# Manual baseline (example):
# python -m sglang.launch_server --model-path $HF --tp $MP ...
# then client warmup 1x$MAX_NEW and record tok/s
EOF
fi

echo "KPI logs under $OUT_DIR — compare warm tok/s; lite must beat SGLang (docs/v4-flash-only.md §5)."
