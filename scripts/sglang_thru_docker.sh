#!/usr/bin/env bash
# Same-host SGLang thruput probe via Docker (PRO6000).
# Compares against sglang-lite scripts/moe_thruput_probe.py on the same weights.
#
# Usage:
#   bash scripts/sglang_thru_docker.sh \
#     --model /home/bodesi/models/Qwen3-30B-A3B-Instruct \
#     --out /home/bodesi/bench/thru_sglang_qwen3_30b.json
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3-30B-A3B-Instruct}"
OUT="${OUT:-$HOME/bench/thru_sglang_qwen3_30b.json}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_NEW="${MAX_NEW:-128}"
INPUT_LEN="${INPUT_LEN:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --input-len) INPUT_LEN="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$OUT")"
HOST_MODEL="$(readlink -f "$MODEL")"
HOST_OUT_DIR="$(readlink -f "$(dirname "$OUT")")"
OUT_NAME="$(basename "$OUT")"

echo "[sglang-docker] image=$IMAGE model=$HOST_MODEL gpu=$GPU out=$OUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="${SCRIPT_DIR}/sglang_offline_bench.py"
if [[ ! -f "$BENCH_PY" ]]; then
  echo "missing $BENCH_PY"
  exit 2
fi

# Mount a real .py file (stdin heredoc breaks SGLang multiproc on some images).
docker run --rm --gpus "\"device=${GPU}\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "${HOST_MODEL}:/model:ro" \
  -v "${HOST_OUT_DIR}:/bench" \
  -v "${BENCH_PY}:/bench_src/sglang_offline_bench.py:ro" \
  "$IMAGE" \
  python /bench_src/sglang_offline_bench.py \
    --model /model \
    --out "/bench/${OUT_NAME}" \
    --max-new "${MAX_NEW}"

echo "[sglang-docker] done -> $OUT"
cat "$OUT"
