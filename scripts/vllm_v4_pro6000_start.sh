#!/usr/bin/env bash
# Start vLLM 0.25.0 DeepSeek-V4-Flash-0731 on 8× RTX PRO 6000 (recipe-aligned).
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash  (RTX PRO 6000 8× section)
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V4-Flash-0731}"
NAME="${NAME:-vllm-v4-flash}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.0}"

sudo docker rm -f "$NAME" 2>/dev/null || true
sudo docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 32g \
  -p "${PORT}:8000" \
  -v "$MODEL_DIR":/model:ro \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --entrypoint bash \
  "$IMAGE" \
  -lc '
    set -e
    /usr/bin/python3.12 -m pip install --no-deps -q flashinfer-python==0.6.14 || true
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    exec vllm serve /model \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --block-size 256 \
      --enable-expert-parallel \
      --tensor-parallel-size 8 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.90
  '
echo "started $NAME on :$PORT — wait for GET /v1/models"
