#!/usr/bin/env bash
# sglang-lite "lite" production-ish env preset (official sparse path).
# Usage: source scripts/env_lite.sh
#
# Does not start the server — only exports stable defaults for deploy / soak.

export SGLANG_LITE_PRESET="${SGLANG_LITE_PRESET:-lite}"

# Product: DeepSeek-V4-Flash only (docs/v4-flash-only.md)
export SGLANG_LITE_V4_ONLY="${SGLANG_LITE_V4_ONLY:-1}"

# Sparse decode: auto → torch gather (fast on PRO6000). FI only if FORCE + FI_PREFIX.
export SGLANG_LITE_V4_SPARSE="${SGLANG_LITE_V4_SPARSE:-auto}"
export SGLANG_LITE_V4_DISABLE_FI_SPARSE="${SGLANG_LITE_V4_DISABLE_FI_SPARSE:-1}"
export SGLANG_LITE_FI_PREFIX="${SGLANG_LITE_FI_PREFIX:-}"
unset SGLANG_LITE_V4_FORCE_FI_SPARSE 2>/dev/null || true

# Prefer in-tree vendored official graph when present
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$_ROOT/engine/vendor/deepseek_infer/model.py" ]]; then
  export SGLANG_LITE_DSV4_INFER="${SGLANG_LITE_DSV4_INFER:-$_ROOT/engine/vendor/deepseek_infer}"
fi

# Structured logs on (request_id JSON). Set 0 to quiet soak.
export SGLANG_LITE_LOG_JSON="${SGLANG_LITE_LOG_JSON:-1}"

# Engine sizing
export SGLANG_LITE_MAX_BATCH_SIZE="${SGLANG_LITE_MAX_BATCH_SIZE:-4}"
export SGLANG_LITE_MAX_CONCURRENT="${SGLANG_LITE_MAX_CONCURRENT:-32}"
export SGLANG_LITE_MAX_TOKENS="${SGLANG_LITE_MAX_TOKENS:-128}"
export SGLANG_LITE_REQUEST_TIMEOUT="${SGLANG_LITE_REQUEST_TIMEOUT:-300}"
export SGLANG_LITE_PORT="${SGLANG_LITE_PORT:-9001}"
export SGLANG_LITE_DEVICE="${SGLANG_LITE_DEVICE:-cuda}"

# CUDA toolchain (PRO6000 / host) — safe no-ops if missing
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
fi
if [[ -d /usr/local/cuda/include ]]; then
  export CPATH="${CPATH:-/usr/local/cuda/include}"
  export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-/usr/local/cuda/include}"
fi

# V4 Hybrid (optional — set if using DeepSeek-V4-Flash)
# export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
# export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8

echo "[env_lite] preset=${SGLANG_LITE_PRESET} device=${SGLANG_LITE_DEVICE} batch=${SGLANG_LITE_MAX_BATCH_SIZE} DISABLE_FI=${SGLANG_LITE_V4_DISABLE_FI_SPARSE}"
