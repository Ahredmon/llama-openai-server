#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
  echo "[warn] .env not found — copy .env.example to .env and configure it."
  exit 1
fi

source venv/bin/activate 2>/dev/null || true

# Pin ROCm to discrete GPU (Device 0). Prevents the iGPU from being enumerated
# as a second ROCm device, which causes segfaults during KV cache allocation.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

# Normalize log level to lowercase (uvicorn rejects uppercase)
_log_level="${LOG_LEVEL:-info}"
exec uvicorn src.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --log-level "${_log_level,,}"
