#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$REPO/xr1"

"$PYTHON_BIN" -B -m unittest discover \
    -s "$REPO/xr1/tests" \
    -p 'test_doosan_*.py' \
    -v
