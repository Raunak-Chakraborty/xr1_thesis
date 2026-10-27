#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$REPO/xr1${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN:-python3}" -B -m unittest discover \
    -s "$REPO/xr1/tests" -p 'test_doosan_force_*.py' -v
