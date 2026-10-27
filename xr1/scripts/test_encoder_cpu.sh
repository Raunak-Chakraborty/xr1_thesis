#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${XR1_ENCODER_PYTHON:-/home/agent/.venvs/xr1-encoder/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing persistent CPU environment: $PYTHON_BIN" >&2
    echo "See xr1/docs/ENCODER_CPU_VALIDATION.md for setup instructions." >&2
    exit 1
fi

export PYTHONPATH="$REPO/xr1"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

"$PYTHON_BIN" -B "$REPO/xr1/tools/check_encoder_cpu_env.py"
exec "$PYTHON_BIN" -B -m pytest \
    -o "cache_dir=${XR1_ENCODER_TEST_CACHE:-/home/agent/.cache/xr1-encoder-pytest}" \
    "$REPO"/xr1/tests/test_force_encoder_*.py \
    "$REPO"/xr1/tests/test_doosan_*.py \
    "$REPO/xr1/tests/test_training_logic.py" \
    -v "$@"
