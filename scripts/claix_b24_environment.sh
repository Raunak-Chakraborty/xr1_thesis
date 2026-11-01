#!/usr/bin/env bash
# Source inside an allocation; use only the previously approved offline setup.
module load Python/3.11.5
source "$HOME/robotics_thesis/configs/xr1_hf_offline.env"
: "${XR1_PYTHON:?Set existing approved CLAIX environment Python; no installation}"
[[ -x "$XR1_PYTHON" ]] || { echo 'Approved Python is not executable' >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled

xr1_require_persistent_directory() {
    "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_filesystem.py" --path "$1"
}
