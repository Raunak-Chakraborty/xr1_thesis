#!/usr/bin/env bash
# Standalone qualification only. Not submitted during development.
#SBATCH --job-name=xr1_b24_combined_qualification
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=claix_b24_combined_%j.out
#SBATCH --error=claix_b24_combined_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set reviewed isolated combined-qualification worktree}"
: "${XR1_PYTHON:?Set existing CLAIX environment Python; no installation/downloads}"
: "${XR1_RUN_ROOT:?Set NEW persistent combined run root; must not exist}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set approved ZIP frame cache}"
: "${XR1_SPLIT_IDENTITIES:?Set existing reviewed frozen split inventory SHA256 JSON}"
args=(--execute --python "$XR1_PYTHON" --run-root "$XR1_RUN_ROOT"
      --split-identities "$XR1_SPLIT_IDENTITIES")
# Default paths are the existing staged transfer paths in the reference overrides.
if [[ -n "${XR1_VALIDATION_ROOT:-}" ]]; then args+=(--validation-root "$XR1_VALIDATION_ROOT"); fi
if [[ -n "${XR1_TEST_ROOT:-}" ]]; then args+=(--test-root "$XR1_TEST_ROOT"); fi
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_combined_qualification.py" "${args[@]}"
