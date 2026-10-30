#!/usr/bin/env bash
# DRAFT: execute only after documented compute-node qualification and release.
#SBATCH --job-name=xr1_b24_production
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=23:45:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=claix_b24_%j.out
#SBATCH --error=claix_b24_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set reviewed isolated worktree}"
: "${XR1_PYTHON:?Set existing CLAIX environment Python}"
: "${XR1_RUN_ROOT:?Set unique persistent run root}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set approved ZIP frame cache}"
: "${XR1_MODE:=qualification}"
args=(--mode "$XR1_MODE" --execute --python "$XR1_PYTHON" --run-root "$XR1_RUN_ROOT" --workers "${XR1_WORKERS:-4}")
if [[ -n "${XR1_RELEASE_RECORD:-}" ]]; then args+=(--release-record "$XR1_RELEASE_RECORD"); fi
if [[ -n "${XR1_RESUME:-}" ]]; then args+=(--resume "$XR1_RESUME"); fi
if [[ -n "${XR1_STOP_AFTER_STEP:-}" ]]; then args+=(--stop-after-step "$XR1_STOP_AFTER_STEP"); fi
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_launch.py" "${args[@]}"
