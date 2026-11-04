#!/usr/bin/env bash
# Not submitted. Submit from a reviewed persistent directory for Slurm logs.
#SBATCH --job-name=xr1_b24_production
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=claix_b24_production_%j.out
#SBATCH --error=claix_b24_production_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set reviewed isolated worktree}"
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
: "${XR1_RUN_ROOT:?Set unique persistent run root}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set approved ZIP frame cache}"
# Launcher logs/provenance are additionally persisted beneath XR1_RUN_ROOT.
xr1_require_persistent_directory "${SLURM_SUBMIT_DIR:?}"
xr1_require_persistent_directory "$PWD"
args=(--mode production --execute --python "$XR1_PYTHON" --run-root "$XR1_RUN_ROOT")
args+=(--workers "${XR1_WORKERS:-4}")
if [[ -n "${XR1_RESUME:-}" ]]; then args+=(--resume "$XR1_RESUME"); fi
if [[ -n "${XR1_STOP_AFTER_STEP:-}" ]]; then args+=(--stop-after-step "$XR1_STOP_AFTER_STEP"); fi
: "${XR1_RELEASE_RECORD:?Production requires a complete reviewed release record}"
args+=(--release-record "$XR1_RELEASE_RECORD")
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_launch.py" "${args[@]}"
