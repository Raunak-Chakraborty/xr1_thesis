#!/usr/bin/env bash
# DRAFT: not submitted. Review site partition/GRES spelling before use.
#SBATCH --job-name=xr1_b24_cpu_qualification
#SBATCH --account=thes2452
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=claix_b24_cpu_qualification_%j.out
#SBATCH --error=claix_b24_cpu_qualification_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set reviewed isolated worktree}"
: "${XR1_PYTHON:?Set existing CLAIX environment Python}"
: "${XR1_RUN_ROOT:?Set unique persistent run root}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set approved ZIP frame cache}"
args=(--mode cpu-qualification --execute --python "$XR1_PYTHON" --run-root "$XR1_RUN_ROOT")
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_launch.py" "${args[@]}"
