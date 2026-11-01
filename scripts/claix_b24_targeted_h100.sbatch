#!/usr/bin/env bash
# Prepared only; never submitted by development tools. Use a NEW persistent root.
#SBATCH --job-name=xr1_b24_final_verify
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=claix_b24_final_verify_%j.out
#SBATCH --error=claix_b24_final_verify_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set final reviewed source}"
: "${XR1_RUN_ROOT:?Set NEW persistent targeted qualification root}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set existing approved ZIP cache}"
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_targeted_h100.py" \
    --execute --python "$XR1_PYTHON" --run-root "$XR1_RUN_ROOT"
