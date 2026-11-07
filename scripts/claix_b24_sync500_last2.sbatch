#!/usr/bin/env bash
#SBATCH --job-name=xr1_sync500_last2
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=xr1_sync500_last2_%j.out
#SBATCH --error=xr1_sync500_last2_%j.err
set -Eeuo pipefail

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${XR1_WORKTREE:?Missing exact source worktree}"
: "${XR1_EXPECTED_SHA:?Missing pinned source commit}"
: "${XR1_PYTHON:?Missing approved Python}"
: "${XR1_COMBINED_ROOT:?Missing new combined run root}"
: "${XR1_PATH_OVERRIDES:?Missing frozen path overrides}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Missing frozen ZIP cache}"
: "${XR1_FILESYSTEM_SOURCE:?Missing Lustre source}"
: "${XR1_LUSTRE_PROJECT_ID:?Missing project ID}"

test "$(git -C "$XR1_WORKTREE" rev-parse HEAD)" = "$XR1_EXPECTED_SHA"
test -z "$(git -C "$XR1_WORKTREE" status --porcelain)"
test ! -e "$XR1_COMBINED_ROOT"

source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"

export PYTHONPATH="$XR1_WORKTREE/xr1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec "$XR1_PYTHON" -B \
    "$XR1_WORKTREE/scripts/claix_b24_sync500_last2.py" \
    --execute \
    --run-root "$XR1_COMBINED_ROOT" \
    --overrides "$XR1_PATH_OVERRIDES" \
    --filesystem-source "$XR1_FILESYSTEM_SOURCE" \
    --project-id "$XR1_LUSTRE_PROJECT_ID" \
    --python "$XR1_PYTHON"
