#!/usr/bin/env bash
# Prepared only. Submission requires reviewed implementation and CPU acceptance.
#SBATCH --job-name=xr1_last2_serious
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=23:45:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=xr1_last2_serious_%j.out
#SBATCH --error=xr1_last2_serious_%j.err
set -Eeuo pipefail

: "${SLURM_JOB_ID:?Allocation required}"
: "${XR1_WORKTREE:?Exact reviewed source checkout}"
: "${XR1_EXPECTED_SHA:?Pinned reviewed commit}"
: "${XR1_PYTHON:?Existing approved offline Python}"
: "${XR1_RUN_ROOT:?New persistent serious root, or same root for continuation}"
: "${XR1_PATH_OVERRIDES:?Existing approved four-path JSON}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Existing immutable ZIP cache}"
: "${XR1_FILESYSTEM_SOURCE:?Verified persistent Lustre source}"
: "${XR1_LUSTRE_PROJECT_ID:?Verified project accounting ID}"
: "${XR1_SERIOUS_AUDIT:?Successful dedicated pre-submission audit JSON}"
test "$(git -C "$XR1_WORKTREE" rev-parse HEAD)" = "$XR1_EXPECTED_SHA"
test -z "$(git -C "$XR1_WORKTREE" status --porcelain --untracked-files=all)"
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
export PYTHONPATH="$XR1_WORKTREE/xr1:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
xr1_require_persistent_directory "${SLURM_SUBMIT_DIR:?}"
xr1_require_persistent_directory "$PWD"
args=(--mode experimental --experimental-unqualified --experimental-checkpoint-interval 500
      --serious-last2 --serious-audit "$XR1_SERIOUS_AUDIT" --execute --python "$XR1_PYTHON"
      --run-root "$XR1_RUN_ROOT" --workers 4 --overrides "$XR1_PATH_OVERRIDES"
      --filesystem-source "$XR1_FILESYSTEM_SOURCE" --project-id "$XR1_LUSTRE_PROJECT_ID")
if [[ -n "${XR1_RESUME:-}" ]]; then args+=(--resume "$XR1_RESUME"); fi
if [[ -n "${XR1_STOP_AFTER_STEP:-}" ]]; then args+=(--stop-after-step "$XR1_STOP_AFTER_STEP"); fi
exec "$XR1_PYTHON" -B "$XR1_WORKTREE/scripts/claix_b24_launch.py" "${args[@]}"
