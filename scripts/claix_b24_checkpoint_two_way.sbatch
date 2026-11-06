#!/usr/bin/env bash
# Task 04B: exactly B sync-500 then C async-500; never a new sync-250 run.
#SBATCH --job-name=xr1_b24_checkpoint_two_way
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=23:45:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=claix_b24_checkpoint_two_way_%j.out
#SBATCH --error=claix_b24_checkpoint_two_way_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set this Task 04B worktree}"
: "${XR1_EXPERIMENTAL_UNQUALIFIED:?Explicitly set to 1}"
[[ "$XR1_EXPERIMENTAL_UNQUALIFIED" == 1 ]] || exit 1
[[ -z "${XR1_RESUME:-}" && -z "${XR1_STOP_AFTER_STEP:-}" ]] || exit 1
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
: "${XR1_TWO_WAY_ROOT:?Set a NEW persistent parent for independent B and C roots}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set existing frozen Lustre ZIP cache}"
: "${XR1_FILESYSTEM_SOURCE:?Set operator-verified persistent findmnt SOURCE}"
: "${XR1_LUSTRE_PROJECT_ID:?Set actual storage project ID}"
: "${XR1_PATH_OVERRIDES:?Set existing four staged path overrides JSON}"
xr1_require_persistent_directory "${SLURM_SUBMIT_DIR:?}"
xr1_require_persistent_directory "$PWD"
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_checkpoint_two_way.py" \
    --execute --python "$XR1_PYTHON" --run-root "$XR1_TWO_WAY_ROOT" \
    --overrides "$XR1_PATH_OVERRIDES" --filesystem-source "$XR1_FILESYSTEM_SOURCE" \
    --project-id "$XR1_LUSTRE_PROJECT_ID"
