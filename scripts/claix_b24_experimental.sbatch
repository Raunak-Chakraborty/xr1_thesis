#!/usr/bin/env bash
# Intended submission only; see LUSTRE_FRESHSTART_TASK02.md.
#SBATCH --job-name=xr1_b24_experimental
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=claix_b24_experimental_%j.out
#SBATCH --error=claix_b24_experimental_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set this experimental worktree}"
: "${XR1_EXPERIMENTAL_UNQUALIFIED:?Explicitly set to 1}"
[[ "$XR1_EXPERIMENTAL_UNQUALIFIED" == 1 ]] || exit 1
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
: "${XR1_RUN_ROOT:?Set a NEW unique persistent root, or the same root for recovery}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set existing frozen Lustre ZIP cache}"
: "${XR1_FILESYSTEM_SOURCE:?Set operator-verified persistent findmnt SOURCE}"
: "${XR1_LUSTRE_PROJECT_ID:?Set actual storage project ID, not inferred from GID}"
: "${XR1_PATH_OVERRIDES:?Set existing JSON containing four staged path overrides}"
xr1_require_persistent_directory "${SLURM_SUBMIT_DIR:?}"
xr1_require_persistent_directory "$PWD"
args=(--mode experimental --experimental-unqualified --execute --python "$XR1_PYTHON"
      --run-root "$XR1_RUN_ROOT" --workers 4 --overrides "$XR1_PATH_OVERRIDES"
      --filesystem-source "$XR1_FILESYSTEM_SOURCE" --project-id "$XR1_LUSTRE_PROJECT_ID")
if [[ -n "${XR1_RESUME:-}" ]]; then args+=(--resume "$XR1_RESUME"); fi
if [[ -n "${XR1_STOP_AFTER_STEP:-}" ]]; then args+=(--stop-after-step "$XR1_STOP_AFTER_STEP"); fi
exec "$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_launch.py" "${args[@]}"
