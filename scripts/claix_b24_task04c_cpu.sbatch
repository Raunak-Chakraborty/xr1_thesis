#!/usr/bin/env bash
# CPU acceptance only; no submission command or training launcher.
#SBATCH --job-name=xr1_task04c_cpu
#SBATCH --account=thes2452
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=xr1_task04c_cpu_%j.out
#SBATCH --error=xr1_task04c_cpu_%j.err
set -Eeuo pipefail
: "${XR1_WORKTREE:?Set exact reviewed Task04C checkout}"
source "$XR1_WORKTREE/scripts/claix_b24_environment.sh"
: "${XR1_CPU_EVIDENCE_ROOT:?Set NEW persistent uniquely named evidence directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
xr1_require_persistent_directory "${SLURM_SUBMIT_DIR:?}"
xr1_require_persistent_directory "$PWD"
exec "$XR1_PYTHON" -B "$XR1_WORKTREE/scripts/claix_b24_task04c_cpu.py" \
    --output "$XR1_CPU_EVIDENCE_ROOT"
