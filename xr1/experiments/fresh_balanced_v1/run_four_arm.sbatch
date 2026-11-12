#!/usr/bin/env bash
#SBATCH --job-name=xr1_fresh_balanced
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=13:00:00
set -Eeuo pipefail
umask 077

ROOT="$1"
BASE="$HOME/robotics_thesis"
WT="$BASE/worktrees/xr1_fresh_balanced_v1"
PY="$BASE/envs/xr1_fp32/bin/python"
CAMPAIGN="$WT/xr1/tools/run_fresh_balanced_campaign.py"
FINAL="$WT/xr1/experiments/fresh_balanced_v1/finalize_campaign.py"

finish() {
  rc=$?
  trap - EXIT
  printf 'JOB_EXIT_CODE=%s\n' "$rc"
  printf 'RUN_ROOT=%s\n' "$ROOT"
  printf '%120s\n' '' | tr ' ' '#'
  exit "$rc"
}
trap finish EXIT

module load Python/3.11.5
export XR1_FRESH_RUN_ROOT="$ROOT"
export PYTHONPATH="$BASE/envs/xr1_motion_addons_v1:$WT/xr1"
source "$BASE/configs/xr1_hf_offline.env"
export XR1_DOOSAN_ZIP_CACHE_ROOT=/hpcwork/zk443626/robotics_thesis/cache/frames_zip/v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_LENGTH=12288
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

(cd "$WT" && sha256sum -c "$ROOT/source_sha256.txt" > "$ROOT/logs/source_integrity.log")
test "$(git -C "$WT" rev-parse HEAD)" = "$(cat "$ROOT/git_commit.txt")"
echo 'FRESH_CAMPAIGN_SOURCE_PINNED=PASS'
# Necessary one-time (not per-validation) check of released XR-1 base model.
"$PY" -B - <<'PYTHON'
import hashlib,json
from pathlib import Path
config=Path('/hpcwork/zk443626/robotics_thesis/runs/xr1-fresh-9b04eb66889347cdb121b9be52b8f7d6/project_xr1_doosan/xr1_doosan_b24_experimental_v1/resolved_compact_config.json')
pre=Path(json.loads(config.read_text())['model']['params']['pretrained'])
h=hashlib.sha256()
with pre.open('rb') as f:
 for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
if h.hexdigest()!='94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7':
 raise RuntimeError('Released pretrained XR-1 identity mismatch')
print('RELEASED_XR1_PRETRAINED_SHA256=PASS',flush=True)
PYTHON

for profile in natural_adamw_l1 balanced_adamw_l1 balanced_muon_l1 balanced_muon_l2; do
  echo "TRAIN_START=$profile"
  TRAIN_LOG="$ROOT/logs/${profile}_train.log"
  if ! "$PY" -m torch.distributed.run --standalone --nproc_per_node=1 "$CAMPAIGN" \
      --mode train --profile "$profile" --steps 5000 > "$TRAIN_LOG" 2>&1; then
    tail -n 50 "$TRAIN_LOG"
    echo "TRAIN_FAILED=$profile"
    exit 1
  fi
  grep 'MOTION_TRAIN_COMPLETED=' "$TRAIN_LOG" | tail -n 1
  echo "TRAIN_END=$profile"
  for step in 1000 3000 5000; do
    echo "VALIDATE_START=$profile step=$step"
    VAL_LOG="$ROOT/logs/${profile}_validation_${step}.log"
    if ! "$PY" -B "$CAMPAIGN" --mode evaluate --profile "$profile" \
        --steps "$step" > "$VAL_LOG" 2>&1; then
      tail -n 50 "$VAL_LOG"
      echo "VALIDATION_FAILED=$profile step=$step"
      exit 1
    fi
    grep 'MOTION_VALIDATION_COMPLETED=' "$VAL_LOG" | tail -n 1
  done
  echo "PROFILE_COMPLETE=$profile"
done
"$PY" -B "$FINAL" "$ROOT"
echo 'FRESH_FOUR_ARM_CAMPAIGN=PASS'
