#!/usr/bin/env bash
#SBATCH --job-name=xr1_nat_l1_48k
#SBATCH --account=thes2452
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=23:45:00
#SBATCH --signal=B:USR1@2700
set -Eeuo pipefail
umask 077

ROOT="$1"
MODE="${2:-fresh}"
RESUME="${3:-}"
BASE="$HOME/robotics_thesis"
WT="$BASE/worktrees/xr1_longrun_natural_l1_v1"
PY="$BASE/envs/xr1_fp32/bin/python"
CAMPAIGN="$WT/xr1/tools/run_natural_longrun.py"

finish() {
  rc=$?
  trap - EXIT
  echo "JOB_EXIT_CODE=$rc"
  echo "RUN_ROOT=$ROOT"
  printf '%120s\n' '' | tr ' ' '#'
  exit "$rc"
}
trap finish EXIT
# Slurm delivers USR1 to the batch shell 45 minutes before walltime.
# The training callback polls the marker at optimizer-step boundaries.
trap 'touch "$ROOT/STOP_REQUESTED"; echo "SLURM_STOP_SIGNAL=RECEIVED"' USR1

module load Python/3.11.5
: "${XR1_SOURCE_COMMIT:?Pin the published GitHub commit using sbatch --export}"
export XR1_FRESH_RUN_ROOT="$ROOT"
export PYTHONPATH="$BASE/envs/xr1_motion_addons_v1:$WT/xr1"
source "$BASE/configs/xr1_hf_offline.env"
export XR1_DOOSAN_ZIP_CACHE_ROOT=/hpcwork/zk443626/robotics_thesis/cache/frames_zip/v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_LENGTH=12288
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ "$MODE" != fresh && "$MODE" != resume ]]; then
  echo 'ERROR=INVALID_TRAIN_MODE'; exit 1
fi
if [[ "$MODE" == fresh && -n "$RESUME" ]]; then
  echo 'ERROR=FRESH_MODE_WITH_RESUME'; exit 1
fi
if [[ "$MODE" == resume && -z "$RESUME" ]]; then
  echo 'ERROR=MISSING_RESUME_CKPT'; exit 1
fi

test "$(git -C "$WT" rev-parse HEAD)" = "$XR1_SOURCE_COMMIT"
(cd "$WT" && sha256sum -c "$ROOT/source_sha256.txt") > "$ROOT/logs/source_integrity.log"
echo 'NEW_LONGRUN_SOURCE_SHA256=PASS'

"$PY" -B - <<'PYTHON'
import hashlib,json
from pathlib import Path
config=Path('/hpcwork/zk443626/robotics_thesis/runs/xr1-fresh-9b04eb66889347cdb121b9be52b8f7d6/project_xr1_doosan/xr1_doosan_b24_experimental_v1/resolved_compact_config.json')
pre=Path(json.loads(config.read_text())['model']['params']['pretrained'])
h=hashlib.sha256()
with pre.open('rb') as f:
 for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
if h.hexdigest()!='94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7':
 raise RuntimeError('Released pretrained model identity mismatch')
print('RELEASED_XR1_PRETRAINED_SHA256=PASS',flush=True)
PYTHON

mkdir -p "$ROOT/logs"

CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node=1
    "$CAMPAIGN" --mode "$MODE")
if [[ "$MODE" == resume ]]; then
  CMD+=(--resume-ckpt "$RESUME")
fi

# Background only so that the batch shell can catch Slurm's USR1.
"${CMD[@]}" > "$ROOT/logs/training.log" 2>&1 &
CHILD=$!
while :; do
  set +e
  wait "$CHILD"
  rc=$?
  set -e
  if kill -0 "$CHILD" 2>/dev/null; then
    # Waiting was interrupted by a signal. The child is still training.
    continue
  fi
  break
done
if (( rc != 0 )); then
  tail -n 70 "$ROOT/logs/training.log"
  echo "TRAIN_EXIT_CODE=$rc"
  exit "$rc"
fi

grep -E 'LONGRUN_FINAL_RECEIPT=' "$ROOT/logs/training.log" | tail -n 1
# Train only: no inference/held-out evaluation on CLAIX.
echo 'CLAIX_INFERENCE_RUNS=0'
echo 'LONGRUN_TRAINING_JOB=PASS'
