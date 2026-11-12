#!/usr/bin/env bash
# Deploy already-published branch to a NEW CLAIX worktree; submit ONE H100 job.
set -Eeuo pipefail
umask 077
BASE="$HOME/robotics_thesis"
SOURCE_WT="$BASE/worktrees/xr1_motion_muon_optuna_v1"
WT="$BASE/worktrees/xr1_fresh_balanced_v1"
BRANCH="thesis/xr1-fresh-balanced-v1"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="/hpcwork/zk443626/robotics_thesis/runs/xr1_fresh_balanced_${STAMP}"
LOGDIR="$BASE/logs/xr1_fresh_balanced_v1"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/submit_$STAMP.log"
finish(){ rc=$?;trap - EXIT;echo "FINAL_EXIT_CODE=$rc";echo "SUBMIT_LOG=$LOG";echo "JOB_ROOT=$ROOT";printf '%120s\n' '' | tr ' ' '#';exit "$rc"; }
trap finish EXIT
{
  echo '===== 1. REQUIRED CLAIX ENVIRONMENT ====='
  test -x "$BASE/envs/xr1_fp32/bin/python"
  test -d "$BASE/envs/xr1_motion_addons_v1"
  test -s "$BASE/configs/xr1_hf_offline.env"
  test -s /hpcwork/zk443626/robotics_thesis/cache/frames_zip/v1/media_manifest.jsonl
  test "$(git -C "$SOURCE_WT" rev-parse HEAD)" = '0cade1ebc9eeb28d942307a051979377ac31a224'
  test ! -e "$WT"
  command -v sbatch >/dev/null
  export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false GCM_INTERACTIVE=never
  git -C "$SOURCE_WT" fetch --no-tags origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" > "$LOGDIR/git_fetch_$STAMP.log" 2>&1 || {
     tail -n 8 "$LOGDIR/git_fetch_$STAMP.log";exit 1;
  }
  git -C "$SOURCE_WT" worktree add --detach "$WT" "refs/remotes/origin/$BRANCH" > "$LOGDIR/git_worktree_$STAMP.log" 2>&1
  test -z "$(git -C "$WT" status --porcelain)"
  COMMIT="$(git -C "$WT" rev-parse HEAD)"
  echo "GIT_SOURCE_COMMIT=$COMMIT"

  echo '===== 2. PIN SOURCE / SCHEDULES ====='
  PY="$BASE/envs/xr1_fp32/bin/python"
  SCRIPT="$WT/xr1/experiments/fresh_balanced_v1/run_four_arm.sbatch"
  CONTRACT="$WT/xr1/experiments/fresh_balanced_v1/experiment_contract.json"
  test -s "$CONTRACT"
  bash -n "$SCRIPT"
  "$PY" -B - "$WT" <<'PYTHON'
import ast,hashlib,json,sys
from pathlib import Path
w=Path(sys.argv[1]);prefix=w/'xr1/experiments/fresh_balanced_v1'
contract=json.loads((prefix/'experiment_contract.json').read_text())
assert contract['steps_per_arm']==2000 and len(contract['profiles'])==4
assert contract['starting_weights']=='released_XR1_5B_pretrained_only'
for p in w.glob('xr1/**/*.py'):
    if p.name in ('doosan_fresh_runner.py','doosan_fresh_optimizer.py','run_fresh_balanced_campaign.py','build_schedules.py','finalize_campaign.py'):
        ast.parse(p.read_text(),filename=str(p))
for name in ('natural','balanced'):
    parent=prefix/'schedules'/name
    manifest=json.loads((parent/'motion_focus_manifest.json').read_text())
    b=(parent/'sample_global_indices.npy').read_bytes()
    if hashlib.sha256(b).hexdigest()!=manifest['sampling']['sample_schedule_sha256']:
        raise RuntimeError('Sample schedule checksum drift: '+name)
    assert (manifest['sampling']['steps'],manifest['sampling']['samples'])==(2000,48000)
    assert len(manifest['train_episode_records'])==255
print('FRESH_CAMPAIGN_STATIC_PREFLIGHT=PASS')
PYTHON
  mkdir -p "$ROOT/logs" "$ROOT/slurm" "$ROOT/trials"
  printf '%s\n' "$COMMIT" > "$ROOT/git_commit.txt"
  cp -- "$CONTRACT" "$ROOT/run_contract.json"
  ln -s "$WT/xr1/experiments/fresh_balanced_v1/schedules" "$ROOT/schedules"
  test -s "$ROOT/schedules/schedule_report.json"
  # Bundle only source paths actually consumed, not unrelated history.
  (
    cd "$WT"
    sha256sum \
      xr1/mibot/models/VLA/XR1.py \
      xr1/mibot/models/runner/doosan_compact_runner.py \
      xr1/mibot/models/runner/doosan_fresh_runner.py \
      xr1/mibot/utils/doosan_motion_loss.py \
      xr1/mibot/utils/doosan_fresh_optimizer.py \
      xr1/mibot/utils/doosan_motion_focus.py \
      xr1/mibot/utils/doosan_last_n.py \
      xr1/mibot/utils/doosan_replay.py \
      xr1/mibot/data/datasets/doosan_json_dataset.py \
      xr1/mibot/data/datamodule/doosan_datamodule.py \
      xr1/tools/run_fresh_balanced_campaign.py \
      xr1/experiments/fresh_balanced_v1/{experiment_contract.json,build_schedules.py,finalize_campaign.py,run_four_arm.sbatch,schedules/natural/sample_global_indices.npy,schedules/natural/motion_focus_manifest.json,schedules/balanced/sample_global_indices.npy,schedules/balanced/motion_focus_manifest.json} \
      > "$ROOT/source_sha256.txt"
    sha256sum -c "$ROOT/source_sha256.txt" > /dev/null
  )
  echo 'IMMUTABLE_INPUTS=PASS'

  echo '===== 3. SUBMIT ONE SEQUENTIAL GPU ALLOCATION ====='
  JOB="$(sbatch --parsable --output="$ROOT/slurm/job-%j.out" --error="$ROOT/slurm/job-%j.err" "$SCRIPT" "$ROOT")"
  JOB="${JOB%%;*}"
  [[ "$JOB" =~ ^[0-9]+$ ]]
  printf '%s\n' "$JOB" > "$ROOT/job_id.txt"
  echo "SUBMITTED_JOB_ID=$JOB"
  echo "H100_WALLTIME=07:00:00"
  echo "TRAINING_PROFILES=4"
  echo "ADDITIONAL_GPU_TRAINING_STEPS=8000"
  echo "RESULT_EVIDENCE=$ROOT/fresh_four_arm_evidence.zip"
  echo 'CLAIX_FRESH_CAMPAIGN_SUBMISSION=PASS'
} 2>&1 | tee "$LOG"
