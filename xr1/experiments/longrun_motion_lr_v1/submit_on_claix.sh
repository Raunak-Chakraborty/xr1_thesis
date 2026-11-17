#!/usr/bin/env bash
# Submit a single training-only H100 48k sampler-and-LR combined experiment. No inference.
set -Eeuo pipefail
umask 077

BASE="$HOME/robotics_thesis"
WT="$BASE/worktrees/xr1_motion_lr_adamw_l1_48k_v1"
OLD_COMMIT="596fb43f7af70e94386b83761895fcca7f6754e6"
PIN="$(git -C "$WT" rev-parse HEAD)"
OLDROOT="/hpcwork/zk443626/robotics_thesis/runs/xr1_fresh_balanced_20261112_213146"
RUNS="/hpcwork/zk443626/robotics_thesis/runs"
PY="$BASE/envs/xr1_fp32/bin/python"
CONFIG="/hpcwork/zk443626/robotics_thesis/runs/xr1-fresh-9b04eb66889347cdb121b9be52b8f7d6/project_xr1_doosan/xr1_doosan_b24_experimental_v1/resolved_compact_config.json"
POINTER="$BASE/xr1_claix/configs/motion_lr_adamw_l1_48k_v1"
BUILDER="$WT/xr1/experiments/longrun_motion_lr_v1/build_motiononly_schedule.py"
SB="$WT/xr1/experiments/longrun_motion_lr_v1/train_only.sbatch"

finish(){
  rc=$?
  trap - EXIT
  echo
  echo "FINAL_EXIT_CODE=$rc"
  printf '%120s\n' '' | tr ' ' '#'
  exit "$rc"
}
trap finish EXIT

echo '===== 1. FAIL-CLOSED SOURCE / JOB CHECK ====='
test -x "$PY" && test -s "$BUILDER" && test -s "$SB"
test -s "$CONFIG" && test -s "$OLDROOT/schedules/natural/motion_focus_manifest.json"
test -s "$BASE/configs/xr1_hf_offline.env"
test -s "/hpcwork/zk443626/robotics_thesis/cache/frames_zip/v1/media_manifest.jsonl"
test ! -e "$POINTER/run_root.txt" || { echo 'ERROR=PRIOR_MOTION_ONLY_JOB_POINTER'; exit 1; }
test -z "$(git -C "$WT" status --porcelain=v1 -uall)" || { echo 'ERROR=WORKTREE_DIRTY'; exit 1; }

git -C "$WT" merge-base --is-ancestor "$OLD_COMMIT" "$PIN"
"$PY" -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read()); print("SAMPLER_SOURCE_SYNTAX=PASS")' "$BUILDER"
bash -n "$SB"
if [ -n "$(squeue -h -u "$USER" -n xr1_motionlr_l1_48k)" ]; then
  echo 'ERROR=ANOTHER_MOTION_ONLY_JOB_QUEUED'; exit 1
fi
FREE_KIB="$(df -Pk "$RUNS" | awk 'NR==2{print $4}')"
[[ "$FREE_KIB" =~ ^[0-9]+$ ]] && (( FREE_KIB >= 65*1024*1024 )) || {
  echo 'ERROR=LESS_THAN_65GIB_CHECKPOINT_STORAGE'; exit 1;
}

TRAIN="$($PY -B - "$CONFIG" <<'PY_CONFIG'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); cfg=json.loads(p.read_text()); m=cfg['model']['params'];d=cfg['data']['params']['train_datasets']
if m['compact_fp32']['last_n']!=1 or d['batch_size']!=24:raise RuntimeError('TRAIN_CONFIG_CHANGED')
if m['compact_fp32']['pretrained_sha256']!='94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7':raise RuntimeError('RELEASED_CHECKPOINT_CHANGED')
if not Path(m['pretrained']).is_file():raise RuntimeError('PRETRAINED_MISSING')
if not Path(d['paths'][0]).is_dir():raise RuntimeError('TRAIN_ROOT_MISSING')
print(d['paths'][0])
PY_CONFIG
)"

# Explicitly ensure model/optimizer/loss/replay/evaluator source is identical
# to the successful Job 5765186 source. Only new wrapper, builder, SBATCH WT path and
# this submit script differ.
for rel in \
  xr1/tools/run_natural_longrun.py \
  xr1/tools/run_fresh_balanced_campaign.py \
  xr1/mibot/models/runner/doosan_fresh_runner.py \
  xr1/mibot/utils/doosan_fresh_optimizer.py \
  xr1/mibot/utils/doosan_motion_loss.py \
  xr1/mibot/utils/doosan_replay.py \
  xr1/mibot/utils/doosan_motion_focus.py \
  xr1/mibot/data/datasets/doosan_json_dataset.py
 do
  test "$(git -C "$WT" rev-parse "HEAD:$rel")" = "$(git -C "$WT" rev-parse "$OLD_COMMIT:$rel")" || {
    echo "ERROR=UNCONTROLLED_CODE_CHANGE:$rel";exit 1;
  }
done
echo 'MODEL_OPTIMIZER_LOSS_SOURCE_IDENTICAL_TO_JOB5765186=PASS'

echo '===== 2. GENERATE FROZEN 48284-STEP MOTION-ONLY SCHEDULE ====='
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN="$RUNS/xr1_motionlr_adamw_l1_${STAMP}"
test ! -e "$RUN"
mkdir -p "$RUN/logs"
"$PY" -B "$BUILDER" \
  --source-manifest "$OLDROOT/schedules/natural/motion_focus_manifest.json" \
  --train "$TRAIN" \
  --out "$RUN/schedules"
"$PY" -B - "$RUN" <<'PY_REPORT'
import json,sys,hashlib
from pathlib import Path
r=Path(sys.argv[1]); d=json.loads((r/'schedules/schedule_report.json').read_text())
m=json.loads((r/'schedules/natural/motion_focus_manifest.json').read_text())
assert (d['steps'],d['batch_size'],d['samples'])==(48284,24,1158816)
assert d['original_natural_schedule_sha256']=='4b96e6eb12b77ad5bb968f0e49028902d851e848829ad61cd751f0cf2ff894b7'
assert d['no_explicit_release_route'] and d['validation_used'] is False and d['test_used'] is False
assert m['model']['starting_checkpoint']=='released_XR1_pretrained_only'
assert d['motion_only_schedule_sha256']==m['sampling']['sample_schedule_sha256']
assert d['route_counts']=={'natural':811173,'translation':115881,'rotation':115881,'quiet':115881}
print('MOTION_ONLY_TRAINING_CONTRACT=PASS')
PY_REPORT

echo '===== 3. PIN SOURCE, SCIENCE CONTRACT, CHECKPOINT POLICY ====='
files=(
 xr1/tools/run_fresh_balanced_campaign.py
 xr1/tools/run_natural_longrun.py
 xr1/tools/run_motion_lr_longrun.py
 xr1/mibot/models/runner/doosan_fresh_runner.py
 xr1/mibot/models/runner/doosan_compact_runner.py
 xr1/mibot/utils/doosan_motion_loss.py
 xr1/mibot/utils/doosan_motion_focus.py
 xr1/mibot/utils/doosan_fresh_optimizer.py
 xr1/mibot/utils/doosan_last_n.py
 xr1/mibot/utils/doosan_replay.py
 xr1/mibot/data/datasets/doosan_json_dataset.py
 xr1/mibot/data/datamodule/doosan_datamodule.py
 xr1/experiments/longrun_motion_lr_v1/build_motiononly_schedule.py
 xr1/experiments/longrun_motion_lr_v1/train_only.sbatch
 xr1/experiments/longrun_motion_lr_v1/submit_on_claix.sh
)
(cd "$WT" && sha256sum "${files[@]}") > "$RUN/source_sha256.txt"
(cd "$WT" && sha256sum -c "$RUN/source_sha256.txt") > "$RUN/logs/source_integrity_preflight.log"
printf '%s\n' "$PIN" > "$RUN/git_commit.txt"
"$PY" -B - "$RUN" "$PIN" "$OLD_COMMIT" <<'PY_CONTRACT'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);rep=json.loads((root/'schedules/schedule_report.json').read_text())
contract=dict(schema='xr1-longrun-motion-lr2e5-combined-v1',source_commit=sys.argv[2],
  control_commit=sys.argv[3],control_job=5765186,experiment_design='combined_sampler_and_higher_lr',starting_weights='released_XR1_pretrained_only',
  sampler_strategy='natural70_translation10_rotation10_quiet10_release0',
  sampler_sha256=rep['motion_only_schedule_sha256'],natural_control_schedule_sha256=rep['original_natural_schedule_sha256'],
  steps=48284,batch_size=24,trainable_last_n=1,optimizer='adamw_control',adamw_lr=2e-5,
  weight_decay=.01,warmup_steps=500,cosine_floor=.25,
  motion_loss={'first_5':2.0,'next_5':1.25,'remaining_20':.75},
  checkpoints_every_steps=2000,checkpoint_full_resumable=True,slurm_walltime='23:45:00',
  claix_inference=False,test_split_used=False)
(root/'ablation_contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
print('COMBINED_CONTRACT_SHA_PINNED=PASS')
PY_CONTRACT
"$PY" -B - "$RUN" "$SB" <<'PY_ZIP'
import hashlib,json,sys,zipfile
from pathlib import Path
r=Path(sys.argv[1]); sb=Path(sys.argv[2]);p=r/'XR1_MOTION_LR2E5_SUBMISSION_EVIDENCE.zip'
with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in ('ablation_contract.json','git_commit.txt','source_sha256.txt','schedules/schedule_report.json','schedules/natural/motion_focus_manifest.json'):
    z.write(r/f,f)
  z.write(sb,'source/train_only.sbatch')
with zipfile.ZipFile(p) as z:assert z.testzip() is None
h=hashlib.sha256(p.read_bytes()).hexdigest();p.with_suffix('.zip.sha256').write_text(h+'  '+p.name+'\n')
print('SUBMISSION_EVIDENCE_ZIP='+str(p))
print('SUBMISSION_EVIDENCE_SHA256='+str(p.with_suffix('.zip.sha256')))
PY_ZIP

echo '===== 4. SUBMIT ONE H100 TRAINING-ONLY JOB ====='
SUBMITTED="$(sbatch --parsable --job-name=xr1_motionlr_l1_48k \
  --export="ALL,XR1_SOURCE_COMMIT=$PIN" \
  --output="$RUN/logs/slurm-%j.out" --error="$RUN/logs/slurm-%j.err" \
  "$SB" "$RUN" fresh)"
JOB="${SUBMITTED%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'ERROR=BAD_SLURM_JOB_ID';exit 1; }
printf '%s\n' "$JOB" > "$RUN/slurm_job_id.txt"
mkdir -p "$POINTER"
printf '%s\n' "$RUN" > "$POINTER/run_root.txt"
printf '%s\n' "$JOB" > "$POINTER/job_id.txt"
printf '%s\n' "$PIN" > "$POINTER/source_commit.txt"
echo "JOB_ID=$JOB"
echo "RUN_ROOT=$RUN"
echo "GITHUB_PINNED_COMMIT=$PIN"
echo 'MODEL_LAST_N=1'
echo 'TRAINING_TARGET=48284'
echo 'CHECKPOINT_EVERY=2000'
echo 'ADAMW_PEAK_LR=2e-5'
echo 'SAMPLER=MOTION_70_10_10_10_NO_RELEASE'
echo 'CLAIX_INFERENCE=0'
echo 'COMBINED_JOB_SUBMISSION=PASS'
squeue -j "$JOB" -o '%.18i %.10T %.10M %.12l %R' || true
