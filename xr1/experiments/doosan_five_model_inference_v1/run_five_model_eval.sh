#!/usr/bin/env bash
# XR-1 validation-only five-model offline action-prediction comparison.
# Installable as a standalone script; does not train, touch ROS, or access the test split.
set -Eeuo pipefail
umask 077

BASE="/mnt/robotics_ssd/ktt_rc_robotics_thesis"
ROOT="$BASE/xr1_motion_job1_5614826"
IN="$ROOT/incoming/job1_5614826"
CK="$IN/checkpoint_transfer"
EVIDENCE="$IN/XR1_JOB1_EVIDENCE_5614826.zip"
OLD="$BASE/xr1_inference_workspace_v1"
ASSETS="$BASE/xr1_inference_assets/xr1_two_successful_inference_v1"
WORK="$OLD/restore_inputs"
VAL="$HOME/robotics_thesis/xr1_lab/datasets/doosan_xr1_force_free_language_v2/splits/val"
PY="$HOME/miniforge3/envs/xr1_inference_py311/bin/python"
OLD_SELECT="$OLD/results/motion_diagnostic_v2_20261110_175035/selection.json"
OLD_EVAL="$OLD/results/motion_diagnostic_v2_20261110_175035/evaluate.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/results/five_models_${STAMP}"
OUT="$ROOT/results/XR1_FIVE_MODEL_HELDOUT_${STAMP}.zip"
STATUS="$RUN/status.txt"
LOG="$RUN/launcher.log"
mkdir -p "$RUN/exec"

finish() {
    rc=$?
    trap - EXIT
    if [ "$rc" -eq 0 ]; then
        printf 'COMPLETE\n' > "$STATUS"
    else
        printf 'FAILED_EXIT_CODE=%s\n' "$rc" > "$STATUS"
    fi
    echo
    echo "===== FINAL BUNDLE (SUCCESS OR PARTIAL FAILURE) ====="
    if [ -s "$RUN/exec/bundle.py" ] && [ -x "$PY" ]; then
        "$PY" -B "$RUN/exec/bundle.py" "$RUN" "$EVIDENCE" "$OUT" "$STATUS" || echo 'BUNDLE_FAILED=YES'
    else
        echo 'BUNDLE_NOT_AVAILABLE_BEFORE_BOOTSTRAP'
    fi
    echo "FINAL_EXIT_CODE=$rc"
    echo "RESULT_DIRECTORY=$RUN"
    echo "AUDIT_LOG=$LOG"
    printf '%120s\n' '' | tr ' ' '#'
    exit "$rc"
}
trap finish EXIT

exec > >(tee -a "$LOG") 2>&1
printf 'XR1_FIVE_MODEL_HELDOUT_START=%s\n' "$(date -Is)"
echo "WORKSTATION_HOST=$(hostname)"
echo 'NO_TRAINING=YES; NO_ROBOT_COMMANDS=YES; HELDOUT_VALIDATION_ONLY=YES'

echo '===== 1. EXACT ASSET PREFLIGHT ====='
mountpoint -q /mnt/robotics_ssd || { echo 'ERROR=Robotics SSD is not mounted'; exit 1; }
for f in \
    "$PY" \
    "$EVIDENCE" \
    "$CK/transfer_manifest.json" \
    "$CK/SHA256SUMS.txt" \
    "$CK/config/resolved_compact_config.json" \
    "$CK/config/sample_global_indices.npy" \
    "$CK/parent/step-048129.ckpt" \
    "$CK/profiles/adamw_control/motion-000500.ckpt" \
    "$CK/profiles/muon_conservative/motion-000500.ckpt" \
    "$CK/profiles/muon_faster/motion-000500.ckpt" \
    "$ASSETS/shared/pretrained/model_states.pt" \
    "$ASSETS/checkpoints/last_two/step-047501.ckpt" \
    "$WORK/source_last_one/xr1_last_one_source/xr1/mibot/models/VLA/XR1.py" \
    "$WORK/source_last_two/xr1_last_two_source/xr1/mibot/models/VLA/XR1.py" \
    "$WORK/metadata/metadata/last_one/resolved_compact_config.json" \
    "$OLD_SELECT" "$OLD_EVAL"
do
    if [ ! -s "$f" ]; then
        echo "MISSING_REQUIRED_FILE=$f"
        exit 1
    fi
done
if [ ! -d "$VAL" ]; then echo "MISSING_VALIDATION_ROOT=$VAL"; exit 1; fi
if [ ! -d "$WORK/processor/processor/Qwen3-VL-4B-Instruct" ]; then echo "MISSING_PROCESSOR_DIR=$WORK/processor/processor/Qwen3-VL-4B-Instruct"; exit 1; fi

export XR1_NEW_JOB1_ROOT="$CK"
export HF_HOME="$OLD/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 MAX_LENGTH=12288
export PYTHONUNBUFFERED=1
unset XR1_DOOSAN_ZIP_CACHE_ROOT XR1_DOOSAN_OFFLINE_FRAME_ROOT

"$PY" -B - <<'PY'
import select, sys
assert getattr(select,'__file__','').endswith(('.so','.pyd')), f'Standard select module shadowed: {select}'
print('PYTHON_SELECT_SHADOW_BUG_AVOIDED=PASS',flush=True)
PY

# Recheck immutable bytes once, recording checksums; never modify source checkpoints.
(
    cd "$CK"
    sha256sum -c SHA256SUMS.txt
) | tee "$RUN/checkpoint_checksums.log"

check_sha() {
    local expected="$1" path="$2" label="$3" actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        echo "ERROR_SHA256_MISMATCH=$label expected=$expected actual=$actual"
        exit 1
    fi
    printf '%s  %s\n' "$actual" "$label" >> "$RUN/preflight_sha256.txt"
    echo "SHA256_PASS=$label"
}
check_sha 'c6760aaf7789acf6053aa8b816b0458a65024ab6450dccbd68301a169e05e91c' "$EVIDENCE" 'CLAIX_JOB1_EVIDENCE'
check_sha '94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7' "$ASSETS/shared/pretrained/model_states.pt" 'PRETRAINED_XR1'
check_sha '179a7f4c7db7827aeacdcd010f33873039dc6a4272c20d746a93d91d86a70001' "$ASSETS/checkpoints/last_two/step-047501.ckpt" 'LAST_TWO_47501'
echo 'CHECKPOINT_AND_BASE_IDENTITY=PASS'

echo
printf '===== 2. GENERATE ISOLATED EVALUATION SCRIPTS =====\n'
cat > "$RUN/exec/plan.py" <<'XR1_PLAN_PYTHON'
"""Make one frozen, validation-only 5-model evaluation schedule."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

val, old_selection, destination = map(Path, sys.argv[1:])
assert val.is_dir() and old_selection.is_file()

def atomic_json(path, payload):
    with path.open('x') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')

episodes = sorted(p for p in val.rglob('episode_*.json') if p.is_file() and p.stem == p.parent.name)
if len(episodes) != 34:
    raise RuntimeError(f'Expected 34 held-out validation episodes, found {len(episodes)}')

old = json.loads(old_selection.read_text())
if old.get('schema') != 'xr1_motion_diagnostic_v2' or old.get('split') != 'val' or len(old.get('windows', [])) != 272:
    raise RuntimeError('Frozen previous 272-window selection is missing or incompatible')
if old['validation_frame_count'] != 34652 or old['validation_episode_count'] != 34 or old['action_horizon'] != 30:
    raise RuntimeError('Previous selection identity differs')

records = []
count = 0
claix_windows = []
release = []
for ep in episodes:
    tr = json.loads(ep.read_text())
    n = int(tr['num_frames'])
    if n < 40:
        raise RuntimeError('Unexpected short episode: ' + str(ep))
    rel = ep.relative_to(val).as_posix()
    semantic = np.asarray(tr['actions']['semantic'], dtype=np.float64)
    if semantic.shape != (n, 7) or not np.isfinite(semantic).all():
        raise RuntimeError('Malformed 7D semantic actions: ' + rel)
    state = (semantic[:, 6] >= .5).astype(np.int8)
    rises = np.flatnonzero(np.diff(state) == 1) + 1
    falls = np.flatnonzero(np.diff(state) == -1)
    if len(rises) != 1 or len(falls) or int(state[0]) != 0:
        raise RuntimeError('Expected exactly one gripper opening: ' + rel)
    transition = int(rises[0]); release_start = min(max(0, transition - 10), n - 30)
    if release_start > transition or not 0 < transition - release_start < 30:
        raise RuntimeError('Bad release window: ' + rel)
    for frac in (.15, .35, .55, .75):
        index = count + int((n - 31) * frac)
        claix_windows.append((rel, index))
    release.append((rel, count + release_start, release_start, transition))
    records.append((rel, n, count))
    count += n
if count != 34652 or len(claix_windows) != 136:
    raise RuntimeError(f'Validation frames/windows differ from frozen contract: {count}/{len(claix_windows)}')
claix_sha = hashlib.sha256(json.dumps(claix_windows, sort_keys=True).encode()).hexdigest()
expected_claix_sha = '66b5d6aa826688a4f733d32d25de3eb12338afeddb08e8f8c7738011ea85519d'
if claix_sha != expected_claix_sha:
    raise RuntimeError(f'CLAIX 136-window schedule identity differs: {claix_sha} != {expected_claix_sha}')

lookup = {rel: (n, offset) for rel, n, offset in records}
combined = []
for i, item in enumerate(old['windows']):
    row = dict(item)
    rel = row['episode_relative']; idx = int(row['sample_index']); frame = int(row['frame_index'])
    if rel not in lookup or idx != lookup[rel][1] + frame:
        raise RuntimeError('Old active selection mismatches new episode/frame mapping')
    if int(row.get('repeat', -1)) not in (0, 1):
        raise RuntimeError('Frozen repeats malformed')
    row.update(suite='active_v2', seed=20261110 + i, prompt_seed=20261110 + idx,
               selection_category=row['selection_category'])
    combined.append(row)

for rel, global_index, frame, transition in release:
    for r in (0, 1):
        i = len(combined)
        combined.append(dict(suite='release', selection_category='gripper_release',
            task_group='/'.join(rel.split('/')[:2]), episode_relative=rel,
            sample_index=global_index, frame_index=frame, repeat=r,
            transition_frame=transition, seed=20261110 + i,
            prompt_seed=20261110 + global_index))

for rel, idx in claix_windows:
    frame = idx - lookup[rel][1]
    seed = int.from_bytes(hashlib.sha256(f'73000:{rel}:{frame}'.encode()).digest()[:4], 'big')
    combined.append(dict(suite='claix_fixed', selection_category='claix_fixed',
        task_group='/'.join(rel.split('/')[:2]), episode_relative=rel,
        sample_index=idx, frame_index=frame, repeat=0,
        seed=seed, prompt_seed=seed))

assert len(combined) == 272 + 68 + 136 == 476
assert len({r['episode_relative'] for r in combined}) == 34
payload = dict(schema='xr1_five_model_heldout_v1', split='val', action_horizon=30,
               validation_episode_count=34, validation_frame_count=count,
               source_active_selection_sha256=hashlib.sha256(old_selection.read_bytes()).hexdigest(),
               claix_fixed_windows_sha256=claix_sha,
               suites=dict(active_v2=272, release=68, claix_fixed=136),
               inference_protocol='instruction_only_no_target_token_leakage',
               windows=combined)
atomic_json(destination, payload)
print('SCHEDULE_CREATED=PASS', flush=True)
print('VALIDATION_EPISODES=34', flush=True)
print('CLAIX_FROZEN_WINDOW_SHA256=' + claix_sha, flush=True)
print('ACTIVE_SEED_REPEATS=272', flush=True)
print('RELEASE_SEED_REPEATS=68', flush=True)
print('CLAIX_FROZEN_SINGLE_SEEDS=136', flush=True)
print('PREDICTIONS_PER_MODEL=476', flush=True)
print('FIVE_MODEL_TOTAL_PREDICTIONS=2380', flush=True)
print('SELECTION_FILE_SHA256='+hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)
print('TEST_SPLIT_ACCESSED=NO', flush=True)
XR1_PLAN_PYTHON
cat > "$RUN/exec/prepare.py" <<'XR1_PREPARE_PYTHON'
"""Create isolated pinned inference sources and patch proven workstation evaluator."""
import ast
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

original, work, evidence, incoming, run = map(Path, sys.argv[1:])
assert original.is_file() and evidence.is_file() and incoming.is_dir() and work.is_dir()
exec_dir = run / 'exec'
exec_dir.mkdir(parents=True, exist_ok=True)

# Extract only trusted model source, loss implementation and actual previous-version evaluator.
with zipfile.ZipFile(evidence) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Job 1 source ZIP CRC failure')
    manifest = json.loads(archive.read('archive_manifest.json'))['files_sha256']
    for suffix in ('xr1/mibot/models/VLA/XR1.py', 'xr1/mibot/utils/doosan_motion_loss.py'):
        key = 'source/' + suffix
        data = archive.read(key)
        if hashlib.sha256(data).hexdigest() != manifest[key]:
            raise RuntimeError('Job1 source integrity mismatch: ' + key)
        target = run / 'source_patches' / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

old_one = work / 'source_last_one/xr1_last_one_source/xr1'
old_two = work / 'source_last_two/xr1_last_two_source/xr1'
for p in (old_one, old_two):
    if not (p / 'mibot/models/VLA/XR1.py').is_file():
        raise RuntimeError('Previous verified XR1 source missing: ' + str(p))

new_source = run / 'isolated_xr1_source'

def ignore_fat(dirpath, names):
    remove = {n for n in names if n in {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', 'logs', 'runs'}}
    for n in names:
        p = Path(dirpath) / n
        if p.is_file() and p.stat().st_size > 20*1024*1024:
            remove.add(n)
    return remove

shutil.copytree(old_one, new_source, ignore=ignore_fat, symlinks=False)
for rel in ('mibot/models/VLA/XR1.py', 'mibot/utils/doosan_motion_loss.py'):
    data = (run / 'source_patches' / 'xr1' / rel).read_bytes()
    dest = new_source / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

# Prove inference-time functions are unchanged between last-one and experimental source.
old_ast = ast.parse((old_one / 'mibot/models/VLA/XR1.py').read_text())
new_ast = ast.parse((new_source / 'mibot/models/VLA/XR1.py').read_text())
def funcs(tree):
    cl = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'xr1')
    return {x.name: ast.dump(x, include_attributes=False) for x in cl.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
a, b = funcs(old_ast), funcs(new_ast)
for name in ('_build_model', '_unpad', '_generate', 'generate', 'dit_forward'):
    if a.get(name) != b.get(name):
        raise RuntimeError('Experimental inference architecture/path differs from last-one: '+name)
print('CANDIDATE_INFERENCE_ARCHITECTURE_MATCH=PASS',flush=True)

s = original.read_text()

def replace_once(old, new, label):
    global s
    c = s.count(old)
    if c != 1:
        raise RuntimeError(f'Cannot safely patch {label}: found {c} occurrences')
    s = s.replace(old, new, 1)

replace_once(
    'outroot = Path(outroot)\n',
    'outroot = Path(outroot)\nincoming = Path(os.environ["XR1_NEW_JOB1_ROOT"])\n',
    'incoming argument')
replace_once(
    'require(len(windows) == 272, "Expected 272 frozen seed-repeated windows")',
    'require(len(windows) == 476, "Expected 476 frozen evaluation predictions")',
    'window count')
start = s.index('details = {\n')
stop = s.index('\nrequire(policy in details, "Unknown policy")', start)
s = s[:start] + '''details = {
    "last_one": {"step": 48129, "last_n": 1,
       "checkpoint": incoming / "parent/step-048129.ckpt",
       "config": work / "metadata/metadata/last_one/resolved_compact_config.json"},
    "last_two": {"step": 47501, "last_n": 2,
       "checkpoint": assets / "checkpoints/last_two/step-047501.ckpt",
       "config": None},
    "adamw_control": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/adamw_control/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
    "muon_conservative": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/muon_conservative/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
    "muon_faster": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/muon_faster/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
}
NEW_POLICIES = {"adamw_control", "muon_conservative", "muon_faster"}
''' + s[stop:]
replace_once(
    'require(model_cfg.pop("type") == "xr1", "Wrong architecture")\n',
    'require(model_cfg.pop("type") == "xr1", "Wrong architecture")\n'
    'if policy in NEW_POLICIES:\n'
    '    model_cfg["motion_focus_loss"] = {"first_5": 2.0, "next_5": 1.25, "remaining_20": 0.75}\n',
    'motion config')

# Fresh generation must exclude teacher-forced action token targets and use zero-prefix policy.
block = '''    sample = dataset[sample_index]
    batch = collator([sample])
'''
new = '''    sample = dataset[sample_index]
    messages = sample["messages"][:2]
    require(len(messages) == 2 and messages[1]["role"] == "assistant",
            "Expected clean inference prompt")
    for message in messages:
        for part in message["content"]:
            if part["type"] == "text":
                require(not any(token in part["text"] for token in ("<a_", "<score>", "<state>")),
                        "Teacher-forced action-token leakage in inference")
    batch = dict(collator.processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True,
        return_tensors="pt", padding=True,
        images_kwargs={"do_resize": False},
    ))
    batch["state"] = sample["state"].unsqueeze(0)
    batch["action"] = torch.zeros_like(sample["action"]).unsqueeze(0)
    batch["action_mask"] = torch.zeros_like(batch["action"], dtype=torch.int32)
    batch["action_mask"][..., :7] = 1
'''
replace_once(block, new, 'no-target-inference payload')
replace_once(
    'random.seed(20261110 + int(window["sample_index"]))',
    'random.seed(int(window["prompt_seed"]))',
    'stable prompt seeding')
replace_once(
    '    seed = 20261110 + index\n',
    '    seed = int(window["seed"])\n',
    'generation seed')

# Keep the exact previously tested FP32 compact partial restore. Extend only metadata validation.
start = s.index('meta = checkpoint["xr1_compact_fp32"]\n')
stop = s.index('\nexpected_full = {"model." + k for k in selected}\n', start)
s = s[:start] + '''state = checkpoint["state_dict"]
require(checkpoint["global_step"] == info["step"], "Checkpoint step mismatch")
expected_base_sha = "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7"
expected_lang_sha = "21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d"

if policy in NEW_POLICIES:
    meta = checkpoint["xr1_motion_compact"]
    require(meta["schema"] == "xr1-motion-compact-lightning-v1", "Wrong experimental schema")
    require(meta["optimizer_kind"] == ("adamw_control" if policy == "adamw_control" else "hybrid_muon"), "Optimizer provenance mismatch")
    require(meta["parent"]["parent_step"] == 48129, "Parent step mismatch")
    require(meta["parent"]["pretrained_sha256"] == expected_base_sha, "Pretrained provenance mismatch")
    require(meta["parent"]["language_sha256"] == expected_lang_sha, "Language provenance mismatch")
    require(meta["parent"]["selected_tensor_count"] == 45, "Parent tensor count mismatch")
    require(meta["parent"]["trainable_elements"] == 141255168, "Parent element count mismatch")
    require(checkpoint["xr1_motion_replay"]["global_step"] == 500, "Replay cursor mismatch")
    require(bool(checkpoint.get("optimizer_states")) and bool(checkpoint.get("lr_schedulers")), "Missing resumable checkpoint optimizer/scheduler")
else:
    meta = checkpoint["xr1_compact_fp32"]
    require(meta["last_n"] == info["last_n"], "Checkpoint last-N mismatch")
    require(meta["pretrained_sha256"] == expected_base_sha, "Baseline provenance mismatch")
    require(meta["language_sha256"] == expected_lang_sha, "Language provenance mismatch")
''' + s[stop:]
replace_once('    key_hash == meta["selected_key_sha256"],',
             '    key_hash == (meta["selected_keys_sha256"] if policy in NEW_POLICIES else meta["selected_key_sha256"]),',
             'selected key sha')
replace_once('    == meta["trainable_parameter_count"],',
             '    == (meta["parent"]["trainable_elements"] if policy in NEW_POLICIES else meta["trainable_parameter_count"]),',
             'selected element count')
replace_once('for name in selected:\n    parameters[name].data',
             '''if policy in NEW_POLICIES:
    require(meta["parent"]["selected_key_sha256"] == key_hash, "Parent trainable-key mismatch")
    require(len(selected) == 45, "New candidate tensor count mismatch")
for name in selected:
    parameters[name].data''',
             'parent selection sha')
replace_once('        "category": window["selection_category"],',
             '        "category": window["selection_category"],\n        "suite": window["suite"],\n        "repeat": int(window["repeat"]),',
             'suite in metrics')
replace_once('        "latency_seconds": elapsed,',
             '''        "latency_seconds": elapsed,
        "first5_normalized_motion_mae": float(np.abs(pnorm[:5,:6] - gnorm[:5,:6]).mean()),
        "first5_translation_error_mm": float(np.linalg.norm(delta[:min(5,len(delta)),:3], axis=1).mean()*1000),
        "first5_rotation_error_deg": float(np.linalg.norm(delta[:min(5,len(delta)),3:6], axis=1).mean()*180/np.pi),
        "net_translation_error_mm": float(np.linalg.norm(delta[:,:3].sum(axis=0))*1000),
        "predicted_translation_mean_mm": float(np.linalg.norm(pred[valid,:3],axis=1).mean()*1000),
        "target_translation_mean_mm": float(np.linalg.norm(truth[valid,:3],axis=1).mean()*1000),
        "predicted_rotation_mean_deg": float(np.linalg.norm(pred[valid,3:6],axis=1).mean()*180/np.pi),
        "target_rotation_mean_deg": float(np.linalg.norm(truth[valid,3:6],axis=1).mean()*180/np.pi),
        "target_gripper_open_fraction": float(np.mean(truth[valid,6] >= .5)),
        "predicted_gripper_open_fraction": float(np.mean(pred[valid,6] >= .5)),''',
             'extra metrics')
replace_once('print(f"MODEL_{policy.upper()}_EVALUATION=PASS", flush=True)',
             'print(f"MODEL_{policy.upper()}_EVALUATION=PASS", flush=True)\nprint("INFERENCE_PROMPT_TARGET_LEAKAGE=NO", flush=True)',
             'inference proof')

out = exec_dir / 'infer_one.py'
out.write_text(s)
ast.parse(s, filename=str(out))
print('EVALUATOR_GENERATED=PASS', flush=True)
print('EVALUATOR_ORIGINAL_SHA256=' + hashlib.sha256(original.read_bytes()).hexdigest(), flush=True)
print('EVALUATOR_NEW_SHA256=' + hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
print('ISOLATED_SOURCE=' + str(new_source), flush=True)
XR1_PREPARE_PYTHON
cat > "$RUN/exec/report.py" <<'XR1_REPORT_PYTHON'
"""Paired held-out metrics; 34 episodes are independent bootstrap units."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run, evidence = map(Path, sys.argv[1:])
policies = ('last_one', 'last_two', 'adamw_control', 'muon_conservative', 'muon_faster')
selection = json.loads((run / 'selection.json').read_text())
windows = selection['windows']
if len(windows) != 476 or selection['split'] != 'val':
    raise RuntimeError('Invalid frozen validation-only evaluation selection')
summary = {}
rows = []
per_model = {}

# Verify that all five models use the same image/token inputs and labels, independent of model type.
reference_inputs = {}
reference_targets = {}
for policy in policies:
    path = run / policy / 'metrics.json'
    if not path.is_file():
        raise RuntimeError(f'Model not completed: {policy}')
    metrics = json.loads(path.read_text())
    found = metrics.get('windows', [])
    if len(found) != len(windows):
        raise RuntimeError(f'Missing predictions for {policy}: {len(found)}')
    sample_rows = []
    for i, (win, metric) in enumerate(zip(windows, found, strict=True)):
        if metric['episode'] != win['episode_relative'] or metric['frame_index'] != win['frame_index']:
            raise RuntimeError(f'Frame identity mismatch {policy} {i}')
        if metric['suite'] != win['suite'] or metric['seed'] != win['seed']:
            raise RuntimeError(f'Frozen suite/seed mismatch {policy} {i}')
        digest = (metric['input_token_sha256'], metric['input_pixel_sha256'])
        if policy == policies[0]:
            reference_inputs[i] = digest
        elif reference_inputs[i] != digest:
            raise RuntimeError(f'CROSS_MODEL_INPUT_MISMATCH {policy} {i}: {reference_inputs[i]} != {digest}')
        with np.load(run / policy / f'window_{i+1:02d}.npz', allow_pickle=False) as z:
            pred = z['prediction_physical'].astype(np.float64)
            true = z['target_physical'].astype(np.float64)
            mask = z['action_mask'].astype(bool)
            if pred.shape != (30,60) or true.shape != (30,60) or mask.shape != (30,60):
                raise RuntimeError(f'Unexpected prediction/target/mask shapes: {policy} {i}')
            if int(z['sample_index']) != win['sample_index'] or int(z['seed']) != win['seed']:
                raise RuntimeError('Stored NPZ input identity mismatch')
        if policy == policies[0]:
            reference_targets[i] = (true[:,:7].copy(), mask[:,:7].copy())
        else:
            baseline_true, baseline_mask = reference_targets[i]
            if not np.allclose(true[:,:7], baseline_true, rtol=0, atol=1e-8) or not np.array_equal(mask[:,:7], baseline_mask):
                raise RuntimeError(f'CROSS_MODEL_TARGET_MISMATCH {policy} {i}')
        valid = mask[:,0]
        if not valid.any() or not valid[:5].all():
            raise RuntimeError('Insufficient valid motion labels in first five steps')
        pred = pred[valid]; true = true[valid]
        delta = pred - true
        pred_t_mm = pred[:,:3] * 1000
        true_t_mm = true[:,:3] * 1000
        norms = np.linalg.norm(true_t_mm, axis=1)
        pnorms = np.linalg.norm(pred_t_mm, axis=1)
        active = norms >= .05
        denom = np.maximum(norms * pnorms, 1e-10)
        cosine = (pred_t_mm * true_t_mm).sum(axis=1) / denom
        t_abs = np.linalg.norm(delta[:,:3],axis=1)*1000
        r_abs = np.linalg.norm(delta[:,3:6],axis=1)*180/np.pi
        release_mask = true[:,6]>=.5
        row = dict(model=policy, suite=win['suite'], role=win['selection_category'],
            episode=win['episode_relative'], task_group=win['task_group'],
            frame_index=win['frame_index'], sample_index=win['sample_index'],
            repeat=win['repeat'], seed=win['seed'],
            first5_normalized_motion_mae=float(metric['first5_normalized_motion_mae']),
            translation_error_mm=float(t_abs.mean()),
            rotation_error_deg=float(r_abs.mean()),
            gripper_error=float(np.abs(delta[:,6]).mean()),
            first5_translation_error_mm=float(t_abs[:5].mean()),
            first5_rotation_error_deg=float(r_abs[:5].mean()),
            last5_translation_error_mm=float(t_abs[-5:].mean()),
            last5_rotation_error_deg=float(r_abs[-5:].mean()),
            net_translation_error_mm=float(np.linalg.norm(delta[:,:3].sum(axis=0))*1000),
            zero_translation_baseline_mm=float(np.linalg.norm(true_t_mm,axis=1).mean()),
            mean_translation_baseline_mm=float(metric['baseline_translation_error_mm']),
            mean_rotation_baseline_deg=float(metric['baseline_rotation_error_deg']),
            predicted_translation_mm=float(pnorms.mean()),
            target_translation_mm=float(norms.mean()),
            reverse_direction_fraction=float(np.mean(cosine[active] < 0)) if active.any() else None,
            directional_cosine=float(np.mean(cosine[active])) if active.any() else None,
            gripper_open_target_fraction=float(release_mask.mean()),
            gripper_open_prediction_fraction=float((pred[:,6]>=.5).mean()),
            gripper_open_recall=float(np.mean(pred[release_mask,6]>=.5)) if release_mask.any() else None,
            latency_seconds=float(metric['latency_seconds']),
            input_token_sha256=digest[0],input_pixel_sha256=digest[1])
        if not all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
            raise RuntimeError('Nonfinite per-window metrics')
        rows.append(row);sample_rows.append(row)
    per_model[policy] = sample_rows
    print(f'MODEL_ANALYZED={policy} WINDOWS={len(sample_rows)}', flush=True)

# Repeated windows must see identical visual/text inputs even when generation seeds differ.
for policy in policies:
    for suite in ('active_v2','release'):
        sampled = [r for r in per_model[policy] if r['suite']==suite]
        keys = defaultdict(list)
        for r in sampled: keys[(r['episode'],r['frame_index'])].append(r)
        if len(keys) != (136 if suite=='active_v2' else 34):
            raise RuntimeError('Expected unique seed-paired windows absent')
        for pair in keys.values():
            if len(pair)!=2 or set(x['repeat'] for x in pair)!={0,1}:
                raise RuntimeError('Seed-repeat pair missing')
            if (pair[0]['input_pixel_sha256'],pair[0]['input_token_sha256']) != (pair[1]['input_pixel_sha256'],pair[1]['input_token_sha256']):
                raise RuntimeError(f'Changed image/text input across seed repeats ({policy}/{suite})')

measures=('first5_normalized_motion_mae','translation_error_mm','rotation_error_deg',
          'gripper_error','first5_translation_error_mm','first5_rotation_error_deg',
          'net_translation_error_mm','predicted_translation_mm','target_translation_mm')

def average(items,key):
    return float(np.mean([float(x[key]) for x in items]))

for policy in policies:
    r=per_model[policy]
    out = dict(checkpoint_name=policy, n_predictions=len(r), by_suite={}, by_role={}, by_task_group={},
               peak_latency_seconds=float(max(x['latency_seconds'] for x in r)))
    for suite in ('claix_fixed','active_v2','release'):
        rr=[x for x in r if x['suite']==suite]
        sums={key:average(rr,key) for key in measures}
        if suite == 'release':
            sums['gripper_open_recall']=float(np.mean([x['gripper_open_recall'] for x in rr if x['gripper_open_recall'] is not None]))
            sums['gripper_open_prediction_fraction']=average(rr,'gripper_open_prediction_fraction')
            sums['gripper_open_target_fraction']=average(rr,'gripper_open_target_fraction')
        if suite=='active_v2':
            q=[x for x in rr if x['role']=='quiet']
            sums['quiet_predicted_translation_mm']=average(q,'predicted_translation_mm')
            sums['translation_windows_better_than_zero_count']=sum(x['translation_error_mm']<x['zero_translation_baseline_mm'] for x in rr)
            sums['translation_windows_better_than_mean_count']=sum(x['translation_error_mm']<x['mean_translation_baseline_mm'] for x in rr)
            d=[x['directional_cosine'] for x in rr if x['directional_cosine'] is not None]
            sums['direction_cosine_mean']=float(np.mean(d)) if d else None
        out['by_suite'][suite]=dict(n_windows=len(rr),n_episodes=len({x['episode'] for x in rr}),**sums)
    for role in sorted(set(x['role'] for x in r)):
        rr=[x for x in r if x['role']==role]
        out['by_role'][role]=dict(n=len(rr),translation_error_mm=average(rr,'translation_error_mm'),rotation_error_deg=average(rr,'rotation_error_deg'),first5_normalized_motion_mae=average(rr,'first5_normalized_motion_mae'),gripper_error=average(rr,'gripper_error'))
    for group in sorted(set(x['task_group'] for x in r)):
        rr=[x for x in r if x['task_group']==group]
        out['by_task_group'][group]=dict(n=len(rr),translation_error_mm=average(rr,'translation_error_mm'),rotation_error_deg=average(rr,'rotation_error_deg'),gripper_error=average(rr,'gripper_error'))
    summary[policy]=out

# Fair optimizer comparisons: every paired analysis resamples the 34 episodes (not 476 correlated frames).
episodes = sorted({row['episode'] for row in rows})
if len(episodes)!=34:raise RuntimeError('Expected 34 independent validation episodes')
comparisons={}
rng=np.random.default_rng(20261111)
for suite in ('claix_fixed','active_v2','release'):
    comparisons[suite]={}
    for candidate in ('last_two','adamw_control','muon_conservative','muon_faster'):
        bases = ('last_one','adamw_control') if candidate not in ('last_two','adamw_control') else ('last_one',)
        for baseline in bases:
            fieldset = ('first5_normalized_motion_mae','translation_error_mm','rotation_error_deg','gripper_error')
            for metric in fieldset:
                diffs=[]
                for ep in episodes:
                    left=[x for x in per_model[candidate] if x['suite']==suite and x['episode']==ep]
                    right=[x for x in per_model[baseline] if x['suite']==suite and x['episode']==ep]
                    if len(left)!=len(right) or not left:
                        raise RuntimeError('Paired episode coverage mismatch')
                    diffs.append(average(left,metric)-average(right,metric))
                d=np.asarray(diffs)
                sampled=np.mean(d[rng.integers(0,len(episodes),(2000,len(episodes)))],axis=1)
                lo,hi=np.quantile(sampled,[.025,.975]);delta=float(d.mean())
                key=f'{candidate}_minus_{baseline}__{metric}'
                comparisons[suite][key]=dict(delta=delta,ci95=[float(lo),float(hi)],lower_is_better=True,
                                              significant_direction='candidate_better' if hi<0 else 'baseline_better' if lo>0 else 'inconclusive')

# Compare workstation checkpoint measurements with original CLAIX aggregate results.
with __import__('zipfile').ZipFile(evidence) as archive:
    claix={}
    for p in ('adamw_control','muon_conservative','muon_faster'):
        orig=json.loads(archive.read(f'campaign/trials/{p}/validation-step500.json'))
        now=summary[p]['by_suite']['claix_fixed']['first5_normalized_motion_mae']
        claix[p]=dict(claix_first5_mae=orig['first5_normalized_motion_mae'], workstation_first5_mae=now,
                      absolute_difference=float(now-orig['first5_normalized_motion_mae']),
                      comparable_scope='matching 136 held-out validation windows, seed 73000',
                      within_0_02=abs(now-orig['first5_normalized_motion_mae'])<.02)

result=dict(schema='xr1-five-model-heldout-evaluation-v1',outcome='FULL_VALIDATION_OFFLINE',
            job1_id=5614826,split='val',n_episodes=34,n_predictions_per_model=476,
            models=list(policies),suites=selection['suites'],
            fixed_validation_window_sha256=selection['claix_fixed_windows_sha256'],
            model_summaries=summary,paired_episode_bootstrap_95ci=comparisons,
            claix_reproduction_check=claix,
            automatic_winner_approved=False,
            interpretation='Offline action prediction only; held-out 34 episodes, no physical assembly success. Small aggregate differences need paired episode analysis and gripper review. No automatic continuation decision.',
            train_data_exposure='Last-one and last-two differ in trainable depth/steps; three new candidates share the same parent, data, frozen schedule and motion-focused loss.',
            test_split_accessed=False)
(run / 'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
with (run/'per_prediction_metrics.csv').open('w',newline='') as f:
    fieldnames=list(rows[0]);w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
text=['XR-1 FIVE-MODEL HELD-OUT VALIDATION','34 episodes; 5 models; 476 predictions per model (2,380 total)','',
      'CLAIX 136 windows + motion/quiet 272 repeated predictions + release 68 repeated predictions.',
      'All five models use instruction-only VLM prompts with no teacher-forced <a_...> tokens.',
      'Errors use held-out ground truth; full-horizon translation mm, rotation degrees, gripper units.','']
for p in policies:
    text.append(p.upper())
    for name in ('claix_fixed','active_v2','release'):
        d=summary[p]['by_suite'][name]
        text.append(f"  {name}: first5 normalized motion MAE={d['first5_normalized_motion_mae']:.6f}, translation mm={d['translation_error_mm']:.6f}, rotation deg={d['rotation_error_deg']:.6f}, gripper MAE={d['gripper_error']:.6f}")
        if name=='release':
            text.append(f"    release recall @ gripper>=0.5: {d['gripper_open_recall']:.3f}")
    text.append('')
text.append('PAIRED 95% EPISODE-LEVEL BOOTSTRAP (negative delta favors candidate)')
for suite in ('claix_fixed','active_v2','release'):
    for key,val in comparisons[suite].items():
        if ('muon_conservative_minus_adamw_control' in key or 'muon_faster_minus_adamw_control' in key or 'adamw_control_minus_last_one' in key) and key.endswith(('__first5_normalized_motion_mae','__translation_error_mm','__rotation_error_deg','__gripper_error')):
            text.append(f"{suite}: {key}: delta={val['delta']:.6g}, CI95={val['ci95']}, {val['significant_direction']}")
text.extend(['','CLAIX aggregate reproduction:'])
for p,v in claix.items():text.append(f"  {p}: CLAIX={v['claix_first5_mae']:.7f}, workstation={v['workstation_first5_mae']:.7f}, delta={v['absolute_difference']:+.7f}")
text.extend(['','No automatic winner approved. Review effect sizes, bootstrap intervals, gripper recall, and held-out task groups before deciding continuation.',
             'No robot commands, no test split, no model updates.'])
(run/'REPORT.txt').write_text('\n'.join(text)+'\n')
print('\n'.join(text),flush=True)
print('SAME_MODEL_INPUTS_ACROSS_FIVE=PASS',flush=True)
print('SEED_REPEAT_INPUTS_IDENTICAL=PASS',flush=True)
print('HELDOUT_STATISTICAL_REPORT=PASS',flush=True)
XR1_REPORT_PYTHON
cat > "$RUN/exec/bundle.py" <<'XR1_BUNDLE_PYTHON'
"""Create integrity-verified, small upload bundle without model weights or datasets."""
from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from pathlib import Path

run, evidence, zip_path, status = map(Path, sys.argv[1:])
if zip_path.exists():
    raise RuntimeError('Refusing to overwrite existing results ZIP')
files = {}

def add(src,arc):
    if not src.is_file() or src.is_symlink():
        return
    if src.suffix in {'.ckpt','.pt','.pth','.mp4','.png','.jpg','.npy'}:
        return
    if src.stat().st_size > 50*1024*1024:
        raise RuntimeError('Unexpectedly large evidence file '+str(src))
    if arc in files:
        raise RuntimeError('Archive duplicate member '+arc)
    files[arc] = src

for p in sorted(run.rglob('*')):
    if not p.is_file() or p.is_symlink():
        continue
    rel = p.relative_to(run)
    if rel.parts[0] == 'isolated_xr1_source':
        continue
    if any(part in ('__pycache__','.git') for part in rel.parts):
        continue
    add(p,'run/'+rel.as_posix())
add(evidence,'original_job1_evidence/XR1_JOB1_EVIDENCE_5614826.zip')
if not files: raise RuntimeError('Nothing to bundle')
manifest = {}
zip_path.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for arc, src in sorted(files.items()):
        data=src.read_bytes()
        z.writestr(arc,data)
        manifest[arc] = hashlib.sha256(data).hexdigest()
    z.writestr('bundle_manifest.json', json.dumps(dict(schema='xr1-five-model-evidence-v1',
        status=status.name, status_text=status.read_text() if status.exists() else 'UNKNOWN',
        files_sha256=manifest),indent=2,sort_keys=True)+'\n')
with zipfile.ZipFile(zip_path) as z:
    if z.testzip(): raise RuntimeError('Evidence ZIP CRC failure')
    check=json.loads(z.read('bundle_manifest.json'))['files_sha256']
    for arc,digest in check.items():
        if hashlib.sha256(z.read(arc)).hexdigest()!=digest:
            raise RuntimeError('Evidence member SHA mismatch '+arc)
digest=hashlib.sha256(zip_path.read_bytes()).hexdigest()
(zip_path.parent/(zip_path.name+'.sha256')).write_text(f'{digest}  {zip_path.name}\n')
print('BUNDLE_ZIP='+str(zip_path))
print('BUNDLE_SIZE_MIB='+str(round(zip_path.stat().st_size/2**20,2)))
print('BUNDLE_FILES='+str(len(files)))
print('BUNDLE_SHA256='+digest)
print('BUNDLE_INTERNAL_INTEGRITY=PASS')
XR1_BUNDLE_PYTHON

echo '===== 3. STATIC SOURCE AND GPU PREFLIGHT ====='
"$PY" -B "$RUN/exec/prepare.py" "$OLD_EVAL" "$WORK" "$EVIDENCE" "$CK" "$RUN"
"$PY" -B "$RUN/exec/plan.py" "$VAL" "$OLD_SELECT" "$RUN/selection.json"
"$PY" -B - "$RUN/exec" <<'PY'
import ast,sys
from pathlib import Path
folder=Path(sys.argv[1]);files=sorted(folder.glob('*.py'))
for f in files:ast.parse(f.read_text(),filename=str(f))
assert len(files)==5, files
print('FIVE_EVAL_CODE_PYTHON_SYNTAX=PASS')
PY
"$PY" -B - <<'PY'
import torch
assert torch.cuda.is_available(), 'No CUDA device'
assert torch.cuda.device_count()>=1
print('GPU='+torch.cuda.get_device_name(0))
free,total=torch.cuda.mem_get_info(0)
print(f'GPU_FREE_GIB={free/2**30:.3f}')
if free < 16*2**30:raise RuntimeError('Need >=16 GiB free on selected A5000, avoid other active processes')
print('GPU_MEMORY_PREFLIGHT=PASS')
PY
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader || true

POLICIES=(last_one last_two adamw_control muon_conservative muon_faster)
for policy in "${POLICIES[@]}"; do
    if [ "$policy" = 'last_one' ]; then
        SRC="$WORK/source_last_one/xr1_last_one_source/xr1"
    elif [ "$policy" = 'last_two' ]; then
        SRC="$WORK/source_last_two/xr1_last_two_source/xr1"
    else
        SRC="$RUN/isolated_xr1_source"
    fi
    echo
    echo "===== 4. INFERENCE $policy — 476 HELD-OUT PREDICTIONS ====="
    POLICYLOG="$RUN/${policy}.log"
    # Full low-noise event stream is saved. Print every 25/50 windows and essential milestones live.
    if PYTHONPATH="$SRC" "$PY" -u "$RUN/exec/infer_one.py" \
        "$policy" "$RUN/selection.json" "$VAL" "$ASSETS" "$WORK" "$RUN" \
        2>&1 | tee "$POLICYLOG" | awk '/MODEL_CONSTRUCTION=|ORIGINAL_BASE_RESTORED=|FINETUNED_RESTORE=|PREPARING_SAMPLES=PASS|MODEL_.*_EVALUATION=PASS|GPU_PEAK_GIB=|INFERENCE_PROMPT_TARGET_LEAKAGE=|SAMPLE_(1|50|100|150|200|250|300|350|400|450|476)=|WINDOW_(1|25|50|75|100|125|150|175|200|225|250|275|300|325|350|375|400|425|450|476)=/ {print;fflush()}'; then
        echo "MODEL_FINISHED=$policy"
    else
        echo "MODEL_FAILED=$policy"
        tail -n 60 "$POLICYLOG"
        exit 1
    fi
done

echo
printf '===== 5. PAIRED MULTI-MODEL SCIENTIFIC ANALYSIS =====\n'
"$PY" -B "$RUN/exec/report.py" "$RUN" "$EVIDENCE" | tee "$RUN/scientific_report.log"
echo 'ALL_FIVE_MODELS_EVALUATED=PASS'
echo 'CROSS_MODEL_AND_REPEAT_INPUTS_VERIFIED=PASS'
echo 'EPISODE_LEVEL_PAIRED_BOOTSTRAP=PASS'
echo 'CLAIX_136_WINDOW_REPRODUCTION_COMPUTED=PASS'
echo 'RELEASE_WINDOWS_EVALUATED=PASS'
echo 'TEST_SPLIT_ACCESSED=NO'
echo 'TRAINING_STARTED=NO'
echo 'ROBOT_COMMANDS_SENT=NO'
echo 'ORIGINAL_MODEL_WEIGHTS_UNMODIFIED=YES'
