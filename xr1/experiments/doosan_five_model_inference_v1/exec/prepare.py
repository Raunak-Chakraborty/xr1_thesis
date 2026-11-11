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
