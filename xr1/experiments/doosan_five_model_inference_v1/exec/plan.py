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
