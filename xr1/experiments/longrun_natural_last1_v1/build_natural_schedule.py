"""Construct a fresh, four-pass, train-only natural replay schedule.

Requires the approved immutable 255-episode manifest from Job 5692780.
Does not read images, val/test frames or source checkpoint weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

STEPS = 48284
BATCH = 24
SEED = 42
TRAIN_FRAMES = 289703
TRAIN_EPISODES = 255


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-manifest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    old = Path(a.source_manifest).resolve(strict=True)
    root = Path(a.out)
    if root.exists():
        raise RuntimeError('Refusing to overwrite existing schedule root: '+str(root))
    source = json.loads(old.read_text())
    rows = source.get('train_episode_records', [])
    if (len(rows), source.get('counts',{}).get('frames')) != (TRAIN_EPISODES,TRAIN_FRAMES):
        raise RuntimeError('Source episode/frame contract does not match')
    # Never sample terminal (nonexecutable) frame. Keep existing bad-label masking.
    kept=[]
    off=0
    for row in rows:
        if (int(row['global_offset']), int(row['num_frames'])) != (off, int(row['num_frames'])):
            raise RuntimeError('Discontinuous episode offsets')
        size=int(row['num_frames'])
        if size < 40:
            raise RuntimeError('Unexpected short episode')
        kept.append(np.arange(off,off+size-1,dtype=np.int32))
        off+=size
    if off != TRAIN_FRAMES:
        raise RuntimeError('Wrong total frame count')
    valid=np.concatenate(kept)
    n=STEPS*BATCH
    rng=np.random.default_rng(SEED)
    cycles=[]
    remaining=n
    while remaining:
        perm=rng.permutation(valid)
        take=min(remaining,len(perm))
        cycles.append(perm[:take]); remaining-=take
    indices=np.concatenate(cycles).astype(np.int32,copy=False)
    if len(indices)!=n or (indices<0).any() or (indices>=TRAIN_FRAMES).any():
        raise RuntimeError('Invalid schedule values')
    if np.isin(indices,[int(r['global_offset'])+int(r['num_frames'])-1 for r in rows]).any():
        raise RuntimeError('Terminal frame included')
    root.mkdir(parents=True,exist_ok=False)
    natural=root/'natural';natural.mkdir()
    schedule=natural/'sample_global_indices.npy'
    np.save(schedule,indices,allow_pickle=False)
    m=json.loads(json.dumps(source))
    m['model']['starting_checkpoint']='released_XR1_pretrained_only'
    m['model']['last_n']=1
    m['model']['new_run_global_step']=0
    m['model']['optimizer']='NEW_ADAMW'
    m['sampling'].update(steps=STEPS,samples=n,seed=SEED,batch_size=BATCH,world_size=1,
        sample_schedule_sha256=sha256(schedule))
    m['status']='DESIGN_ONLY_NOT_CONNECTED_TO_DATALOADER' # required by unchanged MotionFocusPlan
    m['test_split_accessed']=False
    (natural/'motion_focus_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    report=dict(schema='xr1-natural-48284-v1',steps=STEPS,batch_size=BATCH,
        samples=n,seed=SEED,unique_frames=int(len(np.unique(indices))),
        eligible_frames=int(len(valid)),total_train_frames=TRAIN_FRAMES,
        source_manifest_sha256=sha256(old),schedule_sha256=sha256(schedule),
        no_explicit_release_oversampling=True,validation_used=False,test_used=False,
        full_permutations=int(n//len(valid)),remainder=int(n%len(valid)))
    (root/'schedule_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('NATURAL_SCHEDULE=PASS '+json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()
