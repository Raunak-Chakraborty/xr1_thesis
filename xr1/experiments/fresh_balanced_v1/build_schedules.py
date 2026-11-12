"""Deterministic CPU-only, TRAIN-only natural/balanced sampler for XR-1.

Preserves frozen original episode order, action-mask metadata and task-group
mix. No validation/test frames and no camera decoding are accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ORIGINAL_INDEX_SHA = '249bb1367d5dc3438809b0438ba99b251630f5f73086ec0ceda15241968fc2b1'
STEPS = 2000
BATCH = 24
N = STEPS * BATCH
SEED = 20261112
MIX = {'natural': .50, 'translation': .15, 'rotation': .15, 'release': .10, 'quiet': .10}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024), b''):h.update(x)
    return h.hexdigest()


def rolling_max5(array):
    x = np.asarray(array, dtype=np.float64)
    z = np.zeros_like(x)
    for d in range(5):
        z[:len(x)-d] = np.maximum(z[:len(x)-d], x[d:])
    return z


def create(original_plan, train, out):
    original_plan = Path(original_plan).resolve(strict=True)
    train = Path(train).resolve(strict=True)
    out = Path(out)
    if out.exists():
        raise RuntimeError('Refusing existing sampling output: '+str(out))
    manifest = json.loads((original_plan/'motion_focus_manifest.json').read_text())
    rows = manifest['train_episode_records']
    if (len(rows), manifest['counts']['episodes'], manifest['counts']['frames']) != (255,255,289703):
        raise RuntimeError('Unexpected train episode inventory')
    old_indices = original_plan/'sample_global_indices.npy'
    if sha(old_indices) != ORIGINAL_INDEX_SHA or manifest['sampling']['sample_schedule_sha256'] != ORIGINAL_INDEX_SHA:
        raise RuntimeError('Original frozen training schedule changed')
    original = np.load(old_indices, allow_pickle=False)
    if original.shape != (120000,) or original.dtype.kind not in 'iu':
        raise RuntimeError('Unexpected original sample array')
    base = original[:N].astype(np.int64,copy=True)
    starts = []
    stops = []
    pools = {name:defaultdict(list) for name in MIX if name!='natural'}
    release_early=defaultdict(list)
    route_stats = Counter()
    last_end = 0
    for row in rows:
        rel = row['relative_episode']
        off, count = int(row['global_offset']),int(row['num_frames'])
        if off!=last_end or count<40:raise RuntimeError('Incorrect original episode offsets')
        last_end = off+count
        path=(train/rel).resolve(strict=True)
        if not path.is_relative_to(train):raise RuntimeError('Unexpected train path')
        traj=json.loads(path.read_text())
        if int(traj['num_frames'])!=count:raise RuntimeError('Frames changed: '+rel)
        motion=np.asarray(traj['actions']['semantic'], dtype=np.float64)
        if motion.shape!=(count,7) or not np.isfinite(motion).all():
            raise RuntimeError('Invalid semantic actions: '+rel)
        gripper=(motion[:,6]>=.5).astype(np.int8)
        rising=np.flatnonzero(np.diff(gripper)==1)+1
        if len(rising)!=1 or gripper[0]!=0 or int(rising[0])!=int(row['release_frame']):
            raise RuntimeError('Gripper release semantics changed: '+rel)
        # Global first-five normalized magnitude estimates use train-only
        # within-task 70th percentile, not a metre/radian shared threshold.
        trans=rolling_max5(np.linalg.norm(motion[:,:3],axis=1))
        rot=rolling_max5(np.linalg.norm(motion[:,3:6],axis=1))
        valid=np.arange(count-1)
        bad=np.zeros(count,dtype=bool)
        bad[-1]=True
        if len(row['invalid_action_indices']):
            bad[np.array(row['invalid_action_indices'],dtype=np.int64)] = True
        # Exclude starts whose first-five window contains an invalid motion label.
        possible=np.array([i for i in valid if not bad[i:min(i+5,count)].any()],dtype=np.int64)
        if len(possible)<50:raise RuntimeError('Too few valid first-five windows: '+rel)
        # Per-episode quantiles prevent long episodes and workpieces from
        # monopolizing high-motion candidates.
        tth=np.quantile(trans[possible],.70)
        rth=np.quantile(rot[possible],.70)
        qth_t=np.quantile(trans[possible],.30)
        qth_r=np.quantile(rot[possible],.30)
        gp='/'.join(rel.split('/')[:3])
        pools['translation'][gp].extend((off+possible[(trans[possible]>=tth)&(trans[possible]>1e-9)]).tolist())
        pools['rotation'][gp].extend((off+possible[(rot[possible]>=rth)&(rot[possible]>1e-9)]).tolist())
        q=possible[(trans[possible]<=qth_t)&(rot[possible]<=qth_r)]
        pools['quiet'][gp].extend((off+q).tolist())
        release=int(rising[0]); lo=max(0,release-29); hi=min(count-2,release-1)
        if lo>hi:raise RuntimeError('Release near episode start: '+rel)
        pools['release'][gp].extend((off+np.arange(lo,hi+1,dtype=np.int64)).tolist())
        release_early[gp].extend((off+np.arange(max(0,release-5),release,dtype=np.int64)).tolist())
        starts.append(off);stops.append(off+count)
        route_stats[gp]+=count
    if last_end!=289703:raise RuntimeError('Last train offset incorrect')
    ends=np.asarray(stops,dtype=np.int64)
    starts=np.asarray(starts,dtype=np.int64)
    if np.any(np.searchsorted(ends,base,side='right')>=len(rows)):
        raise RuntimeError('Out-of-range original sample')
    rng=np.random.default_rng(SEED)
    balanced=base.copy()
    episode_index=np.searchsorted(ends,base,side='right')
    names=np.asarray(['/'.join(row['relative_episode'].split('/')[:3]) for row in rows],dtype='U200')
    positions=rng.permutation(N)
    # Exact route quotas; natural positions retain the original sample.
    ordered=[('translation',int(.15*N)),('rotation',int(.15*N)),('release',int(.10*N)),('quiet',int(.10*N))]
    consumed=0; fallback=Counter(); role_count=Counter()
    for role,quota in ordered:
        for j in positions[consumed:consumed+quota]:
            gp=names[episode_index[j]]
            pool=pools[role].get(gp,[])
            if not pool:
                fallback[role]+=1
                continue
            idx=int(rng.integers(len(pool)))
            selected=int(pool[idx])
            if role=='release' and release_early[gp] and rng.integers(100)<65:
                epool=release_early[gp]
                selected=int(epool[int(rng.integers(len(epool)))])
            balanced[j]=selected;role_count[role]+=1
        consumed+=quota
    role_count['natural']=N-consumed
    if fallback:raise RuntimeError('Missing task-group sampler pools: '+str(fallback))
    if any(np.isin(balanced,ends-1)):raise RuntimeError('Terminal frame sampled')
    if balanced.shape != base.shape or np.any(balanced<0) or np.any(balanced>=289703):
        raise RuntimeError('Sampler shape mismatch')
    # Verify task-group preservation exactly, not merely approximately.
    before=Counter(names[np.searchsorted(ends,base,side='right')].tolist())
    after=Counter(names[np.searchsorted(ends,balanced,side='right')].tolist())
    if before!=after:raise RuntimeError('Task group sample counts changed')
    out.mkdir(parents=True, exist_ok=False)
    report={'schema':'xr1-fresh-balanced-2000-v1','steps':STEPS,'batch_size':BATCH,'samples':N,
      'seed':SEED,'source_original_sha256':ORIGINAL_INDEX_SHA,'mixture':MIX,
      'route_counts':dict(role_count),'task_group_counts':dict(after),'training_episodes':255,
      'train_frames':289703,'validation_used_for_sampling':False,'test_accessed':False,
      'semantic_action_columns':'translation 0:3; rotation 3:6; gripper 6',
      'candidate_metric':'first-five rolling norm with per-episode 70th/30th quantiles'}
    for plan_name,array in [('natural',base),('balanced',balanced)]:
        path=out/plan_name
        path.mkdir()
        np.save(path/'sample_global_indices.npy',array,allow_pickle=False)
        m=json.loads(json.dumps(manifest))
        m['sampling']['steps']=STEPS
        m['sampling']['samples']=N
        m['sampling']['sample_schedule_sha256']=sha(path/'sample_global_indices.npy')
        m['model']['last_n']=1  # model depth independently selected by runner
        (path/'motion_focus_manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
        report[plan_name+'_sha256']=m['sampling']['sample_schedule_sha256']
    (out/'schedule_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('FRESH_TRAIN_SCHEDULES=PASS samples='+str(N),flush=True)
    print('BALANCED_TASK_GROUP_COUNTS_IDENTICAL=PASS',flush=True)
    print('ORIGINAL_SCHEDULE_IMMUTABLE=PASS',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-plan',required=True)
    p.add_argument('--train',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    print(json.dumps(create(a.original_plan,a.train,a.out),indent=2,sort_keys=True))
