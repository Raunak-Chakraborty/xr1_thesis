#!/usr/bin/env python3
"""XR-1 long run: combined sampling + LR experiment against natural Job 5765186.

Writes a sampling plan in schedules/natural so the *unchanged* pinned
run_natural_longrun.py can be used (its code profile name stays natural_adamw_l1).
Sampling and AdamW peak LR change in this experiment. No validation/test content accessed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

STEPS = 48284
BATCH = 24
SAMPLES = STEPS * BATCH
TRAIN_FRAMES = 289703
TRAIN_EPISODES = 255
EXPECTED_NATURAL_SHA = '4b96e6eb12b77ad5bb968f0e49028902d851e848829ad61cd751f0cf2ff894b7'
BASE_SEED = 42
ROUTE_SEED = 20261117
MIX = {'natural': .70, 'translation': .10, 'rotation': .10, 'quiet': .10, 'release': 0.0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rolling_max5(values):
    v = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(v)
    for delta in range(5):
        result[:len(v)-delta] = np.maximum(result[:len(v)-delta], v[delta:])
    return result


def build(source_manifest, train, out):
    source_manifest = Path(source_manifest).resolve(strict=True)
    train = Path(train).resolve(strict=True)
    out = Path(out)
    if out.exists():
        raise FileExistsError('REFUSE_EXISTING_SCHEDULE_ROOT=' + str(out))
    meta = json.loads(source_manifest.read_text())
    rows = meta['train_episode_records']
    if (len(rows), meta['counts']['episodes'], meta['counts']['frames']) != (
        TRAIN_EPISODES, TRAIN_EPISODES, TRAIN_FRAMES
    ):
        raise RuntimeError('SOURCE_MANIFEST_COUNTS_DIFFER')
    valid_indices = []
    ends=[]
    group_of_episode=[]
    off=0
    for r in rows:
        start, n = int(r['global_offset']),int(r['num_frames'])
        if start != off or n < 40:
            raise RuntimeError('BAD_EPISODE_OFFSETS')
        valid_indices.append(np.arange(off,off+n-1,dtype=np.int32))
        group_of_episode.append('/'.join(r['relative_episode'].split('/')[:3]))
        off += n
        ends.append(off)
    if off != TRAIN_FRAMES:
        raise RuntimeError('MISSING_TRAIN_FRAMES')
    valid = np.concatenate(valid_indices)
    # Exactly reproduce prior long-run Natural schedule (seed 42, 4 permutations).
    rng = np.random.default_rng(BASE_SEED)
    streams=[]
    remaining=SAMPLES
    while remaining:
        permutation=rng.permutation(valid)
        take=min(remaining,len(permutation))
        streams.append(permutation[:take])
        remaining-=take
    natural=np.concatenate(streams).astype(np.int32,copy=False)
    if natural.ndim!=1 or len(natural)!=SAMPLES:
        raise RuntimeError('NATURAL_BASE_SHAPE_MISMATCH')
    # Check byte identity with approved long-run schedule before any modification.
    import io
    memory=io.BytesIO()
    np.save(memory,natural,allow_pickle=False)
    base_sha=hashlib.sha256(memory.getvalue()).hexdigest()
    if base_sha != EXPECTED_NATURAL_SHA:
        raise RuntimeError('PRIOR_NATURAL_BASE_SHA_MISMATCH=' + base_sha)

    ends_arr = np.array(ends,dtype=np.int64)
    names = np.asarray(group_of_episode)
    source_episode = np.searchsorted(ends_arr,natural,side='right')
    if np.any(source_episode >= len(rows)):
        raise RuntimeError('SCHEDULE_REFERENCES_BAD_EPISODES')

    pools = {role:defaultdict(list) for role in ('translation','rotation','quiet')}
    pool_lengths=Counter()
    release_windows = []
    for r in rows:
        rel = r['relative_episode']
        path = (train/rel).resolve(strict=True)
        if not path.is_relative_to(train):
            raise RuntimeError('DATA_PATH_OUTSIDE_TRAIN_ROOT')
        episode = json.loads(path.read_text())
        n = int(r['num_frames'])
        if int(episode['num_frames'])!=n:
            raise RuntimeError('TRAIN_EPISODE_CHANGED=' + rel)
        motion = np.asarray(episode['actions']['semantic'],dtype=np.float64)
        if motion.shape != (n,7) or not np.isfinite(motion).all():
            raise RuntimeError('TRAIN_ACTIONS_INVALID=' + rel)
        g=(motion[:,6]>=.5).astype(np.int8)
        rising=np.flatnonzero(np.diff(g)==1)+1
        release=int(r['release_frame'])
        if len(rising)!=1 or int(rising[0])!=release or int(g[0])!=0:
            raise RuntimeError('GRIPPER_RELEASE_SEMANTICS_CHANGED=' + rel)
        # Explicitly prohibit extra route draws near the release transition.
        # The retained natural 70% remains unmodified (including legitimate releases).
        good=np.ones(n,dtype=bool)
        good[-1]=False
        for idx in r['invalid_action_indices']:
            good[int(idx)] = False
        valid_local=np.arange(n-1)
        valid_local=valid_local[np.array([good[i:min(n,i+5)].all() for i in valid_local])]
        valid_local=valid_local[(valid_local < release-30) | (valid_local > release+30)]
        if len(valid_local) < 40:
            raise RuntimeError('MOTION_ONLY_POOL_TOO_SMALL=' + rel)
        t=rolling_max5(np.linalg.norm(motion[:,:3],axis=1))
        rot=rolling_max5(np.linalg.norm(motion[:,3:6],axis=1))
        hi_t=np.quantile(t[valid_local],.70)
        hi_r=np.quantile(rot[valid_local],.70)
        lo_t=np.quantile(t[valid_local],.30)
        lo_r=np.quantile(rot[valid_local],.30)
        off=int(r['global_offset']);group='/'.join(rel.split('/')[:3])
        picks={
            'translation':valid_local[(t[valid_local]>=hi_t)&(t[valid_local]>1e-9)],
            'rotation':valid_local[(rot[valid_local]>=hi_r)&(rot[valid_local]>1e-9)],
            'quiet':valid_local[(t[valid_local]<=lo_t)&(rot[valid_local]<=lo_r)]
        }
        for role,values in picks.items():
            if len(values):
                pools[role][group].extend((off+values).tolist())
                pool_lengths[role]+=len(values)
        release_windows.append((off+release-30,off+release+30))
    # Change only the schedule: within each original task-group, replace exactly
    # 10% with high translation, 10% with rotation and 10% with quiet.
    mixed=natural.copy()
    choice=np.random.default_rng(ROUTE_SEED)
    selected_positions=choice.permutation(SAMPLES)
    groups=names[source_episode]
    counts=Counter()
    used=0
    for role in ('translation','rotation','quiet'):
        quota=int(MIX[role]*SAMPLES)
        for j in selected_positions[used:used+quota]:
            gp=groups[j]
            pool=pools[role].get(str(gp),[])
            if not pool:
                raise RuntimeError('MISSING_SAMPLING_POOL='+role+':'+str(gp))
            mixed[j]=pool[int(choice.integers(len(pool)))]
            counts[role]+=1
        used+=quota
    counts['natural']=SAMPLES-used
    if (counts['natural'] != 811173 or counts['translation'] != 115881
        or counts['rotation'] != 115881 or counts['quiet'] != 115881):
        raise RuntimeError('ROUTE_QUOTAS_UNEXPECTED=' + str(counts))
    terminals=ends_arr-1
    if np.isin(mixed,terminals).any() or (mixed<0).any() or (mixed>=TRAIN_FRAMES).any():
        raise RuntimeError('BAD_MIXED_SAMPLE_INDICES')
    before=Counter(groups.tolist())
    after=Counter(names[np.searchsorted(ends_arr,mixed,side='right')].tolist())
    if before != after:
        raise RuntimeError('TRAIN_TASK_GROUP_DISTRIBUTION_DRIFT')
    out.mkdir(parents=True,exist_ok=False)
    dest=out/'natural';dest.mkdir()
    np.save(dest/'sample_global_indices.npy',mixed,allow_pickle=False)
    mixed_sha=sha(dest/'sample_global_indices.npy')
    newmeta=json.loads(json.dumps(meta))
    newmeta['model'].update(starting_checkpoint='released_XR1_pretrained_only',last_n=1,new_run_global_step=0,optimizer='NEW_ADAMW')
    newmeta['sampling'].update(steps=STEPS,samples=SAMPLES,seed=BASE_SEED,batch_size=BATCH,world_size=1,sample_schedule_sha256=mixed_sha)
    # Preserve existing MotionFocusPlan schema/status exactly; scientific intent is in report.
    (dest/'motion_focus_manifest.json').write_text(json.dumps(newmeta,sort_keys=True,indent=2)+'\n')
    report=dict(schema='xr1-longrun-motion-only-sampler-v1',steps=STEPS,batch_size=BATCH,samples=SAMPLES,
                base_seed=BASE_SEED,route_seed=ROUTE_SEED,mixture=MIX,route_counts=dict(counts),
                source_manifest_sha256=sha(source_manifest),
                original_natural_schedule_sha256=base_sha,motion_only_schedule_sha256=mixed_sha,
                task_group_counts=dict(after),no_explicit_release_route=True,
                extra_route_avoids_release_plus_minus_30=True,
                candidate_pool_counts=dict(pool_lengths),
                unique_frames=int(len(np.unique(mixed))),training_episodes=TRAIN_EPISODES,
                train_frames=TRAIN_FRAMES,validation_used=False,test_used=False)
    (out/'schedule_report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('MOTION_ONLY_SCHEDULE=PASS '+json.dumps(report,sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-manifest',required=True)
    ap.add_argument('--train',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    build(a.source_manifest,a.train,a.out)

if __name__=='__main__':
    main()
