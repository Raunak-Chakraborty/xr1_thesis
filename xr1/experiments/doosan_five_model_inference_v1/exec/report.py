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
