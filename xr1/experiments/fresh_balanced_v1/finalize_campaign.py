"""Verify all checkpoints and validation receipts; write portable light evidence."""
from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

root=Path(sys.argv[1]).resolve(strict=True)
contract=json.loads((root/'run_contract.json').read_text())
profiles=contract['profiles']
if len(profiles)!=4 or contract['steps_per_arm']!=5000:
    raise RuntimeError('Four-arm contract missing')
result={'schema':'xr1-fresh-balanced-v2-result','completed':True,'profiles':{},
        'training_step_count':20000,'validation_type':'offline 34 episodes / fixed 136 windows + release 34 at 1000,3000,5000',
        'scientific_warning':'Offline action metrics only. This is not physical insertion success.'}
for p in profiles:
    name=p['name']; folder=root/'trials'/name
    rec=json.loads((folder/'checkpoint-step5000.json').read_text())
    val1=json.loads((folder/'validation-step1000.json').read_text())
    val3=json.loads((folder/'validation-step3000.json').read_text())
    val2=json.loads((folder/'validation-step5000.json').read_text())
    if (rec['profile'],rec['step'],rec['training_valid'])!=(name,5000,True):
        raise RuntimeError('Invalid final train receipt: '+name)
    if digest(rec['checkpoint'])!=rec['sha256']:
        raise RuntimeError('Final checkpoint changed: '+name)
    for v,step in ((val1,1000),(val3,3000),(val2,5000)):
        if (v['profile'],v['step'],v['episodes'],v['windows'])!=(name,step,34,136):
            raise RuntimeError('Wrong validation receipt: '+name)
        if v['fixed_validation_windows_sha256']!='66b5d6aa826688a4f733d32d25de3eb12338afeddb08e8f8c7738011ea85519d':
            raise RuntimeError('Unexpected validation windows: '+name)
    if val2['checkpoint_sha256']!=rec['sha256'] or val2['release_windows']!=34:
        raise RuntimeError('Missing real step5000 release validation: '+name)
    result['profiles'][name]={'model':p,'checkpoint_receipt':rec,'validation_step1000':val1,'validation_step3000':val3,'validation_step5000':val2}
    for v,step in ((val1,1000),(val3,3000),(val2,5000)):
        receipt=root/'trials'/name/f'validation-detail-step{step}.json'
        detail=json.loads(receipt.read_text())
        if (detail['profile'],detail['step'],len(detail['fixed']),len(detail['release']))!=(name,step,136,34):
            raise RuntimeError('Per-episode evaluation detail malformed: '+name+str(step))
        fixed_sum=sum(float(row['first5_motion_mae']) for row in detail['fixed'])/136
        if abs(fixed_sum-v['first5_normalized_motion_mae'])>1.e-6:
            raise RuntimeError('Per-window MAE does not reproduce checkpoint validation: '+name)
        tpos=sum(int(x['true_positive']) for x in detail['release'])
        pos=sum(int(x['positive_frames']) for x in detail['release'])
        fp=sum(int(x['false_positive']) for x in detail['release'])
        neg=sum(int(x['negative_frames']) for x in detail['release'])
        if pos<=0 or neg<=0 or abs(tpos/pos-v['release_recall'])>1e-9 or abs(fp/neg-v['release_false_open_fraction'])>1e-9:
            raise RuntimeError('Per-window gripper recall does not match summary: '+name)

    print(f'FINAL_VERIFIED_PROFILE={name} motion={val2["first5_normalized_motion_mae"]:.6f} release_recall={val2["release_recall"]:.6f}',flush=True)

# Episode-paired bootstrap, not a frame-level pseudo-replication.
# Intended for arm-to-arm screening; intervals are exploratory and unadjusted.
import numpy as np
names=[x['name'] for x in profiles]
comparisons=(
    ('balanced_vs_natural_adamw', names[1], names[0]),
    ('muon_vs_adamw_balanced', names[2], names[1]),
    ('muon_last2_vs_last1', names[3], names[2]),
)
result['paired_episode_bootstrap_95ci']={}
result['bootstrap_notes']='34 held-out episodes, 4000 fixed-seed paired resamples, 95% percentile intervals; exploratory, not corrected for multiple testing'
for step in (1000,3000,5000):
    per={n:json.loads((root/'trials'/n/f'validation-detail-step{step}.json').read_text()) for n in names}
    expected_fixed=[(x['episode'],x['frame_index']) for x in per[names[0]]['fixed']]
    expected_release=[(x['episode'],x['frame_index']) for x in per[names[0]]['release']]
    if len(set(ep for ep,_ in expected_fixed))!=34 or len(expected_release)!=34:
        raise RuntimeError('Unexpected validation episode coverage')
    for n in names[1:]:
        if [(x['episode'],x['frame_index']) for x in per[n]['fixed']]!=expected_fixed or [(x['episode'],x['frame_index']) for x in per[n]['release']]!=expected_release:
            raise RuntimeError('Per-model validation window identity changed: '+n)
    episode_ids=sorted(set(ep for ep,_ in expected_fixed))
    rng=np.random.default_rng(20261112+step)
    draws=rng.integers(0,len(episode_ids),size=(4000,len(episode_ids)))
    for label,a,b in comparisons:
        output={}
        for metric in ('first5_motion_mae','translation_mm','rotation_deg','gripper_mae'):
            def arr_for(n):
                return np.asarray([np.mean([float(x[metric]) for x in per[n]['fixed'] if x['episode']==ep]) for ep in episode_ids],dtype=np.float64)
            delta=arr_for(a)-arr_for(b)
            dist=delta[draws].mean(axis=1)
            output[metric]={'delta_first_minus_second':float(delta.mean()),'ci95':list(map(float,np.quantile(dist,[.025,.975])))}
        def release_counts(n):
            return np.array([[int(x['true_positive']),int(x['positive_frames']),int(x['false_positive']),int(x['negative_frames'])] for x in per[n]['release']],dtype=np.float64)
        pa=release_counts(a);pb=release_counts(b)
        for label_metric,num,den in (('release_recall',0,1),('release_false_open_fraction',2,3)):
            avg=pa[:,num].sum()/pa[:,den].sum()-pb[:,num].sum()/pb[:,den].sum()
            da=pa[draws].sum(axis=1);db=pb[draws].sum(axis=1)
            valid=(da[:,den]>0)&(db[:,den]>0)
            dist=da[valid,num]/da[valid,den]-db[valid,num]/db[valid,den]
            if len(dist)<3800:raise RuntimeError('Insufficient valid bootstrap samples')
            output[label_metric]={'delta_first_minus_second':float(avg),'ci95':list(map(float,np.quantile(dist,[.025,.975])))}
        result['paired_episode_bootstrap_95ci'][f'step{step}:{label}']=dict(candidate=a,reference=b,metrics=output)
print('PAIRED_34_EPISODE_BOOTSTRAP=PASS',flush=True)

resultfile=root/'final_comparison.json'
if resultfile.exists():raise RuntimeError('Refusing to replace existing final comparison')
resultfile.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
paths=[root/'run_contract.json',root/'git_commit.txt',root/'source_sha256.txt',root/'final_comparison.json',root/'schedules/schedule_report.json']
paths+=sorted((root/'logs').glob('*.log'))
for p in profiles:
    f=root/'trials'/p['name']
    paths += [f/'checkpoint-step5000.json']
    paths += [f/f'validation-step{step}.json' for step in (1000,3000,5000)]
    paths += [f/f'validation-detail-step{step}.json' for step in (1000,3000,5000)]
archive=root/'fresh_four_arm_evidence.zip'
if archive.exists():raise RuntimeError('Refusing existing archive')
manifest={}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for file in paths:
        if not file.is_file() or file.is_symlink():raise RuntimeError('Untrusted evidence input: '+str(file))
        if file.stat().st_size>20*1024**2:raise RuntimeError('Unexpected large evidence input')
        arc=file.relative_to(root).as_posix();data=file.read_bytes()
        z.writestr(arc,data);manifest[arc]=hashlib.sha256(data).hexdigest()
    z.writestr('bundle_manifest.json',json.dumps({'schema':'xr1-fresh-balanced-v1-bundle','files_sha256':manifest},sort_keys=True,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('ZIP CRC error')
    for name,expected in json.loads(z.read('bundle_manifest.json'))['files_sha256'].items():
        if hashlib.sha256(z.read(name)).hexdigest()!=expected:
            raise RuntimeError('Evidence member hash mismatch: '+name)
(root/'fresh_four_arm_evidence.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
print('FRESH_FOUR_ARM_EVIDENCE=PASS',flush=True)
print('FINAL_EVIDENCE_ZIP='+str(archive),flush=True)
