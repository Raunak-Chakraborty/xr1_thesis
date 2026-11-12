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
if len(profiles)!=4 or contract['steps_per_arm']!=2000:
    raise RuntimeError('Four-arm contract missing')
result={'schema':'xr1-fresh-balanced-v1-result','completed':True,'profiles':{},
        'training_step_count':8000,'validation_type':'offline 34 episodes / fixed 136 windows + release 34 at step2000',
        'scientific_warning':'Offline action metrics only. This is not physical insertion success.'}
for p in profiles:
    name=p['name']; folder=root/'trials'/name
    rec=json.loads((folder/'checkpoint-step2000.json').read_text())
    val1=json.loads((folder/'validation-step1000.json').read_text())
    val2=json.loads((folder/'validation-step2000.json').read_text())
    if (rec['profile'],rec['step'],rec['training_valid'])!=(name,2000,True):
        raise RuntimeError('Invalid final train receipt: '+name)
    if digest(rec['checkpoint'])!=rec['sha256']:
        raise RuntimeError('Final checkpoint changed: '+name)
    for v,step in ((val1,1000),(val2,2000)):
        if (v['profile'],v['step'],v['episodes'],v['windows'])!=(name,step,34,136):
            raise RuntimeError('Wrong validation receipt: '+name)
        if v['fixed_validation_windows_sha256']!='66b5d6aa826688a4f733d32d25de3eb12338afeddb08e8f8c7738011ea85519d':
            raise RuntimeError('Unexpected validation windows: '+name)
    if val2['checkpoint_sha256']!=rec['sha256'] or val2['release_windows']!=34:
        raise RuntimeError('Missing real step2000 release validation: '+name)
    result['profiles'][name]={'model':p,'checkpoint_receipt':rec,'validation_step1000':val1,'validation_step2000':val2}
    print(f'FINAL_VERIFIED_PROFILE={name} motion={val2["first5_normalized_motion_mae"]:.6f} release_recall={val2["release_recall"]:.6f}',flush=True)

resultfile=root/'final_comparison.json'
if resultfile.exists():raise RuntimeError('Refusing to replace existing final comparison')
resultfile.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
paths=[root/'run_contract.json',root/'git_commit.txt',root/'source_sha256.txt',root/'final_comparison.json',root/'schedules/schedule_report.json']
paths+=sorted((root/'logs').glob('*.log'))
for p in profiles:
    f=root/'trials'/p['name']
    paths += [f/'checkpoint-step2000.json',f/'validation-step1000.json',f/'validation-step2000.json']
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
