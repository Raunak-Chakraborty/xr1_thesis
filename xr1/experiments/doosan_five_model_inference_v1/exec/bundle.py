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
