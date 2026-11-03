#!/usr/bin/env python3
"""CPU-only cache sealing/staging. Never changes a training environment or submits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from common import beneath, digest, read_json, require, safe_relative, sha, write_new


def seal(cache, manifest_sha256):
    cache = Path(cache).resolve(strict=True)
    manifest = beneath(cache, 'media_manifest.jsonl')
    require(sha(manifest) == manifest_sha256, 'Wrong media manifest identity')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    require(len(rows) == 646, 'Expected all 646 manifest rows')
    require(len({r['original_source_path'] for r in rows}) == 646, 'Duplicate source')
    files = {}
    for row in rows:
        rel = safe_relative(row['archive'])
        require(rel.suffix == '.zip' and str(rel).startswith('archives/'), 'Invalid archive path')
        require(str(rel) not in files, 'Duplicate archive')
        archive = beneath(cache, str(rel))
        sidecar = beneath(cache, str(rel.with_suffix('.metadata.json')))
        meta = read_json(sidecar)
        require(meta.get('schema') == 'doosan-xr1-rgb-png-zip-v1'
                and meta.get('original_source_path') == row['original_source_path']
                and meta.get('frames') == row['frames'] and row['frames'] > 0
                and meta.get('crop_bbox') == row.get('crop_bbox')
                and meta.get('archive_bytes') == archive.stat().st_size,
                'Sidecar/manifest mismatch: ' + str(rel))
        checksum = sha(archive)
        require(checksum == meta.get('archive_sha256'), 'Archive integrity mismatch: ' + str(rel))
        for path in (archive, sidecar):
            files[str(path.relative_to(cache))] = dict(bytes=path.stat().st_size,
                                                     sha256=checksum if path == archive else sha(path))
    files['media_manifest.jsonl'] = dict(bytes=manifest.stat().st_size, sha256=manifest_sha256)
    return dict(schema='xr1-cache-seal-v1', manifest_sha256=manifest_sha256,
                archives=646, files=files, total_bytes=sum(r['bytes'] for r in files.values()),
                provenance='Archive hashes checked against existing sidecars; requires independent review')


def load_index(path, expected_sha):
    require(sha(path) == expected_sha, 'Cache seal checksum mismatch')
    index = read_json(path)
    files = index['files']
    require(index.get('schema') == 'xr1-cache-seal-v1' and index.get('archives') == 646
            and len(files) == 1293 and sum(n.endswith('.zip') for n in files) == 646,
            'Incomplete cache seal')
    for name, row in files.items():
        safe_relative(name)
        require(type(row['bytes']) is int and row['bytes'] >= 0
                and len(row['sha256']) == 64
                and set(row['sha256']) <= set('0123456789abcdef'), 'Invalid cache seal entry')
        if name.endswith('.zip'):
            require(str(Path(name).with_suffix('.metadata.json')) in files, 'Missing sidecar in seal')
        else:
            require(name == 'media_manifest.jsonl' or name.endswith('.metadata.json'), 'Unexpected cache file')
    require(files['media_manifest.jsonl']['sha256'] == index['manifest_sha256']
            and index['total_bytes'] == sum(r['bytes'] for r in files.values()), 'Inconsistent cache seal')
    return index


def verify(cache, index):
    for name, row in index['files'].items():
        path = beneath(cache, name)
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Cache integrity mismatch: ' + name)
    rows = [json.loads(line) for line in beneath(cache, 'media_manifest.jsonl').read_text().splitlines()
            if line.strip()]
    expected = {'media_manifest.jsonl'}
    for row in rows:
        expected.update((row['archive'], str(Path(row['archive']).with_suffix('.metadata.json'))))
    require(len(rows) == 646 and expected == set(index['files']), 'Seal/manifest file set mismatch')


def stage(source, destination, index, *, reserve_bytes=8 * 1024**3, report_path=None, filesystem=None):
    source, destination = Path(source).resolve(strict=True), Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), 'Staging destination already exists')
    if report_path is not None:
        require(not Path(report_path).exists(), 'Persistent staging report already exists')
    parent = destination.parent.resolve(strict=True)
    require(not parent.is_relative_to(source), 'Staging cannot be inside source cache')
    free = shutil.disk_usage(parent).free
    stat = os.statvfs(parent)
    reserve = max(reserve_bytes, index['total_bytes'] // 20)
    require(free >= index['total_bytes'] + reserve, 'Insufficient actual staging space plus reserve')
    require(stat.f_favail > len(index['files']) * 3,
            'Insufficient staging inodes')
    destination.mkdir(mode=0o700)
    started = time.monotonic()
    # No reusable success marker on any partial failure; do not erase partial evidence.
    for name, row in index['files'].items():
        src = beneath(source, name)
        require(src.stat().st_size == row['bytes'], 'Source cache size changed: ' + name)
        dst = destination / safe_relative(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as reader, dst.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024**2)
            writer.flush()
            os.fsync(writer.fileno())
        dst.chmod(0o444)
    copied_at = time.monotonic()
    verify(destination, index)
    verified_at = time.monotonic()
    result = dict(schema='xr1-cache-stage-v1', cache_seal_digest=digest(index),
                  source=str(source), destination=str(destination.resolve()),
                  copied_bytes=index['total_bytes'], available_bytes_before=free,
                  reserve_bytes=reserve, seconds=time.monotonic()-started,
                  copy_fsync_seconds=copied_at-started,
                  destination_verify_seconds=verified_at-copied_at,
                  copy_bytes_per_second=index['total_bytes']/max(copied_at-started, 1e-9),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if report_path is not None:
        result.update(persistent_report=str(Path(report_path).resolve()), filesystem=filesystem)
        # Publish persistent provenance before enabling this temporary copy.
        write_new(report_path, result)
    write_new(destination / 'STAGING_READY.json', result)
    return result


def allocation_scratch(destination, *, scratch_root=None, beeond=False, allow_tmpfs=False):
    """Validate an existing allocation-owned root; never create/mount BeeOND.

    The caller/operator supplies allocation ownership. A beegfs filesystem alone
    is not proof of BeeOND; require an explicitly selected root as well.
    """
    root_value = scratch_root or os.environ.get('SLURM_TMPDIR')
    require(root_value, 'Specify --scratch-root or an allocation SLURM_TMPDIR')
    root = Path(root_value).resolve(strict=True)
    dest = Path(destination).absolute()
    require(not dest.is_symlink() and dest.parent.resolve(strict=True).is_relative_to(root),
            'Destination must be beneath the allocation scratch root')
    require(not str(root).startswith(('/hpcwork/', '/rwthfs/', '/work/', '/home/')),
            'Scratch root must not be persistent home/project storage')
    fs = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(root),
                     '--output', 'SOURCE,TARGET,FSTYPE'], text=True, timeout=10))['filesystems']
    require(len(fs) == 1, 'Ambiguous scratch mount')
    kind = fs[0]['fstype']
    if beeond:
        require(scratch_root and kind == 'beegfs', 'Explicit BeeOND requires an existing beegfs scratch root')
    else:
        require(kind in (('ext4', 'xfs', 'tmpfs') if allow_tmpfs else ('ext4', 'xfs')),
                'Ordinary scratch requires ext4/xfs; BeeOND and RAM-backed tmpfs need explicit opt-in')
    # Check the destination mount too: an interior mount may differ from root.
    target_fs = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(dest.parent),
                     '--output', 'SOURCE,TARGET,FSTYPE'], text=True, timeout=10))['filesystems']
    require(target_fs == fs, 'Destination crosses an interior scratch mount')
    return dict(**fs[0], allocation_root=str(root), beeond_explicit=beeond,
                storage_kind='beeond' if beeond else 'ram' if kind == 'tmpfs' else 'local_disk')


def check_stage(cache, index, *, persistent_report=False):
    result = read_json(Path(cache) / 'STAGING_READY.json')
    require(result.get('schema') == 'xr1-cache-stage-v1'
            and result.get('cache_seal_digest') == digest(index)
            and result.get('destination') == str(Path(cache).resolve())
            and result.get('slurm_job_id') == os.environ.get('SLURM_JOB_ID'), 'Incomplete/foreign staging')
    if persistent_report:
        from common import persistent
        require(result.get('persistent_report'), 'Missing persistent staging provenance')
        persistent(result['persistent_report'])
        require(read_json(result['persistent_report']) == result, 'Persistent staging provenance mismatch')
    verify(cache, index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('seal')
    s.add_argument('--cache', required=True)
    s.add_argument('--manifest-sha256', required=True)
    s.add_argument('--output', required=True)
    for action in ('verify', 'stage'):
        s = sub.add_parser(action)
        s.add_argument('--cache', required=True)
        s.add_argument('--index', required=True)
        s.add_argument('--index-sha256', required=True)
        if action == 'stage':
            s.add_argument('--destination', required=True)
            s.add_argument('--report', required=True, help='Persistent HPCWORK audit copy')
            s.add_argument('--scratch-root', help='Existing allocation-owned root; defaults to SLURM_TMPDIR')
            s.add_argument('--beeond', action='store_true', help='Explicit existing allocation BeeOND, never mounted by this tool')
            s.add_argument('--allow-tmpfs', action='store_true', help='Explicit RAM-backed scratch; not SSD capacity')
    args = p.parse_args()
    if args.action == 'seal':
        write_new(args.output, seal(args.cache, args.manifest_sha256))
        print('CACHE_SEAL_SHA256=' + sha(args.output))
        return
    index = load_index(args.index, args.index_sha256)
    if args.action == 'verify':
        verify(args.cache, index)
        print('CACHE_INTEGRITY=PASS')
        return
    require(os.environ.get('SLURM_JOB_ID'), 'Actual staging requires an existing allocation')
    from common import persistent
    persistent(args.report, missing=True)
    dest = Path(args.destination).resolve()
    require(not str(dest).startswith(('/hpcwork/', '/rwthfs/', '/work/', '/home/')),
            'Destination must be allocation scratch, not persistent storage')
    fs = allocation_scratch(dest, scratch_root=args.scratch_root,
                            beeond=args.beeond, allow_tmpfs=args.allow_tmpfs)
    # Record mount identity; operator must establish allocation ownership of scratch.
    result = stage(args.cache, dest, index, report_path=args.report, filesystem=fs)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
