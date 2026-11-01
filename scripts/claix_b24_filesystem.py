#!/usr/bin/env python3
"""Read-only verification of approved CLAIX persistent directories and mounts."""
import argparse
import json
import os
from pathlib import Path
import subprocess

# Deliberately not /rwthfs: only this documented canonical hpcwork alias.
APPROVED_ROOTS = (Path('/hpcwork'), Path('/work'), Path('/rwthfs/rz/cluster/hpcwork'))
PERSISTENT_TYPES = {'lustre', 'nfs', 'nfs4', 'gpfs', 'beegfs'}


def approved(path):
    return any(path.is_relative_to(root) for root in APPROVED_ROOTS)


def inspect_directory(path, *, allow_missing=False, expected_source=None):
    requested = Path(path)
    if not requested.is_absolute() or not approved(Path(os.path.abspath(requested))):
        raise RuntimeError('Directory must be under an approved CLAIX persistent root')
    resolved = requested.resolve(strict=not allow_missing)
    if not approved(resolved):
        raise RuntimeError('Resolved directory escapes approved CLAIX persistent roots')
    parent = resolved
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise RuntimeError('Persistent path parent is not a directory')
    result = subprocess.run(['findmnt', '--json', '--target', str(parent),
                             '--output', 'SOURCE,TARGET,FSTYPE'],
                            capture_output=True, text=True, timeout=10, check=True)
    mounts = json.loads(result.stdout)['filesystems']
    if len(mounts) != 1:
        raise RuntimeError('Cannot identify one persistent filesystem')
    fs = mounts[0]
    mount = Path(fs['target']).resolve(strict=True)
    if (not approved(mount) or not resolved.is_relative_to(mount)
            or fs.get('fstype') not in PERSISTENT_TYPES or not fs.get('source')
            or (expected_source is not None and fs['source'] != expected_source)):
        raise RuntimeError('Unapproved persistent mount type, target or source')
    return dict(requested_path=str(requested), resolved_path=str(resolved),
                existing_parent=str(parent), filesystem=fs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path', required=True)
    args = p.parse_args()
    try:
        print(json.dumps(dict(status='PASS', **inspect_directory(args.path)), sort_keys=True))
    except Exception as error:
        print(json.dumps(dict(status='FAIL', reason=str(error)), sort_keys=True))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
