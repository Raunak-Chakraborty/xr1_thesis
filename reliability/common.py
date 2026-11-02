"""Independent engineering tools; never imported by the production trainer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '16da563ce61e2b58952281b86c474a0490e99a52'
SOURCE = '852945709dac80152459c26ab5dab5e70e158b631ea27612d28ee2d172ef3d0b'
CHECKPOINT = 'a7476e50eee739b2acf6823125dda0f41fc25596a02a98d056eee9fe131c2c11'
CONFIG = '87ec83b0200ec9f8ab9e3cafa75079dd0d7881dcea0913c0d03ca0a1571687ea'
INVENTORY = '1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548'
INCIDENT = '9ab26a84c9ca3e4cd123bbda89d73cd1de7bcadd82ed669ca3bd0c20708b3912'
CPU_EVIDENCE = '8234090f6aaffeca718d38a9e28c7ec356850340f406e8d7a632df20142b71fa'
COMPACT_CONFIG = '3b83c90e7ccb3717cbbeecaece7d7b5602083f3093254892f6ef87fabd7fb7c5'
MEDIA_MANIFEST = 'aabf3cae952aaf36fca827bf9d85c3df8430bbfc2aa8348b2feada797ddcff25'
TARGET = Path('project_xr1_doosan/xr1_doosan_b24_results_v1')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regular(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), f'Not a regular non-symlink file: {path}')
    return path


def sha(path):
    h = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(regular(path).read_bytes())


def write_new(path, value):
    """Exclusive metadata creation; incomplete files never count as a success marker."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe_relative(value):
    path = Path(value)
    require(value and not path.is_absolute() and '..' not in path.parts
            and str(path) == value and value != '.', 'Unsafe relative path: ' + value)
    return path


def beneath(root, relative):
    root = Path(root).resolve(strict=True)
    path = root / safe_relative(relative)
    require(path.resolve(strict=True).is_relative_to(root), 'Path escapes root: ' + str(path))
    # Refuse interior symlinks as well as final symlinks.
    require(all(not p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)),
            'Symlink inside verified tree: ' + str(path))
    return regular(path)


def module(name):
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def tools_identity():
    files = sorted((ROOT / 'reliability').rglob('*.py'))
    files += [ROOT / 'docs/claix_b24' / name for name in
              ('PRODUCTION_5363645_ROOT_CAUSE.md', 'RECOVERY_RELIABILITY_HANDOFF.md')]
    rows = {str(p.relative_to(ROOT)): sha(p) for p in files}
    return dict(sha256=digest(rows), files=rows)


def source_check():
    """Keep the engineering bytes frozen; identify reliability checkout truthfully.

    Deployment is allowed only from a clean committed descendant. Separate
    lineage and reviewed tool hashes are checked by the recovery checker.
    """
    launch = module('claix_b24_launch')
    require(launch.tree_digest() == SOURCE, 'Engineering source fingerprint changed')
    checkout = launch.checkout_identity()
    require(checkout['git_status'] == '', 'Uncommitted or untracked reliability checkout')
    ancestor = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', COMMIT, checkout['git_commit']],
        cwd=ROOT, env={**os.environ, 'GIT_NO_REPLACE_OBJECTS': '1'},
        capture_output=True, check=False)
    require(ancestor.returncode == 0, 'Original engineering commit is not an ancestor')
    return launch


def persistent(path, *, missing=False):
    path = Path(path)
    roots = (Path('/hpcwork/zk443626/robotics_thesis'),
             Path('/rwthfs/rz/cluster/hpcwork/zk443626/robotics_thesis'))
    require(any(path.resolve().is_relative_to(r) for r in roots), 'Recovery artifacts require persistent thesis HPCWORK')
    return module('claix_b24_filesystem').inspect_directory(path, allow_missing=missing)
