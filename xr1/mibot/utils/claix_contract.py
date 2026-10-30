"""Data identity and resolved-configuration contracts without model construction."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from mibot.utils.claix_checkpoint import config_digest, file_sha256

BASELINE = '52676ccfa3b52f05de51d3d7407f33855159ac95'


def engineering_source_digest():
    root = Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for path in sorted([*root.joinpath('xr1').rglob('*.py'), *root.joinpath('xr1/configs').rglob('*.yaml'),
                        *root.joinpath('scripts').glob('claix_b24*')]):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def split_inventory(root):
    root = Path(root).resolve()
    files = sorted(root.rglob('*.json'))
    if not files:
        raise RuntimeError(f'No episode JSON in {root}')
    rows = []
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError('Episode symlink escapes split root')
        traj = json.loads(path.read_text())
        count = int(traj['num_frames'])
        if count <= 0:
            raise RuntimeError('Empty episode in frozen split')
        # Media sources detect copies of episodes across split directories.
        cameras = traj.get('videos', {})
        sources = sorted(str(info['path']) for infos in cameras.values()
                         for info in infos if isinstance(info, dict) and 'path' in info)
        rows.append(dict(episode=path.relative_to(root).as_posix(), frames=count,
                         sha256=file_sha256(path), sources=sources))
    return dict(episodes=len(rows), frames=sum(r['frames'] for r in rows), rows=rows,
                sha256=config_digest(rows))


def audit_splits(train, validation, test):
    """Inventory only, never decode/use test labels for model selection."""
    inventories = [split_inventory(p) for p in (train, validation, test)]
    if [x['episodes'] for x in inventories] != [255, 34, 34] or inventories[0]['frames'] != 289703:
        raise RuntimeError('Frozen split episode/frame counts mismatch')
    seen_hashes, seen_media = set(), set()
    for inv in inventories:
        hashes = {r['sha256'] for r in inv['rows']}
        media = {s for r in inv['rows'] for s in r['sources']}
        if seen_hashes & hashes or seen_media & media:
            raise RuntimeError('Episode/media leakage across frozen splits')
        seen_hashes |= hashes
        seen_media |= media
    return inventories


def resume_fingerprint(conf, inventory, max_length):
    data = json.loads(json.dumps(conf['data']))
    data['params']['train_datasets'].pop('resume_step', None)
    return config_digest(dict(data=data, model=conf['model'],
                              engineering_source_sha256=engineering_source_digest(),
                              optimizer=conf['trainer']['optimizer'],
                              scheduler=conf['trainer']['scheduler'],
                              precision=conf['trainer']['precision'],
                              devices=conf['trainer']['devices'],
                              accumulate_grad_batches=conf['trainer']['accumulate_grad_batches'],
                              max_length=max_length, dataset_sha256=inventory['sha256']))
