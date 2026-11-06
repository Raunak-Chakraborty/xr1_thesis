"""Experimental identity checks before model/dataset construction; no release gates."""
from __future__ import annotations

import json
import os
from pathlib import Path

from mibot.utils.claix_checkpoint import file_sha256, load_verified
from mibot.utils.claix_contract import FROZEN_TRAIN_SHA256, engineering_source_digest

EXPERIMENT_NAME = 'xr1_doosan_b24_experimental_v1'
CHECKPOINT500_NAME = 'xr1_doosan_b24_checkpoint500_v1'


def bind_experiment(conf, production, inventory):
    if (os.environ.get('XR1_EXPERIMENTAL_UNQUALIFIED') != '1'
            or os.environ.get('XR1_OPTION_B_RESUME_BRIDGE')):
        raise RuntimeError('Experimental entrypoint requires explicit launcher opt-in; historical bridge forbidden')
    run_id = production.get('experimental_run_id', '')
    if (not isinstance(run_id, str) or len(run_id) != 32
            or any(c not in '0123456789abcdef' for c in run_id)):
        raise RuntimeError('Invalid experimental run ID')
    root = Path(conf['trainer']['default_root_dir']).resolve(strict=True)
    path = root/'experiment.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise RuntimeError('Missing/invalid experimental manifest')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'claix-b24-experiment-v1'
            or manifest.get('experimental_unqualified') is not True
            or manifest.get('run_id') != run_id
            or manifest.get('plan', {}).get('run_root') != str(root)
            or manifest.get('source_tree_sha256') != engineering_source_digest()):
        raise RuntimeError('Experimental source/run identity mismatch')
    if inventory.get('sha256') != FROZEN_TRAIN_SHA256:
        raise RuntimeError('Experimental frozen dataset identity mismatch')
    loader = conf['data']['params']['train_datasets']
    for key, expected in dict(batch_size=24, num_workers=4, prefetch_factor=2,
                              persistent_workers=True, in_order=True, worker_timeout=600,
                              zip_integrity_policy='immutable', sampler_seed=42).items():
        if loader.get(key) != expected:
            raise RuntimeError('Experimental frozen loader setting changed: ' + key)
    variant = production.get('checkpoint_variant', 'checkpoint250_v1')
    if variant not in ('checkpoint250_v1', 'checkpoint500_v1'):
        raise RuntimeError('Unreviewed experimental checkpoint variant')
    interval, name = (500, CHECKPOINT500_NAME) if variant == 'checkpoint500_v1' else (250, EXPERIMENT_NAME)
    if conf['trainer']['save_interval'] != interval or conf['trainer']['exp_name'] != name:
        raise RuntimeError('Experimental checkpoint/output configuration changed')
    if interval == 500 and (manifest.get('plan', {}).get('checkpoint_interval') != interval
                            or manifest.get('plan', {}).get('experiment_name') != name):
        raise RuntimeError('Experimental checkpoint variant/manifest mismatch')
    if interval == 250 and (manifest.get('plan', {}).get('checkpoint_interval', 250) != 250
                            or manifest.get('plan', {}).get('experiment_name', name) != name):
        raise RuntimeError('Experimental checkpoint variant/manifest mismatch')
    cache = Path(os.environ['XR1_DOOSAN_ZIP_CACHE_ROOT']).resolve(strict=True)
    identity = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                run_id=run_id, run_root=str(root),
                zip_cache_root=str(cache), zip_manifest_sha256=file_sha256(cache/'media_manifest.jsonl'))
    if interval == 500:
        identity.update(checkpoint_interval=interval, experiment_name=name)
    return identity


def load_experimental_resume(path, contract, fingerprint, experiment):
    path = Path(path)
    interval = experiment.get('checkpoint_interval', 250)
    if type(interval) is not int or interval not in (250, 500):
        raise RuntimeError('Unreviewed experimental recovery interval')
    name = CHECKPOINT500_NAME if interval == 500 else EXPERIMENT_NAME
    if interval == 500 and experiment.get('experiment_name') != name:
        raise RuntimeError('Experimental recovery variant mismatch')
    target = Path(experiment['run_root'])/'project_xr1_doosan'/name
    if path.is_symlink() or path.resolve(strict=True).parent != target.resolve(strict=True):
        raise RuntimeError('Experimental recovery accepts this run directory only')
    return load_verified(path, contract, expected_config=fingerprint, expected_experiment=experiment)
