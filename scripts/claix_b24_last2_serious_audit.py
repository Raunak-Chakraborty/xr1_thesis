#!/usr/bin/env python3
"""Fail-closed CPU pre-submission audit. Never submits or starts training.

Prints an audit JSON to stdout; --output additionally creates exclusive evidence.
Paths/checksums/project accounting and CPU acceptance must be supplied, not guessed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'xr1'))
from mibot.utils.claix_checkpoint import atomic_json, file_sha256
from mibot.utils.claix_contract import audit_splits, FROZEN_TRAIN_SHA256
from mibot.utils.claix_last2_serious import (
    NAME, CONFIG, VARIANT, BATCH_SCRIPT, PRETRAINED_SHA256, LANGUAGE_SHA256,
    source_identity, validate_science, MIN_GPU_BYTES,
)
from claix_b24_filesystem import inspect_directory
from claix_b24_storage_preflight import serious_evaluate


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise RuntimeError('Invalid audit input: ' + str(path))
    return json.loads(path.read_text())


def compose(overrides, run_root):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf
    allowed = {'data.params.train_datasets.paths', 'model.params.pretrained',
               'model.params.language_policy_path', 'trainer.default_root_dir'}
    if (not isinstance(overrides, list) or len(overrides) != 4
            or any(not isinstance(x, str) for x in overrides)
            or {x.split('=', 1)[0] for x in overrides} != allowed):
        raise RuntimeError('Serious audit requires exactly four approved path overrides')
    with initialize_config_dir(config_dir=str(ROOT / 'xr1/configs'), version_base=None):
        cfg = OmegaConf.to_container(hydra_compose(config_name=CONFIG,
            overrides=overrides + ['trainer.default_root_dir=' + json.dumps(str(run_root))]), resolve=True)
    validate_science(cfg, cfg['production'])
    return cfg


def check_cpu_acceptance(path, source):
    record = read_json(path)
    required = ('hydra_composition', 'lightning_same_run_resume', 'ordered_four_worker_replay',
                'serious_checkpoint_roundtrip', 'legacy_sync_regressions', 'dataset_io')
    if (record.get('schema') != 'claix-last2-serious-cpu-acceptance-v1'
            or record.get('status') != 'PASS' or record.get('framework_stubs') is not False
            or any(record.get(k) != v for k, v in source.items())
            or record.get('batch_script_sha256') != file_sha256(ROOT / 'scripts' / BATCH_SCRIPT)
            or not record.get('lightning_version') or not record.get('hydra_version')):
        raise RuntimeError('Missing real-framework CLAIX CPU acceptance for this source/wrapper')
    from importlib.metadata import version
    for package, field in [('torch', 'torch_version'), ('lightning', 'lightning_version'),
                           ('hydra-core', 'hydra_version')]:
        if record.get(field) != version(package):
            raise RuntimeError('CPU acceptance environment version mismatch: ' + package)
    for gate in required:
        row = record.get('gates', {}).get(gate, {})
        if row.get('status') != 'PASS' or file_sha256(row['evidence']) != row.get('sha256'):
            raise RuntimeError('Unverified CLAIX CPU acceptance gate: ' + gate)
    return file_sha256(path)


def check_active_runs(current_job=None):
    """Conservative: any queued/running dedicated serious wrapper blocks another."""
    output = subprocess.check_output(['squeue', '--noheader', '--user', str(os.getuid()),
                                      '--format=%i|%j'], text=True, timeout=15)
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) != 2:
            raise RuntimeError('Cannot parse active Slurm jobs')
        if fields[1] == 'xr1_last2_serious' and fields[0] != current_job:
            raise RuntimeError('Duplicate active serious run/allocation')


def collect_inputs(request, *, allocation=False):
    source = source_identity()
    if source['source_commit'] != request['expected_commit']:
        raise RuntimeError('Unexpected serious source commit')
    for key in ('overrides', 'validation_root', 'test_root', 'split_identities', 'cpu_acceptance', 'zip_cache_root'):
        if not Path(request[key]).is_absolute():
            raise RuntimeError('Serious audit input paths must be absolute: ' + key)
    python = Path(request['python'])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError('Approved Python must be an existing absolute executable')
    if python.resolve() != Path(sys.executable).resolve():
        raise RuntimeError('Audit must execute with the selected approved Python')
    run_root = Path(request['run_root'])
    if not run_root.is_absolute() or run_root.is_symlink():
        raise RuntimeError('Serious run root must be absolute and non-symlink')
    resume = request.get('resume')
    if run_root.exists() and not resume:
        raise RuntimeError('Fresh serious run requires a never-used run root')
    if resume and not (run_root / 'experiment.json').is_file():
        raise RuntimeError('Continuation requires the same serious experiment manifest')
    overrides = read_json(request['overrides'])
    cfg = compose(overrides, run_root)
    science = validate_science(cfg, cfg['production'])
    loader = cfg['data']['params']['train_datasets']
    if len(loader['paths']) != 1:
        raise RuntimeError('Exactly one frozen training root required')
    roots = [loader['paths'][0], request['validation_root'], request['test_root']]
    inventories = audit_splits(*roots)
    identities = read_json(request['split_identities'])
    if set(identities) != {'train', 'validation', 'test'} or identities['train'] != FROZEN_TRAIN_SHA256:
        raise RuntimeError('Missing reviewed frozen split identities')
    for name, inv in zip(('train', 'validation', 'test'), inventories):
        if inv['sha256'] != identities[name]:
            raise RuntimeError('Frozen split identity mismatch: ' + name)
    for key, checksum in [('pretrained', PRETRAINED_SHA256), ('language_policy_path', LANGUAGE_SHA256)]:
        path = Path(cfg['model']['params'][key])
        if not path.is_absolute() or not path.is_file() or file_sha256(path) != checksum:
            raise RuntimeError('Original pretrained/language identity mismatch: ' + key)
        inspect_directory(path.parent)
    cache = Path(request['zip_cache_root'])
    if os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'):
        raise RuntimeError('Serious run requires immutable ZIP cache only')
    for path in [cache, *map(Path, roots)]:
        if inspect_directory(path)['filesystem']['fstype'] != 'lustre':
            raise RuntimeError('Serious frozen data/cache requires persistent Lustre')
    manifest_hash = file_sha256(cache / 'media_manifest.jsonl')
    if manifest_hash != request['zip_manifest_sha256']:
        raise RuntimeError('Frozen ZIP manifest differs from the reviewed original cache')
    acceptance = check_cpu_acceptance(request['cpu_acceptance'], source)
    check_active_runs(os.environ.get('SLURM_JOB_ID') if allocation else None)
    if resume:
        from claix_b24_launch import check_resume_directory
        target = run_root / 'project_xr1_doosan' / NAME
        raw = Path(resume)
        if raw.is_symlink():
            raise RuntimeError('Symlink serious checkpoint refused')
        with (run_root / '.production.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_resume_directory(target, raw)
    storage = serious_evaluate(run_root, request['filesystem_source'], request['project_id'],
                               stage='allocation' if allocation else 'submission')
    if storage['status'] != 'PASS':
        raise RuntimeError('Serious storage audit failed: ' + json.dumps(storage))
    inputs = dict(**source, scientific_config_sha256=science,
        train_sha256=identities['train'], split_identities=identities,
        split_roots=[str(Path(x).resolve()) for x in roots],
        pretrained_sha256=PRETRAINED_SHA256, language_sha256=LANGUAGE_SHA256,
        pretrained_path=cfg['model']['params']['pretrained'], language_path=cfg['model']['params']['language_policy_path'],
        zip_cache_root=str(cache.resolve()), zip_manifest_sha256=manifest_hash,
        python=str(python), python_sha256=file_sha256(python),
        overrides=overrides, batch_script_sha256=file_sha256(ROOT / 'scripts' / BATCH_SCRIPT),
        cpu_acceptance_sha256=acceptance, min_gpu_bytes=MIN_GPU_BYTES)
    return inputs, storage


def validate_report(path, plan, python, resume, *, allocation=True):
    report = read_json(path)
    request = report['request']
    if (report.get('schema') != 'claix-last2-serious-audit-v1' or report.get('status') != 'PASS'
            or str(Path(request['run_root']).resolve()) != plan['run_root']
            or request['python'] != python or request.get('resume') != resume
            or request['filesystem_source'] != plan['filesystem_source']
            or request['project_id'] != plan['project_id']
            or report['inputs']['overrides'] != plan['overrides']
            or str(Path(os.environ.get('XR1_DOOSAN_ZIP_CACHE_ROOT', '')).resolve()) != report['inputs']['zip_cache_root']):
        raise RuntimeError('Serious audit does not bind this invocation')
    inputs, storage = collect_inputs(request, allocation=allocation)
    if inputs != report['inputs']:
        raise RuntimeError('Serious inputs changed after audit')
    return inputs, storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'overrides', 'filesystem-source', 'expected-commit', 'python',
                 'zip-cache-root', 'zip-manifest-sha256', 'validation-root', 'test-root', 'split-identities', 'cpu-acceptance'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--project-id', type=int, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    request = vars(args).copy()
    request.pop('output')
    try:
        inputs, storage = collect_inputs(request)
        if args.resume:
            from claix_b24_launch import check_experimental_resume, read_experiment
            plan = read_experiment(args.run_root)['plan']
            if plan.get('serious_variant') != VARIANT or plan.get('serious_inputs') != inputs:
                raise RuntimeError('Serious continuation input identity changed')
            check_experimental_resume(args.run_root, args.resume, plan)
        report = dict(schema='claix-last2-serious-audit-v1', status='PASS', request=request,
                      inputs=inputs, storage=storage)
    except Exception as error:
        report = dict(schema='claix-last2-serious-audit-v1', status='BLOCKED', request=request, reason=str(error))
    if args.output:
        atomic_json(args.output, report, exclusive=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
