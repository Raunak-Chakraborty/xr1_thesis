#!/usr/bin/env python3
"""Read-only, exact-input Task04C preflight. No submission capability."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import claix_b24_launch as launch
from claix_b24_filesystem import inspect_directory
from claix_b24_storage_preflight import experimental_evaluate
from claix_b24_task04c_cpu import environment, require, sha, validate_report

START = 'c3b2c6895fe2d3bfd19c3d301ef5b49a02841686'
BRANCH = 'thesis/xr1-checkpoint-benchmark-hardening-v1'
KEYS = {'schema', 'worktree', 'branch', 'commit', 'source_tree_sha256', 'python',
        'overrides', 'overrides_sha256', 'run_root', 'run_ids', 'filesystem_source',
        'project_id', 'zip_cache_root', 'zip_manifest_sha256', 'split_roots',
        'split_identities', 'cpu_report', 'cpu_report_sha256', 'wrapper_sha256',
        'log_directory', 'project_budget', 'project_budget_sha256', 'reviewer'}


def exact_path(value, *, exists=True, canonical=False):
    path = Path(value)
    require(path.is_absolute() and not path.is_symlink()
            and (not canonical or str(path.resolve(strict=exists)) == value),
            'Require exact absolute reviewed path: ' + value)
    if exists:
        require(path.exists(), 'Missing reviewed asset: ' + value)
    return path


def git(*args):
    return subprocess.check_output(['git', *args], cwd=launch.ROOT, text=True).strip()


def protected_science():
    # Complete inherited scientific recipes must stay byte-identical to Task04B.
    paths = [*sorted((launch.ROOT/'xr1/configs').rglob('*.yaml')),
             launch.ROOT/'scripts/claix_b24_checkpoint500.sbatch',
             launch.ROOT/'scripts/claix_b24_experimental.sbatch',
             launch.ROOT/'scripts/claix_b24_production.sbatch',
             launch.ROOT/'xr1/mibot/utils/claix_runtime.py',
             launch.ROOT/'xr1/mibot/utils/claix_checkpoint.py']
    for path in paths:
        name = path.relative_to(launch.ROOT).as_posix()
        require(path.read_bytes() == subprocess.check_output(['git', 'show', START+':'+name], cwd=launch.ROOT),
                'Scientific/sync contract changed: ' + name)


def environment_probe(python):
    result = subprocess.run([python, '-B', str(Path(__file__).resolve()), '--environment-only'],
                            check=True, capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), timeout=120)
    return json.loads(result.stdout)


def compose_science(review):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    raw = json.loads(exact_path(review['overrides']).read_text())
    allowed = {'data.params.train_datasets.paths', 'model.params.pretrained',
               'model.params.language_policy_path', 'trainer.default_root_dir'}
    require(isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, str) for v in raw)
            and {v.split('=', 1)[0] for v in raw} == allowed, 'Require exact four path overrides')
    with initialize_config_dir(config_dir=str(launch.ROOT/'xr1/configs'), version_base=None):
        configs = [OmegaConf.to_container(compose(config_name=launch.experiment_profile(500, leg)['config'],
                    overrides=raw + ['trainer.default_root_dir='+json.dumps(review['run_root']+'/'+name),
                                     'production.stop_after_step=1500']), resolve=True)
                   for leg, name in (('B', 'B_sync500'), ('C', 'C_async500'))]
    for cfg, leg in zip(configs, ('B', 'C')):
        loader = cfg['data']['params']['train_datasets']
        require(loader['paths'] == [review['split_roots']['train']], 'Train root mismatch')
        require(cfg['trainer']['ckpt_path'] is None and cfg['trainer']['max_steps'] == 48284
                and cfg['data']['params']['max_steps'] == 48284
                and cfg['trainer']['scheduler']['params']['num_training_steps'] == 48284
                and cfg['production']['stop_after_step'] == 1500, 'Benchmark horizon/initialization changed')
        for key, expected in dict(batch_size=24, num_workers=4, prefetch_factor=2, persistent_workers=True,
                                  in_order=True, sampler_seed=42, zip_integrity_policy='immutable',
                                  deterministic_replay=True, action_length=30, use_wrench=False).items():
            require(loader[key] == expected, 'Changed loader science: ' + key)
        require(cfg['model']['params']['last_n'] == 1 and cfg['trainer']['seed'] == 42
                and cfg['trainer']['precision'] == 'bf16-mixed', 'Last-block/seed/precision changed')
        for field, hash_field in (('pretrained', 'pretrained_sha256'), ('language_policy_path', 'language_sha256')):
            asset = exact_path(cfg['model']['params'][field])
            require(asset.is_file() and sha(asset) == cfg['model']['params'][hash_field], 'Missing/changed ' + field)
            inspect_directory(asset.parent, expected_source=review['filesystem_source'])
    # Real locally cached processor/config only. Runtime constructs the backbone
    # from config and original XR1 payload; Qwen backbone safetensors are NOT needed.
    from transformers import Qwen3VLConfig
    from mibot.data.collate.custom_collate import CustomCollate
    Qwen3VLConfig.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', local_files_only=True)
    CustomCollate()  # uses approved offline env and validates original special tokens
    return configs


def assets(review):
    from mibot.utils.claix_contract import audit_splits, FROZEN_TRAIN_SHA256
    from mibot.data.datasets.doosan_zip_images import _manifest, ZipArchiveCache
    require(set(review['split_roots']) == set(review['split_identities']) == {'train', 'validation', 'test'},
            'Missing frozen split identities')
    roots = [exact_path(review['split_roots'][s]) for s in ('train', 'validation', 'test')]
    for root in roots:
        require(inspect_directory(root, expected_source=review['filesystem_source'])['filesystem']['fstype'] == 'lustre',
                'Frozen split inventories must remain on persistent Lustre')
    inventories = audit_splits(*roots)
    require(review['split_identities']['train'] == FROZEN_TRAIN_SHA256, 'Wrong frozen training identity')
    for name, inv in zip(('train', 'validation', 'test'), inventories):
        require(inv['sha256'] == review['split_identities'][name], 'Frozen split changed: ' + name)
    cache = exact_path(review['zip_cache_root'])
    require(os.environ.get('XR1_DOOSAN_ZIP_CACHE_ROOT') == str(cache)
            and not os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'), 'Runtime cache environment mismatch')
    require(inspect_directory(cache, expected_source=review['filesystem_source'])['filesystem']['fstype'] == 'lustre',
            'Cache must remain on Lustre')
    require(sha(cache/'media_manifest.jsonl') == review['zip_manifest_sha256'], 'ZIP manifest changed')
    _manifest.cache_clear()
    records = _manifest(str(cache))  # actual runtime manifest schema, exactly 646 records
    # Verify referenced files/sidecars and ZIP directories, one handle at a time.
    # No frame decode, dataset download, raw video existence or cache rewrite.
    required_sources = {s for inv in inventories for row in inv['rows'] for s in row['sources']}
    require(required_sources <= records.keys(), 'Dataset camera source absent from ZIP manifest')
    archive_cache = ZipArchiveCache(max_open=1, integrity_policy='immutable')
    try:
        for row in records.values():
            archive = exact_path(str(cache/row['archive']))
            require(archive.resolve().is_relative_to(cache.resolve()), 'ZIP archive escapes cache')
            # Reuse runtime validation with its bounded sidecar and member checks.
            opened = archive_cache.get(cache, row)
            archive_cache.release(opened)
    finally:
        archive_cache.close()
    return [inv['sha256'] for inv in inventories]


def budget(review):
    path = exact_path(review['project_budget'])
    require(sha(path) == review['project_budget_sha256'], 'Project accounting evidence changed')
    data = json.loads(path.read_text())
    require(data['schema'] == 'task04c-reviewed-project-budget-v1' and data['account'] == 'thes2452'
            and data['reviewer'] == review['reviewer'] and bool(data['reviewer'].strip()),
            'Project budget requires named review of authoritative site accounting')
    age = time.time() - data['observed_epoch']
    require(math.isfinite(age) and 0 <= age <= 900, 'Project accounting evidence older than 15 minutes')
    require(sha(exact_path(data['raw_evidence'])) == data['raw_sha256'], 'Missing raw project accounting output')
    for key, minimum in (('cpu_hours_remaining', 12*23.75), ('gpu_hours_remaining', 23.75)):
        value = data[key]
        require(type(value) in (float, int) and math.isfinite(value) and value >= minimum,
                'Insufficient or unavailable project allowance: ' + key)
    return data


def active_jobs(root):
    ids = subprocess.check_output(['squeue', '-h', '-A', 'thes2452', '-o', '%i'], text=True, timeout=30).split()
    for job in ids:
        require(job.replace('_', '').isdigit(), 'Unknown active-job identifier')
        if job == os.environ.get('SLURM_JOB_ID'):
            continue
        raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
        saved = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True, timeout=30)
        # Serial benchmark policy is conservative when Slurm hides exported env.
        require(root not in raw and 'xr1_b24_checkpoint_two_way' not in raw
                and 'claix_b24_checkpoint_two_way' not in raw
                and root not in saved and 'claix_b24_checkpoint_two_way' not in saved,
                'An active benchmark may use these roots')
    return ids


def verify(review_path, review_sha256, *, expected_root=None, expected_overrides=None, expected_python=None,
           expected_filesystem_source=None, expected_project_id=None):
    require(bool(review_path) and bool(review_sha256), 'Exact reviewed input path and SHA256 are mandatory')
    path = exact_path(review_path)
    require(sha(path) == review_sha256, 'Reviewed input SHA256 mismatch')
    review = json.loads(path.read_text())
    require(set(review) == KEYS and review['schema'] == 'task04c-benchmark-review-v1'
            and bool(review['reviewer'].strip()), 'Incomplete or unknown reviewed input contract')
    require(exact_path(review['worktree']) == launch.ROOT and review['branch'] == BRANCH
            and git('branch', '--show-current') == BRANCH, 'Wrong source root/branch')
    require(git('rev-parse', 'HEAD') == review['commit'] and review['commit'] != START,
            'Pin the final reviewed Task04C commit, not its unmodified parent')
    subprocess.run(['git', 'merge-base', '--is-ancestor', START, review['commit']], cwd=launch.ROOT, check=True)
    require(not git('status', '--porcelain', '--untracked-files=all'), 'Checkout has tracked/untracked source changes')
    require(launch.tree_digest() == review['source_tree_sha256'], 'Exact source digest mismatch')
    protected_science()
    root = exact_path(review['run_root'], exists=False, canonical=True)
    require(not root.exists() and not os.path.lexists(root), 'Benchmark root already exists')
    for name in ('B_sync500', 'C_async500'):
        require(not os.path.lexists(root/name), 'Leg root already exists')
    ids = review['run_ids']
    require(set(ids) == {'B', 'C'} and len(set(ids.values())) == 2
            and all(len(v) == 32 and all(c in '0123456789abcdef' for c in v) for v in ids.values()),
            'B/C require distinct reviewed UUIDs')
    require(review['python'] == os.environ.get('XR1_PYTHON') == str(Path(sys.executable).absolute()),
            'Use exact approved XR1_PYTHON')
    for actual, expected in ((review['run_root'], expected_root), (review['overrides'], expected_overrides),
                             (review['python'], expected_python),
                             (review['filesystem_source'], expected_filesystem_source),
                             (review['project_id'], expected_project_id)):
        require(expected is None or actual == expected, 'Coordinator/review input path mismatch')
    require(sha(exact_path(review['overrides'])) == review['overrides_sha256'], 'Path overrides changed')
    wrapper = launch.ROOT/'scripts/claix_b24_checkpoint_two_way.sbatch'
    require(sha(wrapper) == review['wrapper_sha256'], 'Wrong reviewed batch wrapper')
    # The entire script remains attested again by allocation_deadline, not only directives.
    directives = ('account=thes2452', 'partition=c23g', 'nodes=1', 'ntasks=1', 'gres=gpu:hopper:1',
                  'cpus-per-task=12', 'mem=96G', 'time=23:45:00', 'signal=B:USR1@1800')
    for directive in directives:
        require(wrapper.read_text().count('#SBATCH --'+directive+'\n') == 1, 'Wrong allocation: ' + directive)
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Run expensive preflight in a compute allocation')
    current = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
    fields = dict(p.split('=', 1) for p in current.split() if '=' in p)
    require(fields.get('Account') == 'thes2452' and fields.get('JobState') == 'RUNNING'
            and fields.get('Partition') in ('c23ms', 'c23g'), 'Wrong compute account/partition')
    if fields['Partition'] == 'c23g':
        for key in ('StdOut', 'StdErr', 'WorkDir'):
            location = Path(fields.get(key, ''))
            require(str(location if key == 'WorkDir' else location.parent) == review['log_directory'],
                    'GPU batch work/log directory differs from reviewed path: ' + key)
    require(not os.environ.get('XR1_RESUME') and not os.environ.get('XR1_STOP_AFTER_STEP')
            and not os.environ.get('XR1_OPTION_B_RESUME_BRIDGE'), 'Foreign resume/stop environment forbidden')
    current_env = environment_probe(review['python'])
    report_path = exact_path(review['cpu_report'])
    require(sha(report_path) == review['cpu_report_sha256'], 'CPU acceptance report changed')
    report = json.loads(report_path.read_text())
    validate_report(report)
    require(report['source_tree_sha256'] == review['source_tree_sha256']
            and report['checkout']['git_commit'] == review['commit'] and report['checkout']['git_status'] == ''
            and report['branch'] == BRANCH and report['environment'] == current_env,
            'CPU report belongs to different/dirty source or framework environment')
    require(report['allocation']['Account'] == 'thes2452' and report['allocation']['Partition'] == 'c23ms',
            'CPU acceptance did not run in reviewed CLAIX CPU partition')
    compose_science(review)
    assets(review)
    storage = experimental_evaluate(str(root), review['filesystem_source'], review['project_id'], stage='submission')
    require(storage['status'] == 'PASS', 'Storage/inodes/user/project quota failed: ' + json.dumps(storage))
    logs = exact_path(review['log_directory'])
    require(inspect_directory(logs, expected_source=review['filesystem_source'])['filesystem']['fstype'] == 'lustre'
            and os.access(logs, os.W_OK), 'Persistent Lustre log directory unavailable')
    # Both legs together: two roots each retain two payloads plus transient writes.
    require(storage['available_bytes'] >= 60*1024**3, 'Two-way filesystem reserve below 60 GiB')
    for row in storage['quota_measurements']:
        require(row['remaining_bytes'] is None or row['remaining_bytes'] >= 60*1024**3,
                'Two-way user/project byte reserve below 60 GiB')
        require(row['remaining_inodes'] is None or row['remaining_inodes'] >= 20000,
                'Two-way user/project inode reserve below 20000')
    require(storage['available_inodes'] >= 20000, 'Two-way filesystem inode reserve below 20000')
    budget(review)
    active_jobs(str(root))
    require(launch.tree_digest() == review['source_tree_sha256']
            and not git('status', '--porcelain', '--untracked-files=all')
            and sha(path) == review_sha256, 'Inputs changed during preflight')
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review')
    parser.add_argument('--review-sha256')
    parser.add_argument('--environment-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.environment_only:
        print(json.dumps(environment(), sort_keys=True))
        return 0
    try:
        review = verify(args.review, args.review_sha256)
        print(json.dumps(dict(status='PASS', scope='exact reviewed inputs; never a job submission',
                              run_root=review['run_root'], source_tree_sha256=review['source_tree_sha256'])))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='BLOCKED', reason=f'{type(exc).__name__}: {exc}')))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
