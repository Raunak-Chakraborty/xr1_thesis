#!/usr/bin/env python3
"""Dedicated actual-environment CPU acceptance. Never schedules or trains on GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import claix_b24_launch as launch
from claix_b24_filesystem import inspect_directory

SUITES = ('run_claix_async_cpu_tests.py', 'run_claix_experimental_cpu_tests.py',
          'run_claix_cpu_tests.py', 'run_doosan_io_cpu_tests.py')
REQUIRED = (
    'test_claix_runtime.RuntimeTest.test_checkpoint500_real_lightning_stop_resume_and_final_verification',
    'test_claix_runtime.RuntimeTest.test_lightning_resume_early_and_near_horizon_workers',
    'test_claix_replay.ReplayTest.test_workers_and_prefetch_resume',
    'test_claix_checkpoint500.Checkpoint500HydraTest.test_real_hydra_composition_preserves_frozen_science',
    'test_claix_async_framework.AsyncFrameworkTest.test_real_lightning_500_stop751_resume_rng_and_final_equality',
    'test_claix_async_framework.AsyncDDPFrameworkTest.test_real_single_rank_cpu_ddp_checkpoint_io_handoff',
    'test_claix_async_framework.AsyncFrameworkTest.test_real_hydra_preserves_entire_scientific_configuration',
    'test_claix_async_framework.BenchmarkTerminationFrameworkTest.test_real_sync_async_1500_final_state',
    'test_claix_async_framework.BenchmarkTerminationFrameworkTest.test_real_sync500_four_worker_resume',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def environment():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    require(Path(os.environ['XR1_PYTHON']).resolve(strict=True) == Path(sys.executable).resolve(strict=True),
            'Must use approved existing XR1_PYTHON')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        require(os.environ.get(key) == '1', 'Approved offline environment missing: ' + key)
    # Real imports, no fallbacks or Callback/CheckpointIO/registry stubs here.
    import torch
    import lightning
    import hydra
    import mmengine
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.fabric.plugins.io import CheckpointIO
    require(not torch.cuda.is_available(), 'CPU acceptance must have no visible CUDA')
    require(torch.distributed.is_gloo_available(), 'Real Gloo is required')
    for module in (torch, lightning, hydra, mmengine):
        require(module.__file__ and Path(module.__file__).is_file(), 'Framework module has no real source')
    paths = {m.__name__: str(Path(m.__file__).resolve()) for m in (torch, lightning, hydra, mmengine)}
    return dict(python=str(Path(sys.executable).resolve()), python_version=sys.version,
                versions={name: importlib.metadata.version(name) for name in
                          ('torch', 'lightning', 'hydra-core', 'mmengine', 'transformers')},
                module_paths=paths, ddp_strategy=str(DDPStrategy), checkpoint_io=str(CheckpointIO))


def allocation():
    job = os.environ.get('SLURM_JOB_ID')
    require(bool(job), 'CPU acceptance requires a compute allocation, never a login node')
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    fields = dict(pair.split('=', 1) for pair in raw.split() if '=' in pair)
    for key, expected in dict(JobId=job, Account='thes2452', Partition='c23ms', JobState='RUNNING',
                              NumNodes='1', NumTasks='1', NumCPUs='12', TimeLimit='04:00:00').items():
        require(fields.get(key) == expected, 'Wrong CPU allocation: ' + key)
    tres = dict(p.split('=', 1) for p in fields.get('AllocTRES', '').split(',') if '=' in p)
    require(not any(int(v) for k, v in tres.items() if k.startswith('gres/gpu')), 'GPU TRES forbidden')
    require(tres.get('cpu') == '12' and tres.get('mem') in ('96G', '98304M'), 'Wrong CPU RAM/cores')
    saved = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
    require(saved == (launch.ROOT/'scripts/claix_b24_task04c_cpu.sbatch').read_text(), 'Wrong saved CPU wrapper')
    for key in ('StdOut', 'StdErr', 'WorkDir'):
        path = Path(fields.get(key, ''))
        inspect_directory(path if key == 'WorkDir' else path.parent)
    return fields


def validate_report(report):
    require(report.get('schema') == 'task04c-real-cpu-v1' and report.get('status') == 'PASS',
            'Real CPU acceptance report is incomplete or failed')
    rows = report.get('suites', [])
    require([r['suite'] for r in rows] == list(SUITES), 'Missing or reordered mandatory CPU harness')
    ids = set()
    for row in rows:
        require(row['returncode'] == 0, 'CPU harness failed: ' + row['suite'])
        path = Path(row['result_path'])
        require(sha(path) == row['result_sha256'] and sha(row['log_path']) == row['log_sha256'],
                'CPU evidence bytes changed')
        detail = json.loads(path.read_text())
        require(detail['returncode'] == 0 and detail['count'] == row['count'] == len(detail['tests'])
                and detail['count'] > 0 and detail.get('error') is None, 'Empty/incomplete CPU result')
        require(all(r['status'] == 'PASS' for r in detail['tests']), 'CPU test failed, skipped or unavailable')
        ids.update(r['id'] for r in detail['tests'])
    require(set(REQUIRED) <= ids, 'Mandatory real framework cases missing: ' + str(set(REQUIRED)-ids))
    require(report['tests_executed'] == sum(r['count'] for r in rows), 'CPU test count mismatch')
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    allocation_fields = allocation()  # fail before any heavyweight import or execution
    location = inspect_directory(args.output, allow_missing=True)
    require(location['filesystem']['fstype'] == 'lustre', 'CPU evidence must be persistent Lustre')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='task04c-real-cpu-v1', status='BLOCKED', suites=[], tests_executed=0,
                  source_tree_sha256=launch.tree_digest(), checkout=launch.checkout_identity(),
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=launch.ROOT, text=True).strip(),
                  allocation=allocation_fields, started_epoch=time.time())
    try:
        report['environment'] = environment()
        for name in SUITES:
            log = output/(name + '.log')
            detail = output/(name + '.json')
            cmd = [sys.executable, '-B', str(launch.ROOT/'xr1/tests/claix_cpu_report.py'),
                   '--suite', str(launch.ROOT/'xr1/tests'/name), '--result', str(detail)]
            with log.open('x') as stream:
                code = subprocess.run(cmd, cwd=launch.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).returncode
            parsed = json.loads(detail.read_text()) if detail.exists() else {}
            row = dict(suite=name, command=cmd, returncode=code, count=parsed.get('count', 0),
                       counts=parsed.get('counts', {}),
                       log_path=str(log), log_sha256=sha(log), result_path=str(detail),
                       result_sha256=sha(detail) if detail.exists() else None)
            report['suites'].append(row)
            report['tests_executed'] += row['count']
            counts = row['counts']
            print(f'{name}: exit={code} tests={row["count"]} failures={counts.get("FAIL", 0)} '
                  f'errors={counts.get("ERROR", 0)} skips={counts.get("SKIP", 0)} log={log}', flush=True)
        require(report['source_tree_sha256'] == launch.tree_digest()
                and report['checkout'] == launch.checkout_identity(), 'Source changed during acceptance')
        report['status'] = 'PASS'
        validate_report(report)
    except Exception as exc:
        report.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['finished_epoch'] = time.time()
        with (output/'cpu-acceptance.json').open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
    print(f'TASK04C_CPU_CLAIX={report["status"]} tests={report["tests_executed"]} report={output}/cpu-acceptance.json')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
