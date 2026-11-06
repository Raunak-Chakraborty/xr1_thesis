#!/usr/bin/env python3
"""Exactly two fresh sequential runs. Planning is default; never submits Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import claix_b24_launch as launch


def commands(args):
    root = Path(args.run_root)
    if not root.is_absolute() or root.is_symlink():
        raise RuntimeError('Two-way root must be absolute and new')
    return [[args.python, str(launch.ROOT/'scripts/claix_b24_launch.py'),
             '--mode', 'experimental', '--experimental-unqualified',
             '--experimental-checkpoint-interval', '500', '--experimental-two-way-leg', leg,
             '--run-root', str(root/name), '--workers', '4', '--overrides', args.overrides,
             '--filesystem-source', args.filesystem_source, '--project-id', str(args.project_id),
             '--python', args.python, *(['--execute'] if args.execute else [])]
            for leg, name in (('B', 'B_sync500'), ('C', 'C_async500'))]


def verify_leg(root, leg):
    """Read-only acceptance of receipts, committed terminal bytes and telemetry."""
    sys.path.insert(0, str(launch.ROOT/'xr1'))
    from mibot.utils.claix_checkpoint import load_verified
    profile = launch.experiment_profile(500, leg)
    target = root/'project_xr1_doosan'/profile['name']
    manifest = launch.read_experiment(root)
    plan = manifest['plan']
    if (plan.get('two_way_leg') != leg or plan.get('benchmark_steps') != 1500
            or manifest['source_tree_sha256'] != launch.tree_digest()):
        raise RuntimeError('Two-way source/plan identity drift')
    rows = [json.loads(line) for line in (target/'telemetry.jsonl').read_text().splitlines()]
    saves = [r for r in rows if r['event'] == 'checkpoint']
    if ([r['step'] for r in rows if r['event'] == 'step'] != list(range(1, 1501))
            or [r['step'] for r in saves] != [500, 1000, 1500]
            or any(r.get('verified') is not True for r in saves)
            or rows[-1]['event'] != 'fit_end' or rows[-1].get('final_checkpoint_verified') is not True
            or any(r.get('reason') != 'qualification_step' for r in rows if r['event'] == 'graceful_stop')
            or list(root.glob('launch-*.stop.json'))):
        raise RuntimeError('Two-way leg incomplete, interrupted, or missing verified checkpoints')
    latest = json.loads((target/'latest.json').read_text())
    if latest != json.loads((target/'safe-stop.json').read_text()) or latest['global_step'] != 1500:
        raise RuntimeError('Two-way final pointers differ')
    expected_experiment = latest['experiment']
    if (expected_experiment['run_id'] != manifest['run_id']
            or expected_experiment['run_root'] != str(root)
            or expected_experiment['experiment_name'] != profile['name']
            or expected_experiment.get('checkpoint_writer', 'sync') != plan['checkpoint_writer']):
        raise RuntimeError('Two-way checkpoint identity mismatch')
    receipts = []
    for step in (500, 1000, 1500):
        path = target/f'step-{step:06d}.ckpt'
        receipt = json.loads(path.with_suffix('.verified.json').read_text())
        if (receipt.get('schema') != 'claix-b24-verified-v1' or receipt['global_step'] != step
                or receipt['path'] != path.name or receipt['config_sha256'] != latest['config_sha256']
                or receipt['experiment'] != expected_experiment):
            raise RuntimeError('Two-way checkpoint receipt identity mismatch')
        # Normal retention may remove 500's payload after 1500, leaving its receipt.
        if path.exists():
            load_verified(path, expected_config=latest['config_sha256'], expected_experiment=expected_experiment)
        elif step != 500:
            raise RuntimeError('Missing retained recovery checkpoint')
        receipts.append(receipt)
    final = load_verified(target/'step-001500.ckpt', expected_experiment=expected_experiment)
    if final['xr1_replay']['rank_sample_offset'] != 1500*24:
        raise RuntimeError('Two-way replay cursor mismatch')
    starts = list(target.glob('experimental-start-*.json'))
    if len(starts) != 1:
        raise RuntimeError('Benchmark requires exactly one fresh training invocation per leg')
    initial = json.loads(starts[0].read_text())
    if (initial['initial_global_step'] != 0 or initial['new_optimizer_and_scheduler'] is not True
            or initial['initial_checkpoint_mode'] != 'original_pretrained_step_zero'):
        raise RuntimeError('Benchmark leg was not initialized fresh')
    scientific_trainer = dict(initial['source_config']['trainer'])
    for key in ('exp_name', 'default_root_dir', 'ckpt_path'):
        scientific_trainer.pop(key)
    science = dict(model=initial['source_config']['model'], data=initial['source_config']['data'],
                   trainer=scientific_trainer, dataset=initial['dataset_identity'],
                   pretrained=initial['pretrained_identity'], language=initial['language_identity'])
    step_rows = [r for r in rows if r['event'] == 'step']
    def peak(key):
        values = [r[key] for r in step_rows if isinstance(r.get(key), (int, float))]
        return max(values) if values else None
    utilization = [r['gpu_utilization_percent'] for r in step_rows
                   if isinstance(r.get('gpu_utilization_percent'), (int, float))]
    phase_totals = {}
    for saved in saves:
        for key, seconds in saved.get('timings', {}).items():
            phase_totals[key] = phase_totals.get(key, 0.) + seconds
    metrics = dict(fit_wall_seconds=rows[-1]['end_to_end_seconds'],
        end_to_end_examples_per_second=rows[-1]['end_to_end_examples_per_second'],
        step_sample_mean_gpu_utilization_percent=sum(utilization)/len(utilization) if utilization else None,
        step_sample_peak_cpu_rss_bytes=peak('cpu_rss_bytes'),
        step_sample_peak_worker_rss_bytes=peak('worker_rss_bytes'),
        gpu_peak_allocated_bytes=peak('gpu_peak_allocated_bytes'),
        gpu_peak_reserved_bytes=peak('gpu_peak_reserved_bytes'), checkpoint_phase_seconds=phase_totals,
        main_thread_checkpoint_blocking_seconds=sum(r['timings'][
            'callback_duration_seconds' if leg == 'B' else 'main_thread_blocking_seconds'] for r in saves),
        backpressure_seconds=sum(r['backpressure_seconds'] for r in rows if r['event'] == 'checkpoint_staged'),
        gpu_monitor_files=[str(p) for p in root.glob('launch-*.gpu.jsonl')])
    return dict(run_id=manifest['run_id'], receipts=receipts, final_telemetry=rows[-1],
                science_sha256=hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest(),
                metrics=metrics,
                deterministic_recovery='PENDING real-framework acceptance')


def verify_pair(root, results):
    """CPU-only final-state comparison after both training processes have exited."""
    import numpy as np
    import torch
    from mibot.utils.claix_checkpoint import load_verified
    if results[0]['science_sha256'] != results[1]['science_sha256']:
        raise RuntimeError('Two-way scientific configurations or initialization differ')
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    checkpoints = [load_verified(root/name/'project_xr1_doosan'/launch.experiment_profile(500, leg)['name']/'step-001500.ckpt')
                   for name, leg in (('B_sync500', 'B'), ('C_async500', 'C'))]
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops', 'xr1_replay'):
        if not equal(checkpoints[0][key], checkpoints[1][key]):
            raise RuntimeError('Two-way deterministic final state differs: ' + key)
    if not equal(checkpoints[0]['xr1_production']['rng_by_rank'], checkpoints[1]['xr1_production']['rng_by_rank']):
        raise RuntimeError('Two-way final RNG differs')
    return 'PASS exact final model/AdamW/scheduler/loop/replay/RNG equality; resume execution remains separate acceptance'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--overrides', required=True)
    parser.add_argument('--filesystem-source', required=True)
    parser.add_argument('--project-id', required=True, type=int)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    cmds = commands(args)
    if not args.execute:
        print(json.dumps(dict(new_runs=['B: Sync-500', 'C: Async-500'], commands=cmds,
            steps_each=1500, historical_sync250_job=5504969, gpu_jobs_submitted=0), indent=2))
        return 0
    if os.environ.get('XR1_RESUME') or os.environ.get('XR1_STOP_AFTER_STEP'):
        raise RuntimeError('Two-way job must start both legs fresh with fixed step budget')
    launch.allocation_deadline('experimental', checkpoint_interval=500, two_way_leg='B')
    from claix_b24_storage_preflight import experimental_evaluate
    root = Path(args.run_root)
    storage = experimental_evaluate(root, args.filesystem_source, args.project_id)
    if storage['status'] != 'PASS':
        raise RuntimeError('Two-way storage/quota preflight failed: ' + json.dumps(storage))
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(launch.ROOT/'xr1'))
    from mibot.utils.claix_checkpoint import atomic_json
    digest = launch.tree_digest()
    atomic_json(root/'two-way-plan.json', dict(commands=cmds, source_tree_sha256=digest,
                storage=storage, **launch.checkout_identity()), exclusive=True)
    pending = []
    def stop(signum, frame):
        pending.append(signum)
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    results = []
    try:
        for leg, cmd in zip(('B', 'C'), cmds):
            if pending or launch.tree_digest() != digest:
                raise RuntimeError('Two-way interrupted or source changed; no next leg')
            started = time.perf_counter()
            child = subprocess.Popen(cmd, start_new_session=True)
            sent = False
            try:
                while True:
                    if pending and not sent:
                        try:
                            child.send_signal(pending[0])
                        except ProcessLookupError:
                            pass
                        sent = True
                    try:
                        code = child.wait(timeout=.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if child.poll() is None:
                    child.terminate()  # controller requests a boundary stop, then reaps
                    child.wait()
            result = dict(leg=leg, returncode=code, wall_seconds=time.perf_counter()-started)
            results.append(result)
            atomic_json(root/'two-way-results.json', dict(status='PENDING', legs=results))
            if code or pending:
                raise RuntimeError(f'Leg {leg} failed/interrupted; preserve evidence, no retry')
            leg_root = root/('B_sync500' if leg == 'B' else 'C_async500')
            result.update(verify_leg(leg_root, leg))
            if leg == 'C' and results[0]['run_id'] == result['run_id']:
                raise RuntimeError('Two-way experiment IDs must be independent')
        pair = verify_pair(root, results)
        atomic_json(root/'two-way-results.json', dict(status='TRANSACTIONS_VERIFIED', legs=results,
                    final_state_comparison=pair,
                    historical_sync250=dict(job=5504969, steps=48129, allocated_gpu_hours_approx=23.26,
                        evidence='PENDING retained historical telemetry/receipt attribution; not a controlled short-run comparison'),
                    acceptance='PENDING deterministic recovery and performance review'))
        return 0
    except BaseException as error:
        atomic_json(root/'two-way-results.json', dict(status='BLOCKED', legs=results, error=str(error),
                    recommendation='Abandon async if C failed; preserve all evidence; no automatic retry'))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
