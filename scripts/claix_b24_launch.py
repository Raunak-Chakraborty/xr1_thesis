#!/usr/bin/env python3
"""Reviewable launcher. Default prints a plan; --execute requires a Slurm allocation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GATES = ('frozen_splits', 'tokens_24_examples_48_views', 'workers_0_4_8',
         'h100_b24_memory', 'end_to_end_throughput', 'real_compact_resume',
         'safe_stop_and_storage', 'held_out_validation', 'lr_and_horizon_approval')


def tree_digest():
    h = hashlib.sha256()
    files = sorted([*ROOT.joinpath('xr1').rglob('*.py'), *ROOT.joinpath('xr1/configs').rglob('*.yaml'),
                    *ROOT.joinpath('scripts').glob('claix_b24*')])
    for path in files:
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def check_release(path, plan):
    record = json.loads(Path(path).read_text())
    if record.get('source_tree_sha256') != tree_digest() or record.get('plan') != plan:
        raise RuntimeError('Release record does not bind this source tree and production plan')
    for gate in GATES:
        row = record.get('gates', {}).get(gate, {})
        if row.get('status') != 'PASS':
            raise RuntimeError('Unqualified production gate: ' + gate)
        evidence = Path(row['evidence'])
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != row.get('sha256'):
            raise RuntimeError('Gate evidence checksum mismatch: ' + gate)


def allocation_deadline():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('--execute requires a compute-node Slurm allocation')
    result = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'],
                            check=True, capture_output=True, text=True)
    fields = dict(pair.split('=', 1) for pair in result.stdout.split() if '=' in pair)
    if fields.get('TimeLimit') != '23:45:00':
        raise RuntimeError('Expected exactly 23:45:00 allocation')
    end = fields.get('EndTime', '')
    if not end or end in ('Unknown', 'N/A'):
        raise RuntimeError('Cannot establish authoritative allocation end time')
    epoch = float(subprocess.check_output(['date', '-d', end, '+%s'], text=True).strip())
    deadline = epoch - 1800
    if time.time() >= deadline:
        raise RuntimeError('Allocation is already inside checkpoint write margin')
    return deadline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('qualification', 'production'), default='qualification')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--workers', type=int, choices=(0, 4, 8), default=4)
    p.add_argument('--run-root', required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--release-record')
    p.add_argument('--overrides', default=str(ROOT / '.claix_b24_reference/configs/hydra_overrides.json'))
    args = p.parse_args(argv)
    base = json.loads(Path(args.overrides).read_text())
    allowed = {'data.params.train_datasets.paths', 'model.params.pretrained',
               'model.params.language_policy_path', 'trainer.default_root_dir'}
    if (not isinstance(base, list) or any(not isinstance(v, str) or v.split('=',1)[0] not in allowed for v in base)
            or {v.split('=',1)[0] for v in base} != allowed):
        raise RuntimeError('Launcher accepts the four staged path overrides only')
    plan = dict(batch_size=24, max_length=12288, data_horizon=48284, scheduler_steps=48284,
                warmup_steps=500, workers=args.workers, world_size=1,
                run_root=str(Path(args.run_root).resolve()), overrides=base)
    overrides = base + [f'trainer.default_root_dir={plan["run_root"]}',
                        f'data.params.train_datasets.num_workers={args.workers}']
    if args.mode == 'qualification':
        overrides += ['trainer.max_steps=6', 'trainer.exp_name=qualification_b24_w' + str(args.workers)]
    if args.resume:
        overrides += ['trainer.ckpt_path=' + str(Path(args.resume).resolve())]
    if args.stop_after_step is not None:
        overrides += ['production.stop_after_step=' + str(args.stop_after_step)]
    cmd = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
           str(ROOT / 'xr1/tools/train_doosan_compact.py'), '--config-name=claix_b24_production', *overrides]
    print(json.dumps(dict(mode=args.mode, production_plan=plan, source_tree_sha256=tree_digest()), indent=2))
    print('COMMAND=' + shlex.join(cmd), flush=True)
    if not args.execute:
        return 0
    if args.mode == 'production':
        if not args.release_record:
            raise RuntimeError('Production requires completed qualification release record')
        check_release(args.release_record, plan)
    deadline = allocation_deadline()
    run_root = Path(plan['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    # One lock across qualification/production sharing the run root. Held by
    # controller AND torchrun, so controller death cannot free an active run lock.
    with (run_root / '.production.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = run_root / 'project_xr1_doosan' / (
            'xr1_doosan_b24_results_v1' if args.mode == 'production' else f'qualification_b24_w{args.workers}')
        if target.exists() and any(target.iterdir()) and not args.resume:
            raise RuntimeError('Run directory already contains state; explicitly resume or use a new root')
        env = dict(os.environ, MAX_LENGTH='12288', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   XR1_SAFE_STOP_EPOCH=str(deadline),
                   PYTHONPATH=str(ROOT / 'xr1') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        if not env.get('XR1_DOOSAN_ZIP_CACHE_ROOT') or env.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'):
            raise RuntimeError('Qualification/production require the approved ZIP cache only')
        stop_file = run_root / f'.stop-request-{os.getpid()}.json'
        env['XR1_STOP_REQUEST_FILE'] = str(stop_file)
        pending = {}
        def request_stop(signum, frame):
            pending['signal'] = signum
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, request_stop)
        log_prefix = run_root / f'launch-{time.time_ns()}'
        stdout = Path(str(log_prefix) + '.out').open('x')
        stderr = Path(str(log_prefix) + '.err').open('x')
        stdout.write(f'LAUNCH_EPOCH={time.time()} SAFE_STOP_EPOCH={deadline}\nCOMMAND={shlex.join(cmd)}\n')
        stdout.flush()
        print(f'PERSISTENT_STDOUT={stdout.name} PERSISTENT_STDERR={stderr.name}', flush=True)
        # Never forward USR1 to torchrun: its agent need not handle that signal.
        # Handlers set flags; ordinary controller code requests a worker stop.
        try:
            child = subprocess.Popen(cmd, cwd=ROOT / 'xr1', env=env, pass_fds=(lock.fileno(),),
                                     stdout=stdout, stderr=stderr)
            while True:
                if pending and not stop_file.exists():
                    with stop_file.open('x') as stream:
                        json.dump(pending, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                try:
                    return child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            stdout.close()
            stderr.close()
            if stop_file.exists():
                stop_file.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
