#!/usr/bin/env python3
"""Reviewable launcher. Default prints a plan; --execute requires a Slurm allocation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
GATES = ('frozen_splits', 'tokens_24_examples_48_views', 'workers_0_4_8',
         'h100_b24_memory', 'end_to_end_throughput', 'real_compact_resume',
         'safe_stop_and_storage', 'held_out_validation', 'lr_and_horizon_approval')


class NvidiaSmiMonitor:
    """Best-effort controller-side sampling; no training hooks or signal handlers."""
    def __init__(self, path):
        self.path = Path(path)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='nvidia-smi-monitor')

    def _append(self, **fields):
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                         slurm_job_id=os.environ.get('SLURM_JOB_ID'), **fields)) + '\n')

    def _run(self):
        try:
            while not self.stopping.is_set():
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=3, start_new_session=True)
                devices = []
                for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    index, uuid, utilization, used, total = [value.strip() for value in row]
                    def number(value):
                        try:
                            return float(value)
                        except ValueError:
                            return None
                    devices.append(dict(index=index, uuid=uuid, utilization_percent=number(utilization),
                                        memory_used_mib=number(used), memory_total_mib=number(total)))
                if not devices:
                    raise RuntimeError('nvidia-smi returned no devices')
                self._append(event='sample', devices=devices)
                self.stopping.wait(15)
        except Exception as error:
            # Missing binary, query timeout, unsupported driver, and log I/O
            # disable monitoring only. subprocess.run kills/reaps timed-out queries.
            try:
                self._append(event='monitor_disabled', error_type=type(error).__name__, error=str(error))
            except Exception:
                pass

    def start(self):
        try:
            self.thread.start()
        except Exception:
            pass

    def close(self):
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=4)


def checkout_identity():
    return dict(
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))


def write_provenance(prefix, plan, release_record=None):
    """Keep exact checkout identity and source checksums per invocation/resume."""
    files = sorted([*ROOT.joinpath('xr1').rglob('*.py'), *ROOT.joinpath('xr1/configs').rglob('*.yaml'),
                    *ROOT.joinpath('scripts').glob('claix_b24*')])
    record = dict(
        **checkout_identity(),
        source_tree_sha256=tree_digest(), plan=plan,
        source_files={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        release_record=json.loads(Path(release_record).read_text()) if release_record else None,
        slurm={key: os.environ.get(key) for key in
               ('SLURM_JOB_ID', 'SLURM_SUBMIT_DIR', 'SLURM_JOB_NODELIST')})
    Path(str(prefix) + '.provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


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


# Qualification requests are deliberately independent of the production lease.
PROFILES = {
    'cpu-qualification': dict(margin=120, max_seconds=7200, gpus=0),
    'h100-qualification': dict(margin=600, max_seconds=7200, gpus=1),
    'production': dict(margin=1800, max_seconds=85500, gpus=1),
}


def allocation_deadline(mode='production'):
    profile = PROFILES[mode]
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('--execute requires a compute-node Slurm allocation')
    result = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'],
                            check=True, capture_output=True, text=True)
    fields = dict(pair.split('=', 1) for pair in result.stdout.split() if '=' in pair)
    limit = fields.get('TimeLimit', '')
    if mode == 'production' and limit != '23:45:00':
        raise RuntimeError('Production requires exactly 23:45:00 allocation')
    try:
        days, clock = limit.split('-') if '-' in limit else ('0', limit)
        hours, minutes, seconds = map(int, clock.split(':'))
        duration = int(days)*86400 + hours*3600 + minutes*60 + seconds
    except ValueError as error:
        raise RuntimeError('Cannot establish finite allocation TimeLimit') from error
    if not profile['margin'] < duration <= profile['max_seconds']:
        raise RuntimeError('Allocation TimeLimit is outside the reviewed mode budget')
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    gpu_count = int(tres.get('gres/gpu', sum(int(v) for k, v in tres.items() if k.startswith('gres/gpu:'))))
    if gpu_count != profile['gpus']:
        raise RuntimeError('Allocation GPU count does not match launch mode')
    if int(tres.get('cpu', '0')) < 12:
        raise RuntimeError('Reviewed allocation requires at least 12 CPUs')
    memory = tres.get('mem', '')
    try:
        units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
        memory_bytes = float(memory[:-1]) * units[memory[-1].upper()]
    except (ValueError, KeyError, IndexError) as error:
        raise RuntimeError('Cannot establish allocation host memory') from error
    if not math.isfinite(memory_bytes) or memory_bytes < (48 if mode == 'cpu-qualification' else 96) * 1024**3:
        raise RuntimeError('Allocation host memory is below the reviewed request')
    end = fields.get('EndTime', '')
    if not end or end in ('Unknown', 'N/A'):
        raise RuntimeError('Cannot establish authoritative allocation end time')
    epoch = float(subprocess.check_output(['date', '-d', end, '+%s'], text=True).strip())
    if not math.isfinite(epoch):
        raise RuntimeError('Invalid allocation end epoch')
    deadline = epoch - profile['margin']
    if time.time() >= deadline:
        raise RuntimeError('Allocation is already inside checkpoint write margin')
    print('ALLOCATION=' + json.dumps(dict(mode=mode, time_limit=limit, end_time=end,
                                         safe_stop_epoch=deadline, alloc_tres=tres)), flush=True)
    return deadline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('cpu-qualification', 'h100-qualification', 'qualification', 'production'), default='h100-qualification')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--workers', type=int, choices=(0, 4, 8), default=4)
    p.add_argument('--run-root', required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--qualification-max-steps', type=int,
                   help='H100 qualification only: bounded trainer limit, 6..300 (default 6)')
    p.add_argument('--qualification-deadline-epoch', type=float,
                   help='H100 qualification only: shorten, never extend, allocation safe deadline')
    p.add_argument('--release-record')
    p.add_argument('--overrides', default=str(ROOT / '.claix_b24_reference/configs/hydra_overrides.json'))
    args = p.parse_args(argv)
    # Retain the old CLI spelling as an explicit short H100 qualification alias.
    if args.mode == 'qualification':
        args.mode = 'h100-qualification'
    if args.mode != 'h100-qualification' and (args.qualification_max_steps is not None
                                            or args.qualification_deadline_epoch is not None):
        p.error('qualification bounds are H100 qualification only')
    qualification_steps = args.qualification_max_steps if args.qualification_max_steps is not None else 6
    if not 6 <= qualification_steps <= 300:
        p.error('qualification-max-steps must be between 6 and 300')
    if args.qualification_deadline_epoch is not None and (
            not math.isfinite(args.qualification_deadline_epoch) or args.qualification_deadline_epoch <= 0):
        p.error('qualification deadline must be a positive finite epoch')
    if args.mode == 'cpu-qualification' and (args.resume or args.stop_after_step is not None):
        p.error('CPU qualification always measures all workers; resume/stop-after-step are GPU options')
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= (qualification_steps if args.mode == 'h100-qualification' else 48284):
        p.error('stop-after-step is outside this mode step budget')
    base = json.loads(Path(args.overrides).read_text())
    allowed = {'data.params.train_datasets.paths', 'model.params.pretrained',
               'model.params.language_policy_path', 'trainer.default_root_dir'}
    if (not isinstance(base, list) or any(not isinstance(v, str) or v.split('=',1)[0] not in allowed for v in base)
            or len(base) != 4 or {v.split('=',1)[0] for v in base} != allowed):
        raise RuntimeError('Launcher accepts the four staged path overrides only')
    plan = dict(batch_size=24, max_length=12288, data_horizon=48284, scheduler_steps=48284,
                warmup_steps=500, workers=args.workers, world_size=1,
                run_root=str(Path(args.run_root).resolve()), overrides=base)
    overrides = base + ['trainer.default_root_dir=' + json.dumps(plan['run_root']),
                        f'data.params.train_datasets.num_workers={args.workers}']
    if args.mode == 'h100-qualification':
        overrides += [f'trainer.max_steps={qualification_steps}', 'trainer.exp_name=qualification_b24_w' + str(args.workers)]
    overrides += ['hydra.run.dir=' + json.dumps(str(Path(plan['run_root']) / 'hydra' / f'{args.mode}-{time.time_ns()}')),
                  'hydra.job.chdir=false']
    if args.resume:
        overrides += ['trainer.ckpt_path=' + json.dumps(str(Path(args.resume).resolve()))]
    if args.stop_after_step is not None:
        overrides += ['production.stop_after_step=' + str(args.stop_after_step)]
    cmd = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
           str(ROOT / 'xr1/tools/train_doosan_compact.py'), '--config-name=claix_b24_production', *overrides]
    if args.mode == 'cpu-qualification':
        cmd = [args.python, str(ROOT / 'xr1/tools/claix_b24_worker_qualification.py'),
               '--overrides', str(Path(args.overrides).resolve()),
               '--output', str(Path(plan['run_root']) / 'workers.json')]
    print(json.dumps(dict(safety_margin_seconds=PROFILES[args.mode]['margin'], mode=args.mode,
                          workers_measured=[0,4,8] if args.mode == 'cpu-qualification' else [args.workers],
                          production_plan=plan, source_tree_sha256=tree_digest()), indent=2))
    print('COMMAND=' + shlex.join(cmd), flush=True)
    if not args.execute:
        return 0
    if args.mode == 'production':
        if not args.release_record:
            raise RuntimeError('Production requires completed qualification release record')
        check_release(args.release_record, plan)
    deadline = allocation_deadline(args.mode)
    if args.qualification_deadline_epoch is not None:
        deadline = min(deadline, args.qualification_deadline_epoch)
        if deadline <= time.time():
            raise RuntimeError('Qualification deadline already passed')
    run_root = Path(plan['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    # One lock across qualification/production sharing the run root. Held by
    # controller AND torchrun, so controller death cannot free an active run lock.
    with (run_root / '.production.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = run_root / 'project_xr1_doosan' / (
            'xr1_doosan_b24_results_v1' if args.mode == 'production' else f'qualification_b24_w{args.workers}')
        if args.mode == 'cpu-qualification':
            target = run_root / 'workers.json'
        if target.is_file() or (target.exists() and any(target.iterdir()) and not args.resume):
            raise RuntimeError('Run directory already contains state; explicitly resume or use a new root')
        log_prefix = run_root / f'launch-{time.time_ns()}'
        write_provenance(log_prefix, plan, args.release_record)
        env = dict(os.environ, MAX_LENGTH='12288', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   XR1_SAFE_STOP_EPOCH=str(deadline),
                   PYTHONPATH=str(ROOT / 'xr1') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        if args.mode == 'cpu-qualification':
            env['CUDA_VISIBLE_DEVICES'] = ''
        if not env.get('XR1_DOOSAN_ZIP_CACHE_ROOT') or env.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'):
            raise RuntimeError('Qualification/production require the approved ZIP cache only')
        stop_file = run_root / f'.stop-request-{os.getpid()}.json'
        env['XR1_STOP_REQUEST_FILE'] = str(stop_file)
        pending = {}
        def request_stop(signum, frame):
            pending['signal'] = signum
        old_handlers = {sig: signal.signal(sig, request_stop)
                        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        stdout = Path(str(log_prefix) + '.out').open('x')
        stderr = Path(str(log_prefix) + '.err').open('x')
        stdout.write(f'LAUNCH_EPOCH={time.time()} SAFE_STOP_EPOCH={deadline}\nCOMMAND={shlex.join(cmd)}\n')
        stdout.flush()
        print(f'PERSISTENT_STDOUT={stdout.name} PERSISTENT_STDERR={stderr.name}', flush=True)
        # Never forward USR1 to torchrun: its agent need not handle that signal.
        # Handlers set flags; ordinary controller code requests a worker stop.
        child = None
        child_finished = False
        monitor = (NvidiaSmiMonitor(str(log_prefix) + '.gpu.jsonl')
                   if args.mode == 'production' and os.environ.get('XR1_NVIDIA_SMI_MONITOR', '1') != '0'
                   else None)
        try:
            if monitor is not None:
                monitor.start()
            child = subprocess.Popen(cmd, cwd=ROOT / 'xr1', env=env, pass_fds=(lock.fileno(),),
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            while True:
                if time.time() >= deadline:
                    pending.setdefault('reason', 'safe_deadline')
                if pending and not stop_file.exists():
                    temporary_request = stop_file.with_suffix('.tmp')
                    with temporary_request.open('x') as stream:
                        json.dump(pending, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_request, stop_file)
                    # Preserve controller evidence after the transient request is removed.
                    evidence = dict(pending, controller_pid=os.getpid(), child_pid=getattr(child, 'pid', None),
                                    requested_epoch=time.time(), safe_stop_epoch=deadline,
                                    timestamp=datetime.now(timezone.utc).isoformat())
                    Path(str(log_prefix) + '.stop.json').write_text(json.dumps(evidence, indent=2) + '\n')
                if args.mode == 'h100-qualification' and time.time() >= deadline + 90:
                    raise RuntimeError('Qualification child exceeded bounded shutdown grace')
                try:
                    result = child.wait(timeout=0.5)
                    child_finished = True
                    return result
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if monitor is not None:
                monitor.close()
            # Only our new-session child group; never signal a Slurm daemon or job group.
            # A stuck qualification must not leave training running behind the controller.
            if args.mode == 'h100-qualification' and child is not None and not child_finished:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                except ProcessLookupError:
                    child.wait(timeout=10)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            stdout.close()
            stderr.close()
            if stop_file.exists():
                stop_file.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
