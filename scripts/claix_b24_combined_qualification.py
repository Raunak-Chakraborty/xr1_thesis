#!/usr/bin/env python3
"""Sequential compute-node qualification. Dry-run by default; never submits jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from claix_b24_launch import ROOT, allocation_deadline, tree_digest

STAGED = Path('/hpcwork/zk443626/robotics_thesis/transfer/xr1_claix_transfer_v1')
PINNED = '5e2dbb8d184899901ba4162d5e6bbb208d281a8c'
PHASES = ('preflight', 'clock_probe', 'uninterrupted_6', 'stopped_3', 'resumed_6',
          'restart_equivalence', 'sustained_and_signal_stop', 'validation_smoke', 'evidence')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def rows_at(directory):
    path = Path(directory) / 'telemetry.jsonl'
    if not path.exists():
        return []
    # Last row can be in flight while monitoring; completed rows are immutable.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n')]


def compare_states(left, right):
    """Exact comparison; retain numerical differences for every required tensor."""
    import torch
    rows = []

    def compare(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            compatible = a.shape == b.shape and a.dtype == b.dtype
            equal = compatible and torch.equal(a, b)
            row = dict(path=path, equal=bool(equal), shape=list(a.shape), dtype=str(a.dtype))
            if compatible:
                # Chunked differences avoid large FP64 temporaries for the 141M masters.
                differing, absolute, relative = 0, 0., 0.
                for x, y in zip(a.reshape(-1).split(262144), b.reshape(-1).split(262144)):
                    x, y = x.double(), y.double()
                    delta = (x-y).abs()
                    differing += int((x != y).sum())
                    if delta.numel():
                        absolute = max(absolute, float(delta.max()))
                        relative = max(relative, float((delta / y.abs().clamp_min(1e-30)).max()))
                row.update(differing_elements=differing, max_abs_difference=absolute,
                           max_relative_difference=relative)
            rows.append(row)
        elif isinstance(a, dict) and isinstance(b, dict):
            rows.append(dict(path=path + '/keys', equal=set(a) == set(b)))
            for key in sorted(set(a) & set(b), key=str):
                compare(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            rows.append(dict(path=path + '/length', equal=len(a) == len(b)))
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + '/' + str(i))
        else:
            rows.append(dict(path=path, equal=type(a) is type(b) and a == b,
                             left=a, right=b))

    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'global_step',
                'xr1_replay', 'xr1_replay_contract', 'xr1_compact_fp32'):
        require(key in left and key in right, 'Missing required comparison: ' + key)
        compare(left[key], right[key], key)
    compare(left['xr1_production']['config_sha256'], right['xr1_production']['config_sha256'],
            'config_sha256')
    return dict(status='PASS' if all(r['equal'] for r in rows) else 'FAIL',
                comparison='exact, zero tolerance', comparisons=rows,
                unequal_comparisons=sum(not r['equal'] for r in rows))


def verify_checkpoint(directory, step, contract, fingerprint, *, latest=True):
    from mibot.utils.claix_checkpoint import load_verified
    directory = Path(directory)
    path = directory / f'step-{step:06d}.ckpt'
    ck = load_verified(path, contract, expected_config=fingerprint, resumable=False)
    require(ck['global_step'] == step, 'Unexpected checkpoint step')
    if latest:
        require(read_json(directory/'latest.json') == read_json(path.with_suffix('.verified.json')),
                'Latest pointer does not match the verified final checkpoint')
    return ck


def verify_shutdown(directory, expected_steps, *, stop_reason=None):
    rows = rows_at(directory)
    require(not any(r['event'] == 'fatal' for r in rows), 'Fatal telemetry event')
    require([r['step'] for r in rows if r['event'] == 'step'] == list(expected_steps),
            'Missing, repeated or skipped completed optimizer step')
    require(rows and rows[-1]['event'] == 'fit_end' and rows[-1].get('final_checkpoint_verified'),
            'Missing verified clean fit_end')
    if stop_reason:
        require(any(r['event'] == 'graceful_stop' and r.get('reason') == stop_reason for r in rows),
                'Missing expected completed-boundary stop reason: ' + stop_reason)
    # Every published checkpoint must have a receipt; latest is verified separately.
    for path in Path(directory).glob('step-*.ckpt'):
        require(path.with_suffix('.verified.json').is_file(), 'Unverified checkpoint left behind')
    return rows


def sustained_summary(rows):
    steps = [r for r in rows if r['event'] == 'step']
    require(len(steps) >= 251 and steps[-1]['step'] < 260, 'Signal stop must finish at step 251..259')
    for r in steps:
        require(r['retained_examples'] == 24 and r['views'] == 48, 'Examples/camera views lost')
        require(0 < r['packed_tokens'] <= 9912, 'Packed token certification failed')
        require(r.get('trainable_fp32') is True and r.get('frozen_bf16') is True,
                'Missing trainable/master FP32 and frozen BF16 invariants')
        for key in ('gpu_allocated_bytes', 'gpu_reserved_bytes', 'gpu_peak_allocated_bytes',
                    'gpu_peak_reserved_bytes', 'cpu_rss_bytes', 'worker_rss_bytes',
                    'pipeline_wait_seconds', 'step_seconds', 'elapsed_seconds', 'loss',
                    'end_to_end_examples_per_second'):
            require(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]) and r[key] >= 0,
                    'Missing/nonfinite required telemetry: ' + key)
        require(r['gpu_reserved_bytes'] >= r['gpu_allocated_bytes'] > 0 and r['cpu_rss_bytes'] > 0
                and r['worker_rss_bytes'] > 0, 'Invalid GPU/CPU memory measurement')
    # Fixed real-data window, including data wait and any intervening callback I/O.
    first, last = steps[9], steps[249]
    duration = last['elapsed_seconds']-first['elapsed_seconds']
    require(duration > 0, 'Invalid sustained time window')
    require(any(r['event'] == 'checkpoint' and r['step'] == 250 and r.get('verified')
                and r.get('reason') == 'interval' for r in rows), 'No verified interval checkpoint 250')
    return dict(steps=len(steps), samples=len(steps)*24, warm_window_steps=[11, 250],
                warm_window_seconds=duration, warm_examples_per_second=240*24/duration,
                fit_end_examples_per_second=rows[-1]['end_to_end_examples_per_second'],
                max_gpu_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in steps),
                max_gpu_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in steps),
                max_cpu_rss_bytes=max(r['cpu_rss_bytes'] for r in steps),
                max_worker_rss_bytes=max(r['worker_rss_bytes'] for r in steps),
                pipeline_wait_seconds=sum(r['pipeline_wait_seconds'] for r in steps),
                packed_tokens_range=[min(r['packed_tokens'] for r in steps), max(r['packed_tokens'] for r in steps)],
                memory_scope='per-step allocator peaks; RSS includes shared pages',
                timing_scope='fit includes loader setup and checkpoint I/O; process startup separately logged')


def validation_gate(result, identities, checkpoint_sha, frames):
    require(result['scope'] == 'SMOKE_SUBSET_NOT_A_VALIDATION_RESULT', 'Expected a smoke subset')
    require(result['checkpoint_sha256'] == checkpoint_sha, 'Evaluation checkpoint provenance mismatch')
    for key, split in (('train_sha256', 'train'), ('validation_sha256', 'validation'),
                       ('test_identity_sha256', 'test')):
        require(result[key] == identities[split], 'Evaluation frozen split drift: ' + split)
    require(result.get('processor_model_compatible') is True, 'Processor/model compatibility unverified')
    metrics = result['metrics']
    require(metrics['frames'] == frames and len(result['generated_frames']) == frames, 'Missing generated outputs')
    for aggregation in ('micro_mae', 'episode_macro_mae'):
        require(set(metrics[aggregation]) == {'translation_m', 'rotation_rad', 'gripper'}, 'Missing masked metric')
        require(all(math.isfinite(v) and v >= 0 for v in metrics[aggregation].values()), 'Nonfinite masked metric')
    for row in result['generated_frames']:
        require(row['shape'] == [30, 60] and len(row['prediction_sha256']) == 64, 'Invalid generated action evidence')
        require(set(row['masked_metrics']) == {'translation_m', 'rotation_rad', 'gripper'}, 'Missing per-frame masked metric')
        require(row['all_finite'] and all(m['count'] > 0 and math.isfinite(m['mae']) and m['mae'] >= 0
                                         for m in row['masked_metrics'].values()), 'Invalid generated actions')
        require(len(row['generated_action']) == 30 and all(len(action) == 7 and all(math.isfinite(v) for v in action)
                                                         for action in row['generated_action']),
                'Missing/nonfinite semantic generated-action evidence')


def clock_probe():
    """Safe CPU probe of the real callback clock; does not wait for Slurm expiry."""
    from mibot.utils.claix_runtime import StopRequest
    original = os.environ.pop('XR1_STOP_REQUEST_FILE', None)
    try:
        deadline = time.time()+10
        stop = StopRequest(deadline)
        require(not stop.due(now=deadline-1) and stop.due(now=deadline), 'Clock deadline boundary failed')
        return dict(scope='CPU callback clock boundary only; controller clock covered by local mocked tests',
                    before_deadline=False, at_deadline=True)
    finally:
        if original is not None:
            os.environ['XR1_STOP_REQUEST_FILE'] = original


class Qualification:
    def __init__(self, args, deadline, report):
        self.args, self.deadline, self.report = args, deadline, report
        self.root = Path(args.run_root).resolve()
        self.source = tree_digest()
        self.pending_signal = None

    def handle(self, signum, frame):
        # Flag only; process control and evidence publication happen in normal code.
        self.pending_signal = signum

    def save(self):
        from mibot.utils.claix_checkpoint import atomic_json
        atomic_json(self.root/'qualification-report.json', self.report)

    def gate(self, name, function):
        self.report['active_phase'] = name
        self.save()
        require(self.pending_signal is None, 'Allocation controller received a stop signal')
        require(time.time() < self.deadline-60, 'Insufficient time before allocation safety deadline')
        require(tree_digest() == self.source, 'Source changed during qualification')
        result = function()
        self.report['phases'][name] = dict(status='PASS', result=result)
        self.save()
        return result

    def abort(self, child, controller, hard_deadline):
        import psutil
        # Process objects guard against PID reuse; only descendants of our Popen.
        try:
            parent = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            child.wait(timeout=10)
            return
        descendants = parent.children(recursive=True)
        owned = descendants + [parent]
        if controller and child.poll() is None:
            child.send_signal(signal.SIGUSR1)
            try:
                child.wait(timeout=max(0., min(120., hard_deadline-time.time())))
            except subprocess.TimeoutExpired:
                pass
        for process in owned:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        _, alive = psutil.wait_procs(descendants, timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=10)
        require(not [p for p in alive if p.status() != psutil.STATUS_ZOMBIE], 'Owned processes did not shut down')
        child.wait(timeout=10)

    def process(self, name, command, *, hard_deadline=None, monitor=None, controller=True):
        from mibot.utils.claix_checkpoint import atomic_json
        hard = min(hard_deadline or self.deadline-60, self.deadline-60)
        require(hard-time.time() >= 240, 'Insufficient phase budget: ' + name)
        if controller:
            command += ['--qualification-deadline-epoch', str(hard-180)]
        log = self.root/'logs'/name
        log.parent.mkdir(exist_ok=True)
        atomic_json(log.with_suffix('.command.json'), dict(command=command, hard_deadline=hard, started_epoch=time.time()))
        with log.with_suffix('.out').open('x') as out, log.with_suffix('.err').open('x') as err:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=out, stderr=err,
                                     start_new_session=True)
            import psutil
            owned = {}
            try:
                while child.poll() is None:
                    try:
                        for process in psutil.Process(child.pid).children(recursive=True):
                            owned[(process.pid, process.create_time())] = process
                    except psutil.NoSuchProcess:
                        pass
                    require(self.pending_signal is None, 'Allocation stop signal received')
                    require(time.time() < hard, 'Phase deadline exceeded: ' + name)
                    if monitor:
                        monitor(child)
                    time.sleep(.2)
                require(child.returncode == 0, f'{name} exited {child.returncode}; inspect persistent logs')
                _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
                require(not [p for p in alive if p.status() != psutil.STATUS_ZOMBIE],
                        'Child/worker processes remained after controller shutdown')
            finally:
                if child.poll() is None:
                    self.abort(child, controller, hard)
                # Clean up known descendants even if the controller died first.
                for process in owned.values():
                    try:
                        if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                            process.kill()
                    except psutil.NoSuchProcess:
                        pass
                atomic_json(log.with_suffix('.exit.json'), dict(exit_code=child.returncode, ended_epoch=time.time()))

    def target(self, name):
        return self.root/name/'project_xr1_doosan/qualification_b24_w4'

    def train(self, name, *, maximum=6, stop=None, resume=None, monitor=None, label=None, hard=None):
        command = [self.args.python, str(ROOT/'scripts/claix_b24_launch.py'), '--execute',
                   '--mode', 'h100-qualification', '--python', self.args.python, '--workers', '4',
                   '--run-root', str(self.root/name), '--overrides', self.args.overrides,
                   '--qualification-max-steps', str(maximum)]
        if stop is not None:
            command += ['--stop-after-step', str(stop)]
        if resume is not None:
            command += ['--resume', str(resume)]
        self.process(label or name, command, monitor=monitor, hard_deadline=hard)

    def run(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from mibot.utils.claix_checkpoint import atomic_json, file_sha256
        from mibot.utils.claix_contract import audit_splits, resume_fingerprint
        from mibot.utils.doosan_replay import make_contract
        import torch
        import lightning

        def preflight():
            require(torch.__version__ == '2.8.0+cu128' and lightning.__version__ == '2.5.3',
                    'Use the demonstrated CLAIX torch 2.8.0+cu128 / Lightning 2.5.3 environment')
            require(torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0), 'One visible H100 required')
            require(os.environ.get('XR1_DOOSAN_ZIP_CACHE_ROOT') and not os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'),
                    'Approved ZIP cache required')
            require(Path(self.args.python).is_file(), 'Existing environment Python required')
            with initialize_config_dir(config_dir=str(ROOT/'xr1/configs'), version_base=None):
                cfg = OmegaConf.to_container(compose(config_name='claix_b24_production',
                                                    overrides=read_json(self.args.overrides)), resolve=True)
            train = cfg['data']['params']['train_datasets']['paths']
            require(len(train) == 1, 'One frozen training root required')
            inventories = audit_splits(train[0], self.args.validation_root, self.args.test_root)
            self.identities = read_json(self.args.split_identities)
            require(set(self.identities) == {'train', 'validation', 'test'}, 'Expected three frozen split SHA256 identities')
            for split, inv in zip(('train', 'validation', 'test'), inventories):
                require(self.identities[split] == inv['sha256'], 'Frozen split identity mismatch: ' + split)
                atomic_json(self.root/f'{split}-inventory.json', inv)
            self.fingerprint = resume_fingerprint(cfg, inventories[0], 12288)
            self.contract = make_contract(data_horizon=48284, batch_size=24, world_size=1, sampler_seed=42, num_workers=4)
            self.report.update(config_sha256=self.fingerprint, replay_contract=self.contract, split_identities=self.identities,
                               split_identity_file_sha256=file_sha256(self.args.split_identities),
                               overrides_sha256=file_sha256(self.args.overrides),
                               environment=dict(torch=torch.__version__, lightning=lightning.__version__,
                                                gpu=torch.cuda.get_device_name(0), python=sys.version))
            atomic_json(self.root/'path-overrides.json', read_json(self.args.overrides))
            atomic_json(self.root/'frozen-split-identities.json', self.identities)
            paths = sorted([*ROOT.joinpath('xr1').rglob('*.py'), *ROOT.joinpath('xr1/configs').rglob('*.yaml'),
                            *ROOT.joinpath('scripts').glob('claix_b24*'),
                            *ROOT.joinpath('.claix_b24_reference').rglob('*')])
            atomic_json(self.root/'source-and-reference-sha256.json',
                        {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths if p.is_file()})
            return dict(identities=self.identities, config_sha256=self.fingerprint)

        self.gate('preflight', preflight)
        self.gate('clock_probe', clock_probe)

        def checkpoint_run(name, step, **kwargs):
            self.train(name, **kwargs)
            ck = verify_checkpoint(self.target(name), step, self.contract, self.fingerprint)
            verify_shutdown(self.target(name), range(1, step+1),
                            stop_reason='qualification_step' if step == 3 else None)
            return dict(step=ck['global_step'], receipt=read_json((self.target(name)/f'step-{step:06d}.verified.json')))

        self.gate('uninterrupted_6', lambda: checkpoint_run('uninterrupted', 6))
        self.gate('stopped_3', lambda: checkpoint_run('restarted', 3, stop=3))
        self.gate('resumed_6', lambda: checkpoint_run('restarted', 6,
                  resume=self.target('restarted')/'step-000003.ckpt', label='restarted-resume'))

        def equivalence():
            a = verify_checkpoint(self.target('uninterrupted'), 6, self.contract, self.fingerprint)
            b = verify_checkpoint(self.target('restarted'), 6, self.contract, self.fingerprint)
            result = compare_states(a, b)
            atomic_json(self.root/'restart-comparison.json', result)
            require(result['status'] == 'PASS', 'Deterministic restart differences; see restart-comparison.json')
            starts = [r for r in rows_at(self.target('restarted')) if r['event'] == 'train_start']
            require([r['step'] for r in starts] == [0, 3] and starts[1]['replay_offset'] == 72,
                    'Actual resumed replay cursor mismatch')
            return dict(comparison='restart-comparison.json', unequal_comparisons=0)

        self.gate('restart_equivalence', equivalence)

        def sustained():
            sent = {}
            directory = self.target('sustained')
            def monitor(child):
                if sent or not (directory/'step-000250.verified.json').exists():
                    return
                verify_checkpoint(directory, 250, self.contract, self.fingerprint, latest=False)
                require(child.poll() is None, 'Controller ended before signal test')
                sent.update(controller_pid=child.pid, signal=int(signal.SIGUSR1), sent_epoch=time.time(),
                            prerequisite=read_json(directory/'step-000250.verified.json'))
                atomic_json(self.root/'signal-test.json', sent)
                child.send_signal(signal.SIGUSR1)  # Only the controller returned by this Popen.
            # Reserve ten minutes for inference/evidence, in addition to Slurm's 600 s margin.
            self.train('sustained', maximum=260, monitor=monitor, hard=self.deadline-660)
            require(sent, 'Step 250 was not verified and signaled')
            rows = rows_at(directory)
            final = rows[-1]['step']
            ck = verify_checkpoint(directory, final, self.contract, self.fingerprint)
            rows = verify_shutdown(directory, range(1, final+1), stop_reason=f'launcher_signal:{signal.SIGUSR1}')
            requests = [read_json(p) for p in (self.root/'sustained').glob('launch-*.stop.json')]
            require(len(requests) == 1 and requests[0].get('signal') == signal.SIGUSR1
                    and requests[0]['controller_pid'] == sent['controller_pid'], 'Controller stop-request evidence missing')
            require(any(r['event'] == 'checkpoint' and r['step'] == final and r.get('verified')
                        and r.get('reason') == f'launcher_signal:{signal.SIGUSR1}' for r in rows), 'No newly verified stop checkpoint')
            require(ck['xr1_compact_fp32']['trainable_parameter_count'] == 141255168, 'Approved FP32 selection count drift')
            self.final_checkpoint = directory/f'step-{final:06d}.ckpt'
            return sustained_summary(rows)

        self.gate('sustained_and_signal_stop', sustained)

        def inference():
            output = self.root/'validation-smoke.json'
            command = [self.args.python, str(ROOT/'xr1/tools/claix_b24_evaluate.py'),
                       '--overrides', self.args.overrides, '--checkpoint', str(self.final_checkpoint),
                       '--validation-root', self.args.validation_root, '--test-root', self.args.test_root,
                       '--output', str(output), '--limit', str(self.args.smoke_frames)]
            self.process('validation', command, controller=False)
            result = read_json(output)
            validation_gate(result, self.identities, file_sha256(self.final_checkpoint), self.args.smoke_frames)
            require(result['config_sha256'] == self.fingerprint, 'Inference configuration identity mismatch')
            return dict(output=str(output), sha256=file_sha256(output), scope=result['scope'], metrics=result['metrics'])

        self.gate('validation_smoke', inference)
        self.gate('evidence', self.evidence)
        require(tree_digest() == self.source, 'Source changed before final report')
        require(self.pending_signal is None and time.time() < self.deadline,
                'Stop signal or safety deadline reached before final report')

    def evidence(self):
        from mibot.utils.claix_checkpoint import atomic_json, file_sha256
        files = {}
        for path in sorted(self.root.rglob('*')):
            if path.is_file() and path.name not in ('qualification-report.json', 'evidence-sha256.json'):
                files[path.relative_to(self.root).as_posix()] = dict(bytes=path.stat().st_size, sha256=file_sha256(path))
        path = self.root/'evidence-sha256.json'
        atomic_json(path, files)
        return dict(path=str(path), sha256=file_sha256(path), files=len(files))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--run-root', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--overrides', default=str(ROOT/'.claix_b24_reference/configs/hydra_overrides.json'))
    p.add_argument('--validation-root', default=str(STAGED/'dataset_v2/splits/val'))
    p.add_argument('--test-root', default=str(STAGED/'dataset_v2/splits/test'))
    p.add_argument('--split-identities', help='Existing reviewed JSON: train/validation/test inventory SHA256s')
    p.add_argument('--smoke-frames', type=int, choices=range(1, 17), default=4)
    args = p.parse_args(argv)
    for name in ('overrides', 'validation_root', 'test_root', 'split_identities'):
        if getattr(args, name) is not None:
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    print(json.dumps(dict(phases=PHASES, run_root=str(Path(args.run_root).resolve()), workers=4,
                          trainer_max_steps=[6, 6, 6, 260], stop_after_step=3,
                          data_horizon=48284, scheduler_horizon=48284, safety_margin_seconds=600,
                          inference_frames=args.smoke_frames, execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return 0
    require(args.split_identities, '--split-identities is required for execution')
    deadline = allocation_deadline('h100-qualification')
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never attach to or overwrite an old experiment.
    sys.path.insert(0, str(ROOT/'xr1'))
    os.environ.update(MAX_LENGTH='12288', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONPATH=str(ROOT/'xr1') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    report = dict(schema='claix-b24-combined-qualification-v1', status='RUNNING', pinned_reviewed_commit=PINNED,
                  source_tree_sha256=tree_digest(), job_id=os.environ['SLURM_JOB_ID'], safe_deadline_epoch=deadline,
                  started=datetime.now(timezone.utc).isoformat(),
                  phases={phase: dict(status='UNVERIFIED') for phase in PHASES},
                  scope='Hardware engineering qualification and validation smoke; not production release or scientific validation')
    q = Qualification(args, deadline, report)
    old = {sig: signal.signal(sig, q.handle) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        q.save()
        q.run()
        report['status'] = 'PASS'
        return 0
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = dict(type=type(error).__name__, message=str(error))
        active = report.get('active_phase')
        if active:
            report['phases'][active] = dict(status='FAIL', error=report['error'])
        # Hash partial evidence as well; never turn an evidence failure into PASS.
        try:
            q.evidence()
        except Exception as evidence_error:
            report['evidence_error'] = str(evidence_error)
        raise
    finally:
        report['finished'] = datetime.now(timezone.utc).isoformat()
        q.save()
        print(json.dumps(dict(status=report['status'], report=str(root/'qualification-report.json'))), flush=True)
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
