#!/usr/bin/env python3
"""One allocation: isolated exact 6/3/6 restart plus 260-step retention/logging. Dry-run by default."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from claix_b24_launch import ROOT, allocation_deadline, tree_digest, write_provenance
from claix_b24_combined_qualification import (
    Qualification, compare_states, sustained_summary, verify_checkpoint, verify_shutdown,
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]


def verify(root, fingerprint, stop_step):
    sys.path.insert(0, str(ROOT/'xr1'))
    from mibot.utils.claix_checkpoint import load_verified
    target = root/'project_xr1_doosan/qualification_b24_w4'
    rows = events(target/'telemetry.jsonl')
    steps = [row for row in rows if row['event'] == 'step']
    require([row['step'] for row in steps] == list(range(1, 261)), 'Incomplete/repeated resumed optimizer sequence')
    import math
    for row in steps:
        require(row['retained_examples'] == 24 and row['views'] == 48
                and row['samples_consumed'] == row['step']*24
                and 0 < row['packed_tokens'] <= 12288
                and math.isfinite(row['learning_rate'])
                and all(key in row for key in ('timestamp', 'pipeline_wait_seconds',
                    'step_seconds', 'end_to_end_examples_per_second', 'gpu_peak_allocated_bytes')),
                'Incomplete optimizer-step telemetry')
    losses = [row for path in sorted((target/'metrics').glob('version_*/metrics.csv'))
              for row in csv.DictReader(path.open()) if row.get('train/loss')]
    require(len(list((target/'metrics').glob('version_*/metrics.csv'))) == 2, 'CSV resume version lost')
    require([int(row['step'])+1 for row in losses] == list(range(1, 261)), 'CSV step loss/overwrite')
    for csv_row, event in zip(losses, steps):
        for key in ('train/loss', 'train/loss_freq', 'train/loss_l1', 'train/loss_mse',
                    'train/loss_score', 'train/token', 'lr'):
            require(math.isfinite(float(csv_row[key])), 'Missing/nonfinite CSV metric: '+key)
        require(math.isclose(float(csv_row['train/loss']), event['loss'], rel_tol=1e-6, abs_tol=1e-8), 'CSV/telemetry loss mismatch')
    stops = [row for row in rows if row['event'] == 'graceful_stop']
    require(any(row['step'] == stop_step and 'signal:' in row['reason'] for row in stops), 'No signal safe-stop evidence')
    contract = json.loads((target/'resume_contract.json').read_text())
    receipts = []
    for path in sorted(target.glob('step-*.ckpt')):
        ck = load_verified(path, contract['replay'], expected_config=contract['config_sha256'], resumable=False)
        receipts.append(dict(path=str(path), step=int(ck['global_step'])))
        del ck
    provenance = list(root.glob('launch-*.provenance.json'))
    require(len(provenance) == 2, 'Missing per-invocation provenance')
    for path in provenance:
        require(json.loads(path.read_text())['source_tree_sha256'] == fingerprint, 'Source changed during qualification')
    monitor_logs = list(root.glob('launch-*.gpu.jsonl'))
    require(len(monitor_logs) == 2 and all(any(row['event'] == 'sample' for row in events(path))
            for path in monitor_logs), 'Missing monitor samples across resume')
    samples = [row for path in monitor_logs for row in events(path) if row['event'] == 'sample']
    require(samples and all(d.get('uuid', '').startswith('GPU-') for row in samples for d in row['devices']), 'No persistent UUID monitor samples')
    require({stop_step, 260} <= {row['step'] for row in receipts}, 'Missing recovery checkpoints')
    latest = json.loads((target/'latest.json').read_text())
    require(latest['global_step'] == 260 and latest == json.loads((target/'safe-stop.json').read_text()),
            'Final receipt pointers inconsistent')
    checkpoints = [row for row in rows if row['event'] == 'checkpoint']
    require(len(checkpoints) >= 3 and all(row['verified'] and 'rotation' in row for row in checkpoints)
            and any(row['rotation']['deleted'] > 0 for row in checkpoints), 'Missing publication/rotation evidence')
    return dict(status='PASS', source_tree_sha256=fingerprint, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), job_id=os.environ['SLURM_JOB_ID'],
        kind='current_h100', completed_steps=260, resumed_from_step=stop_step,
        checkpoint_events=checkpoints, retained_receipts_verified=receipts,
        monitor_samples=len(samples), exact_resumed_sequence=True, scope='logging_retention_only')



def compare_exact(left, right):
    """Reuse the qualified training-state comparator; add all saved rank RNGs."""
    import numpy as np
    import torch
    result = compare_states(left, right)
    rows = result['comparisons']

    def rng(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            equal = set(a) == set(b)
            for key in sorted(set(a) & set(b)):
                rng(a[key], b[key], path+'/'+str(key))
        elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            equal = len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                rng(x, y, path+'/'+str(i))
        else:
            equal = type(a) is type(b) and a == b
        rows.append(dict(path=path, equal=bool(equal)))

    require('rng_by_rank' in left['xr1_production'] and 'rng_by_rank' in right['xr1_production'],
            'Missing required RNG state')
    rng(left['xr1_production']['rng_by_rank'], right['xr1_production']['rng_by_rank'], 'rng_by_rank')
    result['unequal_comparisons'] = sum(not row['equal'] for row in rows)
    result['status'] = 'FAIL' if result['unequal_comparisons'] else 'PASS'
    return result


class Targeted(Qualification):
    """Reuse bounded process/signal cleanup without running the combined suite."""
    def save(self):
        from mibot.utils.claix_checkpoint import atomic_json
        atomic_json(self.root/'targeted-report.json', self.report)

    def target(self, name):
        return self.root/name/'project_xr1_doosan/qualification_b24_w4'

    def train(self, name, *, maximum=6, stop=None, resume=None, monitor=None, label=None):
        command = [self.args.python, str(ROOT/'scripts/claix_b24_launch.py'), '--execute',
                   '--mode', 'h100-qualification', '--python', self.args.python, '--workers', '4',
                   '--run-root', str(self.root/name), '--overrides', self.args.overrides,
                   '--qualification-max-steps', str(maximum)]
        if name == 'logging-retention':
            command += ['--qualification-gpu-monitor']
        if stop is not None:
            command += ['--stop-after-step', str(stop)]
        if resume is not None:
            command += ['--resume', str(resume)]
        self.process(label or name.replace('/', '-'), command, monitor=monitor)

    def checked(self, name, step):
        contract = json.loads((self.target(name)/'resume_contract.json').read_text())
        ck = verify_checkpoint(self.target(name), step, contract['replay'], contract['config_sha256'])
        require(contract['replay']['num_workers'] == 4, 'Replay workers changed')
        return ck

    def run_exact(self):
        from mibot.utils.claix_checkpoint import atomic_json
        reference, restarted = 'exact-restart/uninterrupted', 'exact-restart/restarted'
        self.train(reference)
        a = self.checked(reference, 6)
        config = a['xr1_production']['config_sha256']
        del a
        verify_shutdown(self.target(reference), range(1, 7))
        self.train(restarted, stop=3)
        b = self.checked(restarted, 3)
        require(b['xr1_production']['config_sha256'] == config, 'Exact-run configuration changed')
        del b
        verify_shutdown(self.target(restarted), range(1, 4), stop_reason='qualification_step')
        self.train(restarted, resume=self.target(restarted)/'step-000003.ckpt', label='exact-resumed')
        verify_shutdown(self.target(restarted), range(1, 7))
        starts = [row for row in events(self.target(restarted)/'telemetry.jsonl') if row['event'] == 'train_start']
        require([row['step'] for row in starts] == [0, 3] and starts[1]['replay_offset'] == 72,
                'Actual compact resume replay cursor mismatch')
        a, b = self.checked(reference, 6), self.checked(restarted, 6)
        comparison = compare_exact(a, b)
        comparison.update(source_tree_sha256=self.source, config_sha256=config,
                          job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(self.root/'exact-restart/restart-comparison.json', comparison)
        require(comparison['status'] == 'PASS', 'Exact H100 restart mismatch; see restart-comparison.json')
        require(a['xr1_production']['config_sha256'] == config, 'Exact-run configuration changed')
        # Each invocation must be this exact source, independent of checkpoint metadata.
        provenance = list((self.root/'exact-restart').rglob('launch-*.provenance.json'))
        require(len(provenance) == 3 and all(json.loads(path.read_text())['source_tree_sha256'] == self.source
                                           for path in provenance), 'Missing or changed exact-run source provenance')
        return dict(status='PASS', comparison='exact-restart/restart-comparison.json',
                    config_sha256=config, unequal_comparisons=0, completed_training_steps=12)

    def run_logging(self):
        sent = []
        name = 'logging-retention'
        target = self.target(name)
        def monitor(child):
            if not sent and any(row['event'] == 'checkpoint' and row['step'] == 250
                                and row.get('verified') and row.get('reason') == 'interval'
                                for row in events(target/'telemetry.jsonl')):
                child.send_signal(signal.SIGUSR1)
                sent.append(True)
        self.train(name, maximum=260, monitor=monitor)
        require(sent, 'First phase did not reach signal checkpoint')
        receipt = json.loads((target/'safe-stop.json').read_text())
        step = receipt['global_step']
        require(250 <= step < 260, 'Signal did not stop before final qualification boundary')
        self.checked(name, step)
        rows = verify_shutdown(target, range(1, step+1), stop_reason=f'launcher_signal:{signal.SIGUSR1}')
        sustained = sustained_summary(rows)
        self.train(name, maximum=260, resume=target/receipt['path'], label='logging-resumed')
        verify_shutdown(target, range(1, 261))
        result = verify(self.root/name, self.source, step)
        result['sustained'] = sustained
        config = json.loads((target/'resume_contract.json').read_text())['config_sha256']
        require(config == self.report['phases']['exact_restart']['result']['config_sha256'],
                'Logging/retention and exact-restart configuration fingerprints differ')
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--run-root', required=True)
    p.add_argument('--overrides', default=str(ROOT/'.claix_b24_reference/configs/hydra_overrides.json'))
    args = p.parse_args(argv)
    root = Path(args.run_root).resolve()
    print(json.dumps(dict(source_tree_sha256=tree_digest(), run_root=str(root),
                         phases=[dict(run='exact-restart/uninterrupted', maximum=6),
                                 dict(run='exact-restart/restarted', maximum=6, stop_after=3),
                                 dict(run='exact-restart/restarted', maximum=6, resume_step=3),
                                 dict(run='logging-retention', maximum=260, signal_at_completed_step=250),
                                 dict(run='logging-retention', maximum=260, resume='verified signal checkpoint')],
                         total_optimizer_steps=272, workers=4, allocations=1,
                         allocation_max_minutes=45, safe_stop_margin_seconds=600,
                         runtime_guarantee='UNVERIFIED; abort on insufficient measured allocation budget'), indent=2), flush=True)
    if not args.execute:
        return 0
    require(not root.exists(), 'Use a new targeted root; preserve earlier experiments')
    deadline = allocation_deadline('h100-qualification', max_seconds=2700)
    sys.path.insert(0, str(ROOT/'xr1'))
    import torch
    require(torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0), 'One visible real H100 required')
    root.mkdir(parents=True)
    fingerprint = tree_digest()
    write_provenance(root/'targeted', dict(run_root=str(root), mode='targeted-h100'))
    report = dict(status='FAIL', source_tree_sha256=fingerprint, job_id=os.environ['SLURM_JOB_ID'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  kind='current_h100', phases={}, exact_tensor_equivalence='UNVERIFIED')
    q = Targeted(args, deadline, report)
    old = {sig: signal.signal(sig, q.handle) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        q.gate('exact_restart', q.run_exact)
        q.gate('logging_retention', q.run_logging)
        require(tree_digest() == fingerprint and q.pending_signal is None and time.time() < deadline,
                'Source drift, stop signal or allocation safety deadline before final evidence')
        report.update(status='PASS', exact_tensor_equivalence='PASS', completed_optimizer_steps=272)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        q.save()
        # Stream hashes; never materialize a large log in controller memory.
        from mibot.utils.claix_checkpoint import file_sha256, atomic_json
        artifacts = {str(path.relative_to(root)): file_sha256(path)
                     for path in root.rglob('*') if path.is_file() and path.suffix != '.ckpt'
                     and path.name != 'targeted-artifacts.json'}
        atomic_json(root/'targeted-artifacts.json', artifacts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
