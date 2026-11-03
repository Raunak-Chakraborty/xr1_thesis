"""Short real multiprocessing fault tests. No GPU/model/downloads/cluster access."""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'xr1'))

import torch
from mibot.utils.loader_diagnostics import DiagnosticDataLoader
from mibot.utils.loader_trace import stage


class Dataset:
    def __init__(self, mode='normal'):
        self.mode = mode

    def __len__(self):
        return 16

    def __getitem__(self, index):
        with stage('sample', dataset_index=index, episode='/logical/episode.json', frame=index):
            if index == 0 and self.mode in ('json_load', 'zip_member_read', 'png_decode', 'augmentation'):
                with stage(self.mode):
                    time.sleep(5)
            if index == 0 and self.mode == 'death':
                time.sleep(2)
                os._exit(23)
            if self.mode == 'pickle_failure':
                return BrokenResult()
            if self.mode == 'pickle_delay' and index == 0:
                return SlowResult()
            if self.mode == 'receive_delay' and index == 0:
                return SlowReceive()
            return torch.tensor([index], dtype=torch.int64)


class BrokenResult:
    def __reduce__(self):
        raise RuntimeError('injected feeder serialization failure')


class SlowResult:
    def __reduce__(self):
        time.sleep(5)
        return int, (0,)


def slow_receive():
    time.sleep(5)
    return 0


class SlowReceive:
    def __reduce__(self):
        return slow_receive, ()


def slow_send_init(worker_id):
    import multiprocessing.connection
    import threading
    original = multiprocessing.connection.Connection._send_bytes
    def send(connection, payload):
        if worker_id == 0 and threading.current_thread().name == 'QueueFeederThread':
            time.sleep(5)
        return original(connection, payload)
    multiprocessing.connection.Connection._send_bytes = send


def collate(items):
    return items


def slow_collate(items):
    if items[0].item() == 0:
        time.sleep(5)
    return items


class DiagnosticTests(unittest.TestCase):
    def run_case(self, mode='normal', method='fork', collator=collate, timeout=10, init=None, batches=8):
        folder = tempfile.mkdtemp(prefix='diagnostics-', dir=ROOT/'reliability/evidence/tmp')
        self.addCleanup(lambda: None)  # Retain all engineering evidence intentionally.
        loader = DiagnosticDataLoader(Dataset(mode), batch_size=1, num_workers=2,
            prefetch_factor=2, persistent_workers=True, pin_memory=False, in_order=True,
            multiprocessing_context=method, timeout=timeout, collate_fn=collator,
            worker_init_fn=init,
            generator=torch.Generator().manual_seed(42), diagnostics=dict(
                local_dir=folder, persistent_dir=str(Path(folder)/'persistent'),
                thresholds=(.5, 1.5, 3), poll_seconds=.1, reserve_sigusr2=True))
        iterator = None
        results, error = [], None
        started = time.monotonic()
        try:
            iterator = iter(loader)
            for _ in range(batches):
                results.append(next(iterator))
        except Exception as exc:
            error = exc
        finally:
            if iterator is not None:
                iterator._shutdown_workers()
            loader.close_diagnostics()
        self.assertIsNotNone(loader._diagnostic_session, repr(error))
        reports = [json.loads(p.read_text()) for p in loader._diagnostic_session.local.glob('incident-*.json')]
        self.assertTrue(reports, folder)
        return results, error, reports, time.monotonic() - started

    def test_order_and_tensor_transport_fork_and_spawn(self):
        for method in ('fork', 'spawn'):
            with self.subTest(method=method):
                results, error, reports, _ = self.run_case(method=method)
                self.assertIsNone(error)
                self.assertEqual([v[0].item() for v in results], list(range(8)))
                events = [row for report in reports for ring in report['rings'] for row in ring]
                self.assertTrue(any(row['event'] == 'ipc_sent' for row in events))
                self.assertTrue(any(row['event'] == 'parent_received' for row in events))
                self.assertTrue(any(row['event'] == 'ordered_delivered' for row in events))

    def test_stage_delays_and_head_of_line_state(self):
        for mode in ('json_load', 'zip_member_read', 'png_decode', 'augmentation'):
            with self.subTest(mode=mode):
                results, error, reports, _ = self.run_case(mode)
                self.assertIsNone(error)
                captures = [r for r in reports if r['reason'].startswith('parent_wait')]
                self.assertTrue(any(ring and ring[-1]['stage'] == mode
                                    for r in captures for ring in r['rings']))
                self.assertTrue(any('test_diagnostics.py' in stack
                                    for r in captures for stack in r['python_stacks'].values()))
                self.assertTrue(any(row.get('buffered_tasks') for r in captures for ring in r['rings'] for row in ring))

    def test_collate_delay(self):
        _, error, reports, _ = self.run_case(collator=slow_collate)
        self.assertIsNone(error)
        self.assertTrue(any(ring and ring[-1]['stage'] == 'collate' for r in reports for ring in r['rings']))

    def test_feeder_delay_and_serialization_failure(self):
        for mode, expected in (('pickle_delay', 'ipc_serialize'), ('pickle_failure', 'ipc_error')):
            with self.subTest(mode=mode):
                _, error, reports, _ = self.run_case(mode, timeout=7)
                if mode == 'pickle_failure':
                    self.assertIn('timed out', str(error))
                else:
                    self.assertIsNone(error)
                self.assertTrue(any(ring and ring[-1]['stage'] == expected
                                    for r in reports for ring in r['rings']))

    def test_worker_exit_preserves_failure(self):
        _, error, reports, _ = self.run_case('death')
        self.assertIsNotNone(error)
        self.assertIn('exited unexpectedly', str(error))
        self.assertTrue(any(p.get('state') == 'exited_or_reused' for r in reports for p in r['processes']))

    def test_pipe_send_and_parent_reconstruction_delays(self):
        for mode, init, expected in (('normal', slow_send_init, 'ipc_pipe_send'),
                                     ('receive_delay', None, 'parent_ipc_receive')):
            with self.subTest(mode=mode):
                _, error, reports, _ = self.run_case(mode, init=init, batches=1)
                self.assertIsNone(error)
                self.assertTrue(any(ring and ring[-1]['stage'] == expected
                                    for r in reports for ring in r['rings']))

    def test_handler_conflict_is_rejected(self):
        previous = signal.signal(signal.SIGUSR2, lambda sig, frame: None)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'reliability/evidence/tmp') as folder:
                loader = DiagnosticDataLoader(Dataset(), num_workers=1, in_order=True, timeout=10,
                    diagnostics=dict(local_dir=folder, thresholds=(1,), reserve_sigusr2=True))
                with self.assertRaisesRegex(RuntimeError, 'SIGUSR2'):
                    iter(loader)
        finally:
            signal.signal(signal.SIGUSR2, previous)

    def test_dead_watchdog_and_full_stack_socket_cannot_block_loader(self):
        folder = tempfile.mkdtemp(prefix='watchdog-death-', dir=ROOT/'reliability/evidence/tmp')
        loader = DiagnosticDataLoader(Dataset(), batch_size=1, num_workers=2, timeout=10,
            persistent_workers=True, in_order=True, multiprocessing_context='fork',
            diagnostics=dict(local_dir=folder, thresholds=(1,), poll_seconds=.1, reserve_sigusr2=True))
        iterator = iter(loader)
        session = loader._diagnostic_session
        try:
            session.process.terminate()
            session.process.join(timeout=2)
            start = time.monotonic()
            # C faulthandler writes must return EAGAIN once the socket is full.
            for _ in range(250):
                os.kill(os.getpid(), signal.SIGUSR2)
            result = [next(iterator).item() for _ in range(16)]
            self.assertEqual(result, list(range(16)))
            self.assertLess(time.monotonic()-start, 5)
            self.assertLessEqual(len(session.shared.read(0)), 64)
        finally:
            iterator._shutdown_workers()
            loader.close_diagnostics()

    def test_usr1_handler_unchanged(self):
        seen = []
        previous = signal.signal(signal.SIGUSR1, lambda sig, frame: seen.append(sig))
        try:
            _, error, _, _ = self.run_case('zip_member_read')
            self.assertIsNone(error)
            self.assertEqual(seen, [])
            os.kill(os.getpid(), signal.SIGUSR1)
            self.assertEqual(seen, [signal.SIGUSR1])
        finally:
            signal.signal(signal.SIGUSR1, previous)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
