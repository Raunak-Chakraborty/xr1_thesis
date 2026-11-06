"""Real torch/thread/filesystem tests; explicit synthetic Lightning adapter."""
from collections import OrderedDict
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from mibot.utils import claix_checkpoint as tx
from mibot.utils.claix_async_checkpoint import owned_cpu_snapshot, AsyncVerifiedCheckpointIO, AsyncProductionRuntime
from mibot.utils.claix_experiment import ASYNC500_NAME, load_experimental_resume
from test_claix_checkpoint import checkpoint_fixture


def identity(root):
    return dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                run_id='c'*32, run_root=str(root), zip_cache_root='/hpcwork/SYNTHETIC/cache',
                zip_manifest_sha256='b'*64, checkpoint_interval=500,
                experiment_name=ASYNC500_NAME, checkpoint_writer='async_thread_v1')


def payload(root, step):
    ck = checkpoint_fixture(step, 48284)
    ck['xr1_production']['experiment'] = identity(root)
    return ck


class AsyncCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.writer = AsyncVerifiedCheckpointIO(expected_config='a'*64, expected_experiment=identity(self.root))
        self.addCleanup(self.cleanup_writer)

    def cleanup_writer(self):
        try:
            self.writer.close()
        except RuntimeError:
            pass  # each fault test asserts sticky propagation explicitly
        self.assertFalse(any(t.name == 'xr1-checkpoint-writer' and t.is_alive() for t in threading.enumerate()))

    def submit(self, step, reason='interval', ck=None):
        self.writer.arm(reason, time.perf_counter())
        self.writer.save_checkpoint(ck if ck is not None else payload(self.root, step),
                                    self.root/f'step-{step:06d}.ckpt')

    def gate(self):
        entered, release = threading.Event(), threading.Event()
        original = tx.publish_checkpoint
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test writer gate timed out')
            return original(*args, **kwargs)
        self.addCleanup(release.set)
        return entered, release, blocked

    def test_snapshot_tensor_views_optimizer_and_nested_ownership(self):
        parameter = torch.nn.Parameter(torch.ones(2, 3))
        opt = torch.optim.AdamW([parameter], lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        parameter.sum().backward(); opt.step(); scheduler.step(); opt.zero_grad()
        ck = payload(self.root, 1)
        ck['state_dict']['model.weight'] = parameter.detach().view(2, 3)
        ck['optimizer_states'] = [opt.state_dict()]
        ck['lr_schedulers'] = [scheduler.state_dict()]
        expected = owned_cpu_snapshot(ck)
        self.assertNotEqual(expected['state_dict']['model.weight'].data_ptr(), parameter.data_ptr())
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertNotEqual(expected['optimizer_states'][0]['state'][0][key].data_ptr(),
                                opt.state[parameter][key].data_ptr())
        entered, release, blocked = self.gate()
        with patch.object(tx, 'publish_checkpoint', side_effect=blocked):
            self.submit(1, ck=ck)
            self.assertTrue(entered.wait(2))
            parameter.square().sum().backward(); opt.step(); scheduler.step()
            ck['loops'].clear()
            ck['xr1_production']['rng_by_rank'][0]['numpy'][1].fill(0)
            release.set()
            result = self.writer.collect(wait=True)
        actual = tx.load_verified(result['path'])
        self.assertTrue(torch.equal(actual['state_dict']['model.weight'], expected['state_dict']['model.weight']))
        self.assertFalse(torch.equal(actual['state_dict']['model.weight'], parameter))
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertTrue(torch.equal(actual['optimizer_states'][0]['state'][0][key],
                                        expected['optimizer_states'][0]['state'][0][key]))
        self.assertEqual(actual['lr_schedulers'], expected['lr_schedulers'])
        self.assertEqual(actual['loops'], expected['loops'])
        np.testing.assert_array_equal(actual['xr1_production']['rng_by_rank'][0]['numpy'][1],
                                      expected['xr1_production']['rng_by_rank'][0]['numpy'][1])

    def test_rng_replay_and_optimizer_continuation(self):
        original = tx.rng_state()
        self.addCleanup(tx.restore_rng, original)
        p = torch.nn.Parameter(torch.ones(2, 3))
        opt = torch.optim.AdamW([p], lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        p.sum().backward(); opt.step(); scheduler.step(); opt.zero_grad()
        ck = payload(self.root, 1)
        ck.update(state_dict={'model.weight': p.detach()}, optimizer_states=[opt.state_dict()],
                  lr_schedulers=[scheduler.state_dict()])
        self.submit(1, ck=ck); self.writer.collect(wait=True)
        # Publication/validation must not consume the training RNG stream.
        expected_draws = (random.random(), np.random.random(), torch.rand(2, 3))
        (p*expected_draws[2]).sum().backward(); opt.step(); scheduler.step()
        restored = tx.load_verified(self.root/'step-000001.ckpt')
        tx.restore_rng(restored['xr1_production']['rng_by_rank'][0])
        draws = (random.random(), np.random.random(), torch.rand(2, 3))
        self.assertEqual(draws[:2], expected_draws[:2]); self.assertTrue(torch.equal(draws[2], expected_draws[2]))
        q = torch.nn.Parameter(restored['state_dict']['model.weight'])
        other = torch.optim.AdamW([q], lr=.01)
        other_scheduler = torch.optim.lr_scheduler.StepLR(other, step_size=1, gamma=.9)
        other.load_state_dict(restored['optimizer_states'][0]); other_scheduler.load_state_dict(restored['lr_schedulers'][0])
        (q*draws[2]).sum().backward(); other.step(); other_scheduler.step()
        self.assertTrue(torch.equal(p, q))
        self.assertEqual(scheduler.state_dict(), other_scheduler.state_dict())
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertTrue(torch.equal(opt.state[p][key], other.state[q][key]))
        self.assertEqual(restored['xr1_replay']['rank_sample_offset'], 24)

    def test_ordered_metadata_and_unsupported_objects(self):
        values = OrderedDict(weight=torch.ones(2))
        values._metadata = {'': {'version': 1}}
        staged = owned_cpu_snapshot(values)
        values._metadata['']['version'] = 2
        self.assertEqual(staged._metadata['']['version'], 1)
        for value in (object(), np.array([object()], dtype=object)):
            with self.assertRaises(TypeError): owned_cpu_snapshot(value)

    def test_staging_storage_released_before_retention_reads(self):
        references = []
        original_publish, original_rotate = tx.publish_checkpoint, tx.rotate_checkpoints
        entered, release = threading.Event(), threading.Event()
        def publish(ck, *args, **kwargs):
            references.append(weakref.ref(ck['state_dict']['model.weight']))
            return original_publish(ck, *args, **kwargs)
        def rotate(*args, **kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError('Test rotation gate timed out')
            return original_rotate(*args, **kwargs)
        self.addCleanup(release.set)
        with patch.object(tx, 'publish_checkpoint', new=publish), patch.object(tx, 'rotate_checkpoints', new=rotate):
            self.submit(500)
            self.assertTrue(entered.wait(2))
            self.assertIsNone(references[0]())
            release.set(); self.writer.collect(wait=True)

    def test_single_slot_cannot_capture_second_snapshot_until_ack(self):
        entered, release, blocked = self.gate()
        with patch.object(tx, 'publish_checkpoint', side_effect=blocked):
            self.submit(500)
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.writer._thread.daemon)
            with self.assertRaisesRegex(RuntimeError, 'single checkpoint slot'):
                self.writer.arm('interval', time.perf_counter())
            self.assertIsNone(self.writer.collect())
            self.assertFalse((self.root/'latest.json').exists())
            release.set(); self.writer.collect(wait=True)
        self.submit(1000); self.assertEqual(self.writer.collect(wait=True)['step'], 1000)

    def test_close_drains_joins_and_prohibits_new_work(self):
        entered, release, blocked = self.gate()
        with patch.object(tx, 'publish_checkpoint', side_effect=blocked):
            self.submit(500); self.assertTrue(entered.wait(2))
            thread = self.writer._thread
            timer = threading.Timer(.05, release.set); timer.start()
            self.writer.close(); timer.join()
            self.assertFalse(thread.is_alive())
        self.writer.close(); self.writer.teardown()
        tx.load_verified(self.root/'step-000500.ckpt')
        with self.assertRaises(RuntimeError): self.writer.arm('interval', time.perf_counter())

    def test_boundary_validation_and_single_rank_fail_closed(self):
        for field in ('loops', 'optimizer_states', 'lr_schedulers'):
            ck = payload(self.root, 500); ck.pop(field)
            self.writer.arm('interval', time.perf_counter())
            with self.assertRaises((RuntimeError, KeyError)):
                self.writer.save_checkpoint(ck, self.root/'step-000500.ckpt')
            self.assertIsNone(self.writer._thread)
        with self.assertRaisesRegex(RuntimeError, 'armed'):
            self.writer.save_checkpoint(payload(self.root, 500), self.root/'step-000500.ckpt')

    def failure(self, phase):
        old = self.root/'step-000500.ckpt'
        new = self.root/'step-001500.ckpt'
        tx.publish_checkpoint(payload(self.root, 500), old)
        tx.publish_checkpoint(payload(self.root, 1000), self.root/'step-001000.ckpt')
        prior = (self.root/'latest.json').read_bytes(); old_bytes = old.read_bytes()
        original_load, original_link, original_replace = torch.load, os.link, os.replace
        original_unlink, original_publish = Path.unlink, tx.publish_checkpoint
        def partial(ck, stream):
            stream.write(b'interrupted bytes'); raise OSError('injected serialization')
        def readback(*args, **kwargs):
            ck = original_load(*args, **kwargs)
            if str(args[0]).endswith('.tmp'):
                ck['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] += 1
            return ck
        def link(src, dst, *args, **kwargs):
            if Path(dst) == new.with_suffix('.verified.json'): raise OSError('injected receipt')
            return original_link(src, dst, *args, **kwargs)
        def replace(src, dst, *args, **kwargs):
            if Path(dst).name == ('safe-stop.json' if phase == 'safe-stop' else 'latest.json'):
                raise OSError('injected pointer')
            return original_replace(src, dst, *args, **kwargs)
        def unlink(path, *args, **kwargs):
            if path == old: raise OSError('injected retention unlink')
            return original_unlink(path, *args, **kwargs)
        def corrupt(*args, **kwargs):
            receipt = original_publish(*args, **kwargs)
            with new.open('r+b') as stream: stream.write(b'bad')
            return receipt
        injections = dict(serialization=lambda: patch.object(torch, 'save', side_effect=partial),
            fsync=lambda: patch.object(tx.os, 'fsync', side_effect=OSError('injected fsync')),
            readback=lambda: patch.object(torch, 'load', side_effect=readback),
            receipt=lambda: patch.object(tx.os, 'link', side_effect=link),
            latest=lambda: patch.object(tx.os, 'replace', side_effect=replace),
            **{'safe-stop': lambda: patch.object(tx.os, 'replace', side_effect=replace)},
            retention=lambda: patch.object(Path, 'unlink', unlink),
            sha256=lambda: patch.object(tx, 'publish_checkpoint', side_effect=corrupt))
        with injections[phase]():
            self.submit(1500, reason='signal' if phase == 'safe-stop' else 'interval')
            with self.assertRaisesRegex(RuntimeError, 'writer failed') as error:
                self.writer.collect(wait=True)
            self.assertIsNotNone(error.exception.__cause__)
            self.assertIsNone(self.writer._result)
        for action in (self.writer.close, lambda: self.writer.arm('interval', time.perf_counter())):
            with self.assertRaisesRegex(RuntimeError, 'no retry or fallback'): action()
        self.assertEqual(old.read_bytes(), old_bytes)
        tx.load_verified(old)
        self.assertFalse(list(self.root.glob('.*.tmp')))
        if phase in ('serialization', 'fsync', 'readback', 'receipt', 'latest'):
            self.assertEqual((self.root/'latest.json').read_bytes(), prior)
        if phase in ('serialization', 'fsync', 'readback', 'receipt'):
            self.assertFalse(new.with_suffix('.verified.json').exists())
        if phase == 'sha256':
            with self.assertRaisesRegex(RuntimeError, 'checksum'): tx.load_verified(new)

    def test_interrupted_serialization(self): self.failure('serialization')
    def test_fsync_failure(self): self.failure('fsync')
    def test_directory_receipt_and_latest_fsync_failures_propagate(self):
        # Calls 2..6 are payload directory, receipt file/directory, latest file/directory.
        # A failure after link/replace can leave visible but not durably acknowledged metadata.
        original = os.fsync
        for failed_call in range(2, 7):
            with self.subTest(fsync_call=failed_call), tempfile.TemporaryDirectory(dir=self.root) as folder:
                root = Path(folder); old = root/'step-000500.ckpt'; new = root/'step-001000.ckpt'
                tx.publish_checkpoint(payload(root, 500), old)
                writer = AsyncVerifiedCheckpointIO(expected_config='a'*64, expected_experiment=identity(root))
                calls = []
                def fsync(fd):
                    calls.append(fd)
                    if len(calls) == failed_call: raise OSError('injected durability boundary')
                    return original(fd)
                with patch.object(tx.os, 'fsync', side_effect=fsync):
                    writer.arm('interval', time.perf_counter()); writer.save_checkpoint(payload(root, 1000), new)
                    with self.assertRaisesRegex(RuntimeError, 'writer failed'): writer.collect(wait=True)
                with self.assertRaisesRegex(RuntimeError, 'writer failed'): writer.close()
                self.assertIsNone(writer._thread); self.assertIsNone(writer._result)
                tx.load_verified(old)
                self.assertFalse(list(root.glob('.*.tmp')))
    def test_readback_validation_failure(self): self.failure('readback')
    def test_receipt_publication_failure(self): self.failure('receipt')
    def test_latest_pointer_failure(self): self.failure('latest')
    def test_safe_stop_pointer_failure(self): self.failure('safe-stop')
    def test_sha256_mismatch(self): self.failure('sha256')
    def test_retention_unlink_failure(self): self.failure('retention')

    def test_historical_pins_and_same_run_recovery_identity(self):
        # Independent output directory follows the actual async recovery contract.
        root = self.root
        target = root/'project_xr1_doosan'/ASYNC500_NAME
        for step in (500, 1000, *sorted(tx.HISTORICAL_STEPS), 40500, 41000):
            self.writer.arm('interval', time.perf_counter())
            self.writer.save_checkpoint(payload(root, step), target/f'step-{step:06d}.ckpt')
            self.writer.collect(wait=True)
        self.assertEqual({int(p.stem.split('-')[1]) for p in target.glob('*.ckpt')},
                         set(tx.HISTORICAL_STEPS) | {40500, 41000})
        path = target/'step-041000.ckpt'
        load_experimental_resume(path, payload(root, 41000)['xr1_replay_contract'], 'a'*64, identity(root))
        for changed in (dict(run_id='d'*32), dict(checkpoint_writer='sync'), dict(run_root=str(root/'other'))):
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                load_experimental_resume(path, payload(root, 41000)['xr1_replay_contract'], 'a'*64,
                                         dict(identity(root), **changed))


class AsyncRuntimeTest(unittest.TestCase):
    setUp = AsyncCheckpointTest.setUp
    cleanup_writer = AsyncCheckpointTest.cleanup_writer
    gate = AsyncCheckpointTest.gate
    def runtime(self):
        module = SimpleNamespace(_replay_contract=payload(self.root, 500)['xr1_replay_contract'],
            _compact_cfg={'config_sha256': 'a'*64},
            parameters=lambda: [torch.nn.Parameter(torch.ones(1)),
                torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)])
        trainer = SimpleNamespace(global_step=500, world_size=1, is_global_zero=True, max_steps=48284,
            optimizers=[SimpleNamespace(param_groups=[dict(lr=2e-5)])], should_stop=False,
            strategy=SimpleNamespace(checkpoint_io=self.writer, barrier=lambda *_: None))
        def save(path, weights_only):
            self.assertFalse(weights_only)
            ck = payload(self.root, trainer.global_step)
            ck['xr1_production']['rng_by_rank'] = module._checkpoint_rng
            self.writer.save_checkpoint(ck, path)
        trainer.save_checkpoint = save
        runtime = AsyncProductionRuntime(self.root, deadline=time.time()+3600)
        runtime.started = runtime.fit_started = runtime.previous_end = time.perf_counter()-.1
        runtime.initial_step = 0; runtime.previous_step = 499
        runtime.retained, runtime.views, runtime.tokens = 24, 48, 100
        runtime.data_pipeline_seconds = .01
        return runtime, trainer, module

    def test_ack_uses_snapshot_step_not_later_trainer_step(self):
        runtime, trainer, module = self.runtime()
        entered, release, blocked = self.gate()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(tx, 'publish_checkpoint', side_effect=blocked):
            runtime.save(trainer, module, 'interval'); self.assertTrue(entered.wait(2))
            self.assertEqual(runtime.last_saved, -1)
            trainer.global_step = 501; release.set(); runtime._collect(trainer, wait=True)
        rows = [json.loads(r) for r in (self.root/'telemetry.jsonl').read_text().splitlines()]
        self.assertFalse(rows[0]['verified'])
        self.assertEqual(rows[-1]['step'], 500)
        self.assertEqual(rows[-1]['trainer_step_at_ack'], 501)
        self.assertTrue(all(v >= 0 for v in rows[-1]['timings'].values()))

    def stop_during_write(self, terminal):
        runtime, trainer, module = self.runtime()
        entered, release, blocked = self.gate()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(tx, 'publish_checkpoint', side_effect=blocked):
            runtime.save(trainer, module, 'interval'); self.assertTrue(entered.wait(2))
            trainer.global_step = 48284 if terminal else 751
            runtime.previous_step = trainer.global_step-1
            if not terminal: runtime.stop.handle(signal.SIGUSR1, None)
            # While main waits for the existing slot, no second capture can occur.
            timer = threading.Timer(.05, release.set); timer.start()
            runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0); timer.join()
            self.assertEqual(runtime.last_saved, trainer.global_step)
            self.assertEqual(trainer.should_stop, not terminal)
            self.assertEqual(json.loads((self.root/'safe-stop.json').read_text())['global_step'], trainer.global_step)
            runtime.on_fit_end(trainer, module)
        tx.load_verified(self.root/f'step-{trainer.global_step:06d}.ckpt', resumable=False)
        self.assertIsNone(self.writer._thread)
        rows = [json.loads(r) for r in (self.root/'telemetry.jsonl').read_text().splitlines()]
        staged = [r for r in rows if r['event'] == 'checkpoint_staged']
        self.assertGreater(staged[-1]['backpressure_seconds'], .02)
        self.assertTrue(rows[-1]['final_checkpoint_verified'])

    def test_safe_stop_during_writing(self): self.stop_during_write(False)
    def test_terminal_completion_during_writing(self): self.stop_during_write(True)

    def test_exception_cleanup_drains_writer_and_restores_handlers(self):
        runtime, trainer, module = self.runtime()
        runtime.stop.install()
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.save(trainer, module, 'interval')
            runtime.on_exception(trainer, module, ValueError('training failed'))
        self.assertIsNone(self.writer._thread)
        self.assertEqual(runtime.stop.handlers, {})

    def test_writer_error_propagates_at_next_batch_before_optimizer(self):
        runtime, trainer, module = self.runtime()
        with patch.object(torch, 'save', side_effect=OSError('disk full')), contextlib.redirect_stdout(io.StringIO()):
            runtime.save(trainer, module, 'interval')
            self.writer._thread.join()
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                runtime.on_train_batch_start(trainer, module, None, 0)
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                runtime.on_exception(trainer, module, ValueError('training failed'))
        self.assertEqual(runtime.last_saved, -1)
        self.assertIsNone(self.writer._thread)

    def test_exception_cleanup_restores_signals_even_when_fatal_telemetry_fails(self):
        runtime, trainer, module = self.runtime(); runtime.stop.install()
        with contextlib.redirect_stdout(io.StringIO()): runtime.save(trainer, module, 'interval')
        with patch.object(runtime, 'emit', side_effect=OSError('telemetry full')):
            with self.assertRaisesRegex(OSError, 'telemetry full'):
                runtime.on_exception(trainer, module, ValueError('training failed'))
        self.assertIsNone(self.writer._thread)
        self.assertEqual(runtime.stop.handlers, {})

    def test_final_verification_rejects_missing_or_corrupt_terminal(self):
        runtime, trainer, module = self.runtime()
        with self.assertRaisesRegex(RuntimeError, 'without a final boundary'):
            runtime.on_fit_end(trainer, module)

    def test_terminal_bytes_reverified_after_writer_ack(self):
        runtime, trainer, module = self.runtime()
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.save(trainer, module, 'natural_finish')
            with (self.root/'step-000500.ckpt').open('r+b') as stream: stream.write(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'checksum'): runtime.on_fit_end(trainer, module)
        self.assertIsNone(self.writer._thread)

    def test_multi_rank_is_rejected_before_capture(self):
        runtime, trainer, module = self.runtime(); trainer.world_size = 2
        with self.assertRaisesRegex(RuntimeError, 'single-rank'): runtime.on_fit_start(trainer, module)
