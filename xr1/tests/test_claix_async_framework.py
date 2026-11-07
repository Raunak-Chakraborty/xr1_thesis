"""Required real Lightning/Hydra acceptance; never included in the stub suite."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from lightning import seed_everything
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from mibot.data.datamodule.doosan_datamodule import shutdown_loader
from mibot.utils.claix_async_checkpoint import AsyncProductionRuntime, AsyncVerifiedCheckpointIO
from mibot.utils.claix_checkpoint import load_verified
import test_claix_runtime as legacy


class AsyncFrameworkTest(unittest.TestCase):
    # The existing offline fixture patches only DataModule's rank lookup when
    # SingleDeviceStrategy is used; actual Trainer/capture/hooks/loaders execute.
    setUpClass = classmethod(legacy.RuntimeTest.setUpClass.__func__)
    tearDownClass = classmethod(legacy.RuntimeTest.tearDownClass.__func__)

    def run_tiny(self, root, *, resume=None, stop_after=None):
        seed_everything(42, workers=True)
        with patch.object(legacy, 'ProductionRuntime', AsyncProductionRuntime), \
             patch.object(legacy, 'VerifiedCheckpointIO', side_effect=lambda **_: AsyncVerifiedCheckpointIO()):
            model, trainer = legacy.tiny_run(root, workers=4, horizon=1001, max_steps=1001,
                                             interval=500, resume=resume, stop_after=stop_after)
        self.assertIsNone(trainer.strategy.checkpoint_io._thread)
        loader = trainer.train_dataloader
        shutdown_loader(loader, getattr(loader, '_iterator', None))
        return model, trainer

    def test_real_lightning_500_stop751_resume_rng_and_final_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, _ = self.run_tiny(root/'full')
            _, stopped = self.run_tiny(root/'split', stop_after=751)
            self.assertEqual(stopped.global_step, 751)
            load_verified(root/'split/step-000500.ckpt')
            resumed, trainer = self.run_tiny(root/'split', resume=(root/'split/step-000751.ckpt', 751))
            self.assertEqual(trainer.global_step, 1001)
            self.assertEqual(full.seen_indices[751:], resumed.seen_indices)
            a = load_verified(root/'full/step-001001.ckpt', resumable=False)
            b = load_verified(root/'split/step-001001.ckpt', resumable=False)
            self.assertTrue(torch.equal(a['state_dict']['model.weight'], b['state_dict']['model.weight']))
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(a['optimizer_states'][0]['state'][0][key],
                                            b['optimizer_states'][0]['state'][0][key]))
            self.assertEqual(a['lr_schedulers'], b['lr_schedulers'])
            self.assertEqual(a['xr1_replay'], b['xr1_replay'])
            from claix_b24_checkpoint_two_way import same_state
            self.assertTrue(same_state(a['xr1_production']['rng_by_rank'], b['xr1_production']['rng_by_rank']))
            self.assertEqual(resumed.model.weight.dtype, torch.float32)
            self.assertEqual(resumed.model.frozen.dtype, torch.bfloat16)
            rows = [json.loads(r) for r in (root/'split/telemetry.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows if r['event'] == 'checkpoint'], [500, 751, 1000, 1001])
            self.assertTrue(rows[-1]['final_checkpoint_verified'])

    def test_real_hydra_preserves_entire_scientific_configuration(self):
        directory = Path(__file__).resolve().parents[1]/'configs'
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            a = OmegaConf.to_container(compose(config_name='claix_b24_checkpoint500'), resolve=True)
            b = OmegaConf.to_container(compose(config_name='claix_b24_checkpoint500_async'), resolve=True)
        self.assertEqual(a['data'], b['data']); self.assertEqual(a['model'], b['model'])
        a['trainer'].pop('exp_name'); b['trainer'].pop('exp_name')
        self.assertEqual(a['trainer'], b['trainer'])


class AsyncDDPFrameworkTest(unittest.TestCase):
    def test_real_single_rank_cpu_ddp_checkpoint_io_handoff(self):
        # Requires Gloo loopback sockets. Failure is a blocker, never a mock/pass.
        class TrackingIO(AsyncVerifiedCheckpointIO):
            def __init__(self):
                super().__init__()
                self.save_steps = []
                self.teardown_calls = 0
            def save_checkpoint(self, checkpoint, path, storage_options=None):
                if threading.get_ident() != self._owner:
                    raise AssertionError('Lightning capture left the owning thread')
                self.save_steps.append(checkpoint['global_step'])
                return super().save_checkpoint(checkpoint, path, storage_options)
            def teardown(self):
                self.teardown_calls += 1
                super().teardown()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(legacy, 'ProductionRuntime', AsyncProductionRuntime), \
             patch.object(legacy, 'VerifiedCheckpointIO', side_effect=lambda **_: TrackingIO()), \
             patch.object(legacy, 'SingleDeviceStrategy', side_effect=lambda device, checkpoint_io:
                          DDPStrategy(process_group_backend='gloo', checkpoint_io=checkpoint_io,
                                      find_unused_parameters=True)):
            model, trainer = legacy.tiny_run(folder, workers=4, horizon=1001, max_steps=501, interval=500)
            self.assertEqual(trainer.global_step, 501)
            self.assertIsNone(trainer.strategy.checkpoint_io._thread)
            self.assertEqual(trainer.strategy.checkpoint_io.save_steps, [500, 501])
            self.assertGreaterEqual(trainer.strategy.checkpoint_io.teardown_calls, 1)
            load_verified(Path(folder)/'step-000500.ckpt', model._replay_contract)
            load_verified(Path(folder)/'step-000501.ckpt', model._replay_contract)
            loader = trainer.train_dataloader
            shutdown_loader(loader, getattr(loader, '_iterator', None))


class BenchmarkTerminationFrameworkTest(unittest.TestCase):
    setUpClass = classmethod(legacy.RuntimeTest.setUpClass.__func__)
    tearDownClass = classmethod(legacy.RuntimeTest.tearDownClass.__func__)

    def test_real_sync500_four_worker_resume(self):
        from claix_b24_checkpoint_two_way import same_state
        def run(root, **options):
            seed_everything(42, workers=True)
            model, trainer = legacy.tiny_run(root, workers=4, horizon=1001, max_steps=1001,
                                             interval=500, instrument=True, **options)
            loader = trainer.train_dataloader
            shutdown_loader(loader, getattr(loader, '_iterator', None))
            return model, trainer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, _ = run(root/'full')
            _, stopped = run(root/'split', stop_after=751)
            self.assertEqual(stopped.global_step, 751)
            resumed, _ = run(root/'split', resume=(root/'split/step-000751.ckpt', 751))
            self.assertEqual(full.seen_indices[751:], resumed.seen_indices)
            a = load_verified(root/'full/step-001001.ckpt', resumable=False)
            b = load_verified(root/'split/step-001001.ckpt', resumable=False)
            for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'xr1_replay'):
                self.assertTrue(same_state(a[key], b[key]), key)
            self.assertTrue(same_state(a['xr1_production']['rng_by_rank'], b['xr1_production']['rng_by_rank']))

    def test_real_sync_async_1500_final_state(self):
        from claix_b24_checkpoint_two_way import same_state
        finals = []
        with tempfile.TemporaryDirectory() as folder:
            for asynchronous in (False, True):
                seed_everything(42, workers=True)
                root = Path(folder)/('async' if asynchronous else 'sync')
                runtime = AsyncProductionRuntime if asynchronous else legacy.ProductionRuntime
                original_io = legacy.VerifiedCheckpointIO
                def io(**kwargs):
                    return AsyncVerifiedCheckpointIO() if asynchronous else original_io(**kwargs)
                with patch.object(legacy, 'ProductionRuntime', runtime), \
                     patch.object(legacy, 'VerifiedCheckpointIO', side_effect=io):
                    model, trainer = legacy.tiny_run(root, workers=4, horizon=1601, max_steps=1601,
                                                     interval=500, stop_after=1500, instrument=True)
                try:
                    self.assertEqual(trainer.global_step, 1500)
                    rows = [json.loads(r) for r in (root/'telemetry.jsonl').read_text().splitlines()]
                    self.assertEqual([r['step'] for r in rows if r['event'] == 'checkpoint'], [500, 1000, 1500])
                    self.assertTrue(rows[-1]['final_checkpoint_verified'])
                    finals.append(load_verified(root/'step-001500.ckpt', model._replay_contract))
                    if asynchronous:
                        self.assertIsNone(trainer.strategy.checkpoint_io._thread)
                        trainer.strategy.checkpoint_io.teardown()  # repeated teardown
                finally:
                    loader = trainer.train_dataloader
                    shutdown_loader(loader, getattr(loader, '_iterator', None))
            for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops', 'xr1_replay'):
                self.assertTrue(same_state(finals[0][key], finals[1][key]), key)
            self.assertTrue(same_state(finals[0]['xr1_production']['rng_by_rank'],
                                       finals[1]['xr1_production']['rng_by_rank']))
