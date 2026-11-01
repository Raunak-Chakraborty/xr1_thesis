"""CPU observability/retention checks. No real nvidia-smi, Slurm or backbone."""
import csv
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.loggers import CSVLogger

from mibot.utils.claix_checkpoint import (
    atomic_json, file_sha256, HISTORICAL_STEPS, load_verified, publish_checkpoint, rotate_checkpoints,
)
from test_claix_checkpoint import checkpoint_fixture
from test_claix_launch import launch, ROOT
from test_claix_runtime import tiny_run, TinyNetwork


def publish(folder, step):
    path = Path(folder) / f'step-{step:06d}.ckpt'
    receipt = publish_checkpoint(checkpoint_fixture(step, horizon=48284), path)
    return path, receipt


class RetentionReleaseTest(unittest.TestCase):
    def test_milestones_two_recovery_stop_and_all_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            steps = [250, 500, 2000, 5000, 5250, 10000, 20000, 30000, 40000, 40251, 40500, 40750, 41000]
            for step in steps:
                path, receipt = publish(folder, step)
                if step == 40251:
                    atomic_json(Path(folder)/'safe-stop.json', receipt)
                rotate_checkpoints(folder)
            expected = HISTORICAL_STEPS | {40251, 40750, 41000}
            self.assertEqual({int(p.stem.split('-')[1]) for p in Path(folder).glob('*.ckpt')}, expected)
            self.assertEqual(len(list(Path(folder).glob('step-*.verified.json'))), len(steps))
            latest = json.loads((Path(folder)/'latest.json').read_text())
            self.assertEqual(latest, receipt)
            self.assertEqual(latest['sha256'], file_sha256(path))
            # A new final stop supersedes the old stop pin; normal rotation can
            # then reclaim that old payload while its receipt stays immutable.
            path, receipt = publish(folder, 48284)
            atomic_json(Path(folder)/'safe-stop.json', receipt)
            rotate_checkpoints(folder)
            self.assertFalse((Path(folder)/'step-040251.ckpt').exists())
            self.assertTrue((Path(folder)/'step-040251.verified.json').exists())
            self.assertEqual(json.loads((Path(folder)/'latest.json').read_text()), receipt)
            load_verified(path, checkpoint_fixture(48284, 48284)['xr1_replay_contract'], resumable=False)

    def test_latest_milestone_counts_as_one_of_two_recovery_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            for step in (4250, 4500, 4750, 5000):
                publish(folder, step)
            rotate_checkpoints(folder)
            self.assertEqual(sorted(p.name for p in Path(folder).glob('*.ckpt')),
                             ['step-004750.ckpt', 'step-005000.ckpt'])

    def test_corrupt_recovery_candidate_cannot_displace_a_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            for step in (250, 500, 750):
                publish(folder, step)
            damaged = Path(folder)/'step-000500.ckpt'
            with damaged.open('r+b') as stream:
                stream.write(b'corrupt')
            rotate_checkpoints(folder)
            self.assertEqual(len(list(Path(folder).glob('*.ckpt'))), 3)
            self.assertTrue((Path(folder)/'step-000250.ckpt').exists())

    def test_unverified_unrelated_corrupt_and_symlink_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            external = Path(outside)/'unrelated.ckpt'
            external.write_bytes(b'unrelated outside experiment')
            orphan = root/'step-000001.ckpt'
            orphan.write_bytes(b'no receipt')
            unrelated = root/'notes.ckpt'
            unrelated.write_bytes(b'unrelated inside experiment')
            malformed = root/'step-000002.verified.json'
            malformed.write_text('{truncated')
            traversal = root/'step-000003.verified.json'
            traversal.write_text(json.dumps(dict(global_step=3, path=str(external), schema='claix-b24-verified-v1')))
            # A receipt/hash for structurally invalid checkpoint data must not
            # authorize deletion, even with the current config fingerprint.
            bad, receipt = publish(folder, 4)
            ck = torch.load(bad, weights_only=False)
            ck['optimizer_states'] = []
            torch.save(ck, bad)
            receipt.update(bytes=bad.stat().st_size, sha256=file_sha256(bad))
            atomic_json(bad.with_suffix('.verified.json'), receipt)
            corrupted, _ = publish(folder, 5)
            with corrupted.open('r+b') as stream:
                stream.write(b'broken')  # same size, invalid checksum
            foreign, receipt = publish(folder, 6)
            receipt['config_sha256'] = 'b'*64
            atomic_json(foreign.with_suffix('.verified.json'), receipt)
            symlink, _ = publish(folder, 7)
            symlink.unlink()
            symlink.symlink_to(external)
            linked_receipt, _ = publish(folder, 8)
            receipt_path = linked_receipt.with_suffix('.verified.json')
            receipt_copy = Path(outside)/'copied.verified.json'
            receipt_copy.write_bytes(receipt_path.read_bytes())
            receipt_path.unlink()
            receipt_path.symlink_to(receipt_copy)
            for step in (250, 500, 750):
                publish(folder, step)
            protected = [orphan, unrelated, malformed, traversal, bad, corrupted, foreign,
                         symlink, linked_receipt, receipt_path, external, receipt_copy]
            before = {str(p): p.read_bytes() for p in protected}
            rotate_checkpoints(folder)
            self.assertFalse((root/'step-000250.ckpt').exists())
            self.assertEqual(before, {str(p): p.read_bytes() for p in protected})
            self.assertTrue(symlink.is_symlink())
            self.assertTrue(receipt_path.is_symlink())

    def test_invalid_latest_or_stop_pointer_prevents_all_deletion(self):
        for pointer in ('latest.json', 'safe-stop.json'):
            with self.subTest(pointer=pointer), tempfile.TemporaryDirectory() as folder:
                _, old = publish(folder, 250)
                publish(folder, 500)
                publish(folder, 750)
                row = dict(old)
                if pointer == 'safe-stop.json':
                    row['sha256'] = '0'*64
                atomic_json(Path(folder)/pointer, row)
                before = {p.name: p.read_bytes() for p in Path(folder).iterdir()}
                with self.assertRaises(RuntimeError):
                    rotate_checkpoints(folder)
                self.assertEqual(before, {p.name: p.read_bytes() for p in Path(folder).iterdir()})


class MetricsReleaseTest(unittest.TestCase):
    def setUp(self):
        lookup = patch('mibot.data.datamodule.doosan_datamodule.dist',
                       SimpleNamespace(is_available=lambda:True, is_initialized=lambda:True,
                                       get_world_size=lambda:1, get_rank=lambda:0))
        lookup.start()
        self.addCleanup(lookup.stop)

    def test_per_step_csv_across_resume_and_identical_optimizer_results(self):
        with initialize_config_dir(config_dir=str(ROOT/'xr1/configs'), version_base=None):
            cfg = compose(config_name='claix_b24_production')
        self.assertEqual(cfg.trainer.log_every_n_steps, 1)
        original = TinyNetwork.forward
        def components(model, batch, return_loss):
            output = original(model, batch, return_loss)
            # Synthetic components exercise the unchanged real BaseRunner.log
            # path. The tiny model's training loss/backward stays identical.
            for name, factor in (('loss_freq', .1), ('loss_l1', .2), ('loss_mse', .3), ('loss_score', .4)):
                output[name] = output['loss'].detach()*factor
            return output
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference, _ = tiny_run(root/'reference', workers=0)
            expected = load_verified(root/'reference/step-000008.ckpt', reference._replay_contract, resumable=False)
            run = root/'logged'
            with patch.object(TinyNetwork, 'forward', components):
                for resume in (None, (run/'step-000003.ckpt', 3)):
                    logger = CSVLogger(save_dir=str(run), name='metrics')
                    model, _ = tiny_run(run, workers=0, stop_after=3 if resume is None else None,
                                        resume=resume, logger=logger, log_every_n_steps=cfg.trainer.log_every_n_steps)
            paths = sorted((run/'metrics').glob('version_*/metrics.csv'))
            self.assertEqual(len(paths), 2)
            rows = [row for path in paths for row in csv.DictReader(path.open())]
            self.assertEqual([int(row['step']) for row in rows], list(range(8)))
            required = {'step', 'train/loss', 'train/loss_freq', 'train/loss_l1',
                        'train/loss_mse', 'train/loss_score', 'lr', 'train/token'}
            for row in rows:
                self.assertTrue(required <= row.keys())
                self.assertEqual(float(row['train/token']), 20.)
                for key in required:
                    self.assertTrue(torch.isfinite(torch.tensor(float(row[key]))))
            events = [json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
            steps = [r for r in events if r['event'] == 'step']
            self.assertEqual([r['step'] for r in steps], list(range(1,9)))
            for row, event in zip(rows, steps):
                self.assertAlmostEqual(float(row['train/loss']), event['loss'])
            actual = load_verified(run/'step-000008.ckpt', model._replay_contract, resumable=False)
            self.assertTrue(torch.equal(expected['state_dict']['model.weight'], actual['state_dict']['model.weight']))
            self.assertEqual(expected['lr_schedulers'], actual['lr_schedulers'])
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(expected['optimizer_states'][0]['state'][0][key],
                                            actual['optimizer_states'][0]['state'][0][key]))
            self.assertEqual(json.loads((run/'safe-stop.json').read_text()),
                             json.loads((run/'latest.json').read_text()))

    def test_scientific_configuration_is_unchanged(self):
        with initialize_config_dir(config_dir=str(ROOT/'xr1/configs'), version_base=None):
            cfg = compose(config_name='claix_b24_production')
        schedule = cfg.trainer.scheduler.params
        self.assertEqual((schedule.max_lr, schedule.min_lr, schedule.warmup_lr_start), (2e-5,5e-6,5e-7))
        self.assertEqual((schedule.num_warmup_steps, schedule.num_training_steps), (500,48284))
        self.assertEqual((cfg.trainer.max_steps, cfg.data.params.max_steps), (48284,48284))
        self.assertEqual((cfg.data.params.train_datasets.batch_size, cfg.data.params.train_datasets.num_workers), (24,4))
        self.assertEqual(cfg.model.params.last_n, 1)
        self.assertEqual(cfg.trainer.precision, 'bf16-mixed')
        self.assertEqual(cfg.trainer.optimizer.type, 'torch.optim.AdamW')
        self.assertEqual(cfg.trainer.save_interval, 250)
        self.assertEqual(launch.PROFILES['production'], dict(margin=1800, max_seconds=85500, gpus=1))

    def test_release_hash_and_resume_source_binding_reject_changes(self):
        from mibot.utils.claix_contract import resume_fingerprint
        conf = dict(data={'params': {'train_datasets': {}}}, model={}, trainer=dict(
            optimizer={}, scheduler={}, precision='bf16-mixed', devices=1, accumulate_grad_batches=1))
        with patch('mibot.utils.claix_contract.engineering_source_digest', return_value='a'*64):
            old = resume_fingerprint(conf, {'sha256':'c'*64}, 12288)
        with patch('mibot.utils.claix_contract.engineering_source_digest', return_value='b'*64):
            self.assertNotEqual(old, resume_fingerprint(conf, {'sha256':'c'*64}, 12288))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old-release.json'
            path.write_text(json.dumps(dict(source_tree_sha256='0'*64, plan={})))
            with self.assertRaisesRegex(RuntimeError, 'source tree'):
                launch.check_release(path, {})


class MonitorReleaseTest(unittest.TestCase):
    def test_sampler_records_devices_and_stops_without_changing_signals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'gpu.jsonl'
            monitor = launch.NvidiaSmiMonitor(path)
            written = threading.Event()
            append = monitor._append
            def notify(**fields):
                append(**fields)
                written.set()
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
            with patch.object(monitor, '_append', side_effect=notify), \
                 patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(
                     stdout='0, GPU-abc, 91, 65000, 81920\n')) as query:
                monitor.start()
                self.assertTrue(written.wait(2))
                monitor.close()
            self.assertFalse(monitor.thread.is_alive())
            self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})
            query.assert_called_once()
            self.assertEqual(query.call_args.kwargs['timeout'], 3)
            self.assertTrue(query.call_args.kwargs['start_new_session'])
            row = json.loads(path.read_text())
            self.assertEqual(row['devices'][0]['utilization_percent'], 91.)
            self.assertEqual(row['devices'][0]['memory_used_mib'], 65000.)
            self.assertIn('timestamp', row)

    def test_missing_binary_timeout_driver_and_log_errors_only_disable_monitor(self):
        errors = [FileNotFoundError('nvidia-smi'), launch.subprocess.TimeoutExpired('nvidia-smi', 3),
                  launch.subprocess.CalledProcessError(1, 'nvidia-smi')]
        with tempfile.TemporaryDirectory() as folder:
            for i, error in enumerate(errors):
                with self.subTest(error=type(error).__name__), \
                     patch.object(launch.subprocess, 'run', side_effect=error):
                    monitor = launch.NvidiaSmiMonitor(Path(folder)/f'gpu-{i}.jsonl')
                    monitor.start()
                    monitor.thread.join(2)
                    monitor.close()
                    self.assertFalse(monitor.thread.is_alive())
                    self.assertEqual(json.loads(monitor.path.read_text())['event'], 'monitor_disabled')
            with patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
                monitor = launch.NvidiaSmiMonitor(Path(folder)/'missing-parent/gpu.jsonl')
                monitor.start()
                monitor.thread.join(2)
                monitor.close()
                self.assertFalse(monitor.thread.is_alive())

    def test_launcher_closes_monitor_on_success_and_child_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder, \
                 patch.object(launch, 'allocation_deadline', return_value=10000000000), \
                 patch.object(launch, 'check_release'), patch.object(launch, 'write_provenance'), \
                 patch.object(launch, 'NvidiaSmiMonitor') as monitor, \
                 patch.object(launch.subprocess, 'Popen') as child, \
                 patch.dict(launch.os.environ, XR1_DOOSAN_ZIP_CACHE_ROOT='/synthetic', XR1_NVIDIA_SMI_MONITOR='1'):
                if failure:
                    child.side_effect = OSError('synthetic child failure')
                    with self.assertRaises(OSError):
                        launch.main(['--execute', '--mode', 'production', '--run-root', folder, '--release-record', 'fixture'])
                else:
                    child.return_value.wait.return_value = 0
                    self.assertEqual(launch.main(['--execute', '--mode', 'production', '--run-root', folder,
                                                  '--release-record', 'fixture']), 0)
                monitor.return_value.start.assert_called_once()
                monitor.return_value.close.assert_called_once()

    def test_provenance_records_exact_commit_source_files_and_release(self):
        identity = launch.checkout_identity()
        self.assertEqual(len(identity['git_commit']), 40)
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder)/'launch-test'
            release = Path(folder)/'release.json'
            release.write_text('{"fixture": true}')
            launch.write_provenance(prefix, {'fixture':1}, release)
            row = json.loads(Path(str(prefix)+'.provenance.json').read_text())
            self.assertEqual(row['git_commit'], identity['git_commit'])
            self.assertEqual(row['source_tree_sha256'], launch.tree_digest())
            self.assertEqual(row['release_record'], {'fixture':True})
            source = 'xr1/configs/claix_b24_production.yaml'
            self.assertEqual(row['source_files'][source], hashlib.sha256((ROOT/source).read_bytes()).hexdigest())
