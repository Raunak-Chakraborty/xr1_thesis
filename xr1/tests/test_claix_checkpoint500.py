"""Task 04A: real CPU transactions; synthetic framework/allocation boundaries only."""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_claix_experimental as experimental_tests
from test_claix_experimental import effective_command, recipe
from test_claix_launch import ROOT, launch
from test_claix_checkpoint import checkpoint_fixture
from mibot.utils import claix_checkpoint as transaction
from mibot.utils.claix_contract import FROZEN_TRAIN_SHA256, resume_fingerprint
from mibot.utils.claix_experiment import bind_experiment, load_experimental_resume
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO


NAME = 'xr1_doosan_b24_checkpoint500_v1'


def identity(root):
    return dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                run_id='1'*32, run_root=str(root), zip_cache_root='/hpcwork/SYNTHETIC/cache',
                zip_manifest_sha256='b'*64, checkpoint_interval=500, experiment_name=NAME)


def payload(root, step):
    ck = checkpoint_fixture(step, 48284)
    ck['xr1_production']['experiment'] = identity(root)
    return ck


class Checkpoint500LaunchTest(unittest.TestCase):
    # Reuse only the explicit synthetic allocation/child fixture, not assertions.
    execute = experimental_tests.ExperimentalLaunchTest.execute

    def setUp(self):
        experimental_tests.ExperimentalLaunchTest.setUp(self)
        self.args += ['--experimental-checkpoint-interval', '500']

    def test_selector_is_closed_and_experimental_only(self):
        for extra in (['--experimental-checkpoint-interval', '1000'],
                      ['--mode', 'production'], ['--mode', 'h100-qualification'],
                      ['--experimental-checkpoint-interval', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main([*self.args, *extra])
        for value in (True, 500.0, '500', 1000):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                launch.experiment_profile(value)
        self.child.assert_not_called()

    def test_dry_run_selects_exact_recipe_and_never_creates_root(self):
        self.assertEqual(launch.main(self.args), 0)
        self.assertFalse(self.run.exists())
        command, cfg = effective_command(self.output.getvalue())
        self.assertEqual(command[6], '--config-name=claix_b24_checkpoint500')
        self.assertEqual(cfg['trainer']['save_interval'], 500)
        self.assertEqual(cfg['trainer']['exp_name'], NAME)
        self.assertIsNone(cfg['trainer']['ckpt_path'])
        self.assertEqual(cfg['trainer']['default_root_dir'], str(self.run))
        self.child.assert_not_called()

    def test_execution_binds_new_plan_root_and_step_zero_provenance(self):
        self.assertEqual(self.execute(), 0)
        manifest = launch.read_experiment(self.run)
        self.assertEqual(manifest['plan']['checkpoint_interval'], 500)
        self.assertEqual(manifest['plan']['experiment_name'], NAME)
        self.assertEqual(len(manifest['run_id']), 32)
        self.mocks[1].assert_called_once_with('experimental', allocation_record={}, checkpoint_interval=500)
        provenance = json.loads(next(self.run.glob('*.provenance.json')).read_text())
        self.assertEqual(provenance['initial_checkpoint_mode'], 'original_pretrained_step_zero')
        self.assertNotIn('21500', ' '.join(self.child.call_args.args[0]))
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_same_run_latest_recovery_and_cross_variant_rejection(self):
        self.execute()
        manifest = launch.read_experiment(self.run)
        target = self.run/'project_xr1_doosan'/NAME
        ck = payload(self.run, 500)
        ck['xr1_production']['experiment']['run_id'] = manifest['run_id']
        path = target/'step-000500.ckpt'
        transaction.publish_checkpoint(ck, path)
        (target/'telemetry.jsonl').write_text(json.dumps(dict(event='train_start', step=0))+'\n'+
            ''.join(json.dumps(dict(event='step', step=step))+'\n' for step in range(1, 501)))
        self.assertEqual(self.execute(['--resume', str(path)]), 0)
        for interval in ('250',):
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                self.execute(['--experimental-checkpoint-interval', interval, '--resume', str(path)])
        altered = copy.deepcopy(manifest)
        altered['plan']['checkpoint_interval'] = 250
        transaction.atomic_json(self.run/'experiment.json', altered)
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            self.execute(['--resume', str(path)])

    def test_foreign_historical_resume_and_unsafe_overrides_fail_before_child(self):
        old = self.root/'step-021500.ckpt'
        transaction.publish_checkpoint(checkpoint_fixture(21500, 48284), old)
        with self.assertRaises(RuntimeError):
            self.execute(['--resume', str(old)])
        with patch.dict(os.environ, XR1_OPTION_B_RESUME_BRIDGE='EXPLICIT_STEP_21500'), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.execute()
        rows = json.loads(self.overrides.read_text())
        rows[0] = 'trainer.save_interval=250'
        self.overrides.write_text(json.dumps(rows))
        with self.assertRaisesRegex(RuntimeError, 'four staged path'):
            self.execute()
        self.child.assert_not_called()


class Checkpoint500ConfigTest(unittest.TestCase):
    def test_full_inheritance_preserves_science_and_legacy_bytes(self):
        baseline = recipe('claix_b24_experimental')
        candidate = recipe('claix_b24_checkpoint500')
        self.assertEqual(candidate['data'], baseline['data'])
        self.assertEqual(candidate['model'], baseline['model'])
        expected = copy.deepcopy(baseline)
        expected['production']['checkpoint_variant'] = 'checkpoint500_v1'
        expected['trainer'].update(save_interval=500, exp_name=NAME, default_root_dir=None)
        self.assertEqual(candidate, expected)
        self.assertEqual(candidate['trainer']['max_steps'], 48284)
        self.assertEqual(candidate['data']['params']['train_datasets']['batch_size'], 24)
        self.assertEqual(candidate['model']['params']['last_n'], 1)
        self.assertEqual(candidate['trainer']['scheduler']['params'], dict(
            num_training_steps=48284, num_warmup_steps=500,
            warmup_lr_start=5e-7, max_lr=2e-5, min_lr=5e-6))
        for name in ('xr1/configs/claix_b24_experimental.yaml', 'xr1/configs/claix_b24_production.yaml',
                     'xr1/configs/doosan_compact_fp32.yaml', 'xr1/configs/model/doosan_fp32_v2.yaml',
                     'xr1/configs/data/doosan_fp32_v2.yaml', 'xr1/configs/trainer/doosan_fp32_v2.yaml',
                     'scripts/claix_b24_experimental.sbatch', 'scripts/claix_b24_production.sbatch',
                     'scripts/claix_b24_release.py', 'xr1/mibot/utils/claix_resume_bridge.py'):
            self.assertEqual((ROOT/name).read_bytes(), subprocess.check_output(
                ['git', 'show', '0106f1a0e0dfbcbf18e9732f4924b8ccd9e937f0:'+name], cwd=ROOT), name)
        self.assertEqual(baseline['trainer']['save_interval'], 250)
        self.assertEqual(recipe('claix_b24_production')['trainer']['save_interval'], 250)

    def test_entrypoint_binding_rejects_variant_name_interval_and_manifest_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root/'cache'; cache.mkdir()
            (cache/'media_manifest.jsonl').write_text('SYNTHETIC\n')
            cfg = recipe('claix_b24_checkpoint500')
            cfg['trainer']['default_root_dir'] = folder
            cfg['production']['experimental_run_id'] = '1'*32
            manifest = dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='1'*32, source_tree_sha256=launch.tree_digest(),
                plan=dict(run_root=folder, checkpoint_interval=500, experiment_name=NAME))
            transaction.atomic_json(root/'experiment.json', manifest)
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache)):
                bound = bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                self.assertEqual(bound['checkpoint_interval'], 500)
                self.assertEqual(bound['experiment_name'], NAME)
                for key, value in (('save_interval', 250), ('exp_name', launch.EXPERIMENT_NAME)):
                    bad = copy.deepcopy(cfg); bad['trainer'][key] = value
                    with self.assertRaisesRegex(RuntimeError, 'checkpoint/output'):
                        bind_experiment(bad, bad['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                bad = copy.deepcopy(cfg); bad['production']['checkpoint_variant'] = 'arbitrary'
                with self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
                    bind_experiment(bad, bad['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                legacy = recipe('claix_b24_experimental')
                legacy['trainer']['default_root_dir'] = folder
                legacy['production']['experimental_run_id'] = '1'*32
                with self.assertRaisesRegex(RuntimeError, 'variant/manifest'):
                    bind_experiment(legacy, legacy['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                manifest['plan']['checkpoint_interval'] = 250
                transaction.atomic_json(root/'experiment.json', manifest)
                with self.assertRaisesRegex(RuntimeError, 'variant/manifest'):
                    bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
            old = dict(bound); old.pop('checkpoint_interval'); old.pop('experiment_name')
            self.assertNotEqual(resume_fingerprint(cfg, dict(sha256=FROZEN_TRAIN_SHA256), 12288, experiment=bound),
                                resume_fingerprint(cfg, dict(sha256=FROZEN_TRAIN_SHA256), 12288, experiment=old))

    def test_exact_new_wrapper_and_allocation_contract(self):
        wrapper = ROOT/'scripts/claix_b24_checkpoint500.sbatch'
        subprocess.run(['bash', '-n', str(wrapper)], check=True)
        text = wrapper.read_text()
        for directive in ('account=thes2452', 'partition=c23g', 'gres=gpu:hopper:1',
                          'cpus-per-task=12', 'mem=96G', 'time=23:45:00', 'signal=B:USR1@1800',
                          'nodes=1', 'ntasks=1'):
            self.assertIn('#SBATCH --'+directive+'\n', text)
        self.assertIn('--experimental-checkpoint-interval 500', text)
        fields = experimental_tests.AllocationTest.fields(self)
        def validate(batch, **changes):
            selected = dict(fields, **changes)
            def output(cmd, **kwargs):
                if cmd[0] == 'date': return '10000'
                if cmd[0] == 'nvidia-smi': return 'NVIDIA H100 80GB HBM3\n'
                if cmd[:3] == ['scontrol', 'write', 'batch_script']: return batch
                raise AssertionError(cmd)
            with patch.dict(os.environ, SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='0'), \
                 patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(
                     stdout=' '.join(k+'='+v for k,v in selected.items()))), \
                 patch.object(launch.subprocess, 'check_output', side_effect=output), \
                 patch.object(launch.time, 'time', return_value=100), \
                 patch('claix_b24_filesystem.inspect_directory'), contextlib.redirect_stdout(io.StringIO()):
                return launch.allocation_deadline('experimental', checkpoint_interval=500)
        self.assertEqual(validate(text), 8200)
        with self.assertRaisesRegex(RuntimeError, 'wrapper'):
            validate((ROOT/'scripts/claix_b24_experimental.sbatch').read_text())
        for key, value in (('TimeLimit', '20:00:00'), ('Account', 'other'), ('Partition', 'other'),
                           ('AllocTRES', 'cpu=12,mem=97G,gres/gpu=1'), ('NumCPUs', '24')):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate(text, **{key:value})


class Checkpoint500EntrypointTest(experimental_tests.EntrypointTest):
    config_name = 'claix_b24_checkpoint500'
    interval = 500
    experiment_name = NAME


class Checkpoint500RuntimeTest(unittest.TestCase):
    def boundary(self, folder, step, *, interval=500, signal_stop=False, instrument=True):
        ck = payload(Path(folder), step)
        module = SimpleNamespace(_replay_contract=ck['xr1_replay_contract'],
            _compact_cfg=dict(config_sha256='a'*64, experiment=identity(Path(folder))),
            parameters=lambda: [torch.nn.Parameter(torch.ones(1)),
                                torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)])
        checkpoint_io = VerifiedCheckpointIO(expected_config='a'*64,
            expected_experiment=identity(Path(folder)), instrument=instrument)
        def save(path, weights_only):
            self.assertFalse(weights_only)
            ck['xr1_production']['rng_by_rank'] = module._checkpoint_rng
            checkpoint_io.save_checkpoint(ck, path)
        trainer = SimpleNamespace(global_step=step, world_size=1, is_global_zero=True,
            max_steps=48284, optimizers=[SimpleNamespace(param_groups=[dict(lr=2e-5)])],
            save_checkpoint=save, strategy=SimpleNamespace(barrier=lambda *_: None, checkpoint_io=checkpoint_io),
            should_stop=False)
        runtime = ProductionRuntime(folder, deadline=time.time()+3600, interval=interval)
        runtime.started = runtime.fit_started = runtime.previous_end = time.perf_counter()-.1
        runtime.previous_step = step-1; runtime.initial_step = 0
        runtime.retained, runtime.views, runtime.tokens = 24, 48, 100
        runtime.data_pipeline_seconds = .01
        if signal_stop: runtime.stop.handle(signal.SIGUSR1, None)
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
        return runtime, trainer, module

    def test_no_ordinary_checkpoint_at_250_or_750(self):
        for step in (250, 750):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as folder:
                runtime, _, _ = self.boundary(folder, step)
                self.assertEqual(runtime.last_saved, -1)
                self.assertFalse(list(Path(folder).glob('*.ckpt')))
                self.assertFalse((Path(folder)/'latest.json').exists())

    def test_periodic_500_1000_and_pinned_2000(self):
        for step in (500, 1000, 2000):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as folder:
                runtime, trainer, module = self.boundary(folder, step)
                self.assertEqual(runtime.last_saved, step)
                self.assertFalse(trainer.should_stop)
                ck = transaction.load_verified(Path(folder)/f'step-{step:06d}.ckpt',
                    module._replay_contract, expected_experiment=identity(Path(folder)))
                self.assertEqual(ck['xr1_replay']['rank_sample_offset'], step*24)
                row = json.loads((Path(folder)/'telemetry.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['reason'], 'interval')

    def test_safe_stop_751_is_immediate_verified_and_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, trainer, module = self.boundary(folder, 751, signal_stop=True)
            self.assertTrue(trainer.should_stop)
            self.assertEqual(runtime.last_saved, 751)
            self.assertEqual(json.loads((Path(folder)/'safe-stop.json').read_text())['global_step'], 751)
            with contextlib.redirect_stdout(io.StringIO()): runtime.on_fit_end(trainer, module)
            for step in (1000, 1500, 2000):
                transaction.publish_checkpoint(payload(Path(folder), step), Path(folder)/f'step-{step:06d}.ckpt')
                transaction.rotate_checkpoints(folder, expected_config='a'*64)
            self.assertTrue((Path(folder)/'step-000751.ckpt').exists())

    def test_terminal_48284_publishes_and_reverifies_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, trainer, module = self.boundary(folder, 48284)
            with contextlib.redirect_stdout(io.StringIO()): runtime.on_fit_end(trainer, module)
            rows = [json.loads(x) for x in (Path(folder)/'telemetry.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['reason'], 'natural_finish')
            self.assertTrue(rows[-1]['final_checkpoint_verified'])
            path = Path(folder)/'step-048284.ckpt'
            transaction.load_verified(path, resumable=False)
            with self.assertRaises(RuntimeError): transaction.load_verified(path)
            with path.open('r+b') as stream: stream.write(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'checksum'): runtime.on_fit_end(trainer, module)

    def test_fit_cannot_succeed_without_boundary_save(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, trainer, module = self.boundary(folder, 750)
            with self.assertRaisesRegex(RuntimeError, 'without a final boundary'): runtime.on_fit_end(trainer, module)

    def test_legacy_250_schedule_and_telemetry_unchanged(self):
        for step in (250, 750):
            with tempfile.TemporaryDirectory() as folder:
                runtime, _, _ = self.boundary(folder, step, interval=250, instrument=False)
                self.assertEqual(runtime.last_saved, step)
                row = json.loads((Path(folder)/'telemetry.jsonl').read_text().splitlines()[-1])
                self.assertNotIn('timings', row)
        self.assertEqual(ProductionRuntime('/unused', deadline=time.time()+1).interval, 250)

    def test_bounded_phase_timings_leave_receipt_schema_and_io_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(transaction.torch, 'load', wraps=torch.load) as loads, \
             patch.object(transaction, 'file_sha256', wraps=transaction.file_sha256) as hashes:
            self.boundary(folder, 500)
            # One read/hash for publication; one read/hash for initial rotation.
            self.assertEqual((loads.call_count, hashes.call_count), (2, 2))
            row = json.loads((Path(folder)/'telemetry.jsonl').read_text().splitlines()[-1])
            timings = row['timings']
            expected = {'state_capture_seconds', 'prewrite_validation_seconds', 'serialization_fsync_seconds',
                'readback_validation_seconds', 'sha256_seconds', 'publication_seconds',
                'safe_stop_pointer_seconds', 'rotation_verification_seconds', 'callback_duration_seconds'}
            self.assertEqual(set(timings), expected)
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in timings.values()))
            self.assertGreaterEqual(timings['callback_duration_seconds'],
                                    sum(v for k,v in timings.items() if k != 'callback_duration_seconds'))
            receipt = json.loads((Path(folder)/'latest.json').read_text())
            self.assertEqual(set(receipt), {'schema','path','global_step','bytes','sha256','config_sha256','experiment'})

    def test_failed_io_never_marks_boundary_saved_or_reports_success(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, trainer, module = self.boundary(folder, 750)
            with patch.object(transaction.os, 'fsync', side_effect=OSError('injected fsync failure')):
                with self.assertRaisesRegex(OSError, 'injected fsync'): runtime.save(trainer, module, 'signal')
            self.assertEqual(runtime.last_saved, -1)
            self.assertIsNone(trainer.strategy.checkpoint_io.last_save_timings)
            with self.assertRaisesRegex(RuntimeError, 'without a final boundary'): runtime.on_fit_end(trainer, module)
            self.assertFalse((Path(folder)/'latest.json').exists())


class Checkpoint500TransactionTest(unittest.TestCase):
    def test_real_verified_recovery_checksum_and_foreign_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root/'project_xr1_doosan'/NAME
            ck = payload(root, 500); path = target/'step-000500.ckpt'
            receipt = transaction.publish_checkpoint(ck, path)
            self.assertEqual(receipt['sha256'], transaction.file_sha256(path))
            self.assertEqual(receipt['bytes'], path.stat().st_size)
            recovered = load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, identity(root))
            self.assertTrue(torch.equal(recovered['state_dict']['model.weight'], ck['state_dict']['model.weight']))
            self.assertEqual(recovered['lr_schedulers'], ck['lr_schedulers'])
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(recovered['optimizer_states'][0]['state'][0][key],
                                            ck['optimizer_states'][0]['state'][0][key]))
            for foreign in (dict(identity(root), run_id='2'*32), dict(identity(root), checkpoint_interval=250)):
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, foreign)
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                load_experimental_resume(path, ck['xr1_replay_contract'], 'b'*64, identity(root))
            with path.open('r+b') as stream: stream.write(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, identity(root))

    def failure(self, phase):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = root/'step-000500.ckpt'; new = root/'step-001000.ckpt'
            transaction.publish_checkpoint(payload(root, 500), old)
            prior = (root/'latest.json').read_bytes()
            old_bytes = old.read_bytes()
            def serialize(ck, stream):
                stream.write(b'partial serialization')
                raise OSError('injected serialization failure')
            original_link, original_replace = os.link, os.replace
            def link(src, dst, *args, **kwargs):
                if Path(dst) == new.with_suffix('.verified.json'):
                    raise OSError('injected receipt failure')
                return original_link(src, dst, *args, **kwargs)
            def replace(src, dst, *args, **kwargs):
                if Path(dst) == root/'latest.json': raise OSError('injected pointer failure')
                return original_replace(src, dst, *args, **kwargs)
            injection = (patch.object(torch, 'save', side_effect=serialize) if phase == 'serialization' else
                         patch.object(transaction.os, 'link', side_effect=link) if phase == 'receipt' else
                         patch.object(transaction.os, 'replace', side_effect=replace))
            with injection, self.assertRaisesRegex(OSError, 'injected'):
                transaction.publish_checkpoint(payload(root, 1000), new, timings={})
            self.assertEqual((root/'latest.json').read_bytes(), prior)
            self.assertEqual(old.read_bytes(), old_bytes)
            transaction.load_verified(old, expected_config='a'*64, expected_experiment=identity(root))
            self.assertFalse(list(root.glob('.*.tmp')))
            self.assertEqual(new.exists(), phase != 'serialization')
            self.assertEqual(new.with_suffix('.verified.json').exists(), phase == 'pointer')
            if phase == 'receipt':
                with self.assertRaises(FileNotFoundError): transaction.load_verified(new)
            if phase == 'pointer':
                transaction.load_verified(new, expected_experiment=identity(root))
                with self.assertRaisesRegex(RuntimeError, 'newest verified'):
                    transaction.rotate_checkpoints(root)
            if phase != 'serialization':
                with self.assertRaises(FileExistsError): transaction.publish_checkpoint(payload(root, 1000), new)

    def test_partial_serialization_failure_preserves_previous_commit(self): self.failure('serialization')
    def test_atomic_receipt_link_failure_preserves_previous_pointer(self): self.failure('receipt')
    def test_atomic_latest_replace_failure_retains_verified_payload_and_old_pointer(self): self.failure('pointer')

    def test_readback_validation_failure_cannot_publish_receipt_or_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = root/'step-000500.ckpt'; new = root/'step-001000.ckpt'
            transaction.publish_checkpoint(payload(root, 500), old)
            prior = (root/'latest.json').read_bytes()
            original_load = torch.load
            def damaged_readback(*args, **kwargs):
                ck = original_load(*args, **kwargs)
                ck['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] += 1
                return ck
            with patch.object(torch, 'load', side_effect=damaged_readback), self.assertRaises(RuntimeError):
                transaction.publish_checkpoint(payload(root, 1000), new)
            self.assertFalse(new.exists())
            self.assertFalse(new.with_suffix('.verified.json').exists())
            self.assertEqual((root/'latest.json').read_bytes(), prior)

    def test_restored_rng_and_replay_cursor_match_verified_boundary(self):
        import random
        import numpy as np
        original = transaction.rng_state()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); path = root/'step-000500.ckpt'
                ck = payload(root, 500)
                expected = (random.random(), np.random.random(), torch.rand(4))
                transaction.publish_checkpoint(ck, path)
                loaded = transaction.load_verified(path, expected_experiment=identity(root))
                transaction.restore_rng(loaded['xr1_production']['rng_by_rank'][0])
                actual = (random.random(), np.random.random(), torch.rand(4))
                self.assertEqual(actual[:2], expected[:2])
                self.assertTrue(torch.equal(actual[2], expected[2]))
                self.assertEqual(loaded['xr1_replay']['rank_sample_offset'], 500*24)
                self.assertEqual(loaded['lr_schedulers'][0]['last_epoch'], 500)
        finally:
            transaction.restore_rng(original)

    def test_pinned_milestones_and_two_usable_recovery_states(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in (500, 1000, *sorted(transaction.HISTORICAL_STEPS), 40500, 41000, 41500):
                transaction.publish_checkpoint(payload(root, step), root/f'step-{step:06d}.ckpt')
                transaction.rotate_checkpoints(root, expected_config='a'*64)
            retained = {int(p.stem.split('-')[1]) for p in root.glob('*.ckpt')}
            self.assertEqual(retained, set(transaction.HISTORICAL_STEPS) | {41000, 41500})
            for step in retained:
                transaction.load_verified(root/f'step-{step:06d}.ckpt', expected_experiment=identity(root))
            with (root/'step-041500.ckpt').open('r+b') as stream: stream.write(b'bad')
            before = {p.name for p in root.iterdir()}
            with self.assertRaisesRegex(RuntimeError, 'Newest checkpoint failed'):
                transaction.rotate_checkpoints(root)
            self.assertEqual({p.name for p in root.iterdir()}, before)


class Checkpoint500HydraTest(unittest.TestCase):
    """Real Hydra acceptance; deliberately excluded from the framework-free harness."""
    def test_real_hydra_composition_preserves_frozen_science(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(ROOT/'xr1/configs'), version_base=None):
            old = OmegaConf.to_container(compose(config_name='claix_b24_experimental'), resolve=True)
            new = OmegaConf.to_container(compose(config_name='claix_b24_checkpoint500'), resolve=True)
        self.assertEqual(old['data'], new['data'])
        self.assertEqual(old['model'], new['model'])
        for key in ('optimizer', 'scheduler', 'precision', 'max_steps', 'accumulate_grad_batches'):
            self.assertEqual(old['trainer'][key], new['trainer'][key])
        self.assertEqual(new['trainer']['save_interval'], 500)
        self.assertEqual(new['trainer']['exp_name'], NAME)
        self.assertIsNone(new['trainer']['default_root_dir'])
