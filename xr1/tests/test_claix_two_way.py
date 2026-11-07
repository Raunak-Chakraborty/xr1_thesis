"""Closed launcher and science regressions; allocation/child boundaries synthetic."""
import contextlib
import ast
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import claix_b24_checkpoint_two_way as benchmark
import claix_b24_storage_preflight as storage
from mibot.utils import claix_checkpoint as tx
from mibot.utils.claix_contract import FROZEN_TRAIN_SHA256, resume_fingerprint
from mibot.utils.claix_experiment import bind_experiment
from test_claix_async_checkpoint import identity, payload
import test_claix_experimental as fixtures
from test_claix_experimental import recipe, effective_command
from test_claix_launch import ROOT, launch


class TwoWayLaunchTest(unittest.TestCase):
    execute = fixtures.ExperimentalLaunchTest.execute

    def setUp(self):
        fixtures.ExperimentalLaunchTest.setUp(self)
        self.args += ['--experimental-checkpoint-interval', '500', '--experimental-two-way-leg', 'C']

    def test_fresh_async_identity_exact_config_and_fixed_science(self):
        self.execute()
        manifest = launch.read_experiment(self.run)
        self.assertEqual(manifest['plan']['checkpoint_writer'], 'async_thread_v1')
        self.assertEqual(manifest['plan']['benchmark_steps'], 1500)
        cfg = effective_command(self.output.getvalue())[1]
        baseline = recipe('claix_b24_checkpoint500')
        for key in ('optimizer', 'scheduler', 'max_steps', 'precision', 'save_interval', 'seed'):
            self.assertEqual(cfg['trainer'][key], baseline['trainer'][key])
        self.assertEqual(cfg['production']['stop_after_step'], 1500)
        self.assertEqual(cfg['production']['checkpoint_variant'], 'checkpoint500_async_v1')
        self.assertIsNone(cfg['trainer']['ckpt_path'])
        self.mocks[1].assert_called_once_with('experimental', allocation_record={}, checkpoint_interval=500, two_way_leg='C')
        self.mocks[3].assert_called_once()
        self.assertTrue(self.child.call_args.kwargs['pass_fds'])
        first = manifest['run_id']
        self.args[self.args.index('--run-root')+1] = str(self.root/'B')
        self.execute(['--experimental-two-way-leg', 'B'])
        second = launch.read_experiment(self.root/'B')
        self.assertNotEqual(first, second['run_id'])
        self.assertEqual(second['plan']['checkpoint_writer'], 'sync')

    def test_no_250_arbitrary_budget_or_cross_mode_escape(self):
        for extra in (['--experimental-checkpoint-interval', '250'], ['--mode', 'production'],
                      ['--stop-after-step', '100'], ['--experimental-two-way-leg', 'A']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.execute(extra)
        self.child.assert_not_called()

    def test_same_run_restore_rejects_changed_writer_source_and_telemetry(self):
        self.execute()
        manifest = launch.read_experiment(self.run)
        target = self.run/'project_xr1_doosan'/launch.ASYNC500_NAME
        ck = payload(self.run, 500); ck['xr1_production']['experiment']['run_id'] = manifest['run_id']
        path = target/'step-000500.ckpt'; tx.publish_checkpoint(ck, path)
        (target/'telemetry.jsonl').write_text(json.dumps(dict(event='train_start', step=0))+'\n'+
            ''.join(json.dumps(dict(event='step', step=s))+'\n' for s in range(1, 501)))
        self.execute(['--resume', str(path)])
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            self.execute(['--resume', str(path), '--experimental-two-way-leg', 'B'])
        altered = copy.deepcopy(manifest); altered['source_tree_sha256'] = '0'*64
        tx.atomic_json(self.run/'experiment.json', altered)
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'): self.execute(['--resume', str(path)])
        tx.atomic_json(self.run/'experiment.json', manifest)
        with (target/'telemetry.jsonl').open('a') as stream: stream.write(json.dumps(dict(event='step', step=501))+'\n')
        with self.assertRaisesRegex(RuntimeError, 'Telemetry'): self.execute(['--resume', str(path)])

    def test_async_storage_failure_does_not_start_child(self):
        self.mocks[3].return_value = dict(status='FAIL', abort_reasons=['quota'])
        with self.assertRaisesRegex(RuntimeError, 'storage preflight'): self.execute()
        self.child.assert_not_called(); self.assertFalse(self.run.exists())


class TwoWayConfigTest(unittest.TestCase):
    def test_actual_entrypoint_selects_async_and_closes_on_fit_failure(self):
        from mibot.utils.claix_async_checkpoint import AsyncProductionRuntime, AsyncVerifiedCheckpointIO
        from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root/'cache'; cache.mkdir()
            (cache/'media_manifest.jsonl').write_text('synthetic\n')
            cfg = recipe('claix_b24_checkpoint500_async')
            cfg['trainer']['default_root_dir'] = folder; cfg['production']['experimental_run_id'] = 'c'*32
            for path_key, hash_key in (('pretrained', 'pretrained_sha256'), ('language_policy_path', 'language_sha256')):
                file = root/path_key; file.write_bytes(b'synthetic')
                cfg['model']['params'][path_key] = str(file); cfg['model']['params'][hash_key] = tx.file_sha256(file)
            tx.atomic_json(root/'experiment.json', dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='c'*32, source_tree_sha256=launch.tree_digest(), plan=dict(run_root=folder,
                    checkpoint_interval=500, experiment_name=launch.ASYNC500_NAME, checkpoint_writer='async_thread_v1')))
            source = ast.parse((ROOT/'xr1/tools/train_doosan_compact.py').read_text())
            main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            main.decorator_list = []; main.args.args[0].annotation = None
            trainer = Mock(); trainer.train_dataloader = None
            trainer.fit.side_effect = ValueError('injected fit failure')
            trainer_type = Mock(return_value=trainer)
            namespace = dict(Config=fixtures.Config, OmegaConf=SimpleNamespace(to_container=lambda value, **_: copy.deepcopy(value)),
                torch=torch, os=os, Path=Path, json=json, BASELINE=fixtures.BASELINE, resume_fingerprint=resume_fingerprint,
                split_inventory=lambda _: dict(episodes=255, frames=289703, sha256=FROZEN_TRAIN_SHA256),
                seed_everything=Mock(), DDPStrategy=lambda **_: 'synthetic-ddp',
                make_contract=__import__('mibot.utils.doosan_replay', fromlist=['make_contract']).make_contract,
                ModelCheckpoint=Mock(), ModelSummary=Mock(), CSVLogger=Mock(), ProductionRuntime=ProductionRuntime,
                VerifiedCheckpointIO=VerifiedCheckpointIO, DATASETS=SimpleNamespace(build=Mock()),
                MIMODEL=SimpleNamespace(build=Mock()), Trainer=trainer_type)
            exec(compile(ast.Module(body=[main], type_ignores=[]), 'train_doosan_compact.py', 'exec'), namespace)
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_SAFE_STOP_EPOCH=str(time.time()+1000),
                            XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache), WORLD_SIZE='1', MAX_LENGTH='12288'), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.cuda, 'get_device_name', return_value='SYNTHETIC H100'), \
                 patch.dict(sys.modules, {'mibot.data.datamodule.doosan_datamodule': SimpleNamespace(shutdown_loader=Mock())}), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'fit failure'):
                namespace['main'](cfg)
            plugin = trainer_type.call_args.kwargs['plugins'][0]
            self.assertIsInstance(plugin, AsyncVerifiedCheckpointIO); self.assertTrue(plugin._closed)
            self.assertIsInstance(trainer_type.call_args.kwargs['callbacks'][1], AsyncProductionRuntime)
            params = namespace['MIMODEL'].build.call_args.args[0]['params']
            self.assertEqual(params['compact_fp32']['resume_step'], 0)
            self.assertEqual(params['optimizer'], cfg['trainer']['optimizer'])
            self.assertEqual(params['scheduler'], cfg['trainer']['scheduler'])
            self.assertIsNone(trainer.fit.call_args.kwargs['ckpt_path'])

    def test_full_science_inheritance_and_unchanged_sync_implementations(self):
        old = recipe('claix_b24_checkpoint500'); new = recipe('claix_b24_checkpoint500_async')
        self.assertEqual(old['data'], new['data']); self.assertEqual(old['model'], new['model'])
        a, b = copy.deepcopy(old['trainer']), copy.deepcopy(new['trainer'])
        a.pop('exp_name'); b.pop('exp_name'); self.assertEqual(a, b)
        for name in ('xr1/mibot/utils/claix_runtime.py', 'xr1/mibot/utils/claix_checkpoint.py',
                     'xr1/configs/claix_b24_checkpoint500.yaml', 'xr1/configs/claix_b24_experimental.yaml',
                     'scripts/claix_b24_checkpoint500.sbatch', 'scripts/claix_b24_experimental.sbatch',
                     'scripts/claix_b24_production.sbatch', 'scripts/claix_b24_release.py',
                     'xr1/tests/test_claix_checkpoint500.py', 'xr1/tests/test_claix_runtime.py'):
            self.assertEqual((ROOT/name).read_bytes(), subprocess.check_output(
                ['git', 'show', '1e528af3ef370f2aa2c46d4e418f185dcb48be32:'+name], cwd=ROOT))

    def test_manifest_writer_and_output_are_bound_to_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root/'cache'; cache.mkdir()
            (cache/'media_manifest.jsonl').write_text('synthetic\n')
            cfg = recipe('claix_b24_checkpoint500_async')
            cfg['trainer']['default_root_dir'] = folder
            cfg['production']['experimental_run_id'] = 'c'*32
            manifest = dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='c'*32, source_tree_sha256=launch.tree_digest(), plan=dict(run_root=folder,
                checkpoint_interval=500, experiment_name=launch.ASYNC500_NAME, checkpoint_writer='async_thread_v1'))
            tx.atomic_json(root/'experiment.json', manifest)
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache)):
                bound = bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                manifest['plan']['checkpoint_writer'] = 'sync'; tx.atomic_json(root/'experiment.json', manifest)
                with self.assertRaisesRegex(RuntimeError, 'writer/manifest'):
                    bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
            other = dict(bound, checkpoint_writer='sync')
            self.assertNotEqual(resume_fingerprint(cfg, dict(sha256=FROZEN_TRAIN_SHA256), 12288, experiment=bound),
                                resume_fingerprint(cfg, dict(sha256=FROZEN_TRAIN_SHA256), 12288, experiment=other))

    def test_allocation_requires_exact_consolidated_wrapper_and_resources(self):
        batch = (ROOT/'scripts/claix_b24_checkpoint_two_way.sbatch').read_text()
        subprocess.run(['bash', '-n', str(ROOT/'scripts/claix_b24_checkpoint_two_way.sbatch')], check=True)
        for directive in ('account=thes2452', 'partition=c23g', 'gres=gpu:hopper:1', 'cpus-per-task=12',
                          'mem=96G', 'time=23:45:00', 'signal=B:USR1@1800', 'nodes=1', 'ntasks=1'):
            self.assertIn('#SBATCH --'+directive+'\n', batch)
        def validate(saved, **changes):
            fields = dict(fixtures.AllocationTest.fields(self), **changes)
            def output(cmd, **kwargs):
                if cmd[0] == 'date': return '10000'
                if cmd[0] == 'nvidia-smi': return 'NVIDIA H100\n'
                if cmd[:3] == ['scontrol', 'write', 'batch_script']: return saved
                raise AssertionError(cmd)
            with patch.dict(os.environ, SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='0'), \
                 patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items()))), \
                 patch.object(launch.subprocess, 'check_output', side_effect=output), \
                 patch.object(launch.time, 'time', return_value=100), patch('claix_b24_filesystem.inspect_directory'), \
                 contextlib.redirect_stdout(io.StringIO()):
                return launch.allocation_deadline('experimental', checkpoint_interval=500, two_way_leg='C')
        self.assertEqual(validate(batch), 8200)
        for saved in (batch+'# changed\n', (ROOT/'scripts/claix_b24_checkpoint500.sbatch').read_text()):
            with self.assertRaisesRegex(RuntimeError, 'wrapper'): validate(saved)
        for key, value in (('Account', 'other'), ('TimeLimit', '20:00:00'), ('TimeLimit', '06:00:00'),
                           ('TimeLimit', '23:44:00'), ('AllocTRES', 'cpu=12,mem=97G,gres/gpu=1')):
            with self.assertRaises(RuntimeError): validate(batch, **{key: value})


class TwoWayCoordinatorTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)/'benchmark'
        self.args = ['--run-root', str(self.root), '--overrides', '/synthetic/paths.json',
                     '--filesystem-source', 'lustre', '--project-id', '44885']

    def test_plan_exactly_two_fresh_500_legs_no_250_or_resume(self):
        output = io.StringIO()
        with patch.object(benchmark.subprocess, 'Popen') as child, contextlib.redirect_stdout(output):
            self.assertEqual(benchmark.main(self.args), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan['commands']), 2)
        self.assertEqual(plan['historical_sync250_job'], 5504969)
        for leg, cmd in zip(('B', 'C'), plan['commands']):
            self.assertEqual(cmd[cmd.index('--experimental-two-way-leg')+1], leg)
            self.assertEqual(cmd[cmd.index('--experimental-checkpoint-interval')+1], '500')
            self.assertNotIn('--resume', cmd); self.assertNotIn('--execute', cmd)
        child.assert_not_called(); self.assertFalse(self.root.exists())

    def run_coordinator(self, codes, *, stop=False, quota='PASS'):
        children = []
        for code in codes:
            child = Mock(); child.poll.return_value = code; child.wait.return_value = code
            children.append(child)
        if stop:
            def wait(timeout=None):
                signal.raise_signal(signal.SIGUSR1)
                return 0
            children[0].wait.side_effect = wait
        with patch.object(benchmark.launch, 'allocation_deadline'), \
             patch('claix_b24_task04c_preflight.verify', return_value=dict(run_ids={'B':'B'*32, 'C':'C'*32})), \
             patch.object(benchmark.launch, 'checkout_identity', return_value=dict(git_commit='synthetic', git_status='')), \
             patch.object(storage, 'experimental_evaluate', return_value=dict(status=quota)), \
             patch.object(benchmark.subprocess, 'Popen', side_effect=children) as start, \
             patch.object(benchmark, 'verify_leg', side_effect=lambda root, leg: dict(run_id=leg*32)), \
             patch.object(benchmark, 'verify_pair', return_value='synthetic comparison'), \
             contextlib.redirect_stdout(io.StringIO()):
            if quota != 'PASS' or stop or any(codes):
                with self.assertRaises(RuntimeError): benchmark.main([*self.args, '--execute'])
            else:
                self.assertEqual(benchmark.main([*self.args, '--execute']), 0)
            return start.call_count

    def test_coordinator_runs_b_then_c_once(self):
        self.assertEqual(self.run_coordinator([0, 0]), 2)
        result = json.loads((self.root/'two-way-results.json').read_text())
        self.assertEqual([r['leg'] for r in result['legs']], ['B', 'C'])

    def test_c_failure_preserves_b_and_never_retries(self):
        self.assertEqual(self.run_coordinator([0, 1]), 2)
        result = json.loads((self.root/'two-way-results.json').read_text())
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertEqual(result['legs'][0]['run_id'], 'B'*32)
        self.assertIn('Abandon async', result['recommendation'])

    def test_b_failure_prevents_c(self): self.assertEqual(self.run_coordinator([1]), 1)
    def test_signal_prevents_next_leg(self): self.assertEqual(self.run_coordinator([0], stop=True), 1)
    def test_quota_failure_prevents_all_children(self):
        self.assertEqual(self.run_coordinator([], quota='FAIL'), 0)
        self.assertFalse(self.root.exists())


class TwoWayEvidenceTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        rng = tx.rng_state()
        for leg, name in (('B', 'B_sync500'), ('C', 'C_async500')):
            root = self.root/name; root.mkdir()
            profile = launch.experiment_profile(500, leg)
            target = root/'project_xr1_doosan'/profile['name']
            ident = identity(root); ident.update(run_id=leg.lower()*32, experiment_name=profile['name'],
                                                 two_way_leg=leg, benchmark_steps=1500)
            if leg == 'B': ident.pop('checkpoint_writer')
            tx.atomic_json(root/'experiment.json', dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id=ident['run_id'], source_tree_sha256=launch.tree_digest(), plan=dict(two_way_leg=leg,
                    benchmark_steps=1500, checkpoint_writer=ident.get('checkpoint_writer', 'sync'),
                    run_root=str(root), run_id=ident['run_id'], checkpoint_interval=500, experiment_name=profile['name'])))
            for step in (500, 1000, 1500):
                ck = payload(root, step); ck['xr1_production'].update(experiment=ident, rng_by_rank=[rng])
                receipt = tx.publish_checkpoint(ck, target/f'step-{step:06d}.ckpt')
                if step == 1500: tx.atomic_json(target/'safe-stop.json', receipt)
                tx.rotate_checkpoints(target)
            cfg = recipe(profile['config'])
            tx.atomic_json(target/'experimental-start-1.json', dict(initial_global_step=0, new_optimizer_and_scheduler=True,
                experiment=ident, config_sha256='a'*64,
                initial_checkpoint_mode='original_pretrained_step_zero', source_config=cfg,
                dataset_identity=FROZEN_TRAIN_SHA256, pretrained_identity={'sha256':'a'*64}, language_identity={'sha256':'b'*64}))
            rows = [dict(event='train_start', step=0)] + [dict(event='step', step=s) for s in range(1,1501)]
            rows += [dict(event='checkpoint', step=s, verified=True,
                     timings={('callback_duration_seconds' if leg == 'B' else 'main_thread_blocking_seconds'): .1}) for s in (500,1000,1500)]
            rows += [dict(event='graceful_stop', step=1500, reason='qualification_step'),
                     dict(event='fit_end', step=1500, final_checkpoint_verified=True,
                          end_to_end_seconds=150., end_to_end_examples_per_second=240.)]
            (target/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_real_receipts_retention_metrics_and_final_state_comparison(self):
        results = [benchmark.verify_leg(self.root/name, leg) for leg, name in (('B','B_sync500'),('C','C_async500'))]
        self.assertNotEqual(results[0]['run_id'], results[1]['run_id'])
        self.assertEqual(len(results[0]['receipts']), 3)
        self.assertAlmostEqual(results[0]['metrics']['main_thread_checkpoint_blocking_seconds'], .3)
        self.assertTrue(benchmark.verify_pair(self.root, results).startswith('PASS exact'))
        results[1]['science_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'scientific'): benchmark.verify_pair(self.root, results)

    def test_retained_corrupt_bytes_and_missing_receipts_block_evidence(self):
        target = self.root/'C_async500'/'project_xr1_doosan'/launch.ASYNC500_NAME
        with (target/'step-001000.ckpt').open('r+b') as stream: stream.write(b'bad')
        with self.assertRaisesRegex(RuntimeError, 'checksum'): benchmark.verify_leg(self.root/'C_async500', 'C')
        (target/'step-000500.verified.json').unlink()
        with self.assertRaises(FileNotFoundError): benchmark.verify_leg(self.root/'C_async500', 'C')

    def test_valid_but_different_model_state_fails_pair_comparison(self):
        results = [benchmark.verify_leg(self.root/name, leg) for leg, name in (('B','B_sync500'),('C','C_async500'))]
        path = self.root/'C_async500'/'project_xr1_doosan'/launch.ASYNC500_NAME/'step-001500.ckpt'
        ck = tx.load_verified(path); ck['state_dict']['model.weight'].add_(1)
        torch.save(ck, path)
        receipt_path = path.with_suffix('.verified.json'); receipt = json.loads(receipt_path.read_text())
        receipt.update(sha256=tx.file_sha256(path), bytes=path.stat().st_size)
        tx.atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(RuntimeError, 'state_dict'): benchmark.verify_pair(self.root, results)
