"""Synthetic CPU tests: never construct XR-1, scan frozen data, or contact Slurm."""
import ast
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from test_claix_launch import ROOT, launch
from test_claix_checkpoint import checkpoint_fixture
from test_claix_final_release import reviewed_fixture, release
from mibot.utils.claix_checkpoint import atomic_json, file_sha256, load_verified, publish_checkpoint, rotate_checkpoints
from mibot.utils.claix_contract import BASELINE, FROZEN_TRAIN_SHA256, resume_fingerprint
from mibot.utils.claix_experiment import bind_experiment, load_experimental_resume
import claix_b24_filesystem as filesystem
import claix_b24_storage_preflight as storage


def merge(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(left.get(key), dict):
            merge(left[key], value)
        else:
            left[key] = copy.deepcopy(value)
    return left


def recipe(name):
    """Compose this repository's global YAML defaults without a Hydra dependency."""
    cfg = yaml.safe_load((ROOT/'xr1/configs'/name).with_suffix('.yaml').read_text())
    defaults = cfg.pop('defaults', [])
    result = {}
    for item in defaults:
        if item == '_self_':
            continue
        path = next(iter(item.items())) if isinstance(item, dict) else None
        merge(result, recipe('/'.join(path) if path else item))
    return merge(result, cfg)


def effective_command(output):
    command = shlex.split(output.split('COMMAND=')[1].splitlines()[0])
    config = recipe(command[6].split('=', 1)[1])
    for override in command[7:]:
        key, value = override.split('=', 1)
        cursor = config
        keys = key.split('.')
        for part in keys[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[keys[-1]] = yaml.safe_load(value)
    return command, config


class ExperimentalLaunchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'new-run'
        self.overrides = self.root/'paths.json'
        self.overrides.write_text(json.dumps([
            'data.params.train_datasets.paths=["/hpcwork/SYNTHETIC/train"]',
            'model.params.pretrained="/hpcwork/SYNTHETIC/model_states.pt"',
            'model.params.language_policy_path="/hpcwork/SYNTHETIC/language.yaml"',
            'trainer.default_root_dir="/hpcwork/SYNTHETIC/unused"']))
        self.args = ['--mode', 'experimental', '--experimental-unqualified', '--run-root', str(self.run),
                     '--overrides', str(self.overrides), '--filesystem-source', 'test-lustre', '--project-id', '44885']
        self.output = io.StringIO()
        patches = [patch.object(launch, 'checkout_identity', return_value=dict(git_commit='synthetic', git_status='')),
                   patch.object(launch, 'allocation_deadline', return_value=time.time()+1000),
                   patch.object(launch, 'check_release', side_effect=AssertionError('release validator called')),
                   patch.object(storage, 'experimental_evaluate', return_value=dict(status='PASS', abort_reasons=[])),
                   patch.object(filesystem, 'inspect_directory', return_value=dict(filesystem=dict(fstype='lustre'))),
                   patch.object(launch.subprocess, 'Popen'), patch.object(launch.NvidiaSmiMonitor, 'start'),
                   patch.object(launch.NvidiaSmiMonitor, 'close'),
                   patch.dict(os.environ, XR1_DOOSAN_ZIP_CACHE_ROOT='/hpcwork/SYNTHETIC/cache',
                              XR1_NVIDIA_SMI_MONITOR='1'), contextlib.redirect_stdout(self.output)]
        self.mocks = [p.start() if hasattr(p, 'start') else p.__enter__() for p in patches]
        for p in patches:
            self.addCleanup(p.stop if hasattr(p, 'stop') else p.__exit__, *(() if hasattr(p, 'stop') else (None, None, None)))
        self.child = self.mocks[5]
        self.child.return_value.wait.return_value = 0

    def execute(self, extra=()):
        return launch.main([*self.args, '--execute', *extra])

    def test_explicit_double_opt_in_and_no_cross_mode_fallback(self):
        for extra in (['--mode', 'experimental'], ['--experimental-unqualified'],
                      ['--mode', 'production', '--experimental-unqualified']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main(['--run-root', str(self.run), '--overrides', str(self.overrides), *extra])
        self.child.assert_not_called()

    def test_initial_launch_bypasses_only_release_and_starts_at_zero(self):
        self.assertEqual(self.execute(), 0)
        cmd = self.child.call_args.args[0]
        cfg = effective_command(self.output.getvalue())[1]
        self.assertIsNone(cfg['trainer']['ckpt_path'])
        self.assertNotIn('--resume', cmd)
        self.assertNotIn('21500', ' '.join(cmd))
        self.assertEqual(cfg['trainer']['max_steps'], 48284)
        self.assertTrue(cfg['production']['experimental_unqualified'])
        self.assertEqual(self.child.call_args.kwargs['env']['XR1_EXPERIMENTAL_UNQUALIFIED'], '1')
        diagnostics = json.loads(Path(self.child.call_args.kwargs['env']['XR1_LOADER_DIAGNOSTICS_JSON']).read_text())
        self.assertEqual(diagnostics['persistent_dir'], str(self.run/'diagnostics'))
        self.assertEqual(diagnostics['thresholds'], [30, 90, 150])
        self.assertTrue(diagnostics['reserve_sigusr2'])
        self.mocks[1].assert_called_once_with('experimental', allocation_record={})
        self.mocks[3].assert_called_once()
        self.mocks[6].assert_called_once()
        provenance = json.loads(next(self.run.glob('launch-*.provenance.json')).read_text())
        self.assertTrue(provenance['experimental_unqualified'])
        self.assertEqual(provenance['initial_checkpoint_mode'], 'original_pretrained_step_zero')
        self.assertEqual(provenance['git_commit'], 'synthetic')
        self.assertEqual(provenance['source_tree_sha256'], launch.tree_digest())
        self.assertIsNone(provenance['release_record'])
        self.assertTrue(list(self.run.glob('launch-*.out')))
        self.assertTrue(list(self.run.glob('launch-*.err')))
        self.assertTrue(list(self.run.glob('launch-*.storage.json')))

    def test_dry_run_is_read_only_and_frozen_recipe_unchanged(self):
        self.assertEqual(launch.main(self.args), 0)
        self.assertFalse(self.run.exists())
        _, cfg = effective_command(self.output.getvalue())
        strict = recipe('claix_b24_production')
        self.assertEqual(cfg['data'], {**strict['data'], 'params': {**strict['data']['params'],
            'train_datasets': {**strict['data']['params']['train_datasets'], 'paths': ['/hpcwork/SYNTHETIC/train']}}})
        self.assertEqual(cfg['model']['params']['model'], strict['model']['params']['model'])
        for key in ('optimizer', 'scheduler', 'precision', 'accumulate_grad_batches', 'gradient_clip_val', 'save_interval'):
            self.assertEqual(cfg['trainer'][key], strict['trainer'][key])
        self.assertEqual(cfg['model']['params']['pretrained_sha256'], '94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7')
        loader = cfg['data']['params']['train_datasets']
        self.assertEqual((loader['num_workers'], loader['prefetch_factor'], loader['worker_timeout']), (4, 2, 600))
        self.assertTrue(loader['persistent_workers'] and loader['in_order'])
        self.assertEqual(loader['zip_integrity_policy'], 'immutable')

    def test_watchdog_preserves_timing_options_and_binds_persistent_destination(self):
        path = self.root/'existing-diagnostics.json'
        path.write_text(json.dumps(dict(local_dir='/tmp', persistent_dir='/tmp/old', thresholds=[20, 60, 120],
                                        poll_seconds=.5, reserve_sigusr2=True)))
        with patch.dict(os.environ, XR1_LOADER_DIAGNOSTICS_JSON=str(path)):
            self.execute()
        effective = json.loads(Path(self.child.call_args.kwargs['env']['XR1_LOADER_DIAGNOSTICS_JSON']).read_text())
        self.assertEqual(effective['persistent_dir'], str(self.run/'diagnostics'))
        self.assertEqual(effective['thresholds'], [20, 60, 120])
        self.assertEqual(effective['poll_seconds'], .5)

    def test_storage_failure_prevents_directory_and_child(self):
        self.mocks[3].return_value = dict(status='FAIL', abort_reasons=['inodes failed'])
        with self.assertRaisesRegex(RuntimeError, 'before model loading'):
            self.execute()
        self.assertFalse(self.run.exists())
        self.child.assert_not_called()
        self.mocks[3].side_effect = RuntimeError('unapproved mount')
        with self.assertRaisesRegex(RuntimeError, 'unapproved'):
            self.execute()

    def test_no_local_cache_no_foreign_bridge_no_release_record(self):
        with patch.dict(os.environ, XR1_DOOSAN_OFFLINE_FRAME_ROOT='/tmp/staged'):
            with self.assertRaisesRegex(RuntimeError, 'Lustre ZIP'):
                self.execute()
        for extra in (['--release-record', '/historical/release.json'], ['--workers', '8']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.execute(extra)
        with patch.dict(os.environ, XR1_OPTION_B_RESUME_BRIDGE='EXPLICIT_STEP_21500'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.execute()
        self.child.assert_not_called()

    def test_unique_root_and_inherited_run_lock(self):
        self.execute()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertIn('pass_fds', self.child.call_args.kwargs)
        self.assertTrue(self.child.call_args.kwargs['pass_fds'])

    def same_run_checkpoint(self):
        self.execute()
        identity = launch.read_experiment(self.run)
        target = self.run/'project_xr1_doosan'/launch.EXPERIMENT_NAME
        target.mkdir(parents=True)
        ck = checkpoint_fixture(2, 48284)
        ck['xr1_production']['experiment'] = dict(schema='claix-b24-experimental-checkpoint-v1',
            experimental_unqualified=True, run_id=identity['run_id'], run_root=str(self.run), zip_manifest_sha256='b'*64)
        path = target/'step-000002.ckpt'
        publish_checkpoint(ck, path)
        (target/'telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in [
            dict(event='train_start', step=0), dict(event='step', step=1), dict(event='step', step=2)]))
        return identity, path

    def test_recovery_requires_same_root_source_config_latest_and_lock(self):
        identity, path = self.same_run_checkpoint()
        self.assertEqual(self.execute(['--resume', str(path)]), 0)
        with (self.run/'.production.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute(['--resume', str(path)])
        changed = copy.deepcopy(identity)
        for key in ('git_commit', 'source_tree_sha256'):
            changed[key] = 'changed'
            atomic_json(self.run/'experiment.json', changed)
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                self.execute(['--resume', str(path)])
            changed = copy.deepcopy(identity)
        atomic_json(self.run/'experiment.json', identity)
        (path.parent/'telemetry.jsonl').write_text('{"event":"train_start","step":0}\n')
        with self.assertRaisesRegex(RuntimeError, 'Telemetry'):
            self.execute(['--resume', str(path)])

    def test_historical_checkpoint_rejected_even_with_forged_step_name(self):
        self.execute()
        foreign = self.root/'step-021500.ckpt'
        publish_checkpoint(checkpoint_fixture(21500, 48284), foreign)
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.execute(['--resume', str(foreign)])
        self.child.reset_mock()
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            launch.main([*self.args, '--resume', '/historical/qualification.ckpt'])
        self.child.assert_not_called()

    def test_controller_signal_still_writes_persistent_safe_stop_request(self):
        env = {}
        class Child:
            calls = 0
            def wait(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    signal.raise_signal(signal.SIGUSR1)
                    raise subprocess.TimeoutExpired('synthetic', timeout)
                request = json.loads(Path(env['XR1_STOP_REQUEST_FILE']).read_text())
                if request['signal'] != signal.SIGUSR1:
                    raise AssertionError(request)
                return 0
        def start(cmd, **kwargs):
            env.update(kwargs['env'])
            return Child()
        self.child.side_effect = start
        self.execute()
        self.assertTrue(list(self.run.glob('launch-*.stop.json')))
        self.assertFalse(list(self.run.glob('.stop-request*')))


class AllocationTest(unittest.TestCase):
    def fields(self):
        return dict(JobId='123', JobState='RUNNING', Account='thes2452', Partition='c23g',
                    NumNodes='1', NumTasks='1', NumCPUs='12', **{'CPUs/Task': '12'},
                    TimeLimit='20:00:00', EndTime='future',
                    AllocTRES='cpu=12,mem=96G,gres/gpu=1,gres/gpu:h100=1',
                    StdOut='/hpcwork/test/log.out', StdErr='/hpcwork/test/log.err', WorkDir='/hpcwork/test')

    def validate(self, fields, gpu='NVIDIA H100 80GB HBM3', batch=None):
        def output(cmd, **kwargs):
            if cmd[0] == 'date': return '10000'
            if cmd[0] == 'nvidia-smi': return gpu+'\n'
            return batch if batch is not None else (ROOT/'scripts/claix_b24_experimental.sbatch').read_text()
        with patch.dict(os.environ, SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='0'), \
             patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items()))), \
             patch.object(launch.subprocess, 'check_output', side_effect=output), \
             patch.object(launch.time, 'time', return_value=100), \
             patch.object(filesystem, 'inspect_directory'):
            return launch.allocation_deadline('experimental')

    def test_exact_allocation_and_1800_margin(self):
        self.assertEqual(self.validate(self.fields()), 8200)

    def test_wrong_allocation_or_missing_slurm_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'Slurm allocation'):
            launch.allocation_deadline('experimental')
        for key, value in [('JobId','456'), ('JobState','PENDING'), ('Account','other'), ('Partition','other'),
                           ('NumNodes','2'), ('NumTasks','2'), ('NumCPUs','24'), ('CPUs/Task','6'),
                           ('TimeLimit','23:45:00'), ('TimeLimit','19:00:00'), ('TimeLimit','UNLIMITED'),
                           ('AllocTRES','cpu=24,mem=96G,gres/gpu=1'), ('AllocTRES','cpu=12,mem=97G,gres/gpu=1'),
                           ('AllocTRES','cpu=12,mem=95G,gres/gpu=1'), ('AllocTRES','cpu=12,mem=96G,gres/gpu=2'),
                           ('EndTime','Unknown')]:
            fields = self.fields(); fields[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.validate(fields)
        for gpu in ('NVIDIA A100', 'NVIDIA H100\nNVIDIA H100'):
            with self.assertRaisesRegex(RuntimeError, 'H100'):
                self.validate(self.fields(), gpu=gpu)
        with self.assertRaisesRegex(RuntimeError, 'wrapper'):
            self.validate(self.fields(), batch='#!/bin/bash\n')

    def test_batch_wrapper_syntax_resources_and_strict_unchanged(self):
        script = ROOT/'scripts/claix_b24_experimental.sbatch'
        subprocess.run(['bash', '-n', str(script)], check=True)
        content = script.read_text()
        for directive in ('account=thes2452', 'partition=c23g', 'gres=gpu:h100:1', 'cpus-per-task=12',
                          'mem=96G', 'time=20:00:00', 'signal=B:USR1@1800', 'nodes=1', 'ntasks=1'):
            self.assertIn('#SBATCH --'+directive+'\n', content)
        self.assertIn('--experimental-unqualified', content)
        self.assertNotIn('--release-record', content)
        for name in ('scripts/claix_b24_production.sbatch', 'scripts/claix_b24_release.py',
                     'xr1/configs/claix_b24_production.yaml'):
            self.assertEqual((ROOT/name).read_bytes(), subprocess.check_output(['git','show','HEAD:'+name], cwd=ROOT))


class ExperimentalStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.location = dict(resolved_path=str(self.root/'new'), existing_parent=str(self.root),
                             filesystem=dict(source='lustre', target=str(self.root), fstype='lustre'))
        self.stat = SimpleNamespace(f_bavail=40, f_frsize=1024**3, f_favail=20000)
        for p in (patch.object(storage, 'inspect_directory', return_value=self.location),
                  patch.object(storage.os, 'statvfs', return_value=self.stat),
                  patch.object(storage, 'query', side_effect=self.query)):
            p.start(); self.addCleanup(p.stop)

    def query(self, cmd):
        if cmd[1] == 'project':
            return dict(returncode=0, stdout=f'44885 P {self.root}\n')
        identity, number = ('uid', os.getuid()) if '-u' in cmd else ('pid', 44885)
        return dict(returncode=0, stdout=f'Disk quotas for test ({identity} {number}):\n'
                    'Filesystem kbytes bquota blimit bgrace files iquota ilimit igrace\n'
                    f'{self.root}\n 1000 100000000 110000000 - 10 200000 210000 -\n')

    def evaluate(self):
        return storage.experimental_evaluate(str(self.root/'new'), 'lustre', 44885)

    def test_live_free_space_inodes_and_user_project_quotas(self):
        row = self.evaluate()
        self.assertEqual(row['status'], 'PASS')
        self.assertEqual(len(row['quota_measurements']), 2)
        self.assertFalse((self.root/'new').exists())
        for field, value in (('f_bavail', 29), ('f_favail', 9999)):
            old = getattr(self.stat, field); setattr(self.stat, field, value)
            self.assertEqual(self.evaluate()['status'], 'FAIL')
            setattr(self.stat, field, old)

    def test_bad_unknown_or_exhausted_quotas_and_wrong_project_fail_closed(self):
        for result in (dict(returncode=1, stdout=''), dict(error='timeout'), dict(returncode=0, stdout='unparsed'),
                       dict(returncode=0, stdout='44886 P '+str(self.root))):
            with patch.object(storage, 'query', return_value=result):
                self.assertEqual(self.evaluate()['status'], 'FAIL')
        for replacement in (' 99999999 100000000', ' 199999 200000'):
            def exhausted(cmd):
                row = self.query(cmd)
                row['stdout'] = row['stdout'].replace(' 1000 100000000' if '99999999' in replacement else ' 10 200000', replacement)
                return row
            with patch.object(storage, 'query', side_effect=exhausted):
                self.assertEqual(self.evaluate()['status'], 'FAIL')
        self.location['filesystem']['fstype'] = 'ext4'
        with self.assertRaisesRegex(RuntimeError, 'Lustre'):
            self.evaluate()

    def test_real_filesystem_guard_rejects_tmp_and_missing_approved_source(self):
        with self.assertRaisesRegex(RuntimeError, 'approved'):
            filesystem.inspect_directory(self.root)
        with self.assertRaises(ValueError):
            storage.experimental_evaluate(self.root, '', 44885)


class ExperimentalCheckpointTest(unittest.TestCase):
    def test_real_runner_hooks_preserve_and_enforce_experimental_identity(self):
        from mibot.utils.doosan_replay import checkpoint_cursor, validate_full_compact_resume
        tree = ast.parse((ROOT/'xr1/mibot/models/runner/doosan_compact_runner.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        hooks = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('on_save_checkpoint', 'on_load_checkpoint')]
        namespace = dict(checkpoint_cursor=checkpoint_cursor, validate_full_compact_resume=validate_full_compact_resume)
        exec(compile(ast.Module(body=hooks, type_ignores=[]), 'doosan_compact_runner.py', 'exec'), namespace)
        ck = checkpoint_fixture(2, 48284)
        experiment = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                          run_id='1'*32, run_root='/hpcwork/SYNTHETIC/run', zip_manifest_sha256='b'*64)
        module = SimpleNamespace(global_step=2, _resume_step=2, _replay_contract=ck['xr1_replay_contract'],
            _checkpoint_rng=ck['xr1_production']['rng_by_rank'], _compact_meta=ck['xr1_compact_fp32'],
            _compact_cfg=dict(config_sha256='a'*64, experiment=experiment))
        del ck['xr1_production']
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['on_save_checkpoint'](module, ck)
            self.assertEqual(ck['xr1_production']['experiment'], experiment)
            namespace['on_load_checkpoint'](module, ck)
            self.assertIs(module._pending_rng, ck['xr1_production']['rng_by_rank'])
            ck['xr1_production']['experiment']['run_id'] = '2'*32
            with self.assertRaisesRegex(RuntimeError, 'experimental run'):
                namespace['on_load_checkpoint'](module, ck)

    def test_runtime_signal_publishes_only_completed_step_and_verifies_finish(self):
        from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
        with tempfile.TemporaryDirectory() as folder:
            experiment = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                              run_id='1'*32, run_root=folder, zip_manifest_sha256='b'*64)
            ck = checkpoint_fixture(1, 48284)
            ck['xr1_production']['experiment'] = experiment
            trainable = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
            module = SimpleNamespace(_replay_contract=ck['xr1_replay_contract'],
                _compact_cfg=dict(config_sha256='a'*64, experiment=experiment),
                parameters=lambda: [trainable, frozen])
            def save(path, weights_only):
                ck['xr1_production']['rng_by_rank'] = module._checkpoint_rng
                publish_checkpoint(ck, path)
            trainer = SimpleNamespace(global_step=1, world_size=1, is_global_zero=True,
                max_steps=48284, optimizers=[SimpleNamespace(param_groups=[dict(lr=5e-7)])],
                save_checkpoint=save, strategy=SimpleNamespace(barrier=lambda *_: None), should_stop=False)
            runtime = ProductionRuntime(folder, deadline=time.time()+1000, interval=250)
            runtime.started = runtime.fit_started = runtime.previous_end = time.perf_counter()-.1
            runtime.previous_step = runtime.initial_step = 0
            runtime.retained, runtime.views, runtime.tokens = 24, 48, 100
            runtime.data_pipeline_seconds = .01
            runtime.stop.handle(signal.SIGUSR1, None)
            runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
            self.assertTrue(trainer.should_stop)
            self.assertEqual(runtime.last_saved, 1)
            path = Path(folder)/'step-000001.ckpt'
            plugin = VerifiedCheckpointIO(expected_config='a'*64, expected_experiment=experiment)
            self.assertEqual(plugin.load_checkpoint(path)['global_step'], 1)
            runtime.on_fit_end(trainer, module)
            rows = [json.loads(line) for line in (Path(folder)/'telemetry.jsonl').read_text().splitlines()]
            self.assertTrue(rows[-1]['final_checkpoint_verified'])
            self.assertEqual(json.loads((Path(folder)/'safe-stop.json').read_text())['global_step'], 1)
            with self.assertRaisesRegex(RuntimeError, 'skipped or repeated'):
                runtime.on_train_batch_end(trainer, module, None, None, 0)
            with self.assertRaisesRegex(RuntimeError, 'experimental run'):
                VerifiedCheckpointIO(expected_experiment={**experiment, 'run_id':'2'*32}).load_checkpoint(path)

    def test_same_run_only_and_all_existing_verification_still_required(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                              run_id='1'*32, run_root=folder, zip_cache_root='/hpcwork/cache', zip_manifest_sha256='b'*64)
            target = Path(folder)/'project_xr1_doosan'/launch.EXPERIMENT_NAME
            target.mkdir(parents=True)
            path = target/'step-000002.ckpt'
            ck = checkpoint_fixture(2, 48284)
            ck['xr1_production']['experiment'] = experiment
            publish_checkpoint(ck, path)
            loaded = load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, experiment)
            self.assertEqual(loaded['global_step'], 2)
            for field in ('run_id', 'zip_manifest_sha256', 'run_root'):
                foreign = dict(experiment); foreign[field] = 'changed'
                with self.subTest(field=field), self.assertRaises((RuntimeError, FileNotFoundError)):
                    load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, foreign)
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                load_experimental_resume(path, ck['xr1_replay_contract'], 'b'*64, experiment)
            old = target/'step-021500.ckpt'
            publish_checkpoint(checkpoint_fixture(21500, 48284), old)
            with patch.object(torch, 'load', side_effect=AssertionError('must reject before deserialization')), \
                 self.assertRaisesRegex(RuntimeError, 'experimental run'):
                load_experimental_resume(old, ck['xr1_replay_contract'], 'a'*64, experiment)
            with path.open('ab') as stream: stream.write(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                load_experimental_resume(path, ck['xr1_replay_contract'], 'a'*64, experiment)

    def test_run_identity_changes_resume_fingerprint(self):
        cfg = recipe('claix_b24_experimental')
        inventory = dict(sha256=FROZEN_TRAIN_SHA256)
        first = resume_fingerprint(cfg, inventory, 12288, experiment=dict(run_id='1'*32))
        second = resume_fingerprint(cfg, inventory, 12288, experiment=dict(run_id='2'*32))
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, resume_fingerprint(cfg, inventory, 12288))

    def test_publication_retention_and_safe_stop_pointer_for_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = dict(schema='claix-b24-experimental-checkpoint-v1', experimental_unqualified=True,
                              run_id='1'*32, run_root=folder, zip_manifest_sha256='b'*64)
            for step in (1, 250, 500, 750):
                ck = checkpoint_fixture(step, 48284); ck['xr1_production']['experiment'] = experiment
                path = Path(folder)/f'step-{step:06d}.ckpt'
                receipt = publish_checkpoint(ck, path)
                if step == 1: atomic_json(Path(folder)/'safe-stop.json', receipt)
                rotate_checkpoints(folder, expected_config='a'*64)
            self.assertEqual(sorted(p.name for p in Path(folder).glob('*.ckpt')),
                             ['step-000001.ckpt', 'step-000500.ckpt', 'step-000750.ckpt'])
            self.assertFalse(list(Path(folder).glob('*.tmp')))
            with self.assertRaises(FileExistsError): publish_checkpoint(ck, path)


class StrictReleaseRegressionTest(unittest.TestCase):
    def test_all_nine_unapproved_gates_remain_blocking(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); record = reviewed_fixture(root); path = root/'release.json'
            for gate in launch.GATES:
                record['gates'][gate]['status'] = 'UNVERIFIED'
                path.write_text(json.dumps(record))
                with self.subTest(gate=gate), self.assertRaisesRegex(RuntimeError, 'Unqualified production gate'):
                    launch.check_release(path, record['plan'])
                record['gates'][gate]['status'] = 'PASS'


class Config(dict):
    """Explicit minimal mmengine boundary stand-in, no registry/model creation."""
    def __init__(self, value):
        super().__init__({k: Config(v) if isinstance(v, dict) else copy.deepcopy(v) for k,v in value.items()})
    def __getattr__(self, key): return self[key]
    __setattr__ = dict.__setitem__
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k,v in self.items()}


class EntrypointTest(unittest.TestCase):
    def test_actual_main_function_uses_original_base_new_optimizer_scheduler_and_step_zero(self):
        from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root/'cache'; cache.mkdir()
            (cache/'media_manifest.jsonl').write_text('SYNTHETIC\n')
            cfg = recipe('claix_b24_experimental')
            cfg['trainer']['default_root_dir'] = folder
            cfg['production']['experimental_run_id'] = '1'*32
            for path_key, hash_key in [('pretrained','pretrained_sha256'), ('language_policy_path','language_sha256')]:
                path = root/path_key; path.write_bytes(b'SYNTHETIC BASE ONLY')
                cfg['model']['params'][path_key] = str(path)
                cfg['model']['params'][hash_key] = file_sha256(path)
            atomic_json(root/'experiment.json', dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                run_id='1'*32, source_tree_sha256=launch.tree_digest(), plan=dict(run_root=folder)))
            inventory = dict(episodes=255, frames=289703, sha256=FROZEN_TRAIN_SHA256)
            model_builder = Mock(return_value=SimpleNamespace())
            trainer = Mock(); trainer.train_dataloader = None
            trainer_type = Mock(return_value=trainer)
            source = ast.parse((ROOT/'xr1/tools/train_doosan_compact.py').read_text())
            main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            main.decorator_list = []; main.returns = None
            main.args.args[0].annotation = None
            namespace = dict(Config=Config, OmegaConf=SimpleNamespace(to_container=lambda value, **_: copy.deepcopy(value)),
                torch=torch, os=os, Path=Path, json=json, BASELINE=BASELINE, resume_fingerprint=resume_fingerprint,
                split_inventory=Mock(return_value=inventory), seed_everything=Mock(),
                DDPStrategy=lambda **_: 'synthetic-ddp', make_contract=__import__('mibot.utils.doosan_replay',fromlist=['make_contract']).make_contract,
                ModelCheckpoint=Mock(), ModelSummary=Mock(), CSVLogger=Mock(), ProductionRuntime=ProductionRuntime,
                VerifiedCheckpointIO=VerifiedCheckpointIO, DATASETS=SimpleNamespace(build=Mock()),
                MIMODEL=SimpleNamespace(build=model_builder), Trainer=trainer_type)
            exec(compile(ast.Module(body=[main], type_ignores=[]), 'train_doosan_compact.py', 'exec'), namespace)
            shutdown_module = SimpleNamespace(shutdown_loader=Mock())
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_SAFE_STOP_EPOCH=str(time.time()+1000),
                            XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache), WORLD_SIZE='1', MAX_LENGTH='12288'), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.cuda, 'get_device_name', return_value='SYNTHETIC H100'), \
                 patch.dict(sys.modules, {'mibot.data.datamodule.doosan_datamodule':shutdown_module}), \
                 contextlib.redirect_stdout(io.StringIO()):
                namespace['main'](cfg)
                # Every failure must occur before registry/model construction.
                for change in ('weights', 'dataset', 'loader', 'run_id', 'historical'):
                    bad = copy.deepcopy(cfg)
                    model_builder.reset_mock()
                    if change == 'weights': bad['model']['params']['pretrained_sha256'] = '0'*64
                    if change == 'dataset': inventory['sha256'] = '0'*64
                    if change == 'loader': bad['data']['params']['train_datasets']['worker_timeout'] = 60
                    if change == 'run_id': bad['production']['experimental_run_id'] = '2'*32
                    if change == 'historical':
                        old = root/'historical.ckpt'
                        publish_checkpoint(checkpoint_fixture(21500, 48284), old)
                        bad['trainer']['ckpt_path'] = str(old)
                    with self.subTest(change=change), self.assertRaises((RuntimeError, FileNotFoundError)):
                        namespace['main'](bad)
                    model_builder.assert_not_called()
                    inventory['sha256'] = FROZEN_TRAIN_SHA256
                # Restore the successful call for the assertions below.
                namespace['main'](cfg)
            fit = trainer.fit.call_args.kwargs
            self.assertIsNone(fit['ckpt_path'])
            params = model_builder.call_args.args[0]['params']
            self.assertEqual(params['compact_fp32']['resume_step'], 0)
            self.assertEqual(params['pretrained'], cfg['model']['params']['pretrained'])
            self.assertEqual(namespace['DATASETS'].build.call_args.args[0]['params']['train_datasets']['resume_step'], 0)
            self.assertEqual(params['optimizer'], cfg['trainer']['optimizer'])
            self.assertEqual(params['scheduler'], cfg['trainer']['scheduler'])
            # The unchanged optimizer/schedule really start empty and at step zero.
            from mibot.utils.cosine_warmup import get_cosine_schedule_with_warmup
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], **params['optimizer']['params'])
            scheduler = get_cosine_schedule_with_warmup(optimizer, **params['scheduler']['params'])
            self.assertFalse(optimizer.state)
            self.assertEqual(scheduler.last_epoch, 0)
            self.assertEqual(optimizer.param_groups[0]['lr'], 5e-7)
            start = json.loads(next(root.rglob('experimental-start-*.json')).read_text())
            self.assertEqual(start['initial_global_step'], 0)
            self.assertTrue(start['new_optimizer_and_scheduler'])
            self.assertEqual(start['dataset_identity'], FROZEN_TRAIN_SHA256)
            self.assertEqual(trainer_type.call_args.kwargs['callbacks'][1].interval, 250)
            self.assertIsInstance(trainer_type.call_args.kwargs['plugins'][0], VerifiedCheckpointIO)
            target = root/'project_xr1_doosan'/launch.EXPERIMENT_NAME
            ck = checkpoint_fixture(2, 48284)
            ck['xr1_production'].update(config_sha256=start['config_sha256'], experiment=start['experiment'])
            ck['xr1_compact_fp32'].update(pretrained_sha256=cfg['model']['params']['pretrained_sha256'],
                language_sha256=cfg['model']['params']['language_sha256'], source_commit=BASELINE)
            path = target/'step-000002.ckpt'
            publish_checkpoint(ck, path)
            recovered = copy.deepcopy(cfg); recovered['trainer']['ckpt_path'] = str(path)
            bridge = SimpleNamespace(authorize_original_resume=Mock(side_effect=AssertionError('historical bridge')))
            with patch.dict(os.environ, XR1_EXPERIMENTAL_UNQUALIFIED='1', XR1_SAFE_STOP_EPOCH=str(time.time()+1000),
                            XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache), WORLD_SIZE='1', MAX_LENGTH='12288'), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.cuda, 'get_device_name', return_value='SYNTHETIC H100'), \
                 patch.dict(sys.modules, {'mibot.data.datamodule.doosan_datamodule':shutdown_module,
                                          'mibot.utils.claix_resume_bridge':bridge}), \
                 contextlib.redirect_stdout(io.StringIO()):
                namespace['main'](recovered)
            self.assertEqual(trainer.fit.call_args.kwargs['ckpt_path'], str(path))
            self.assertEqual(model_builder.call_args.args[0]['params']['compact_fp32']['resume_step'], 2)
            starts = [json.loads(p.read_text()) for p in target.glob('experimental-start-*.json')]
            self.assertTrue(any(row['initial_global_step'] == 2 and not row['new_optimizer_and_scheduler'] for row in starts))
            bridge.authorize_original_resume.assert_not_called()
