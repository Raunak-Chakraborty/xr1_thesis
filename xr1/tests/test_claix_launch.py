import contextlib
import fcntl
import importlib.util
import io
import json
import os
import signal
import shlex
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('claix_b24_launch',ROOT/'scripts/claix_b24_launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


class LaunchTest(unittest.TestCase):
    def test_plan_does_not_launch_or_require_slurm(self):
        from mibot.utils.claix_contract import engineering_source_digest
        self.assertEqual(launch.tree_digest(),engineering_source_digest())
        with contextlib.redirect_stdout(io.StringIO()), patch.object(launch.subprocess,'Popen',side_effect=AssertionError('launch')):
            self.assertEqual(launch.main(['--run-root','/tmp/dry-plan-only']),0)

    def test_execute_fails_without_allocation_or_release(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'Slurm allocation'):
                launch.main(['--execute','--run-root','/tmp/dry-plan-only'])
            with self.assertRaisesRegex(RuntimeError,'release record'):
                launch.main(['--execute','--mode','production','--run-root','/tmp/dry-plan-only'])

    def test_unverified_gates_and_hyperparameter_override_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            record=Path(folder)/'release.json'
            record.write_text(json.dumps(dict(source_tree_sha256=launch.tree_digest(),plan={'test':True},gates={})))
            with self.assertRaisesRegex(RuntimeError,'Unqualified production gate'):
                launch.check_release(record,{'test':True})
            overrides=Path(folder)/'overrides.json';overrides.write_text('["trainer.max_steps=1"]')
            with self.assertRaisesRegex(RuntimeError,'four staged path'):
                launch.main(['--run-root',folder,'--overrides',str(overrides)])

    def test_allocation_walltime_and_deadline(self):
        with patch.dict(os.environ,SLURM_JOB_ID='synthetic'), patch.object(launch.subprocess,'run') as run:
            run.return_value.stdout='JobId=1 TimeLimit=24:00:00 EndTime=2026-10-31T18:00:00'
            with self.assertRaisesRegex(RuntimeError,'23:45:00'):
                launch.allocation_deadline()
            run.return_value.stdout='JobId=1 TimeLimit=23:45:00 EndTime=2026-10-31T18:00:00 AllocTRES=cpu=12,mem=96G,gres/gpu=1,gres/gpu:h100=1'
            with patch.object(launch.subprocess,'check_output',return_value='10000'),patch.object(launch.time,'time',return_value=100):
                self.assertEqual(launch.allocation_deadline(),8200)

    def test_short_qualification_allocations_and_failures(self):
        with patch.dict(os.environ,SLURM_JOB_ID='synthetic'), \
             patch.object(launch.subprocess,'run') as run, \
             patch.object(launch.subprocess,'check_output',return_value='10000'), \
             patch.object(launch.time,'time',return_value=100):
            for mode, limit, tres, expected in (
                ('cpu-qualification','00:20:00','cpu=12,mem=48G',9880),
                ('h100-qualification','00:45:00','cpu=12,mem=98304M,gres/gpu=1',9400)):
                with self.subTest(mode=mode):
                    run.return_value.stdout=f'TimeLimit={limit} EndTime=2026-10-31T18:00:00 AllocTRES={tres}'
                    self.assertEqual(launch.allocation_deadline(mode),expected)
            for fields, mode in (
                ('TimeLimit=23:45:00','h100-qualification'),
                ('TimeLimit=UNLIMITED','cpu-qualification'),
                ('TimeLimit=00:01:00','cpu-qualification'),
                ('TimeLimit=00:20:00 AllocTRES=cpu=12,mem=48G,gres/gpu=1','cpu-qualification'),
                ('TimeLimit=00:20:00 AllocTRES=cpu=8,mem=48G','cpu-qualification'),
                ('TimeLimit=00:45:00 AllocTRES=cpu=12,mem=48G,gres/gpu=1','h100-qualification'),
                ('TimeLimit=00:20:00 AllocTRES=cpu=12,mem=48G','cpu-qualification')):
                run.return_value.stdout=fields
                with self.subTest(fields=fields), self.assertRaises(RuntimeError):
                    launch.allocation_deadline(mode)
            run.return_value.stdout='TimeLimit=00:20:00 EndTime=2026-10-31T18:00:00 AllocTRES=cpu=12,mem=48G'
            with patch.object(launch.time,'time',return_value=9900), self.assertRaisesRegex(RuntimeError,'margin'):
                launch.allocation_deadline('cpu-qualification')

    def test_gpu_command_hydra_composition_with_paths_and_resume(self):
        from hydra import compose, initialize_config_dir
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            launch.main(['--mode','h100-qualification','--workers','8','--run-root','/tmp/qualification space',
                         '--resume','/tmp/checkpoint space/step-000003.ckpt','--stop-after-step','5'])
        cmd=shlex.split(output.getvalue().split('COMMAND=')[1])
        self.assertEqual(cmd[1:6],['-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(ROOT/'xr1/tools/train_doosan_compact.py')])
        with initialize_config_dir(config_dir=str(ROOT/'xr1/configs'),version_base=None):
            cfg=compose(config_name='claix_b24_production',overrides=cmd[7:])
        self.assertEqual(cfg.trainer.max_steps,6)
        self.assertEqual(cfg.data.params.max_steps,48284)
        self.assertEqual(cfg.trainer.scheduler.params.num_training_steps,48284)
        self.assertEqual(cfg.data.params.train_datasets.num_workers,8)
        self.assertEqual(cfg.trainer.default_root_dir,'/tmp/qualification space')
        self.assertEqual(cfg.trainer.ckpt_path,'/tmp/checkpoint space/step-000003.ckpt')
        self.assertEqual(cfg.production.stop_after_step,5)

    def test_cpu_qualification_needs_no_release_and_hides_cuda(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(launch,'allocation_deadline',return_value=10000000000) as deadline, \
             patch.object(launch.subprocess,'Popen') as child, \
             patch.dict(os.environ,XR1_DOOSAN_ZIP_CACHE_ROOT='/synthetic/cache'):
            child.return_value.wait.return_value=0
            self.assertEqual(launch.main(['--mode','cpu-qualification','--execute','--run-root',folder]),0)
            deadline.assert_called_once_with('cpu-qualification')
            cmd=child.call_args.args[0]; env=child.call_args.kwargs['env']
            self.assertIn('claix_b24_worker_qualification.py',cmd[1])
            self.assertNotIn('torch.distributed.run',cmd)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['HF_DATASETS_OFFLINE'],'1')
            self.assertEqual(env['WANDB_MODE'],'disabled')
            self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            self.assertEqual(env['PYTHONPATH'].split(os.pathsep)[0],str(ROOT/'xr1'))
            self.assertEqual(env['OMP_NUM_THREADS'],'1')

    def test_production_checks_every_gate_and_evidence_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence=Path(folder)/'evidence.txt'; evidence.write_text('synthetic gate fixture only')
            record=Path(folder)/'release.json'
            row=dict(status='PASS',evidence=str(evidence),sha256=launch.hashlib.sha256(evidence.read_bytes()).hexdigest())
            release=dict(source_tree_sha256=launch.tree_digest(),plan={'fixture':True},
                         gates={key:dict(row) for key in launch.GATES})
            record.write_text(json.dumps(release))
            launch.check_release(record,{'fixture':True})
            for gate in launch.GATES:
                release['gates'][gate]['status']='UNVERIFIED'; record.write_text(json.dumps(release))
                with self.subTest(gate=gate), self.assertRaisesRegex(RuntimeError,'Unqualified'):
                    launch.check_release(record,{'fixture':True})
                release['gates'][gate]['status']='PASS'
            record.write_text(json.dumps(release)); evidence.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                launch.check_release(record,{'fixture':True})

    def test_duplicate_run_lock(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with (Path(folder)/'.production.lock').open('a') as first:
                fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(launch,'allocation_deadline',return_value=10000), \
                     patch.object(launch,'check_release'):
                    for mode in launch.PROFILES:
                        with self.subTest(mode=mode), self.assertRaises(BlockingIOError):
                            launch.main(['--execute','--mode',mode,'--run-root',folder,'--release-record','synthetic'])

    def test_existing_results_fail_before_starting_child(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(launch,'allocation_deadline',return_value=10000000000), \
             patch.object(launch.subprocess,'Popen',side_effect=AssertionError('must not start')):
            (Path(folder)/'workers.json').write_text('synthetic prior evidence')
            with self.assertRaisesRegex(RuntimeError,'contains state'):
                launch.main(['--execute','--mode','cpu-qualification','--run-root',folder])
            target=Path(folder)/'project_xr1_doosan/qualification_b24_w4'
            target.mkdir(parents=True); (target/'telemetry.jsonl').write_text('synthetic prior run')
            with self.assertRaisesRegex(RuntimeError,'contains state'):
                launch.main(['--execute','--mode','h100-qualification','--run-root',folder])

    def test_controller_signal_and_persistent_logs_without_training(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            environment={}
            class ChildFixture:
                calls=0
                def wait(self,timeout):
                    self.calls+=1
                    if self.calls==1:
                        signal.raise_signal(signal.SIGUSR1)
                        raise launch.subprocess.TimeoutExpired('mock worker',timeout)
                    request=Path(environment['XR1_STOP_REQUEST_FILE'])
                    self.assert_request=json.loads(request.read_text())
                    if self.assert_request['signal']!=signal.SIGUSR1:
                        raise AssertionError('wrong signal')
                    return 0
            def fake_child(command,**kwargs):
                environment.update(kwargs['env'])
                kwargs['stdout'].write('SYNTHETIC_WORKER_ONLY\n');kwargs['stdout'].flush()
                return ChildFixture()
            old={sig:signal.getsignal(sig) for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
            try:
                with patch.object(launch,'allocation_deadline',return_value=10000000000), \
                     patch.object(launch.subprocess,'Popen',side_effect=fake_child), \
                     patch.dict(os.environ,XR1_DOOSAN_ZIP_CACHE_ROOT='/synthetic/cache'):
                    self.assertEqual(launch.main(['--execute','--run-root',folder]),0)
                self.assertIn('SYNTHETIC_WORKER_ONLY',next(Path(folder).glob('launch-*.out')).read_text())
                self.assertTrue(list(Path(folder).glob('launch-*.err')))
                self.assertFalse(list(Path(folder).glob('.stop-request*')))
            finally:
                for sig,handler in old.items():
                    signal.signal(sig,handler)
