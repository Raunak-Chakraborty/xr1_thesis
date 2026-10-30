import contextlib
import fcntl
import importlib.util
import io
import json
import os
import signal
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
            run.return_value.stdout='JobId=1 TimeLimit=23:45:00 EndTime=2026-10-31T18:00:00'
            with patch.object(launch.subprocess,'check_output',return_value='10000'),patch.object(launch.time,'time',return_value=100):
                self.assertEqual(launch.allocation_deadline(),8200)

    def test_duplicate_run_lock(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with (Path(folder)/'.production.lock').open('a') as first:
                fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(launch,'allocation_deadline',return_value=10000):
                    with self.assertRaises(BlockingIOError):
                        launch.main(['--execute','--run-root',folder])

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
