"""Independent Task04C CPU regressions; site measurements are explicit fixtures."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import weakref

import torch

import claix_b24_launch as launch
import claix_b24_checkpoint_two_way as two_way
import claix_b24_task04c_cpu as acceptance
import claix_b24_task04c_preflight as preflight
from mibot.utils import claix_checkpoint as tx
from mibot.utils.claix_async_checkpoint import AsyncVerifiedCheckpointIO
from mibot.utils.claix_process import process_identity, process_alive, writer_lifetime_lock
import test_claix_async_checkpoint as async_tests


class LifecycleHardeningTest(unittest.TestCase):
    setUp = async_tests.AsyncCheckpointTest.setUp
    cleanup_writer = async_tests.AsyncCheckpointTest.cleanup_writer
    submit = async_tests.AsyncCheckpointTest.submit
    gate = async_tests.AsyncCheckpointTest.gate
    def test_chained_failure_releases_snapshot(self):
        references = []
        def broken(snapshot, *args, **kwargs):
            references.append(weakref.ref(snapshot['state_dict']['model.weight']))
            try:
                raise OSError('inner I/O failure')
            except OSError as inner:
                raise RuntimeError('outer publication failure') from inner
        with patch.object(tx, 'publish_checkpoint', new=broken):
            self.submit(500)
            self.writer._thread.join()
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                self.writer.collect(wait=True)
        self.assertIsNone(references[0]())
        self.assertIsNone(self.writer._error.__cause__.__traceback__)

    def test_stuck_writer_close_waits_without_abandoning_or_daemonizing(self):
        # Timer observes owner blocked in close, then releases simulated I/O.
        # No deliberately unkillable process or filesystem hang is created.
        entered, release, blocked = self.gate()
        observed = []
        with patch.object(tx, 'publish_checkpoint', new=blocked):
            self.submit(500)
            self.assertTrue(entered.wait(2))
            with self.assertRaises(BlockingIOError): writer_lifetime_lock(self.root, probe=True)
            thread = self.writer._thread
            def observe_and_release():
                observed.append((thread.is_alive(), thread.daemon, self.writer._closed))
                release.set()
            timer = threading.Timer(.05, observe_and_release)
            timer.start()
            self.writer.close()
            timer.join()
        self.assertEqual(observed, [(True, False, True)])
        self.assertFalse(thread.is_alive())
        writer_lifetime_lock(self.root, probe=True)
        self.writer.close(); self.writer.teardown()

    def test_failed_staging_cannot_be_retried(self):
        self.writer.arm('interval', time.perf_counter())
        with self.assertRaises(TypeError):
            self.writer.save_checkpoint({'opaque': object()}, self.root/'step-000500.ckpt')
        with self.assertRaisesRegex(RuntimeError, 'writer failed'):
            self.writer.arm('interval', time.perf_counter())

    def test_owner_thread_enforced(self):
        errors = []
        def close():
            try: self.writer.close()
            except Exception as exc: errors.append(str(exc))
        thread = threading.Thread(target=close); thread.start(); thread.join()
        self.assertIn('training thread', errors[0])

    def test_thread_start_failure_is_sticky_and_retains_no_snapshot(self):
        with patch.object(threading.Thread, 'start', side_effect=RuntimeError('cannot start thread')):
            with self.assertRaisesRegex(RuntimeError, 'cannot start thread'): self.submit(500)
        self.assertIsNone(self.writer._thread)
        self.assertIsNone(self.writer._error.__traceback__)
        with self.assertRaisesRegex(RuntimeError, 'writer failed'): self.writer.close()

    def test_payload_link_and_hash_read_errors_fail_without_success(self):
        from test_claix_async_checkpoint import payload
        for phase in ('payload_link', 'hash_read'):
            writer = AsyncVerifiedCheckpointIO(expected_config='a'*64)
            destination = self.root/(phase+'.ckpt')
            # Valid name in independent directories; no publication overwritten.
            destination = self.root/phase/'step-000500.ckpt'
            original_link = os.link
            def link(source, target, *a, **kw):
                if Path(target) == destination: raise OSError('injected payload link')
                return original_link(source, target, *a, **kw)
            injection = (patch.object(tx.os, 'link', side_effect=link) if phase == 'payload_link' else
                         patch.object(tx, 'file_sha256', side_effect=OSError('injected hash read')))
            with injection:
                writer.arm('interval', time.perf_counter())
                writer.save_checkpoint(payload(self.root,500), destination)
                with self.assertRaisesRegex(RuntimeError, 'writer failed'): writer.collect(wait=True)
            with self.assertRaisesRegex(RuntimeError, 'writer failed'): writer.teardown()
            self.assertIsNone(writer._thread)
            self.assertFalse((destination.parent/'latest.json').exists())


class RuntimeHardeningTest(unittest.TestCase):
    setUp = async_tests.AsyncRuntimeTest.setUp
    cleanup_writer = async_tests.AsyncRuntimeTest.cleanup_writer
    gate = async_tests.AsyncRuntimeTest.gate
    runtime = async_tests.AsyncRuntimeTest.runtime
    def test_normal_sync1500_budget_is_a_verified_boundary(self):
        from mibot.utils.claix_benchmark_runtime import BenchmarkSyncRuntime
        from mibot.utils.claix_runtime import VerifiedCheckpointIO
        from test_claix_async_checkpoint import payload
        previous, trainer, module = self.runtime()
        self.writer.close()
        runtime = BenchmarkSyncRuntime(self.root, run_root=self.root, deadline=time.time()+3600,
                                       interval=500, stop_after_step=1500)
        runtime.__dict__.update(previous.__dict__)
        runtime.stop_after_step = 1500
        plugin = VerifiedCheckpointIO(instrument=True)
        trainer.strategy.checkpoint_io = plugin
        def save(path, weights_only):
            ck = payload(self.root, trainer.global_step)
            ck['xr1_production']['rng_by_rank'] = module._checkpoint_rng
            plugin.save_checkpoint(ck, path)
        trainer.save_checkpoint = save
        with contextlib.redirect_stdout(io.StringIO()):
            for step in (500,1000,1500):
                trainer.global_step = step; runtime.previous_step = step-1
                runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
            runtime.on_fit_end(trainer, module)
        self.assertTrue(trainer.should_stop)
        self.assertEqual(runtime.last_saved, 1500)
        writer_lifetime_lock(self.root, probe=True)
        tx.load_verified(self.root/'step-001500.ckpt')

    def test_strategy_cannot_replace_active_plugin(self):
        runtime, trainer, module = self.runtime()
        with contextlib.redirect_stdout(io.StringIO()): runtime.save(trainer, module, 'interval')
        other = AsyncVerifiedCheckpointIO()
        trainer.strategy.checkpoint_io = other
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            runtime._io(trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.on_exception(trainer, module, RuntimeError('replaced plugin'))
        self.assertIsNone(self.writer._thread)
        self.assertTrue(self.writer._closed)
        other.close()

    def test_real_usr1_during_active_writer_commits_current_boundary(self):
        runtime, trainer, module = self.runtime()
        entered, release, blocked = self.gate()
        runtime.stop.install()
        self.addCleanup(runtime.stop.restore)
        with patch.object(tx, 'publish_checkpoint', new=blocked), contextlib.redirect_stdout(io.StringIO()):
            runtime.save(trainer, module, 'interval')
            self.assertTrue(entered.wait(2))
            signal.raise_signal(signal.SIGUSR1)
            trainer.global_step = 751; runtime.previous_step = 750
            timer = threading.Timer(.05, release.set); timer.start()
            try:
                runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
                runtime.on_fit_end(trainer, module)
            finally:
                release.set(); timer.join()
        self.assertEqual(runtime.last_saved, 751)
        self.assertTrue(trainer.should_stop)
        self.assertEqual(json.loads((self.root/'safe-stop.json').read_text())['global_step'], 751)

    def test_dead_controller_requests_verified_boundary_stop(self):
        runtime, trainer, module = self.runtime()
        with patch.dict(os.environ, XR1_CONTROLLER_ID='999999999:0'), contextlib.redirect_stdout(io.StringIO()):
            runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
        self.assertTrue(trainer.should_stop)
        self.assertEqual(runtime.stop.reason, 'controller_lost')
        self.assertEqual(runtime.last_saved, 500)
        tx.load_verified(self.root/'step-000500.ckpt')

    def test_normal_async1500_budget_is_a_verified_boundary(self):
        runtime, trainer, module = self.runtime()
        runtime.stop_after_step = 1500
        with contextlib.redirect_stdout(io.StringIO()):
            for step in (500, 1000, 1500):
                trainer.global_step = step; runtime.previous_step = step-1
                runtime.on_train_batch_end(trainer, module, torch.tensor(1.), None, 0)
            runtime.on_fit_end(trainer, module)
        self.assertEqual(runtime.last_saved, 1500)
        self.assertTrue(trainer.should_stop)
        rows = [json.loads(r) for r in (self.root/'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in rows if r['event'] == 'checkpoint'], [500,1000,1500])
        self.assertTrue(rows[-1]['final_checkpoint_verified'])


class ProcessIdentityTest(unittest.TestCase):
    def test_live_pid_reuse_and_dead_controller(self):
        identity = process_identity()
        self.assertTrue(process_alive(identity))
        self.assertFalse(process_alive(identity.split(':')[0]+':0'))
        self.assertFalse(process_alive('999999999:1'))
        self.assertFalse(process_alive('malformed'))

    def test_leg_exit_codes_and_strict_state_types(self):
        self.assertEqual(two_way.LegFailure('failed B', 17).returncode, 17)
        self.assertEqual(two_way.LegFailure('signal', -signal.SIGTERM).returncode, 143)
        self.assertFalse(two_way.same_state(1, True))
        self.assertFalse(two_way.same_state(torch.ones(1), torch.ones(1, dtype=torch.float64)))


class LauncherFailureTest(unittest.TestCase):
    def setUp(self):
        import test_claix_experimental as fixtures
        fixtures.ExperimentalLaunchTest.setUp(self)
        self.child.return_value.pid = 12345

    def test_failure_keeps_stop_request_until_child_is_reaped(self):
        fixture_launch = __import__('test_claix_experimental').launch
        self.child.return_value.pid = 12345
        observed = []
        def wait(timeout=None):
            if timeout is not None:
                raise ValueError('injected controller failure')
            paths = list(self.run.glob('.stop-request-*.json'))
            observed.extend(json.loads(p.read_text()) for p in paths)
            return 17
        self.child.return_value.wait.side_effect = wait
        with self.assertRaisesRegex(ValueError, 'controller failure'):
            fixture_launch.main([*self.args, '--execute'])
        self.assertEqual(observed, [dict(reason='controller_failure')])
        self.assertFalse(list(self.run.glob('.stop-request-*.json')))
        self.assertTrue(list(self.run.glob('launch-*.stop.json')))
        self.assertEqual(self.child.return_value.wait.call_count, 2)

    def test_dead_coordinator_requests_boundary_stop(self):
        fixture_launch = __import__('test_claix_experimental').launch
        observed = []
        def wait(timeout=None):
            observed.extend(json.loads(p.read_text()) for p in self.run.glob('.stop-request-*.json'))
            return 0
        self.child.return_value.wait.side_effect = wait
        with patch.dict(os.environ, XR1_COORDINATOR_ID='999999999:0'):
            self.assertEqual(fixture_launch.main([*self.args, '--execute']), 0)
        self.assertEqual(observed, [dict(reason='coordinator_lost')])


class ActiveJobTest(unittest.TestCase):
    def test_live_query_rejects_benchmark_and_root_reuse(self):
        for description in ('JobName=xr1_b24_checkpoint_two_way', 'Command=/hpcwork/reviewed/NEW/runner'):
            with patch.object(preflight.subprocess, 'check_output', side_effect=['123\n', description, '# unrelated script']), \
                 patch.dict(os.environ, SLURM_JOB_ID='124'), self.assertRaisesRegex(RuntimeError, 'active benchmark'):
                preflight.active_jobs('/hpcwork/reviewed/NEW')
        with patch.object(preflight.subprocess, 'check_output', side_effect=['123\n', 'JobName=overridden',
                                'python claix_b24_checkpoint_two_way.py']), \
             patch.dict(os.environ, SLURM_JOB_ID='124'), self.assertRaisesRegex(RuntimeError, 'active benchmark'):
            preflight.active_jobs('/hpcwork/reviewed/NEW')
        with patch.object(preflight.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, 'squeue')), \
             self.assertRaises(subprocess.CalledProcessError):
            preflight.active_jobs('/hpcwork/reviewed/NEW')


class RealCoordinatorProcessTest(unittest.TestCase):
    def test_interrupt_forwards_to_launcher_reaps_and_prevents_c(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder); root = parent/'NEW'
            ready = parent/'ready'; stopped = parent/'stopped'
            child_code = '''import os, signal, sys, time
from pathlib import Path
def stop(signum, frame):
    Path(sys.argv[2]).write_text(str(signum))
    sys.exit(0)
signal.signal(signal.SIGUSR1, stop)
Path(sys.argv[1]).write_text(str(os.getpid()))
os.kill(os.getppid(), signal.SIGUSR1)
time.sleep(10)
sys.exit(99)
'''
            cmd = [sys.executable, '-c', child_code, str(ready), str(stopped)]
            args = ['--run-root', str(root), '--overrides', '/synthetic', '--filesystem-source', 'synthetic',
                    '--project-id', '44885', '--execute']
            with patch.object(two_way, 'commands', return_value=[cmd, [sys.executable, '-c', 'raise RuntimeError("C must not run")']]), \
                 patch.object(preflight, 'verify', return_value=dict(run_ids={'B':'b'*32, 'C':'c'*32})), \
                 patch.object(two_way.launch, 'allocation_deadline'), \
                 patch('claix_b24_storage_preflight.experimental_evaluate', return_value=dict(status='PASS')), \
                 self.assertRaises(two_way.LegFailure) as failure:
                two_way.main(args)
            self.assertEqual(failure.exception.returncode, 128+signal.SIGUSR1)
            self.assertEqual(stopped.read_text(), str(signal.SIGUSR1))
            pid = int(ready.read_text())
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
            result = json.loads((root/'two-way-results.json').read_text())
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual([r['leg'] for r in result['legs']], ['B'])

    def test_zero_child_codes_cannot_hide_final_equality_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'NEW'
            args = ['--run-root', str(root), '--overrides', '/synthetic', '--filesystem-source', 'synthetic',
                    '--project-id', '44885', '--execute']
            with patch.object(two_way, 'commands', return_value=[[sys.executable, '-c', 'pass'] for _ in range(2)]), \
                 patch.object(preflight, 'verify', return_value=dict(run_ids={'B':'b'*32, 'C':'c'*32})), \
                 patch.object(two_way.launch, 'allocation_deadline'), \
                 patch('claix_b24_storage_preflight.experimental_evaluate', return_value=dict(status='PASS')), \
                 patch.object(two_way, 'verify_leg', side_effect=lambda root, leg: dict(run_id=leg.lower()*32)), \
                 patch.object(two_way, 'verify_pair', side_effect=RuntimeError('final RNG differs')), \
                 self.assertRaisesRegex(RuntimeError, 'final RNG differs'):
                two_way.main(args)
            result = json.loads((root/'two-way-results.json').read_text())
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual([r['returncode'] for r in result['legs']], [0,0])


class CPUReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = dict(schema='task04c-real-cpu-v1', status='PASS', suites=[], tests_executed=0)
        for index, name in enumerate(acceptance.SUITES):
            ids = acceptance.REQUIRED if index == 0 else [f'explicit_fixture.{index}']
            detail = self.root/(name+'.json'); log = self.root/(name+'.log')
            log.write_text('SYNTHETIC test report validation fixture\n')
            detail.write_text(json.dumps(dict(returncode=0, count=len(ids), error=None,
                                              tests=[dict(id=i, status='PASS') for i in ids])))
            self.report['suites'].append(dict(suite=name, returncode=0, count=len(ids), result_path=str(detail),
                result_sha256=acceptance.sha(detail), log_path=str(log), log_sha256=acceptance.sha(log)))
            self.report['tests_executed'] += len(ids)

    def test_complete_evidence_and_changed_log(self):
        self.assertTrue(acceptance.validate_report(self.report))
        Path(self.report['suites'][0]['log_path']).write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'bytes changed'): acceptance.validate_report(self.report)

    def test_missing_harness_count_failure_and_blocked_claim(self):
        for change in ('missing', 'count', 'failure', 'blocked'):
            report = copy.deepcopy(self.report)
            if change == 'missing': report['suites'].pop()
            if change == 'count': report['tests_executed'] += 1
            if change == 'failure': report['suites'][1]['returncode'] = 1
            if change == 'blocked': report['status'] = 'BLOCKED'
            with self.subTest(change=change), self.assertRaises(RuntimeError): acceptance.validate_report(report)

    def test_skipped_or_missing_real_framework_case_fails_closed(self):
        row = self.report['suites'][0]; path = Path(row['result_path'])
        original = json.loads(path.read_text())
        for change in ('skip', 'missing'):
            detail = copy.deepcopy(original)
            detail['tests'][0].update(status='SKIP' if change == 'skip' else 'PASS', id='unavailable')
            path.write_text(json.dumps(detail)); row['result_sha256'] = acceptance.sha(path)
            with self.subTest(change=change), self.assertRaises(RuntimeError): acceptance.validate_report(self.report)

    def test_cpu_runner_never_executes_on_login_node(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(acceptance.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'compute allocation'):
                acceptance.main(['--output', str(self.root/'no')])
            run.assert_not_called()

    def test_reporter_records_actual_failures_skips_and_return_code(self):
        suite = self.root/'tiny_suite.py'
        suite.write_text('''import unittest
class Results(unittest.TestCase):
    def test_pass(self): pass
    def test_failure(self): self.fail('injected')
    @unittest.skip('injected unavailable test')
    def test_skip(self): pass
if __name__ == '__main__': unittest.main()
''')
        result_path = self.root/'tiny_result.json'
        command = [sys.executable, '-B', str(launch.ROOT/'xr1/tests/claix_cpu_report.py'),
                   '--suite', str(suite), '--result', str(result_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result_path.read_text())
        self.assertEqual(report['returncode'], 1)
        self.assertEqual(report['count'], 3)
        self.assertEqual(sorted(r['status'] for r in report['tests']), ['FAIL','PASS','SKIP'])


class PreflightTest(unittest.TestCase):
    def setUp(self):
        CPUReportTest.setUp(self)
        self.commit = 'd'*40
        self.python = str(Path(sys.executable).absolute())
        self.overrides = self.root/'paths.json'; self.overrides.write_text('[]')
        self.cpu = self.root/'cpu.json'
        self.report.update(source_tree_sha256=launch.tree_digest(), checkout=dict(git_commit=self.commit, git_status=''),
            branch=preflight.BRANCH, environment={'fixture': 'not a real CLAIX environment'},
            allocation=dict(Account='thes2452', Partition='c23ms'))
        self.cpu.write_text(json.dumps(self.report))
        self.budget_file = self.root/'budget.json'
        raw = self.root/'accounting.txt'; raw.write_text('SYNTHETIC accounting fixture\n')
        self.budget_file.write_text(json.dumps(dict(schema='task04c-reviewed-project-budget-v1', account='thes2452',
            reviewer='fixture reviewer', observed_epoch=time.time(), raw_evidence=str(raw), raw_sha256=acceptance.sha(raw),
            cpu_hours_remaining=1000, gpu_hours_remaining=100)))
        self.review = dict(schema='task04c-benchmark-review-v1', worktree=str(launch.ROOT), branch=preflight.BRANCH,
            commit=self.commit, source_tree_sha256=launch.tree_digest(), python=self.python, overrides=str(self.overrides),
            overrides_sha256=acceptance.sha(self.overrides), run_root=str(self.root/'NEW'), run_ids={'B':'b'*32,'C':'c'*32},
            filesystem_source='SYNTHETIC', project_id=44885, zip_cache_root=str(self.root/'cache'),
            zip_manifest_sha256='a'*64, split_roots={}, split_identities={}, cpu_report=str(self.cpu),
            cpu_report_sha256=acceptance.sha(self.cpu),
            wrapper_sha256=acceptance.sha(launch.ROOT/'scripts/claix_b24_checkpoint_two_way.sbatch'),
            log_directory=str(self.root), project_budget=str(self.budget_file),
            project_budget_sha256=acceptance.sha(self.budget_file), reviewer='fixture reviewer')
        self.review_file = self.root/'review.json'
        self.storage = dict(status='PASS', available_bytes=60*1024**3, available_inodes=20000,
                            quota_measurements=[dict(remaining_bytes=60*1024**3, remaining_inodes=20000)])
        def git(*args):
            return self.commit if args[:2] == ('rev-parse','HEAD') else preflight.BRANCH if args[0] == 'branch' else ''
        self.patches = [patch.object(preflight, 'git', side_effect=git),
            patch.object(preflight, 'protected_science'), patch.object(preflight.subprocess, 'run'),
            patch.object(preflight.subprocess, 'check_output', return_value='Account=thes2452 JobState=RUNNING Partition=c23ms'),
            patch.object(preflight, 'environment_probe', return_value=self.report['environment']),
            patch.object(preflight, 'compose_science'), patch.object(preflight, 'assets'),
            patch.object(preflight, 'experimental_evaluate', side_effect=lambda *a, **k: self.storage),
            patch.object(preflight, 'inspect_directory', return_value={'filesystem':{'fstype':'lustre'}}),
            patch.object(preflight, 'active_jobs', return_value=[]),
            patch.dict(os.environ, XR1_PYTHON=self.python, SLURM_JOB_ID='123', XR1_RESUME='', XR1_STOP_AFTER_STEP='',
                       XR1_OPTION_B_RESUME_BRIDGE='')]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def verify(self):
        self.review_file.write_text(json.dumps(self.review))
        return preflight.verify(str(self.review_file), acceptance.sha(self.review_file))

    def test_bound_inputs_then_root_source_wrapper_and_identity_failures(self):
        self.assertEqual(self.verify(), self.review)
        for field, bad in (('worktree', str(self.root)), ('branch', 'foreign'), ('source_tree_sha256','0'*64),
                           ('wrapper_sha256','0'*64), ('overrides_sha256','0'*64),
                           ('run_ids', {'B':'b'*32,'C':'b'*32}), ('commit', preflight.START)):
            saved = self.review[field]; self.review[field] = bad
            with self.subTest(field=field), self.assertRaises(RuntimeError): self.verify()
            self.review[field] = saved
        Path(self.review['run_root']).mkdir()
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.verify()

    def test_incomplete_cpu_acceptance_blocks(self):
        self.report['suites'].pop()
        self.cpu.write_text(json.dumps(self.report)); self.review['cpu_report_sha256'] = acceptance.sha(self.cpu)
        with self.assertRaisesRegex(RuntimeError, 'mandatory CPU harness'): self.verify()

    def test_insufficient_storage_inodes_and_quota(self):
        for key, value in (('status', 'FAIL'), ('available_bytes', 1024), ('available_inodes', 1)):
            saved = self.storage[key]; self.storage[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.verify()
            self.storage[key] = saved
        self.storage['quota_measurements'][0]['remaining_inodes'] = 1
        with self.assertRaisesRegex(RuntimeError, 'inode reserve'): self.verify()

    def test_missing_assets_are_not_ignored(self):
        for error in (FileNotFoundError('original model missing'), FileNotFoundError('dataset/cache missing')):
            with patch.object(preflight, 'assets', side_effect=error), self.assertRaises(FileNotFoundError): self.verify()
        self.review.pop('zip_manifest_sha256')
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'): self.verify()

    def test_dirty_checkout_wrong_python_and_root_argument_mismatch(self):
        with patch.object(preflight, 'git', side_effect=lambda *a: 'dirty' if a[0]=='status' else
                          self.commit if a[0]=='rev-parse' else preflight.BRANCH), self.assertRaisesRegex(RuntimeError, 'source changes'):
            self.verify()
        self.verify()
        with self.assertRaisesRegex(RuntimeError, 'input path mismatch'):
            preflight.verify(str(self.review_file), acceptance.sha(self.review_file), expected_root='/foreign')
        self.review['python'] = '/foreign/python'
        with self.assertRaisesRegex(RuntimeError, 'approved XR1_PYTHON'): self.verify()

    def test_budget_missing_expired_or_insufficient_is_closed(self):
        for key, bad in (('observed_epoch', time.time()-1000), ('gpu_hours_remaining',1),
                         ('cpu_hours_remaining',1), ('raw_sha256','0'*64)):
            data = json.loads(self.budget_file.read_text()); original = copy.deepcopy(data)
            data[key] = bad; self.budget_file.write_text(json.dumps(data))
            self.review['project_budget_sha256'] = acceptance.sha(self.budget_file)
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.verify()
            self.budget_file.write_text(json.dumps(original))

    def test_active_benchmark_fails_closed(self):
        with patch.object(preflight, 'active_jobs', side_effect=RuntimeError('active benchmark')), self.assertRaisesRegex(RuntimeError,'active benchmark'):
            self.verify()


class ProtectedContractsTest(unittest.TestCase):
    def test_two_way_identity_binds_leg_writer_and_1500_stop(self):
        from test_claix_experimental import recipe
        from mibot.utils.claix_experiment import bind_experiment
        from mibot.utils.claix_contract import FROZEN_TRAIN_SHA256
        for leg in ('B','C'):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder); cache = root/'cache'; cache.mkdir()
                (cache/'media_manifest.jsonl').write_text('SYNTHETIC\n')
                profile = launch.experiment_profile(500, leg)
                cfg = recipe(profile['config'])
                cfg['trainer']['default_root_dir'] = folder
                cfg['production'].update(experimental_run_id='c'*32, stop_after_step=1500)
                manifest = dict(schema='claix-b24-experiment-v1', experimental_unqualified=True,
                    run_id='c'*32, source_tree_sha256=launch.tree_digest(), plan=dict(run_root=folder,
                    checkpoint_interval=500, experiment_name=profile['name'], two_way_leg=leg,
                    checkpoint_writer='sync' if leg=='B' else 'async_thread_v1', benchmark_steps=1500))
                tx.atomic_json(root/'experiment.json', manifest)
                with patch.dict(os.environ, XR1_DOOSAN_ZIP_CACHE_ROOT=str(cache), XR1_EXPERIMENTAL_UNQUALIFIED='1'):
                    bound = bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))
                    self.assertEqual(bound['two_way_leg'], leg)
                    cfg['production']['stop_after_step'] = 1000
                    with self.assertRaisesRegex(RuntimeError, 'leg/step/writer'):
                        bind_experiment(cfg, cfg['production'], dict(sha256=FROZEN_TRAIN_SHA256))

    def test_all_science_and_sync_bytes_still_match_start(self):
        preflight.protected_science()

    def test_changed_optimizer_scheduler_or_data_contract_is_rejected(self):
        original = subprocess.check_output
        for selected in ('xr1/configs/trainer/doosan_fp32_v2.yaml', 'xr1/configs/claix_b24_production.yaml',
                         'xr1/configs/data/doosan_fp32_v2.yaml'):
            def output(cmd, **kwargs):
                data = original(cmd, **kwargs)
                return data+b'\n# synthetic changed contract\n' if cmd[-1].endswith(':'+selected) else data
            with patch.object(preflight.subprocess, 'check_output', side_effect=output), self.assertRaisesRegex(RuntimeError, 'contract changed'):
                preflight.protected_science()
