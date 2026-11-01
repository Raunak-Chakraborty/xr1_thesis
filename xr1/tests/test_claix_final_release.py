"""Final release engineering tests; all hardware inputs here are synthetic."""
import contextlib
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
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
from unittest.mock import patch

import torch
import mibot.utils.claix_checkpoint as checkpoints
from test_claix_checkpoint import checkpoint_fixture
from test_claix_launch import launch, ROOT


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'claix_b24_{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = script('release')
storage = script('storage_preflight')


def reviewed_fixture(root):
    artifact = root/'synthetic-artifact.txt'
    artifact.write_text('SYNTHETIC TEST ONLY: no hardware qualification')
    plan = dict(batch_size=24, max_length=12288, data_horizon=48284, scheduler_steps=48284,
                warmup_steps=500, workers=4, world_size=1, run_root=str(root/'run'), overrides=[])
    record = dict(schema='claix-b24-release-v2', plan=plan, source_tree_sha256=launch.tree_digest(),
                  reviewed_by='synthetic reviewer', reviewed_at='2026-11-01',
                  scientific_parameters=release.SCIENCE, scientific_approved_by='synthetic scientist', gates={})
    provenance=root/'synthetic-provenance.json'
    provenance.write_text(json.dumps(dict(source_commit='synthetic-commit',
        source_tree_sha256=record['source_tree_sha256'],job_id='SYNTHETIC')))
    for gate in launch.GATES:
        envelope = root/(gate+'.json')
        envelope.write_text(json.dumps(dict(gate=gate, source_commit='synthetic-commit',
            source_tree_sha256=record['source_tree_sha256'], kind='current_h100', job_id='SYNTHETIC',
            artifacts=[dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                       dict(path=str(provenance),sha256=hashlib.sha256(provenance.read_bytes()).hexdigest())],
            provenance_bindings={key:dict(path=str(provenance),key=key)
                for key in ('source_commit','source_tree_sha256','job_id')})))
        record['gates'][gate] = dict(status='PASS', evidence=str(envelope),
            sha256=hashlib.sha256(envelope.read_bytes()).hexdigest(),
            applicability_reviewed_by='synthetic reviewer', applicability_rationale='test fixture only')
    review = root/'synthetic-quota-review.json'
    now = datetime.now(timezone.utc).isoformat()
    review.write_text(json.dumps(dict(timestamp=now, reviewed_by='synthetic operator',
        run_root=plan['run_root'], filesystem_source='synthetic-lustre', project_id=42,
        uid=os.getuid(), user_and_project_reviewed=True, quota_query_sha256='a'*64)))
    report = root/'synthetic-storage-report.json'
    report.write_text(json.dumps(dict(timestamp=now, status='PASS', stage='submission',
        quota_verification='explicit_operator_review', run_root=plan['run_root'], project_id=42,
        quota_query_sha256='a'*64, filesystem=dict(source='synthetic-lustre'),
        quota_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest())))
    record['storage_preflight']=dict(report=str(report), report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        quota_review=str(review), quota_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
        filesystem_source='synthetic-lustre', project_id=42)
    return record


class RotationIOTest(unittest.TestCase):
    def publish(self, root, step):
        path = root/f'step-{step:06d}.ckpt'
        return checkpoints.publish_checkpoint(checkpoint_fixture(step, horizon=48284), path)

    def counts(self, root):
        with patch.object(checkpoints, 'load_verified', wraps=checkpoints.load_verified) as verify, \
             patch.object(checkpoints, 'file_sha256', wraps=checkpoints.file_sha256) as sha, \
             patch.object(torch, 'load', wraps=torch.load) as load:
            result = checkpoints.rotate_checkpoints(root)
        names = [Path(call.args[0]).name for call in verify.call_args_list]
        self.assertEqual(len(names), len(set(names)), 'No payload is verified/loaded twice per rotation')
        return result, (load.call_count, sha.call_count, verify.call_count), names

    def test_initial_publication_and_resume_keep_full_verification_without_preload(self):
        from mibot.utils.claix_runtime import VerifiedCheckpointIO
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(torch,'load',wraps=torch.load) as loads, \
                 patch.object(checkpoints,'file_sha256',wraps=checkpoints.file_sha256) as hashes, \
                 patch.object(checkpoints,'validate_production_checkpoint',wraps=checkpoints.validate_production_checkpoint) as validate:
                self.publish(root,250)
                self.assertEqual((loads.call_count,hashes.call_count,validate.call_count),(1,1,2))
            with patch.object(torch,'load',wraps=torch.load) as loads, \
                 patch.object(checkpoints,'file_sha256',wraps=checkpoints.file_sha256) as hashes:
                ck=VerifiedCheckpointIO().load_checkpoint(root/'step-000250.ckpt')
                self.assertEqual(ck['global_step'],250)
                self.assertEqual((loads.call_count,hashes.call_count),(1,1))

    def test_counts_independent_of_historical_milestones_and_receipt_history(self):
        for milestones in ((), tuple(sorted(checkpoints.HISTORICAL_STEPS))):
            with self.subTest(milestones=milestones), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for step in (*milestones, 40250, 40500, 40750):
                    self.publish(root, step)
                # Deleted historical payload receipts and unexpected files require
                # no payload reads; milestones also require none here.
                for i in range(1, 201):
                    (root/f'step-{i:06d}.verified.json').write_text('{}')
                result, counts, names = self.counts(root)
                self.assertEqual(counts, (3, 3, 3))
                self.assertEqual(result, dict(verified=3, deleted=1))
                self.assertEqual(set(names), {'step-040250.ckpt','step-040500.ckpt','step-040750.ckpt'})

    def test_safe_stop_resume_supersession_has_bounded_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            stop=self.publish(root, 40251)
            checkpoints.atomic_json(root/'safe-stop.json', stop)
            for step in (40500,40750,41000):
                self.publish(root,step)
            _, counts, _=self.counts(root)
            self.assertEqual(counts,(4,4,4))
            self.assertTrue((root/stop['path']).exists())
            new=self.publish(root,41251)
            checkpoints.atomic_json(root/'safe-stop.json',new)
            _, counts, _=self.counts(root)
            self.assertEqual(counts,(4,4,4))
            self.assertFalse((root/stop['path']).exists())
            self.assertEqual(json.loads((root/'safe-stop.json').read_text()),new)

    def test_corrupt_newest_or_safe_stop_never_authorizes_deletion(self):
        for damage_stop in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                for step in (250,500,750):
                    row=self.publish(root,step)
                    if step==250 and damage_stop:
                        checkpoints.atomic_json(root/'safe-stop.json',row)
                damaged=root/('step-000250.ckpt' if damage_stop else 'step-000750.ckpt')
                with damaged.open('r+b') as stream:
                    stream.write(b'bad')
                with self.assertRaises(RuntimeError):
                    checkpoints.rotate_checkpoints(root)
                self.assertEqual(len(list(root.glob('*.ckpt'))),3)

    def test_size_changed_previous_cannot_masquerade_as_first_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.publish(root,250);self.publish(root,500)
            with (root/'step-000250.ckpt').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError,'usable recovery'):
                checkpoints.rotate_checkpoints(root)
            self.assertEqual(len(list(root.glob('*.ckpt'))),2)

    def test_previous_corrupt_uses_older_recovery_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (250,500,750,1000):
                self.publish(root,step)
            damaged=root/'step-000750.ckpt'
            with damaged.open('r+b') as stream:
                stream.write(b'bad')
            _, counts, _=self.counts(root)
            self.assertEqual(counts,(3,4,4)) # checksum rejects corruption before torch.load
            self.assertTrue(damaged.exists())
            self.assertTrue((root/'step-000500.ckpt').exists())
            self.assertFalse((root/'step-000250.ckpt').exists())

    def test_backlog_verification_is_capped_and_recovery_failure_is_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in range(1,21):
                self.publish(root,step)
            result, counts, _=self.counts(root)
            self.assertEqual(counts,(8,8,8))
            self.assertEqual(result['deleted'],6)
            for path in list(root.glob('*.ckpt')):
                if path.name!='step-000020.ckpt':
                    with path.open('r+b') as stream:
                        stream.write(b'bad')
            before={p.name for p in root.iterdir()}
            with patch.object(checkpoints,'load_verified',wraps=checkpoints.load_verified) as verify:
                with self.assertRaisesRegex(RuntimeError,'budget|usable recovery'):
                    checkpoints.rotate_checkpoints(root)
            self.assertEqual(verify.call_count,8)
            self.assertEqual(before,{p.name for p in root.iterdir()})

    def test_receipt_types_foreign_latest_and_dangling_symlinks_fail_safely(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (250,500,750):
                self.publish(root,step)
            for i, value in enumerate(([], None, True, {'global_step':True}, {'global_step':'x'}),1):
                (root/f'step-{i:06d}.verified.json').write_text(json.dumps(value))
            _, counts, _=self.counts(root)
            self.assertEqual(counts,(3,3,3))
            with self.assertRaisesRegex(RuntimeError,'Incompatible'):
                checkpoints.rotate_checkpoints(root,expected_config='b'*64)
            for name in ('step-001000.ckpt','step-001001.verified.json'):
                path=root/name; path.symlink_to(root/'missing-target')
                with self.assertRaises(FileExistsError):
                    self.publish(root,1000 if name.endswith('.ckpt') else 1001)
                self.assertTrue(path.is_symlink())

    def test_write_failures_and_symlink_pointers_preserve_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (250,500,750):
                self.publish(root,step)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with patch.object(Path,'unlink',side_effect=OSError('read only filesystem')):
                with self.assertRaises(OSError):
                    checkpoints.rotate_checkpoints(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            latest=root/'latest.json'
            latest.rename(root/'pointer-original.json')
            latest.symlink_to(root/'pointer-original.json')
            with self.assertRaisesRegex(RuntimeError,'Symlink'):
                checkpoints.rotate_checkpoints(root)
            with self.assertRaisesRegex(RuntimeError,'symlink'):
                self.publish(root,1000)
            self.assertTrue(latest.is_symlink())

    def test_null_stop_large_receipt_and_fifo_are_fail_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (250,500,750):self.publish(root,step)
            (root/'step-000001.verified.json').write_bytes(b' '*20000)
            os.mkfifo(root/'step-000002.verified.json')
            (root/'safe-stop.json').write_text('null')
            with self.assertRaisesRegex(RuntimeError,'Malformed safe-stop'):
                checkpoints.rotate_checkpoints(root)
            self.assertEqual(len(list(root.glob('*.ckpt'))),3)
            (root/'safe-stop.json').unlink()
            _,counts,_=self.counts(root)
            self.assertEqual(counts,(3,3,3))
            self.assertTrue((root/'step-000002.verified.json').exists())

    def test_atomic_receipt_failure_keeps_previous_pointer_and_orphan(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old=self.publish(root,250)
            original=checkpoints.atomic_json
            def fail_receipt(path,*args,**kwargs):
                if Path(path).suffixes==['.verified','.json']:
                    raise OSError('receipt failure')
                return original(path,*args,**kwargs)
            with patch.object(checkpoints,'atomic_json',side_effect=fail_receipt):
                with self.assertRaises(OSError):
                    self.publish(root,500)
            self.assertEqual(json.loads((root/'latest.json').read_text()),old)
            self.assertTrue((root/'step-000500.ckpt').exists())
            with self.assertRaises(FileExistsError):
                self.publish(root,500)


class ReleasePreparationTest(unittest.TestCase):
    def test_reviewed_release_checks_all_artifacts_and_rejects_old_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            record=reviewed_fixture(root)
            path=root/'release.json'
            path.write_text(json.dumps(record))
            launch.check_release(path,record['plan'])
            with self.assertRaisesRegex(RuntimeError,'read-only'):
                launch.check_release(path,record['plan'],require_immutable=True)
            path.chmod(0o444)
            launch.check_release(path,record['plan'],require_immutable=True)
            path.chmod(0o644)
            artifact=root/'synthetic-artifact.txt'
            artifact.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'artifact checksum'):
                launch.check_release(path,record['plan'])
            record=reviewed_fixture(root)
            gate='real_compact_resume'
            envelope=Path(record['gates'][gate]['evidence'])
            row=json.loads(envelope.read_text()); row.update(kind='historical_h100',job_id='5357168',source_tree_sha256='a'*64)
            envelope.write_text(json.dumps(row))
            record['gates'][gate]['sha256']=hashlib.sha256(envelope.read_bytes()).hexdigest()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'Final-source H100'):
                launch.check_release(path,record['plan'])
            self.assertEqual(json.loads(envelope.read_text())['job_id'],'5357168')

    def test_original_provenance_job_and_commit_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for field in ('job_id','source_commit'):
                record=reviewed_fixture(root)
                path=Path(record['gates']['h100_b24_memory']['evidence'])
                envelope=json.loads(path.read_text());envelope[field]='mismatched'
                path.write_text(json.dumps(envelope))
                with self.assertRaisesRegex(RuntimeError,'provenance mismatch'):
                    release.validate_review(record)

    def test_scientific_approval_extra_gate_and_cpu_claims_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for change in ('science','approval','extra','cpu','mismatch','missing'):
                record=reviewed_fixture(root)
                if change=='science':
                    record['scientific_parameters']=dict(release.SCIENCE,max_lr=1e-3)
                elif change=='approval':
                    record['scientific_approved_by']=''
                elif change=='extra':
                    record['gates']['extra']={}
                else:
                    envelope=Path(record['gates']['h100_b24_memory']['evidence'])
                    row=json.loads(envelope.read_text())
                    if change=='cpu': row['kind']='cpu'
                    if change=='mismatch': row['source_tree_sha256']='0'*64
                    if change=='missing': row['artifacts']=[]
                    envelope.write_text(json.dumps(row))
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    release.validate_review(record)

    def test_builder_never_approves_and_exclusively_creates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); plan=root/'plan.json'; output=root/'draft.json'
            plan.write_text(json.dumps(reviewed_fixture(root)['plan']))
            # CLI uses the ordinary script import path, no model/hardware imports.
            result=subprocess.run([sys.executable,str(ROOT/'scripts/claix_b24_release.py'),
                '--plan',str(plan),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            draft=json.loads(output.read_text())
            self.assertEqual(set(draft['gates']),set(launch.GATES))
            self.assertTrue(all(row['status']=='UNVERIFIED' for row in draft['gates'].values()))
            self.assertFalse(draft['reviewed_by'])
            with self.assertRaises(FileExistsError):
                release.main(['--plan',str(plan),'--output',str(output)])


class StoragePreflightTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run = '/hpcwork/account/new-run'
        self.mount = self.root  # synthetic existing mount only, never CLAIX
        self.location = dict(resolved_path=self.run, existing_parent=str(self.root),
            filesystem=dict(source='approved', target=str(self.mount), fstype='lustre'))
        self.stat = SimpleNamespace(f_bavail=40, f_frsize=storage.GIB, f_favail=20000)
        self.user_output = self.snapshot('uid', os.getuid())
        self.project_output = self.snapshot('projid', 42)
        for mock in (patch.object(storage, 'inspect_directory', return_value=self.location),
                     patch.object(storage.os, 'statvfs', return_value=self.stat),
                     patch.object(storage, 'query', side_effect=self.query)):
            mock.start()
            self.addCleanup(mock.stop)

    def snapshot(self, identity, value):
        return (f'Disk quotas for test ({identity} {value}):\n'
            'Filesystem kbytes quota limit grace files quota limit grace\n'
            f'{self.mount} 1000 100000000 110000000 - 10 200000 210000 -\n')

    def query(self, cmd):
        return dict(command=cmd, returncode=0, stderr='', stdout=(self.user_output
            if '-u' in cmd and cmd[0]=='lfs' else self.project_output if '-p' in cmd else 'generic quota output'))

    def attest(self, age=0, policy=False):
        report = storage.evaluate(self.run, 'approved', project_id=42)
        reviewed = datetime.now(timezone.utc)-timedelta(seconds=age)
        row = dict(reviewed_by='synthetic operator', timestamp=reviewed.isoformat(),
            run_root=self.run, filesystem_source='approved', uid=os.getuid(), project_id=42,
            user_and_project_reviewed=True, usable_bytes=40*storage.GIB, usable_inodes=20000,
            quota_query_sha256=report['quota_query_sha256'])
        if policy:
            row.update(allocation_refresh_policy=storage.REFRESH_POLICY,
                       allocation_valid_until=(reviewed+timedelta(days=7)).isoformat())
        path = self.root/'review.json'
        path.write_text(json.dumps(row))
        return path

    def evaluate(self, path=None, stage='submission'):
        return storage.evaluate(self.run, 'approved', path, 42, stage=stage)

    def test_quota_unknown_and_low_headroom_fail_closed_without_writes(self):
        self.assertEqual(self.evaluate()['status'], 'UNVERIFIED')
        self.assertEqual(list(self.root.iterdir()), [])
        review = self.attest()
        self.assertEqual(self.evaluate(review)['status'], 'PASS')
        for change in ('bytes', 'inodes', 'snapshot'):
            self.stat.f_bavail, self.stat.f_favail = 40, 20000
            if change=='bytes': self.stat.f_bavail=29
            if change=='inodes': self.stat.f_favail=1
            if change=='snapshot': self.user_output += 'changed quota output'
            result = self.evaluate(review, 'allocation')
            self.assertNotEqual(result['status'], 'PASS')
            self.assertTrue(result['abort_reasons'])
        self.assertEqual({p.name for p in self.root.iterdir()}, {'review.json'})

    def test_long_queue_requires_explicit_policy_and_fresh_identical_lustre_quotas(self):
        for age in (7200, 3*86400):
            review = self.attest(age)
            result = self.evaluate(review, 'allocation')
            self.assertEqual(result['quota_status'], 'UNVERIFIED')
            self.assertIn('stale', result['reason'])
            review = self.attest(age, policy=True)
            self.assertEqual(self.evaluate(review)['status'], 'UNVERIFIED')
            result = self.evaluate(review, 'allocation')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['quota_verification'], storage.REFRESH_POLICY)
            self.assertEqual(result['available_bytes'], 40*storage.GIB)
        self.stat.f_bavail=1
        self.assertEqual(self.evaluate(review, 'allocation')['status'], 'FAIL')

    def test_expiry_future_wrong_identity_and_ambiguous_queries_abort(self):
        for change in ('expired', 'fresh_expired', 'future', 'long_policy', 'uid', 'project', 'user_failed',
                       'project_failed', 'rounded', 'grace', 'warning', 'format', 'nfs'):
            self.user_output=self.snapshot('uid', os.getuid())
            self.project_output=self.snapshot('projid', 42)
            self.location['filesystem']['fstype']='lustre'
            review=self.attest(7200, policy=True)
            row=json.loads(review.read_text())
            if change=='expired': row['allocation_valid_until']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
            if change=='fresh_expired':
                row['timestamp']=datetime.now(timezone.utc).isoformat()
                row['allocation_valid_until']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
            if change=='future': row['timestamp']=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
            if change=='long_policy': row['allocation_valid_until']=(datetime.now(timezone.utc)+timedelta(days=8)).isoformat()
            if change=='uid': row['uid']=-1
            if change=='project': row['project_id']=99
            if change=='rounded': self.user_output=self.user_output.replace('1000 ', '1M ')
            if change=='grace': self.project_output=self.project_output.replace(' - ', ' 6days ', 1)
            if change=='format': self.user_output='unknown site format'
            if change=='nfs': self.location['filesystem']['fstype']='nfs'
            if change in ('rounded', 'grace', 'format'):
                # Even an operator-reviewed unknown snapshot cannot be auto-refreshed.
                row['quota_query_sha256']=self.evaluate()['quota_query_sha256']
            review.write_text(json.dumps(row))
            original=self.query
            def changed(cmd):
                result=original(cmd)
                if change=='warning': result['stderr']='warning: incomplete quota information'
                if change=='user_failed' and '-u' in cmd: result['returncode']=1
                if change=='project_failed' and '-p' in cmd: result['returncode']=1
                return result
            with patch.object(storage, 'query', side_effect=changed):
                if change in ('warning', 'user_failed', 'project_failed'):
                    row['quota_query_sha256']=self.evaluate()['quota_query_sha256']
                    review.write_text(json.dumps(row))
                with self.subTest(change=change):
                    self.assertNotEqual(self.evaluate(review, 'allocation')['status'], 'PASS')

    def test_bound_allocation_review_cannot_be_changed(self):
        review=self.attest(7200, policy=True)
        record=dict(plan=dict(run_root=self.run), storage_preflight=dict(quota_review=str(review),
            quota_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(), filesystem_source='approved', project_id=42))
        self.assertEqual(storage.allocation_recheck(record)['status'], 'PASS')
        review.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            storage.allocation_recheck(record)


class MonitorQualificationTest(unittest.TestCase):
    def test_monitor_qualification_opt_in_and_cpu_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                launch.main(['--run-root','/tmp/none','--mode','cpu-qualification','--qualification-gpu-monitor'])
            self.assertEqual(launch.main(['--run-root','/tmp/none','--qualification-gpu-monitor']),0)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(launch,'allocation_deadline',return_value=time.time()+1000), \
             patch.object(launch,'write_provenance'), patch.object(launch,'NvidiaSmiMonitor') as monitor, \
             patch.object(launch.subprocess,'Popen') as child, \
             patch.dict(os.environ,XR1_DOOSAN_ZIP_CACHE_ROOT='/synthetic'):
            child.return_value.wait.return_value=0
            launch.main(['--execute','--run-root',folder,'--qualification-gpu-monitor'])
            monitor.return_value.start.assert_called_once()
            monitor.return_value.close.assert_called_once()

    def test_real_timeout_kills_and_reaps_query_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); pidfile=root/'pid'; executable=root/'nvidia-smi'
            executable.write_text(f'#!{sys.executable}\nimport os,time\nopen({str(pidfile)!r},"w").write(str(os.getpid()))\ntime.sleep(30)\n')
            executable.chmod(0o755)
            monitor=launch.NvidiaSmiMonitor(root/'gpu.jsonl')
            started=time.monotonic()
            with patch.dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH']):
                monitor.start()
                monitor.thread.join(5)
                monitor.close()
            self.assertLess(time.monotonic()-started,5)
            self.assertFalse(monitor.thread.is_alive())
            self.assertEqual(json.loads(monitor.path.read_text())['error_type'],'TimeoutExpired')
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()),0)

    def test_production_sbatch_and_bootstrap_contract(self):
        text=(ROOT/'scripts/claix_b24_production.sbatch').read_text()
        for directive in ('--account=thes2452','--partition=c23g','--ntasks=1','--gres=gpu:1',
                          '--cpus-per-task=12','--mem=96G','--time=23:45:00','--signal=B:USR1@1800'):
            self.assertIn('#SBATCH '+directive,text)
        self.assertIn('--mode production --execute',text)
        self.assertIn('XR1_RELEASE_RECORD:?',text)
        bootstrap=(ROOT/'scripts/claix_b24_environment.sh').read_text()
        self.assertIn('module load Python/3.11.5',bootstrap)
        self.assertIn('$HOME/robotics_thesis/configs/xr1_hf_offline.env',bootstrap)
        runtime=(ROOT/'xr1/tools/train_doosan_compact.py').read_text()
        self.assertIn("'H100' not in torch.cuda.get_device_name(0)",runtime)

    def test_targeted_verifier_checks_persistent_step_metrics_receipts_and_provenance(self):
        targeted=script('targeted_h100')
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,SLURM_JOB_ID='SYNTHETIC_CPU_FIXTURE'):
            root=Path(folder);target=root/'project_xr1_doosan/qualification_b24_w4'
            target.mkdir(parents=True)
            rows=[dict(event='step',step=step,retained_examples=24,views=48,samples_consumed=step*24,
                       packed_tokens=100,learning_rate=1e-5,timestamp='synthetic',pipeline_wait_seconds=0.,
                       step_seconds=1.,end_to_end_examples_per_second=24.,gpu_peak_allocated_bytes=0,loss=.1)
                  for step in range(1,261)]
            rows.append(dict(event='graceful_stop',step=251,reason='launcher_signal:10'))
            for step in (250,251,260):
                receipt=checkpoints.publish_checkpoint(checkpoint_fixture(step,horizon=48284),target/f'step-{step:06d}.ckpt')
                if step in (251,260):checkpoints.atomic_json(target/'safe-stop.json',receipt)
                rotation=checkpoints.rotate_checkpoints(target)
                rows.append(dict(event='checkpoint',step=step,verified=True,rotation=rotation))
            (target/'telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (target/'resume_contract.json').write_text(json.dumps(dict(replay=checkpoint_fixture()['xr1_replay_contract'],config_sha256='a'*64)))
            # Use the actual fixture contract for the 48284-step synthetic payloads.
            contract=json.loads((target/'resume_contract.json').read_text());contract['replay']=checkpoint_fixture(260,48284)['xr1_replay_contract']
            (target/'resume_contract.json').write_text(json.dumps(contract))
            for version,steps in enumerate((range(1,252),range(252,261))):
                metrics=target/f'metrics/version_{version}';metrics.mkdir(parents=True)
                keys=['step','train/loss','train/loss_freq','train/loss_l1','train/loss_mse','train/loss_score','train/token','lr']
                import csv
                with (metrics/'metrics.csv').open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
                    for step in steps:writer.writerow(dict.fromkeys(keys,.1)|dict(step=step-1))
                (root/f'launch-{version}.provenance.json').write_text(json.dumps(dict(source_tree_sha256='a'*64)))
                (root/f'launch-{version}.gpu.jsonl').write_text(json.dumps(dict(event='sample',devices=[dict(uuid='GPU-SYNTHETIC')]))+'\n')
            result=targeted.verify(root,'a'*64,251)
            self.assertEqual(result['completed_steps'],260)
            self.assertTrue(result['exact_resumed_sequence'])
            with (target/'metrics/version_1/metrics.csv').open('a') as stream:
                stream.write('259,.1,.1,.1,.1,.1,.1,.1\n')
            with self.assertRaisesRegex(RuntimeError,'CSV step'):
                targeted.verify(root,'a'*64,251)

    def test_targeted_dry_run_never_launches(self):
        targeted=script('targeted_h100')
        with patch.object(targeted.subprocess,'Popen',side_effect=AssertionError('launch')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(targeted.main(['--run-root','/tmp/new-targeted']),0)
