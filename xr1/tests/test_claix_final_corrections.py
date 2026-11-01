"""Targeted negative paths; all filesystem/H100 evidence in these tests is synthetic."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from test_claix_final_release import script, reviewed_fixture, launch, release, ROOT
from test_claix_checkpoint import checkpoint_fixture
from mibot.utils.claix_checkpoint import publish_checkpoint, atomic_json

paths = script('filesystem')
targeted = script('targeted_h100')


class PersistentPathsTest(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.canonical=self.root/'rwthfs/rz/cluster/hpcwork'
        self.canonical.mkdir(parents=True)
        self.alias=self.root/'hpcwork'
        self.alias.symlink_to(self.canonical, target_is_directory=True)
        self.work=self.root/'work'
        self.work.mkdir()
        self.approved=(self.alias, self.work, self.canonical)
        self.fs=dict(source='real-reviewed-source',target=str(self.canonical),fstype='lustre')
        for mock in (patch.object(paths, 'APPROVED_ROOTS', self.approved),
                     patch.object(paths.subprocess, 'run', side_effect=self.mount)):
            mock.start(); self.addCleanup(mock.stop)

    def mount(self, *args, **kwargs):
        return SimpleNamespace(stdout=json.dumps(dict(filesystems=[self.fs])))

    def test_documented_aliases_only(self):
        with patch.object(paths, 'APPROVED_ROOTS', (Path('/hpcwork'), Path('/work'), Path('/rwthfs/rz/cluster/hpcwork'))):
            for path in ('/hpcwork/user/logs', '/work/user/logs', '/rwthfs/rz/cluster/hpcwork/user/logs'):
                self.assertTrue(paths.approved(Path(path)))
            for path in ('/rwthfs', '/rwthfs/rz/cluster/other/user', '/hpcwork-other/user', '/tmp', '/tmp/hpcwork'):
                self.assertFalse(paths.approved(Path(path)))

    def test_alias_and_canonical_paths_use_same_actual_mount(self):
        (self.canonical/'user/logs').mkdir(parents=True)
        for path in (self.alias/'user/logs', self.canonical/'user/logs'):
            result=paths.inspect_directory(path, expected_source='real-reviewed-source')
            self.assertEqual(result['resolved_path'], str(self.canonical/'user/logs'))
        self.fs['target']=str(self.work)
        self.assertEqual(paths.inspect_directory(self.work)['resolved_path'], str(self.work))

    def test_missing_file_dangling_and_escape_paths_rejected(self):
        (self.canonical/'file').write_text('not a directory')
        (self.canonical/'escape').symlink_to(self.root)
        (self.canonical/'dangling').symlink_to(self.root/'missing')
        for path in (self.canonical/'missing', self.canonical/'file', self.canonical/'escape',
                     self.canonical/'dangling', self.root, Path('relative')):
            with self.subTest(path=path), self.assertRaises((RuntimeError, OSError)):
                paths.inspect_directory(path)
        new=self.canonical/'new/run'
        self.assertEqual(paths.inspect_directory(new,allow_missing=True)['existing_parent'], str(self.canonical))
        self.assertFalse(new.exists())
        with self.assertRaises(RuntimeError):
            paths.inspect_directory(self.canonical/'escape/missing',allow_missing=True)

    def test_tmpfs_ramfs_local_overlay_wrong_mount_and_source_rejected(self):
        for kind in ('tmpfs', 'ramfs', 'overlay', 'ext4', None):
            self.fs['fstype']=kind
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                paths.inspect_directory(self.canonical)
        self.fs['fstype']='lustre'
        with self.assertRaises(RuntimeError): paths.inspect_directory(self.canonical,expected_source='wrong')
        self.fs['target']=str(self.root)
        with self.assertRaises(RuntimeError): paths.inspect_directory(self.canonical)
        with patch.object(paths.subprocess, 'run', return_value=SimpleNamespace(stdout='{}')):
            with self.assertRaises(KeyError): paths.inspect_directory(self.canonical)


class ExactRestartTest(unittest.TestCase):
    def test_exact_comparator_includes_all_training_states_and_all_saved_rng(self):
        original=checkpoint_fixture(6,48284)
        original['xr1_production']['rng_by_rank'][0]['cuda']=[torch.arange(16,dtype=torch.uint8)]
        self.assertEqual(targeted.compare_exact(original,copy.deepcopy(original))['status'], 'PASS')
        changes=(('state_dict','model.weight'), ('optimizer_states',0,'state',0,'exp_avg'),
            ('optimizer_states',0,'state',0,'exp_avg_sq'), ('optimizer_states',0,'state',0,'step'),
            ('lr_schedulers',0,'last_epoch'), ('global_step',), ('xr1_replay','rank_sample_offset'),
            ('xr1_production','config_sha256'), ('xr1_compact_fp32','source_commit'),
            ('xr1_production','rng_by_rank',0,'torch'), ('xr1_production','rng_by_rank',0,'cuda',0))
        for keys in changes:
            changed=copy.deepcopy(original); parent=changed
            for key in keys[:-1]: parent=parent[key]
            key=keys[-1]
            if isinstance(parent[key],torch.Tensor): parent[key].reshape(-1)[0] += 1
            elif isinstance(parent[key],str): parent[key]='c'*64
            else: parent[key]+=1
            with self.subTest(keys=keys):
                self.assertEqual(targeted.compare_exact(original,changed)['status'], 'FAIL')
        for kind in ('python','numpy'):
            changed=copy.deepcopy(original)
            state=changed['xr1_production']['rng_by_rank'][0]
            if kind=='python':
                value=list(state[kind]); value[2]=.5; state[kind]=tuple(value)
            else: state[kind][1][0] ^= 1
            self.assertEqual(targeted.compare_exact(original,changed)['status'], 'FAIL')
        changed=copy.deepcopy(original)
        changed['xr1_production']['rng_by_rank'][0]['cuda']=[]
        self.assertEqual(targeted.compare_exact(original,changed)['status'], 'FAIL')
        changed=copy.deepcopy(original);del changed['xr1_production']['rng_by_rank']
        with self.assertRaisesRegex(RuntimeError,'RNG'): targeted.compare_exact(original,changed)
        changed=copy.deepcopy(original)
        changed.update(timestamp='unrelated', log_metadata={'different':True})
        changed['xr1_production']['timestamp']='unrelated'
        self.assertEqual(targeted.compare_exact(original,changed)['status'], 'PASS')

    def test_exact_orchestration_verifies_before_resume_and_saves_real_comparison(self):
        for failure in (None, 'tensor', 'corrupt_checkpoint', 'source', 'cursor'):
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,SLURM_JOB_ID='SYNTHETIC_CPU'):
                q=targeted.Targeted(SimpleNamespace(run_root=folder,python='python',overrides='unused'),time.time()+1000,{'phases':{}})
                calls=[]
                base=checkpoint_fixture(6,48284)
                def train(name, **kwargs):
                    calls.append((name,kwargs))
                    resume=kwargs.get('resume')
                    if resume:
                        # A published checksum-bound step-3 checkpoint precedes this invocation.
                        self.assertTrue(resume.with_suffix('.verified.json').is_file())
                    step=3 if kwargs.get('stop') else 6
                    ck=checkpoint_fixture(step,48284)
                    ck['xr1_production']['rng_by_rank']=copy.deepcopy(base['xr1_production']['rng_by_rank'])
                    if failure=='tensor' and resume: ck['state_dict']['model.weight'][0,0]+=.01
                    directory=q.target(name);directory.mkdir(parents=True,exist_ok=True)
                    receipt=publish_checkpoint(ck,directory/f'step-{step:06d}.ckpt')
                    if failure=='corrupt_checkpoint' and step==3:
                        with (directory/receipt['path']).open('ab') as stream: stream.write(b'corrupt')
                    atomic_json(directory/'resume_contract.json',dict(replay=ck['xr1_replay_contract'],config_sha256='a'*64))
                    rows=[dict(event='train_start',step=3 if resume else 0,replay_offset=0 if failure=='cursor' else 72 if resume else 0)]
                    rows += [dict(event='step',step=i) for i in range(4 if resume else 1,step+1)]
                    if step==3: rows += [dict(event='graceful_stop',step=step,reason='qualification_step')]
                    rows += [dict(event='fit_end',step=step,final_checkpoint_verified=True)]
                    with (directory/'telemetry.jsonl').open('a') as stream:
                        stream.write(''.join(json.dumps(row)+'\n' for row in rows))
                    provenance=q.root/name/f'launch-{step}.provenance.json'
                    atomic_json(provenance,dict(source_tree_sha256='bad' if failure=='source' else q.source))
                with patch.object(q,'train',side_effect=train):
                    if failure:
                        with self.subTest(failure=failure), self.assertRaises(RuntimeError): q.run_exact()
                        if failure=='corrupt_checkpoint': self.assertEqual(len(calls),2)
                    else:
                        result=q.run_exact()
                        self.assertEqual(result['status'],'PASS')
                        self.assertEqual([item[0] for item in calls],['exact-restart/uninterrupted', 'exact-restart/restarted', 'exact-restart/restarted'])
                        self.assertEqual(calls[1][1]['stop'],3)
                        self.assertEqual(calls[2][1]['resume'].name,'step-000003.ckpt')
                        comparison=json.loads((q.root/'exact-restart/restart-comparison.json').read_text())
                        self.assertEqual(comparison['status'],'PASS')
                        self.assertTrue(any(row['path'].startswith('rng_by_rank') for row in comparison['comparisons']))

    def test_commands_same_allocation_workers_four_isolated_roots_and_bounds(self):
        q=targeted.Targeted(SimpleNamespace(run_root='/unused',python='approved-python',overrides='approved-overrides'),time.time()+1000,{'phases':{}})
        with patch.object(q,'process') as process:
            q.train('exact-restart/uninterrupted')
            q.train('exact-restart/restarted',stop=3)
            q.train('exact-restart/restarted',resume=Path('/unused/step-000003.ckpt'))
            q.train('logging-retention',maximum=260)
            for call in process.call_args_list:
                cmd=call.args[1]
                self.assertEqual(cmd[cmd.index('--workers')+1],'4')
                self.assertEqual(cmd[cmd.index('--mode')+1],'h100-qualification')
                self.assertNotIn('sbatch',cmd)
            self.assertIn('--qualification-gpu-monitor',process.call_args_list[-1].args[1])
        q.pending_signal=signal.SIGUSR1
        with tempfile.TemporaryDirectory() as folder, patch.object(q,'process',side_effect=AssertionError('must not launch')):
            q.root=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'stop signal'): q.gate('exact_restart',q.run_exact)

    def test_dry_run_explicitly_includes_all_five_phases_and_no_child(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output),patch.object(targeted.subprocess,'Popen',side_effect=AssertionError('child')):
            targeted.main(['--run-root','/hpcwork/NEW/targeted'])
        plan=json.loads(output.getvalue())
        self.assertEqual(plan['total_optimizer_steps'],272)
        self.assertEqual(len(plan['phases']),5)
        self.assertEqual(plan['allocation_max_minutes'],45)
        self.assertEqual(plan['safe_stop_margin_seconds'],600)
        self.assertIn('UNVERIFIED',plan['runtime_guarantee'])


class AllocationStorageGateTest(unittest.TestCase):
    def test_release_binds_submission_report_and_attestation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for change in ('missing','report_changed','review_changed','unverified','old_at_submission'):
                record=reviewed_fixture(root)
                binding=record['storage_preflight']
                if change=='missing': del record['storage_preflight']
                elif change=='report_changed': Path(binding['report']).write_text('{}')
                elif change=='review_changed': Path(binding['quota_review']).write_text('{}')
                else:
                    path=Path(binding['report']); row=json.loads(path.read_text())
                    if change=='unverified': row['status']='UNVERIFIED'
                    else: row['timestamp']='2020-01-01T00:00:00+00:00'
                    path.write_text(json.dumps(row));binding['report_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(change=change), self.assertRaises(RuntimeError): release.validate_review(record)

    def test_allocation_failure_is_machine_readable_and_never_loads_model(self):
        import claix_b24_storage_preflight as storage_module
        for outcome in (dict(status='UNVERIFIED',abort_reasons=['Quota review stale']),
                        dict(status='FAIL',abort_reasons=['inodes failed']),RuntimeError('query unavailable')):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder); record=reviewed_fixture(root); record['plan']['run_root']=folder
                path=root/'release.json';path.write_text(json.dumps(record))
                with patch.object(launch,'check_release',return_value=None),patch.object(launch,'allocation_deadline',return_value=time.time()+1000), \
                     patch.object(launch.subprocess,'Popen',side_effect=AssertionError('must not load model')), \
                     patch.object(storage_module,'allocation_recheck',side_effect=outcome if isinstance(outcome,Exception) else None,
                                  return_value=outcome if isinstance(outcome,dict) else None), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'before model loading'):
                        launch.main(['--mode','production','--execute','--run-root',folder,'--release-record',str(path)])
                reports=list(root.glob('launch-*.storage.json'))
                self.assertEqual(len(reports),1)
                result=json.loads(reports[0].read_text())
                self.assertNotEqual(result['status'],'PASS')
                self.assertTrue(result['abort_reasons'])


class ResumeDirectorySafetyTest(unittest.TestCase):
    def test_latest_resume_only_with_complete_contiguous_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for step in (3,6): publish_checkpoint(checkpoint_fixture(step,48284), root/f'step-{step:06d}.ckpt')
            good=[dict(event='train_start',step=0)]+[dict(event='step',step=i) for i in range(1,7)]
            path=root/'telemetry.jsonl'
            def write(rows): path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            write(good)
            launch.check_resume_directory(root,root/'step-000006.ckpt')
            with self.assertRaisesRegex(RuntimeError,'latest verified'): launch.check_resume_directory(root,root/'step-000003.ckpt')
            for bad in (good+[dict(event='step',step=7)], good[:-1], good+[dict(event='step',step=6)],
                        good+[dict(event='train_start',step=3)]):
                write(bad)
                with self.assertRaises(RuntimeError): launch.check_resume_directory(root,root/'step-000006.ckpt')
            write(good)
            with path.open('a') as stream: stream.write('{"event":')
            with self.assertRaisesRegex(RuntimeError,'Incomplete'): launch.check_resume_directory(root,root/'step-000006.ckpt')
            with tempfile.TemporaryDirectory() as other:
                external=Path(other)/'step-000006.ckpt';publish_checkpoint(checkpoint_fixture(6,48284),external)
                with self.assertRaisesRegex(RuntimeError,'own latest'): launch.check_resume_directory(root,external)
            write(good)
            # Repeat two clean resumes; each boundary is preserved in the same stream.
            rows=good[:4]+[dict(event='train_start',step=3)]+good[4:6]+[dict(event='train_start',step=5)]+good[6:]
            write(rows)
            launch.check_resume_directory(root,root/'step-000006.ckpt')


class TargetedAllocationLimitTest(unittest.TestCase):
    def test_targeted_limit_cannot_extend_to_old_two_hour_allocation(self):
        with patch.dict(os.environ,SLURM_JOB_ID='SYNTHETIC_CPU'),patch.object(launch.subprocess,'run') as run, \
             patch.object(launch.subprocess,'check_output',return_value='10000'),patch.object(launch.time,'time',return_value=100):
            for limit, passes in (('00:45:00',True),('00:46:00',False),('02:00:00',False)):
                run.return_value.stdout=f'TimeLimit={limit} EndTime=unused AllocTRES=cpu=12,mem=96G,gres/gpu=1'
                if passes: self.assertEqual(launch.allocation_deadline('h100-qualification',max_seconds=2700),9400)
                else:
                    with self.assertRaisesRegex(RuntimeError,'budget'): launch.allocation_deadline('h100-qualification',max_seconds=2700)
