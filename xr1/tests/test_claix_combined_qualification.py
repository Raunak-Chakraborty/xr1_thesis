"""Synthetic orchestration/receipt tests; no CLAIX data, Slurm calls or GPU."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import claix_b24_combined_qualification as combined
from mibot.utils.claix_checkpoint import publish_checkpoint
from test_claix_checkpoint import checkpoint_fixture


def telemetry(final=251):
    rows=[dict(event='train_start',step=0)]
    for step in range(1,final+1):
        rows.append(dict(event='step',step=step,retained_examples=24,views=48,packed_tokens=9912,
                         trainable_fp32=True,frozen_bf16=True,gpu_allocated_bytes=64,gpu_reserved_bytes=80,
                         gpu_peak_allocated_bytes=64,gpu_peak_reserved_bytes=80,cpu_rss_bytes=100,
                         worker_rss_bytes=200,pipeline_wait_seconds=.1,step_seconds=1.,elapsed_seconds=float(step),
                         end_to_end_examples_per_second=24.,loss=1.))
    rows += [dict(event='checkpoint',step=250,verified=True,reason='interval'),
             dict(event='graceful_stop',step=final,reason=f'launcher_signal:{signal.SIGUSR1}'),
             dict(event='fit_end',step=final,final_checkpoint_verified=True,end_to_end_examples_per_second=23.)]
    return rows


class CombinedTest(unittest.TestCase):
    def test_dry_run_is_read_only_and_execute_needs_slurm_and_identities(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(combined.subprocess,'Popen',side_effect=AssertionError('launch')):
            target=Path(folder)/'new'
            self.assertEqual(combined.main(['--run-root',str(target)]),0)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(RuntimeError,'split-identities'):
                combined.main(['--execute','--run-root',str(target)])
            with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,'Slurm allocation'):
                combined.main(['--execute','--run-root',str(target),'--split-identities','not-read'])
            self.assertFalse(target.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            for n in (0,17,-1):
                with self.assertRaises(SystemExit):
                    combined.main(['--run-root','/tmp/no-run','--smoke-frames',str(n)])

    def test_exact_comparison_detects_weights_moments_counters_scheduler_and_identities(self):
        original=checkpoint_fixture(step=6)
        self.assertEqual(combined.compare_states(original,copy.deepcopy(original))['status'],'PASS')
        changes=(('state_dict','model.weight'),('optimizer_states',0,'state',0,'exp_avg'),
                 ('optimizer_states',0,'state',0,'exp_avg_sq'),('optimizer_states',0,'state',0,'step'),
                 ('lr_schedulers',0,'last_epoch'),('global_step',),('xr1_replay','rank_sample_offset'),
                 ('xr1_replay_contract','sampler_seed'),('xr1_compact_fp32','pretrained_sha256'),
                 ('xr1_production','config_sha256'))
        for keys in changes:
            modified=copy.deepcopy(original)
            parent=modified
            for key in keys[:-1]: parent=parent[key]
            key=keys[-1]
            if isinstance(parent[key],torch.Tensor): parent[key].view(-1)[0] += .125
            elif isinstance(parent[key],str): parent[key]='c'*64
            else: parent[key]+=1
            result=combined.compare_states(original,modified)
            with self.subTest(keys=keys):
                self.assertEqual(result['status'],'FAIL')
                self.assertGreater(result['unequal_comparisons'],0)
                numeric=[r for r in result['comparisons'] if not r['equal'] and 'max_abs_difference' in r]
                if isinstance(parent[key],torch.Tensor):
                    self.assertEqual(numeric[0]['differing_elements'],1)
                    self.assertGreater(numeric[0]['max_abs_difference'],0)
        missing=copy.deepcopy(original);del missing['state_dict']['model.weight']
        self.assertEqual(combined.compare_states(original,missing)['status'],'FAIL')

    def test_verified_receipt_and_latest_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            ck=checkpoint_fixture(step=3)
            path=Path(folder)/'step-000003.ckpt'
            publish_checkpoint(ck,path)
            combined.verify_checkpoint(folder,3,ck['xr1_replay_contract'],'a'*64)
            with self.assertRaisesRegex(RuntimeError,'configuration'):
                combined.verify_checkpoint(folder,3,ck['xr1_replay_contract'],'b'*64)
            (Path(folder)/'latest.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Latest pointer'):
                combined.verify_checkpoint(folder,3,ck['xr1_replay_contract'],'a'*64)
            path.with_suffix('.verified.json').unlink()
            with self.assertRaises(FileNotFoundError):
                combined.verify_checkpoint(folder,3,ck['xr1_replay_contract'],'a'*64)

    def test_sustained_gate_requires_real_250_steps_and_complete_metrics(self):
        rows=telemetry()
        summary=combined.sustained_summary(rows)
        self.assertEqual(summary['warm_examples_per_second'],24.)
        self.assertEqual(summary['warm_window_steps'],[11,250])
        self.assertEqual(summary['samples'],251*24)
        for key,value in (('retained_examples',23),('views',47),('packed_tokens',9913),
                          ('gpu_reserved_bytes',None),('gpu_peak_reserved_bytes',float('nan')),
                          ('worker_rss_bytes',0),('trainable_fp32',False),('loss',float('inf'))):
            broken=copy.deepcopy(rows);broken[11][key]=value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                combined.sustained_summary(broken)
        for broken in (telemetry(6),telemetry(250),telemetry(260),[r for r in rows if r['event']!='checkpoint']):
            with self.assertRaises(RuntimeError): combined.sustained_summary(broken)

    def test_shutdown_rejects_missing_steps_fatal_orphan_and_wrong_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'telemetry.jsonl'
            good=telemetry()
            def write(rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(good)
            combined.verify_shutdown(folder,range(1,252),stop_reason=f'launcher_signal:{signal.SIGUSR1}')
            for bad in (good[:-1],good+[dict(event='fatal')],good[:10]+good[11:]):
                write(bad)
                with self.assertRaises(RuntimeError): combined.verify_shutdown(folder,range(1,252))
            write(good)
            with self.assertRaises(RuntimeError): combined.verify_shutdown(folder,range(1,252),stop_reason='safe_deadline')
            (Path(folder)/'step-000252.ckpt').write_text('orphan')
            with self.assertRaisesRegex(RuntimeError,'Unverified'): combined.verify_shutdown(folder,range(1,252))

    def test_validation_gate_checks_split_provenance_generated_outputs_and_metrics(self):
        identities=dict(train='a'*64,validation='b'*64,test='c'*64)
        metrics={key:{name:1. for name in ('translation_m','rotation_rad','gripper')}
                 for key in ('micro_mae','episode_macro_mae')}
        good=dict(scope='SMOKE_SUBSET_NOT_A_VALIDATION_RESULT',checkpoint_sha256='d'*64,
                  train_sha256=identities['train'],validation_sha256=identities['validation'],
                  test_identity_sha256=identities['test'],processor_model_compatible=True,
                  metrics=dict(metrics,frames=1),generated_frames=[dict(shape=[30,60],prediction_sha256='e'*64,
                    all_finite=True,generated_action=[[0.]*7 for _ in range(30)],
                    masked_metrics={key:dict(count=3,mae=.1) for key in ('translation_m','rotation_rad','gripper')})])
        combined.validation_gate(good,identities,'d'*64,1)
        for key in ('checkpoint_sha256','train_sha256','validation_sha256','test_identity_sha256',
                    'processor_model_compatible','generated_frames'):
            bad=copy.deepcopy(good);bad[key]=[] if key=='generated_frames' else False
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                combined.validation_gate(bad,identities,'d'*64,1)
        bad=copy.deepcopy(good);bad['metrics']['micro_mae']['gripper']=float('nan')
        with self.assertRaises(RuntimeError): combined.validation_gate(bad,identities,'d'*64,1)

    def test_clock_probe_and_gate_refuse_deadline_source_drift_and_signals(self):
        self.assertTrue(combined.clock_probe()['at_deadline'])
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            q=combined.Qualification(SimpleNamespace(run_root=folder),time.time()+1000,{'phases':{}})
            self.assertEqual(q.gate('example',lambda:{'fixture':True}),{'fixture':True})
            self.assertEqual(q.report['phases']['example']['status'],'PASS')
            with patch.object(combined,'tree_digest',return_value='drift'),self.assertRaisesRegex(RuntimeError,'Source changed'):
                q.gate('drift',lambda:self.fail('must not execute'))
            q.deadline=time.time()+59
            with self.assertRaisesRegex(RuntimeError,'Insufficient time'): q.gate('deadline',lambda:None)
            q.deadline=time.time()+1000;q.handle(signal.SIGTERM,None)
            with self.assertRaisesRegex(RuntimeError,'stop signal'): q.gate('signal',lambda:None)

    def test_phase_process_persists_logs_and_propagates_nonzero_exit(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            q=combined.Qualification(SimpleNamespace(run_root=folder),time.time()+1000,{'phases':{}})
            q.process('cpu-only',[sys.executable,'-c','print("SYNTHETIC_ORCHESTRATOR_ONLY")'],controller=False)
            self.assertIn('SYNTHETIC_ORCHESTRATOR_ONLY',(Path(folder)/'logs/cpu-only.out').read_text())
            self.assertEqual(json.loads((Path(folder)/'logs/cpu-only.exit.json').read_text())['exit_code'],0)
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                q.process('failure',[sys.executable,'-c','raise SystemExit(7)'],controller=False)
            self.assertEqual(json.loads((Path(folder)/'logs/failure.exit.json').read_text())['exit_code'],7)
            with self.assertRaisesRegex(RuntimeError,'Insufficient phase budget'):
                q.process('too-late',[],hard_deadline=time.time()+1)

    def test_existing_run_root_and_failed_phase_cannot_publish_pass(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()), \
             patch.object(combined,'allocation_deadline',return_value=time.time()+1000), \
             patch.dict(os.environ,SLURM_JOB_ID='synthetic'):
            with self.assertRaises(FileExistsError):
                combined.main(['--execute','--run-root',folder,'--split-identities','not-read'])
            target=Path(folder)/'fresh'
            with patch.object(combined.Qualification,'run',side_effect=RuntimeError('synthetic phase failed')):
                with self.assertRaisesRegex(RuntimeError,'synthetic phase failed'):
                    combined.main(['--execute','--run-root',str(target),'--split-identities','not-read'])
            self.assertEqual(json.loads((target/'qualification-report.json').read_text())['status'],'FAIL')

    def test_complete_sequence_and_restart_difference_gate_with_mock_hardware(self):
        from types import SimpleNamespace
        from mibot.utils.doosan_replay import make_contract
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as folder:
                args=SimpleNamespace(run_root=folder,python=sys.executable,overrides='unused',
                                     validation_root='validation-only',test_root='identity-only',smoke_frames=4)
                q=combined.Qualification(args,time.time()+3000,{'phases':{}})
                calls=[]
                actual_gate=q.gate
                def gate(name,function):
                    if name=='preflight':
                        q.identities=dict(train='a'*64,validation='b'*64,test='c'*64)
                        q.fingerprint='a'*64
                        q.contract=make_contract(data_horizon=48284,batch_size=24,world_size=1,
                                                 sampler_seed=42,num_workers=4)
                        return actual_gate(name,lambda:{'scope':'MOCK HARDWARE AND SPLITS ONLY'})
                    return actual_gate(name,function)
                def verify(directory,step,contract,fingerprint,**kwargs):
                    ck=checkpoint_fixture(step=step,horizon=48284)
                    if step>=250: ck['xr1_compact_fp32']['trainable_parameter_count']=141255168
                    if mismatch and step==6 and Path(directory)==q.target('restarted'):
                        ck['state_dict']['model.weight'][0,0]+=.125
                    return ck
                def train(name,maximum=6,stop=None,resume=None,monitor=None,label=None,hard=None):
                    calls.append((name,maximum,stop,resume is not None))
                    directory=q.target(name);directory.mkdir(parents=True,exist_ok=True)
                    final=stop or maximum
                    if name=='sustained':
                        (directory/'step-000250.verified.json').write_text('{"step":250}')
                        (directory/'step-000250.ckpt').write_text('MOCK CHECKPOINT')
                        child=SimpleNamespace(pid=12345,poll=lambda:None,send_signal=lambda sig:calls.append(('signal',sig)))
                        monitor(child)
                        rows=telemetry(251)
                        rows.append(dict(event='checkpoint',step=251,verified=True,reason=f'launcher_signal:{signal.SIGUSR1}'))
                        rows[-1],rows[-2]=rows[-2],rows[-1]  # fit_end must be final
                        (Path(folder)/name/'launch-mock.stop.json').write_text(json.dumps(dict(signal=int(signal.SIGUSR1),controller_pid=12345)))
                        final=251
                    else:
                        rows=combined.rows_at(directory)
                        initial=3 if resume else 0
                        rows += [dict(event='train_start',step=initial,replay_offset=initial*24)]
                        rows += [dict(event='step',step=step) for step in range(initial+1,final+1)]
                        if stop: rows.append(dict(event='graceful_stop',step=final,reason='qualification_step'))
                        rows.append(dict(event='fit_end',step=final,final_checkpoint_verified=True))
                    (directory/f'step-{final:06d}.ckpt').write_text('MOCK CHECKPOINT')
                    (directory/f'step-{final:06d}.verified.json').write_text(json.dumps(dict(step=final)))
                    (directory/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                def inference(name,command,**kwargs):
                    calls.append(('inference',))
                    result=dict(scope='SMOKE_SUBSET_NOT_A_VALIDATION_RESULT')
                    from mibot.utils.claix_checkpoint import file_sha256
                    result.update(checkpoint_sha256=file_sha256(q.final_checkpoint),config_sha256=q.fingerprint,
                        train_sha256=q.identities['train'],validation_sha256=q.identities['validation'],
                        test_identity_sha256=q.identities['test'],processor_model_compatible=True,
                        metrics=dict(frames=4,micro_mae=dict(translation_m=1.,rotation_rad=1.,gripper=1.),
                                     episode_macro_mae=dict(translation_m=1.,rotation_rad=1.,gripper=1.)),
                        generated_frames=[dict(shape=[30,60],prediction_sha256='e'*64,all_finite=True,
                                               generated_action=[[0.]*7 for _ in range(30)],
                                               masked_metrics={key:dict(count=3,mae=1.) for key in ('translation_m','rotation_rad','gripper')})]*4)
                    (Path(folder)/'validation-smoke.json').write_text(json.dumps(result))
                with patch.object(q,'gate',side_effect=gate),patch.object(q,'train',side_effect=train), \
                     patch.object(q,'process',side_effect=inference),patch.object(combined,'verify_checkpoint',side_effect=verify):
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError,'restart differences'): q.run()
                        self.assertFalse(any(row[0]=='sustained' for row in calls))
                        comparison=json.loads((Path(folder)/'restart-comparison.json').read_text())
                        self.assertEqual(comparison['status'],'FAIL')
                    else:
                        q.run()
                        self.assertEqual(list(q.report['phases']),list(combined.PHASES))
                        self.assertTrue(all(row['status']=='PASS' for row in q.report['phases'].values()))
                        self.assertEqual(calls[:3],[('uninterrupted',6,None,False),('restarted',6,3,False),('restarted',6,None,True)])
                        self.assertEqual(calls[3],('sustained',260,None,False))
                        self.assertEqual(calls[4],('signal',signal.SIGUSR1))
                        self.assertEqual(calls[5],('inference',))

    def test_failed_monitor_terminates_only_its_owned_cpu_subprocess(self):
        from types import SimpleNamespace
        import psutil
        with tempfile.TemporaryDirectory() as folder:
            q=combined.Qualification(SimpleNamespace(run_root=folder),time.time()+1000,{'phases':{}})
            owned=[]
            def fail_monitor(child):
                owned.append(child.pid)
                raise RuntimeError('synthetic monitor failure')
            with self.assertRaisesRegex(RuntimeError,'synthetic monitor failure'):
                q.process('aborted',[sys.executable,'-c','import time; time.sleep(60)'],
                          controller=False,monitor=fail_monitor)
            self.assertFalse(psutil.pid_exists(owned[0]))
            result=json.loads((Path(folder)/'logs/aborted.exit.json').read_text())
            self.assertEqual(result['exit_code'],-signal.SIGTERM)
