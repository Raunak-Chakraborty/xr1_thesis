import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from claix_test_fixtures import write_path_overrides

from mibot.utils.claix_contract import audit_splits, resume_fingerprint, split_inventory
from mibot.utils.claix_evaluation import action_errors, evaluate_items, inference_payload
from mibot.utils.io import ACTION_EPS


def eval_item(target=0.,steps=30):
    mask=torch.zeros(30,60,dtype=torch.int32);mask[:steps,:7]=1
    return dict(messages=[dict(role='user',content=[dict(type='text',text='assemble /no_cot')]),
                          dict(role='assistant',content=[dict(type='text',text='<cot></cot>')]),
                          dict(role='user',content=[dict(type='text',text='Robot state: <state>')]),
                          dict(role='assistant',content=[dict(type='text',text='<a_0><score>')])],
                action=torch.full((30,60),float(target)),action_mask=mask,state=torch.zeros(1,60))


class ProcessorFixture:
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages
        return dict(input_ids=torch.ones(1,10,dtype=torch.long),image_grid_thw=torch.ones(2,3,dtype=torch.long))


class ModelFixture(torch.nn.Module):
    def generate(self,batch):
        assert torch.count_nonzero(batch['action'])==0
        assert not any('target' in key or 'vlm_action' in key for key in batch)
        return torch.ones(1,30,60)


class EvaluationTest(unittest.TestCase):
    def test_no_teacher_forcing_and_masked_physical_metrics(self):
        processor=ProcessorFixture()
        item=eval_item(0.,steps=3)
        payload=inference_payload(item,processor)
        self.assertEqual(len(processor.messages[0]),2)
        self.assertEqual(torch.count_nonzero(payload['action']).item(),0)
        prediction=torch.ones(30,60);prediction[3:]=float('nan')
        row=action_errors(prediction,item['action'],item['action_mask'],torch.ones(30,60)*2)
        for value in row.values():
            self.assertAlmostEqual(value['mae'],2.+ACTION_EPS,places=6)
        self.assertEqual(row['translation_m']['count'],9)
        broken=copy.deepcopy(item);broken['messages'][0]['content'][0]['text']='<a_0>'
        with self.assertRaisesRegex(ValueError,'leakage'):
            inference_payload(broken,processor)

    def test_full_evaluation_aggregation_and_state_restoration(self):
        model=ModelFixture().train()
        random.seed(125);torch.manual_seed(99)
        py_state,torch_state=random.getstate(),torch.get_rng_state()
        result=evaluate_items(model,[('episode-a',0,eval_item(0.,1)),('episode-b',0,eval_item(1.,3))],
                              ProcessorFixture(),torch.ones(30,60))
        self.assertTrue(model.training)
        self.assertEqual(py_state,random.getstate());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(result['episodes'],2)
        self.assertAlmostEqual(result['micro_mae']['translation_m'],0.25*(1+ACTION_EPS))
        self.assertAlmostEqual(result['episode_macro_mae']['translation_m'],0.5*(1+ACTION_EPS))

    def test_generated_smoke_evidence_and_nonfinite_output(self):
        evidence=[]
        evaluate_items(ModelFixture(), [('validation-only',2,eval_item(0.,3))], ProcessorFixture(),
                       torch.ones(30,60), evidence=evidence)
        row=evidence[0]
        self.assertEqual(row['episode'],'validation-only')
        self.assertEqual(row['frame'],2)
        self.assertEqual(row['shape'],[30,60])
        self.assertEqual(len(row['prediction_sha256']),64)
        self.assertEqual(len(row['generated_action']),30)
        self.assertEqual(row['masked_metrics']['translation_m']['count'],9)
        class BadModel(ModelFixture):
            def generate(self,batch):
                return torch.full((1,30,60),float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite generated'):
            evaluate_items(BadModel(), [('validation-only',2,eval_item())], ProcessorFixture(), torch.ones(30,60))

    def test_split_counts_leakage_and_inventory_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            roots=[Path(folder)/name for name in ('train','val','test')]
            for root,n in zip(roots,(255,34,34)):
                root.mkdir()
                for i in range(n):
                    frames=(289703-254 if i==0 else 1) if root.name=='train' else 1
                    (root/f'{i:03}.json').write_text(json.dumps(dict(num_frames=frames,
                        videos={'external_camera_2':[dict(path=f'/{root.name}/{i}/ext.mp4')],
                                'tcp_camera':[dict(path=f'/{root.name}/{i}/tcp.mp4')]})))
            inventories=audit_splits(*roots)
            self.assertEqual(inventories[0]['frames'],289703)
            old=inventories[0]['sha256']
            path=roots[0]/'000.json';obj=json.loads(path.read_text());obj['other']='changed';path.write_text(json.dumps(obj))
            self.assertNotEqual(old,split_inventory(roots[0])['sha256'])
            (roots[1]/'000.json').write_text((roots[2]/'000.json').read_text())
            with self.assertRaisesRegex(RuntimeError,'leakage'):
                audit_splits(*roots)

    def test_production_hydra_and_resume_fingerprint(self):
        root=Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
            cfg=OmegaConf.to_container(compose(config_name='claix_b24_production'),resolve=True)
        self.assertEqual(cfg['production']['max_length'],12288)
        self.assertEqual(cfg['data']['params']['max_steps'],48284)
        self.assertEqual(cfg['trainer']['scheduler']['params']['num_training_steps'],48284)
        original=resume_fingerprint(cfg,{'sha256':'a'*64},12288)
        allowed=copy.deepcopy(cfg);allowed['trainer']['ckpt_path']='new.ckpt';allowed['trainer']['max_steps']=6
        self.assertEqual(original,resume_fingerprint(allowed,{'sha256':'a'*64},12288))
        changed=copy.deepcopy(cfg);changed['trainer']['scheduler']['params']['num_warmup_steps']=501
        self.assertNotEqual(original,resume_fingerprint(changed,{'sha256':'a'*64},12288))

    def test_evaluator_entrypoint_integration_with_mock_model_and_held_out_data(self):
        import contextlib
        import importlib.util
        import io
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_claix_checkpoint import checkpoint_fixture
        path=Path(__file__).resolve().parents[1]/'tools/claix_b24_evaluate.py'
        spec=importlib.util.spec_from_file_location('claix_evaluator_entry',path)
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            entry.main(['--overrides','unused','--checkpoint','unused','--validation-root','unused',
                        '--test-root','unused','--output','unused','--limit','-1'])
        ck=checkpoint_fixture(step=6,horizon=48284)
        class Runner:
            def __init__(self,params):
                self._compact_meta=ck['xr1_compact_fp32'];self.model=ModelFixture()
                self.params=params
            def configure_model(self): pass
            def load_state_dict(self,state):
                if set(state)!=set(ck['state_dict']): raise AssertionError('wrong checkpoint')
        class Dataset:
            def __init__(self,params):
                # Entry must point inference only at held-out validation data.
                assert params.train_datasets.paths==[validation]
                self.samples=[dict(file=str(Path(validation)/'episode.json'),frame_index=i) for i in range(3)]
                self.std=torch.ones(30,60).numpy()
            def __len__(self): return len(self.samples)
            def __getitem__(self,index): return eval_item(0.,3)
        def cpu_evaluate(model,items,processor,std,**kwargs):
            return evaluate_items(model,items,processor,std,device='cpu',evidence=kwargs['evidence'])
        def checksum(path):
            if str(path).endswith('model_states.pt'):
                return '94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7'
            if str(path).endswith('.yaml'):
                return '21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d'
            return 'd'*64
        with tempfile.TemporaryDirectory() as folder:
            validation=str(Path(folder)/'validation')
            output=Path(folder)/'result.json'
            with patch.dict(os.environ,SLURM_JOB_ID='synthetic'), \
                 patch('mibot.utils.claix_contract.audit_splits',return_value=[{'sha256':v*64} for v in 'abc']) as audit, \
                 patch('mibot.utils.claix_contract.resume_fingerprint',return_value='a'*64), \
                 patch('mibot.utils.claix_checkpoint.load_verified',return_value=ck) as verified, \
                 patch('mibot.utils.claix_checkpoint.file_sha256',side_effect=checksum), \
                 patch('mibot.models.runner.doosan_compact_runner.DoosanCompactRunner',Runner), \
                 patch('mibot.data.datasets.doosan_json_dataset.DoosanValidationDataset',Dataset), \
                 patch('mibot.data.collate.custom_collate.CustomCollate',return_value=SimpleNamespace(processor=ProcessorFixture())), \
                 patch('mibot.utils.claix_evaluation.evaluate_items',side_effect=cpu_evaluate), \
                 patch('torch.autocast',return_value=contextlib.nullcontext()):
                entry.main(['--overrides',str(write_path_overrides(folder)),
                            '--checkpoint',str(Path(folder)/'mock.ckpt'),'--validation-root',validation,
                            '--test-root','IDENTITY_ONLY_NEVER_INFER','--output',str(output),'--limit','1'])
            self.assertEqual(audit.call_args.args[2],'IDENTITY_ONLY_NEVER_INFER')
            self.assertEqual(verified.call_args.kwargs,dict(expected_config='a'*64,resumable=False))
            result=json.loads(output.read_text())
            self.assertEqual(result['scope'],'SMOKE_SUBSET_NOT_A_VALIDATION_RESULT')
            self.assertEqual(result['metrics']['frames'],1)
            self.assertEqual(result['generated_frames'][0]['episode'],'episode.json')
            self.assertTrue(result['processor_model_compatible'])
