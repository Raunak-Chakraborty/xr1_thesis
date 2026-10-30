import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
