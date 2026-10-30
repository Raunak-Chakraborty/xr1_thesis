import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mibot.utils.claix_checkpoint import (
    load_verified, publish_checkpoint, rng_state, rotate_checkpoints, validate_production_checkpoint,
)
from mibot.utils.doosan_replay import make_contract, checkpoint_cursor


def checkpoint_fixture(step=2, horizon=3000):
    contract = make_contract(data_horizon=horizon,batch_size=24,world_size=1,sampler_seed=42,num_workers=4)
    progress = dict(ready=step, started=step, processed=step, completed=step-1)
    return dict(global_step=step, epoch=0, state_dict={'model.weight': torch.ones(2,3)},
                xr1_compact_fp32=dict(schema='xr1-doosan-fp32-trainable-compact-lightning-v1',
                                      selected_tensor_count=1, trainable_parameter_count=6,
                                      pretrained_sha256='a'*64,language_sha256='b'*64,source_commit='synthetic',last_n=1,
                                      trainable_storage_dtype='torch.float32',
                                      frozen_weights_source='external_original_pretrained_checkpoint',
                                      optimizer='torch.optim.AdamW with fp32 exp_avg and exp_avg_sq',
                                      selected_key_sha256=hashlib.sha256(b'model.weight').hexdigest()),
                optimizer_states=[dict(state={0: dict(step=torch.tensor(float(step)),
                                                       exp_avg=torch.ones(2,3), exp_avg_sq=torch.ones(2,3))},
                                       param_groups=[dict(params=[0], lr=1e-5)])],
                lr_schedulers=[dict(last_epoch=step)], loops={'fit_loop': {
                    'epoch_loop.automatic_optimization.optim_progress': {'optimizer': {'step': {'total': {'completed': step}}}},
                    'epoch_loop.batch_progress': {'total': progress, 'current': dict(progress)}}},
                xr1_replay=checkpoint_cursor(contract,step), xr1_replay_contract=contract,
                xr1_production=dict(schema='claix-b24-runtime-v1',config_sha256='a'*64,rng_by_rank=[rng_state()]))


class CheckpointTest(unittest.TestCase):
    def test_approved_last_one_policy_preserves_frozen_stacks(self):
        from types import SimpleNamespace
        from mibot.utils.doosan_last_n import apply_last_n_policy
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(24)])
                self.language=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(36)])
                self.dit=torch.nn.Module()
                self.dit.layers=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(36)])
                self.vlm=SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(blocks=self.vision),
                    language_model=SimpleNamespace(layers=self.language)))
                for name in ('state_projector','action_projector','action_output_layer','t_embedder','t_projector','sink'):
                    setattr(self,name,torch.nn.Linear(1,1))
        model=Model();report=apply_last_n_policy(model,1)
        for stack in (model.vision,model.language,model.dit.layers):
            self.assertTrue(all(not p.requires_grad for block in stack[:-1] for p in block.parameters()))
            self.assertTrue(all(p.requires_grad for p in stack[-1].parameters()))
        self.assertEqual(report['n'],1)
        self.assertTrue(all(p.requires_grad for p in model.state_projector.parameters()))

    def test_verified_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'step-000002.ckpt'
            ck = checkpoint_fixture()
            receipt = publish_checkpoint(ck,path)
            loaded = load_verified(path,ck['xr1_replay_contract'],expected_config='a'*64)
            self.assertEqual(loaded['global_step'],2)
            self.assertEqual(receipt['bytes'],path.stat().st_size)
            with self.assertRaises(FileExistsError):
                publish_checkpoint(ck,path)
            with path.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                load_verified(path,ck['xr1_replay_contract'])

    def test_required_state_and_counters(self):
        ck = checkpoint_fixture()
        keys = ('state_dict','optimizer_states','lr_schedulers','loops','xr1_replay','xr1_production')
        for key in keys:
            damaged = copy.deepcopy(ck); damaged.pop(key)
            with self.subTest(key=key), self.assertRaises((RuntimeError,KeyError)):
                validate_production_checkpoint(damaged,ck['xr1_replay_contract'])
        for change in ('optimizer_step','scheduler_step','nonfinite','rng','config'):
            damaged = copy.deepcopy(ck)
            if change == 'optimizer_step':
                damaged['optimizer_states'][0]['state'][0]['step'] += 1
            elif change == 'scheduler_step':
                damaged['lr_schedulers'][0]['last_epoch'] += 1
            elif change == 'nonfinite':
                damaged['state_dict']['model.weight'][0,0] = float('nan')
            elif change == 'rng':
                damaged['xr1_production']['rng_by_rank'] = []
            else:
                damaged['xr1_production']['config_sha256'] = 'b'*64
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_production_checkpoint(damaged,ck['xr1_replay_contract'],expected_config='a'*64)

    def test_batch_end_boundary_and_fractional_counters_fail_closed(self):
        ck = checkpoint_fixture()
        for change in ('current', 'after_batch_end', 'epoch', 'adam_fraction', 'scheduler_fraction'):
            damaged = copy.deepcopy(ck)
            progress = damaged['loops']['fit_loop']['epoch_loop.batch_progress']
            if change == 'current':
                progress['current']['ready'] -= 1
            elif change == 'after_batch_end':
                progress['total']['completed'] += 1
                progress['current']['completed'] += 1
            elif change == 'epoch':
                damaged['epoch'] = 1
            elif change == 'adam_fraction':
                damaged['optimizer_states'][0]['state'][0]['step'] += .5
            else:
                damaged['lr_schedulers'][0]['last_epoch'] += .5
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_production_checkpoint(damaged,ck['xr1_replay_contract'])

    def test_receipt_configuration_must_match_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'step-000002.ckpt'
            ck = checkpoint_fixture()
            publish_checkpoint(ck,path)
            receipt_path = path.with_suffix('.verified.json')
            row = json.loads(receipt_path.read_text())
            row['config_sha256'] = 'b'*64
            receipt_path.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, 'Receipt configuration'):
                load_verified(path,ck['xr1_replay_contract'])

    def test_interrupted_write_keeps_previous_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'step-000002.ckpt', Path(folder)/'step-000003.ckpt'
            ck = checkpoint_fixture(); publish_checkpoint(ck, old)
            with patch('mibot.utils.claix_checkpoint.torch.save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    publish_checkpoint(checkpoint_fixture(3),new)
            self.assertFalse(new.exists())
            self.assertEqual(json.loads((Path(folder)/'latest.json').read_text())['global_step'],2)
            load_verified(old,ck['xr1_replay_contract'])
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_orphan_filename_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'step-000002.ckpt';torch.save(checkpoint_fixture(),path)
            with self.assertRaises(FileNotFoundError):
                load_verified(path,checkpoint_fixture()['xr1_replay_contract'])

    def test_terminal_checkpoint_and_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            for step in (1999,2000,2250,2500,3000):
                ck=checkpoint_fixture(step)
                publish_checkpoint(ck,Path(folder)/f'step-{step:06d}.ckpt')
                rotate_checkpoints(folder)
            self.assertEqual(sorted(p.name for p in Path(folder).glob('*.ckpt')),
                             ['step-002000.ckpt','step-002500.ckpt','step-003000.ckpt'])
            final=Path(folder)/'step-003000.ckpt'
            load_verified(final,ck['xr1_replay_contract'],resumable=False)
            with self.assertRaises(RuntimeError):
                load_verified(final,ck['xr1_replay_contract'])
