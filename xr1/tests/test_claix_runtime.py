import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from lightning import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy
from mmengine import Config

from mibot.data.datamodule.doosan_datamodule import DoosanDataModule
from mibot.models.runner.doosan_compact_runner import DoosanCompactRunner
from mibot.utils.claix_checkpoint import load_verified
from mibot.utils.claix_runtime import ProductionRuntime, StopRequest, VerifiedCheckpointIO
from mibot.utils.doosan_compact_fp32 import install_compact_fp32
from test_claix_replay import AugmentedFixture


def tiny_collate(items):
    indices=torch.tensor([i['index'] for i in items],dtype=torch.float32)
    batch = dict(input_ids=torch.ones(1,len(items)*10,dtype=torch.long),
                action_segments=torch.zeros(len(items),2,dtype=torch.long),
                image_grid_thw=torch.ones(2*len(items),3,dtype=torch.long),
                x=torch.stack([indices/20,torch.ones_like(indices)],dim=-1),
                target=indices/40)
    return {key:value.numpy() for key,value in batch.items()}


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([0.1,0.2],dtype=torch.bfloat16))
        self.frozen=torch.nn.Parameter(torch.tensor(1.,dtype=torch.bfloat16),requires_grad=False)
    def forward(self,batch,return_loss):
        # Exercise model RNG and per-step seed on uninterrupted/resumed training.
        y=(batch['x']*self.weight).sum(dim=-1)+torch.rand(())/10
        return {'loss':((y-batch['target'])*self.frozen).square().mean()}


class TinyRunner(DoosanCompactRunner):
    def training_step(self, batch, batch_idx):
        if not hasattr(self, 'seen_indices'):
            self.seen_indices = []
        self.seen_indices.append(batch['x'][:,0].tolist())
        return super().training_step(batch, batch_idx)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return {key:torch.from_numpy(value) for key,value in batch.items()}
    def configure_model(self):
        self.model=TinyNetwork()
        self._compact_meta=install_compact_fp32(self,selected_keys={'model.weight'},
                                               pretrained_sha256='a'*64,language_sha256='b'*64,
                                               source_commit='synthetic',last_n=1)


def tiny_run(directory, *, workers, max_steps=8, stop_after=None, resume=None, deadline=None):
    params=Config(dict(max_steps=8,train_datasets=dict(batch_size=2,num_workers=workers,
                  prefetch_factor=2,persistent_workers=bool(workers),pin_memory=False,
                  deterministic_replay=True,sampler_seed=42,resume_step=resume[1] if resume else 0)))
    with patch('mibot.data.datamodule.base_datamodule.CustomCollate'):
        dm=DoosanDataModule(params)
    dm.collate_fn=tiny_collate
    dm.train_set=AugmentedFixture(length=16)
    model=TinyRunner(dict(compact_fp32=dict(last_n=1,data_horizon=8,batch_size=2,world_size=1,
                                          sampler_seed=42,num_workers=workers,
                                          resume_step=resume[1] if resume else 0,config_sha256='c'*64),
                          optimizer=dict(type='torch.optim.AdamW',params=dict(lr=1.0,weight_decay=0.1)),
                          scheduler=dict(type='mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup',
                                         params=dict(num_warmup_steps=2,num_training_steps=8,
                                                     max_lr=0.02,min_lr=0.005,warmup_lr_start=0.001))))
    runtime=ProductionRuntime(directory,deadline=deadline if deadline else time.time()+3600,
                              interval=250,stop_after_step=stop_after)
    trainer=Trainer(accelerator='cpu',devices=1,max_steps=max_steps,max_epochs=1,
                    strategy=SingleDeviceStrategy(device='cpu',checkpoint_io=VerifiedCheckpointIO()),
                    callbacks=[runtime],logger=False,enable_checkpointing=False,enable_progress_bar=False,
                    enable_model_summary=False,use_distributed_sampler=False,num_sanity_val_steps=0)
    trainer.fit(model,datamodule=dm,ckpt_path=str(resume[0]) if resume else None)
    return model,trainer


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Network namespace denies even loopback Gloo sockets. Only the
        # DataModule rank lookup is mocked; Trainer is real single-device CPU.
        cls.dist_patch=patch('mibot.data.datamodule.doosan_datamodule.dist',
                             SimpleNamespace(is_available=lambda:True,is_initialized=lambda:True,
                                             get_world_size=lambda:1,get_rank=lambda:0))
        cls.dist_patch.start()
    @classmethod
    def tearDownClass(cls):
        cls.dist_patch.stop()

    def test_lightning_resume_early_and_near_horizon_workers(self):
        for workers in (0,4,8):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                full,_=tiny_run(root/'full',workers=workers)
                expected=load_verified(root/'full/step-000008.ckpt',full._replay_contract,resumable=False)
                for split in (1,3,6,7):
                    run=root/f'split{split}'
                    _,first=tiny_run(run,workers=workers,stop_after=split)
                    self.assertEqual(first.global_step,split)
                    resumed,last=tiny_run(run,workers=workers,resume=(run/f'step-{split:06d}.ckpt',split))
                    self.assertEqual(last.global_step,8)
                    self.assertEqual(resumed.seen_indices,full.seen_indices[split:])
                    actual=load_verified(run/'step-000008.ckpt',resumed._replay_contract,resumable=False)
                    self.assertTrue(torch.equal(expected['state_dict']['model.weight'],actual['state_dict']['model.weight']))
                    self.assertEqual(expected['lr_schedulers'],actual['lr_schedulers'])
                    for key in ('step','exp_avg','exp_avg_sq'):
                        self.assertTrue(torch.equal(expected['optimizer_states'][0]['state'][0][key],
                                                    actual['optimizer_states'][0]['state'][0][key]))
                    self.assertEqual(float(resumed.model.frozen),1.)
                    self.assertEqual(resumed.model.frozen.dtype,torch.bfloat16)
                    self.assertEqual(resumed.model.weight.dtype,torch.float32)
                    rows=[json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
                    steps=[r['step'] for r in rows if r['event']=='step']
                    self.assertEqual(steps,list(range(1,9)))
                    self.assertTrue(rows[-1]['final_checkpoint_verified'])

    def test_telemetry_includes_loader_wait_and_final_checkpoint_io(self):
        original_getitem=AugmentedFixture.__getitem__
        original_save=ProductionRuntime.save
        def slow_getitem(dataset,index):
            time.sleep(.02)
            return original_getitem(dataset,index)
        def slow_save(runtime,*args):
            time.sleep(.03)
            return original_save(runtime,*args)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(AugmentedFixture,'__getitem__',slow_getitem), \
             patch.object(ProductionRuntime,'save',slow_save):
            tiny_run(folder,workers=0,max_steps=1)
            rows=[json.loads(line) for line in (Path(folder)/'telemetry.jsonl').read_text().splitlines()]
            step=next(r for r in rows if r['event']=='step')
            final=rows[-1]
            self.assertGreaterEqual(step['pipeline_wait_seconds'],.04)
            self.assertGreaterEqual(final['end_to_end_seconds'],.07)
            self.assertLess(final['end_to_end_examples_per_second'],step['end_to_end_examples_per_second'])

    def test_deadline_stops_on_completed_optimizer_step(self):
        with tempfile.TemporaryDirectory() as folder:
            model,trainer=tiny_run(folder,workers=0,deadline=time.time()-1)
            self.assertEqual(trainer.global_step,1)
            ck=load_verified(Path(folder)/'step-000001.ckpt',model._replay_contract)
            self.assertEqual(ck['xr1_replay']['rank_sample_offset'],2)
            rows=[json.loads(line) for line in (Path(folder)/'telemetry.jsonl').read_text().splitlines()]
            self.assertIn('graceful_stop',[r['event'] for r in rows])
            step=next(r for r in rows if r['event']=='step')
            self.assertGreater(step['end_to_end_examples_per_second'],0)

    def test_signal_handler_only_sets_flag(self):
        request=StopRequest(time.time()+100)
        request.install()
        try:
            with patch('torch.save',side_effect=AssertionError('asynchronous save')):
                signal.raise_signal(signal.SIGUSR1)
        finally:
            request.restore()
        self.assertTrue(request.due())
        self.assertEqual(request.reason,f'signal:{signal.SIGUSR1}')

    def test_nonfinite_gradient_rejected(self):
        module=TinyRunner.__new__(TinyRunner)
        torch.nn.Module.__init__(module)
        module.model=TinyNetwork()
        module.model.weight.grad=torch.full_like(module.model.weight,float('nan'))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite gradient'):
            module.on_before_optimizer_step(None)

    def test_skipped_step_and_sampler_replacement_fail_closed(self):
        runtime=ProductionRuntime('/tmp/unused-claix-test',deadline=time.time()+100)
        trainer=SimpleNamespace(train_dataloader=SimpleNamespace(sampler=object()),global_step=0)
        module=SimpleNamespace(_replay_contract={'batch_size':24,'num_workers':4,'sampler_seed':42,'world_size':1})
        with self.assertRaisesRegex(RuntimeError,'sampler contract drift'):
            runtime.on_train_start(trainer,module)
        runtime.previous_step=1
        with self.assertRaisesRegex(RuntimeError,'skipped or repeated'):
            runtime.on_train_batch_end(SimpleNamespace(global_step=1),module,None,None,0)

    def test_provisional_cosine_endpoints(self):
        from mibot.utils.cosine_warmup import get_cosine_schedule_with_warmup
        parameter=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.AdamW([parameter],lr=1.)
        scheduler=get_cosine_schedule_with_warmup(optimizer,500,48284,max_lr=2e-5,
                                                 warmup_lr_start=5e-7,min_lr=5e-6)
        factor=scheduler.lr_lambdas[0]
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],5e-7)
        self.assertAlmostEqual(factor(500),2e-5)
        self.assertAlmostEqual(factor(48284),5e-6)
