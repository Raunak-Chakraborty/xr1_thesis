"""Real CPU Lightning/AdamW recovery at 8750; tiny synthetic network, no XR1 weights.

The existing offline harness stubs mmengine registration and the unavailable
backbone import only. No optimizer, scheduler, checkpoint or Trainer mock is used.
SingleDeviceStrategy avoids claiming Gloo/NCCL or H100 qualification.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', MAX_LENGTH='12288', MPLCONFIGDIR='/tmp/xr1-reliability-mpl')
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'reliability'),str(REPO/'xr1/tests'),str(REPO/'xr1')]
from run_claix_cpu_tests import bootstrap
bootstrap()
import numpy as np
import torch
from lightning import Trainer, LightningDataModule
from lightning.pytorch.strategies import SingleDeviceStrategy
from torch.utils.data import DistributedSampler
from common import sha, module, CONFIG, INVENTORY
from recover import copy_exact, incident, recompose
from test_claix_runtime import TinyRunner, tiny_collate
from mibot.data.datamodule.doosan_datamodule import ReplayDataLoader
from mibot.utils.doosan_replay import RankOffsetSampler, validate_full_compact_resume
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
from mibot.utils.claix_checkpoint import load_verified, restore_rng, publish_checkpoint

torch.set_num_threads(1)
HORIZON, BATCH = 48284, 24


class Numbers:
    def __len__(self):
        return HORIZON * BATCH

    def __getitem__(self,index):
        return dict(index=index)


class Data(LightningDataModule):
    def __init__(self,step):
        super().__init__()
        self.step = step

    def train_dataloader(self):
        dataset = Numbers()
        sampler = DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True,seed=42)
        return ReplayDataLoader(dataset,batch_size=BATCH,
                                sampler=RankOffsetSampler(sampler,self.step*BATCH,batch_size=BATCH),
                                num_workers=0,collate_fn=tiny_collate,
                                generator=torch.Generator().manual_seed(42))


class FailableRunner(TinyRunner):
    fail_at = None

    def training_step(self,batch,batch_idx):
        if self.fail_at is not None and self.global_step == self.fail_at:
            raise RuntimeError('injected fatal after optimizer step 8758')
        return super().training_step(batch,batch_idx)


def run(directory, *, resume=None, fail=False, stop=None):
    step = 8750 if resume else 0
    model = FailableRunner(dict(compact_fp32=dict(last_n=1,data_horizon=HORIZON,batch_size=BATCH,
                    world_size=1,sampler_seed=42,num_workers=0,resume_step=step,config_sha256='c'*64),
                    optimizer=dict(type='torch.optim.AdamW',params=dict(lr=1.,weight_decay=.1,betas=(.9,.95),eps=1e-8)),
                    scheduler=dict(type='mibot.utils.cosine_warmup.get_cosine_schedule_with_warmup',
                        params=dict(num_warmup_steps=500,num_training_steps=HORIZON,
                                    max_lr=2e-5,min_lr=5e-6,warmup_lr_start=5e-7))))
    model.fail_at = 8758 if fail else None
    runtime = ProductionRuntime(directory,deadline=time.time()+3600,interval=250,stop_after_step=stop)
    trainer = Trainer(accelerator='cpu',devices=1,max_steps=8760,max_epochs=1,
                      strategy=SingleDeviceStrategy(device='cpu',checkpoint_io=VerifiedCheckpointIO()),
                      callbacks=[runtime],logger=False,enable_checkpointing=False,
                      enable_progress_bar=False,enable_model_summary=False,
                      gradient_clip_val=1.,gradient_clip_algorithm='norm',
                      use_distributed_sampler=False,num_sanity_val_steps=0)
    try:
        trainer.fit(model,datamodule=Data(step),ckpt_path=str(resume) if resume else None)
    except RuntimeError as error:
        if not fail or str(error) != 'injected fatal after optimizer step 8758':
            raise
    else:
        if fail:
            raise AssertionError('Expected injected fatal exception')
    return model,trainer


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.original = cls.root/'original'
        # Suppress thousands of ordinary per-step lines in the test result artifact.
        with tempfile.TemporaryFile(mode='w+') as output, contextlib.redirect_stdout(output):
            cls.failed_model,cls.failed_trainer = run(cls.original,fail=True)
        cls.path = cls.original/'step-008750.ckpt'
        cls.ck = load_verified(cls.path,expected_config='c'*64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fatal_tail_recovery_exact_weights_moments_scheduler_and_cursor(self):
        self.assertEqual(self.failed_trainer.global_step,8758)
        self.assertEqual(self.ck['global_step'],8750)
        self.assertEqual(self.ck['xr1_replay']['rank_sample_offset'],210000)
        self.assertEqual(self.ck['lr_schedulers'][0]['last_epoch'],8750)
        self.assertEqual(self.ck['optimizer_states'][0]['state'][0]['step'].item(),8750)
        self.assertFalse((self.original/'safe-stop.json').exists())
        telemetry = self.original/'telemetry.jsonl'
        before = sha(telemetry)
        parent_checkpoint_hash = sha(self.path)
        with self.assertRaisesRegex(RuntimeError,'Telemetry extends beyond'):
            module('claix_b24_launch').check_resume_directory(self.original,self.path)
        recovery = self.root/'recovery'
        recovery.mkdir()
        seed = recovery/'seed'
        seed.mkdir()
        copy_exact(self.path,seed/self.path.name)
        copy_exact(self.path.with_suffix('.verified.json'),seed/self.path.with_suffix('.verified.json').name)
        with tempfile.TemporaryFile(mode='w+') as output, contextlib.redirect_stdout(output):
            model,trainer = run(recovery/'target',resume=seed/self.path.name,stop=8758)
        self.assertEqual(trainer.global_step,8758)
        self.assertEqual(model.seen_indices,self.failed_model.seen_indices[8750:8758])
        self.assertTrue(torch.equal(model.model.weight,self.failed_model.model.weight))
        self.assertEqual(trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                         self.failed_trainer.lr_scheduler_configs[0].scheduler.state_dict())
        actual = trainer.optimizers[0].state_dict()['state'][0]
        expected = self.failed_trainer.optimizers[0].state_dict()['state'][0]
        for key in ('step','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(actual[key],expected[key]),key)
        rows = [json.loads(line) for line in (recovery/'target/telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['step'],8750)
        self.assertEqual([r['step'] for r in rows if r['event']=='step'],list(range(8751,8759)))
        self.assertIn('graceful_stop',[r['event'] for r in rows])
        self.assertTrue(rows[-1]['final_checkpoint_verified'])
        self.assertTrue((recovery/'target/safe-stop.json').exists())
        self.assertEqual(sha(telemetry),before)
        self.assertEqual(sha(self.path),parent_checkpoint_hash)
        self.assertEqual(sha(seed/self.path.name),parent_checkpoint_hash)

    def test_saved_cpu_rng_restores_exactly(self):
        state = self.ck['xr1_production']['rng_by_rank'][0]
        restore_rng(state)
        a = (random.random(),np.random.rand(),torch.rand(4))
        restore_rng(state)
        b = (random.random(),np.random.rand(),torch.rand(4))
        self.assertEqual(a[:2],b[:2])
        self.assertTrue(torch.equal(a[2],b[2]))

    def test_corrupt_incomplete_config_and_foreign_metadata_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'configuration'):
            load_verified(self.path,expected_config='d'*64)
        wrong = copy.deepcopy(self.ck['xr1_compact_fp32'])
        wrong['pretrained_sha256'] = 'd'*64
        with self.assertRaisesRegex(RuntimeError,'provenance'):
            validate_full_compact_resume(self.ck,self.ck['xr1_replay_contract'],expected_metadata=wrong)
        for key in ('optimizer_states','lr_schedulers','loops','xr1_replay'):
            bad = copy.deepcopy(self.ck)
            del bad[key]
            with self.subTest(missing=key),self.assertRaises(RuntimeError):
                publish_checkpoint(bad,self.root/(key+'.ckpt'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/self.path.name
            path.write_bytes(self.path.read_bytes()[:-64])
            path.with_suffix('.verified.json').write_bytes(self.path.with_suffix('.verified.json').read_bytes())
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                load_verified(path)
            path.with_suffix('.verified.json').unlink()
            with self.assertRaises(FileNotFoundError):
                load_verified(path)


@unittest.skipUnless(os.environ.get('XR1_INCIDENT_EVIDENCE'), 'Set XR1_INCIDENT_EVIDENCE to archived failure directory')
class IncidentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['XR1_INCIDENT_EVIDENCE'])
        cls.evidence = incident(root/'xr1_production_5363645_root_cause_evidence_v1.tar.gz',root/'cpu_rootcause_results.tar.gz')

    def test_actual_config_recomposes_and_run_root_is_excluded_only(self):
        from mibot.utils.claix_contract import resume_fingerprint
        conf = recompose(self.evidence,'/tmp/new-reviewed-root','/tmp/step-008750.ckpt')
        self.assertEqual(resume_fingerprint(conf,self.evidence['inventory'],12288),CONFIG)
        conf['trainer']['scheduler']['params']['num_training_steps'] += 1
        self.assertNotEqual(resume_fingerprint(conf,self.evidence['inventory'],12288),CONFIG)
        conf = recompose(self.evidence,'/tmp/new-reviewed-root')
        inv = copy.deepcopy(self.evidence['inventory'])
        inv['sha256'] = 'd'*64
        self.assertNotEqual(resume_fingerprint(conf,inv,12288),CONFIG)

    def test_reconstructed_batches_match_actual_sampler_and_frame_table(self):
        from bisect import bisect_right
        import tarfile
        from mibot.data.datasets.doosan_json_dataset import FrameSampleTable
        root = Path(os.environ['XR1_INCIDENT_EVIDENCE'])
        with tarfile.open(root/'cpu_rootcause_results.tar.gz') as archive:
            batches = json.load(archive.extractfile('./reconstructed_batches.json'))
        rows = sorted(self.evidence['inventory']['rows'],key=lambda row:row['episode'])
        rng,selected = random.Random(42),[]
        while len(selected) < HORIZON*BATCH:
            cycle = list(range(289703))
            rng.shuffle(cycle)
            selected.extend(cycle[:HORIZON*BATCH-len(selected)])
        table = FrameSampleTable([r['episode'] for r in rows],[r['frames'] for r in rows],selected)
        base = DistributedSampler(table,num_replicas=1,rank=0,shuffle=True,seed=42)
        iterator = iter(RankOffsetSampler(base,8758*24,batch_size=24))
        for step in range(8759,8767):
            for expected in batches[str(step)]:
                index = next(iterator)
                self.assertEqual(index,expected['dataset_index'])
                self.assertEqual(table[index],dict(file=expected['episode'],frame_index=expected['frame']))


if __name__ == '__main__':
    print(f'CPU_ENV torch={torch.__version__}; lightning={__import__("lightning").__version__}',flush=True)
    unittest.main(verbosity=2)
