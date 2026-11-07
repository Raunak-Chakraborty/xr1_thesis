"""Required real Hydra/Lightning CPU acceptance; excluded from the local stub suite.

Uses a tiny model with the real 24/36/36 stack layout and runner hooks. Never
loads XR1 assets. The scientific horizon stays 48284; safe-stop bounds CPU work.
"""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lightning import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy
from mmengine import Config

from mibot.data.datamodule.doosan_datamodule import DoosanDataModule, shutdown_loader
from mibot.models.runner.doosan_compact_runner import DoosanCompactRunner
from mibot.utils.claix_checkpoint import load_verified, SERIOUS_LAST2_MILESTONES
from mibot.utils.claix_contract import BASELINE
from mibot.utils.claix_last2_serious import CONFIG, NAME, PRETRAINED_SHA256, LANGUAGE_SHA256, validate_science
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO
from mibot.utils.doosan_compact_fp32 import install_compact_fp32
from mibot.utils.doosan_last_n import apply_last_n_policy
from test_claix_last2_serious import identity
from test_claix_replay import AugmentedFixture
from test_claix_runtime import tiny_collate


def tensor_collate(items):
    # Actual CPU tensor IPC on CLAIX, rather than the laptop NumPy workaround.
    return {key: torch.from_numpy(value) for key, value in tiny_collate(items).items()}


class TinyLastTwoNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Module(); self.vlm.model = torch.nn.Module()
        self.vlm.model.visual = torch.nn.Module(); self.vlm.model.language_model = torch.nn.Module()
        self.vlm.model.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(24)])
        self.vlm.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(36)])
        self.dit = torch.nn.Module()
        self.dit.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(36)])
        for key in ('state_projector', 'action_projector', 'action_output_layer', 't_embedder', 't_projector', 'sink'):
            setattr(self, key, torch.nn.Linear(1, 1))
        with torch.no_grad():
            for parameter in self.parameters(): parameter.fill_(.1)
        self.bfloat16()

    def forward(self, batch, return_loss):
        total = sum(parameter.float().sum() for parameter in self.parameters() if parameter.requires_grad)
        x = batch['x'][:, 0] / 100000
        return dict(loss=((total * x + torch.rand(()) / 10) - .1).square().mean())


class TinyLastTwoRunner(DoosanCompactRunner):
    def configure_model(self):
        # Substitute only tiny backbone construction, not save/load/replay hooks.
        self.model = TinyLastTwoNetwork()
        selected = apply_last_n_policy(self.model, 2)
        self._compact_meta = install_compact_fp32(self,
            selected_keys={'model.' + name for name in selected['trainable_names']},
            pretrained_sha256=PRETRAINED_SHA256, language_sha256=LANGUAGE_SHA256,
            source_commit=BASELINE, last_n=2)

    def training_step(self, batch, batch_idx):
        if not hasattr(self, 'seen'): self.seen = []
        self.seen.append(batch['x'][:, 0].tolist())
        return super().training_step(batch, batch_idx)


def run(root, *, stop, resume=None):
    from mibot.utils.claix_last2_serious import recipe
    cfg = recipe()
    step = int(resume.stem[5:]) if resume else 0
    params = Config(dict(max_steps=48284, train_datasets=dict(batch_size=24, num_workers=4,
        prefetch_factor=2, persistent_workers=True, in_order=True, pin_memory=False,
        deterministic_replay=True, sampler_seed=42, resume_step=step)))
    with patch('mibot.data.datamodule.base_datamodule.CustomCollate'):
        data = DoosanDataModule(params)
    data.collate_fn = tensor_collate
    data.train_set = AugmentedFixture(length=48284 * 24)
    experiment = identity(root)
    model = TinyLastTwoRunner(dict(compact_fp32=dict(last_n=2, data_horizon=48284,
        batch_size=24, world_size=1, sampler_seed=42, num_workers=4, resume_step=step,
        config_sha256='a' * 64, experiment=experiment),
        optimizer=copy.deepcopy(cfg['trainer']['optimizer']), scheduler=copy.deepcopy(cfg['trainer']['scheduler'])))
    runtime = ProductionRuntime(root, deadline=time.time() + 600, interval=500, stop_after_step=stop,
                                retention_milestones=SERIOUS_LAST2_MILESTONES)
    trainer = Trainer(accelerator='cpu', devices=1, max_steps=48284, max_epochs=1,
        strategy=SingleDeviceStrategy(device='cpu', checkpoint_io=VerifiedCheckpointIO(
            expected_config='a' * 64, expected_experiment=experiment)),
        callbacks=[runtime], logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, use_distributed_sampler=False, num_sanity_val_steps=0)
    root.mkdir(parents=True, exist_ok=True)
    try:
        trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)
    finally:
        loader = trainer.train_dataloader
        if loader is not None: shutdown_loader(loader, getattr(loader, '_iterator', None))
    return model, trainer


class SeriousFrameworkAcceptanceTest(unittest.TestCase):
    def test_real_hydra_composition(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
            cfg = OmegaConf.to_container(compose(config_name=CONFIG), resolve=True)
        validate_science(cfg, cfg['production'])
        self.assertEqual(cfg['trainer']['exp_name'], NAME)

    def test_real_lightning_last_two_same_run_four_worker_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            'mibot.data.datamodule.doosan_datamodule.dist',
            SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                            get_world_size=lambda: 1, get_rank=lambda: 0)):
            root = Path(folder)
            continuous, _ = run(root / 'continuous', stop=6)
            first, _ = run(root / 'split', stop=3)
            resumed, trainer = run(root / 'split', stop=6, resume=root / 'split/step-000003.ckpt')
            self.assertEqual(trainer.global_step, 6)
            self.assertEqual(first.seen + resumed.seen, continuous.seen)
            expected = load_verified(root / 'continuous/step-000006.ckpt')
            actual = load_verified(root / 'split/step-000006.ckpt')
            self.assertEqual(actual['xr1_replay']['rank_sample_offset'], 6 * 24)
            self.assertEqual(actual['lr_schedulers'], expected['lr_schedulers'])
            for name in expected['state_dict']:
                self.assertTrue(torch.equal(expected['state_dict'][name], actual['state_dict'][name]))
            for index, state in expected['optimizer_states'][0]['state'].items():
                for name in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertTrue(torch.equal(state[name], actual['optimizer_states'][0]['state'][index][name]))
            events = [json.loads(row) for row in (root / 'split/telemetry.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in events if row['event'] == 'step'], list(range(1, 7)))
            self.assertTrue(events[-1]['final_checkpoint_verified'])
