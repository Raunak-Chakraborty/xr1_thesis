"""Opt-in Doosan last-N FP32 master-weight and compact-resume Lightning runner."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from mibot.models import MIMODEL
from mibot.models.runner.base_runner import BaseRunner
from mibot.utils.doosan_compact_fp32 import install_compact_fp32
from mibot.utils.doosan_last_n import apply_last_n_policy
from mibot.utils.doosan_replay import (
    make_contract, checkpoint_cursor, validate_cursor, validate_full_compact_resume,
)


@MIMODEL.register_module()
class DoosanCompactRunner(BaseRunner):
    def __init__(self, params):
        super().__init__(params)
        cfg = params.get('compact_fp32')
        if not isinstance(cfg, dict):
            raise ValueError('DoosanCompactRunner requires compact_fp32 config')
        self._compact_cfg = dict(cfg)
        self._last_n = int(cfg['last_n'])
        self._replay_contract = make_contract(
            data_horizon=int(cfg['data_horizon']), batch_size=int(cfg['batch_size']),
            world_size=int(cfg['world_size']), sampler_seed=int(cfg['sampler_seed']),
            accumulate_grad_batches=int(cfg.get('accumulate_grad_batches', 1)),
            num_workers=int(cfg.get('num_workers', 0)),
        )
        self._resume_step = int(cfg.get('resume_step', 0))
        if self._resume_step:
            validate_cursor(checkpoint_cursor(self._replay_contract, self._resume_step),
                            self._replay_contract, checkpoint_step=self._resume_step)

    def configure_model(self) -> None:
        # Must load immutable pretrained BF16 model before promoting trainables.
        super().configure_model()
        selected_report = apply_last_n_policy(self.model, self._last_n)
        selected = {'model.' + name for name in selected_report['trainable_names']}
        self._compact_meta = install_compact_fp32(
            self, selected_keys=selected,
            pretrained_sha256=self._compact_cfg['pretrained_sha256'],
            language_sha256=self._compact_cfg['language_sha256'],
            source_commit=self._compact_cfg['source_commit'],
            last_n=self._last_n,
        )
        if self._compact_meta['trainable_parameter_count'] != selected_report['trainable_parameters']:
            raise RuntimeError('Selected last-N trainable parameter count drift')
        if any(p.dtype != torch.bfloat16 for p in self.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen weights must remain in BF16')
        if any(p.dtype != torch.float32 for p in self.parameters() if p.requires_grad):
            raise RuntimeError('Selected weights must be FP32')
        print('XR1_PRODUCTION_FP32=PASS last_n=' + str(self._last_n) +
              ' tensors=' + str(len(selected)) +
              ' elements=' + str(selected_report['trainable_parameters']), flush=True)

    def training_step(self, batch, batch_idx):
        # Same deterministic per-(optimizer step, rank) policy as Stage 11.
        step = int(self.global_step)
        if step < self._resume_step or step >= self._replay_contract['data_horizon']:
            raise RuntimeError('Global step violates compact replay horizon')
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        if not 0 <= rank < self._replay_contract['world_size']:
            raise RuntimeError('Rank inconsistent with checkpoint replay contract')
        seed = 900100 + 997 * step + rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        return super().training_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        # install_compact_fp32 wraps and invokes this method, adding masters and
        # optimizer validation. The replay cursor is reconstructed from the
        # *completed optimizer step*, not an in-flight DataLoader iterator.
        step = int(self.global_step)
        if step <= 0:
            raise RuntimeError('Cannot save optimizer-resumable checkpoint before step 1')
        checkpoint['xr1_replay'] = checkpoint_cursor(self._replay_contract, step)
        if int(checkpoint.get('global_step', step)) != step:
            raise RuntimeError('Lightning checkpoint step inconsistent with sampler cursor')
        if not checkpoint.get('lr_schedulers') or not checkpoint.get('optimizer_states'):
            raise RuntimeError('Cannot save resume checkpoint without optimizer/scheduler')
        for opt in checkpoint['optimizer_states']:
            if not opt.get('state'):
                raise RuntimeError('AdamW state is empty')
        print(f'XR1_COMPACT_REPLAY_SAVED=PASS global_step={step} '
              f'offset={checkpoint["xr1_replay"]["rank_sample_offset"]}', flush=True)

    def on_load_checkpoint(self, checkpoint):
        validate_full_compact_resume(checkpoint, self._replay_contract,
                                     expected_metadata=self._compact_meta)
        step = int(checkpoint['global_step'])
        if step != self._resume_step:
            raise RuntimeError('Dataloader resume offset was not preconfigured from checkpoint')
        print('XR1_COMPACT_REPLAY_RESTORED=PASS global_step=' + str(step), flush=True)
