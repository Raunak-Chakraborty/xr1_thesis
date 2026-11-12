"""Independent fresh-XR-1-pretrained Muon/AdamW training runner.

Not compatible with historical CLAIX production checkpoint metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import types
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from mibot.models import MIMODEL
from mibot.models.runner.base_runner import BaseRunner
from mibot.models.runner.doosan_compact_runner import DoosanCompactRunner
from mibot.utils.doosan_last_n import apply_last_n_policy
from mibot.utils.doosan_fresh_optimizer import create_motion_optimizer, route_last_n
from mibot.utils.doosan_replay import checkpoint_cursor, validate_cursor


def _hash_keys(keys):
    return hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()


def _hash_config(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def install_motion_partial_state(runner, selected):
    """Persist only FP32 trainable parameters in Lightning state_dict."""
    selected = set(selected)

    actual = {
        n for n, p in runner.named_parameters()
        if p.requires_grad
    }

    if not selected or actual != selected:
        raise RuntimeError('Trainable freeze policy drift')

    all_keys = set(runner.state_dict())

    if not selected.issubset(all_keys):
        raise RuntimeError('Selected keys missing from model state')

    params = dict(runner.named_parameters())

    raw_state = runner.state_dict
    raw_load = runner.load_state_dict

    def compact_state(self, *args, **kwargs):
        full = raw_state(*args, **kwargs)

        if set(full) != all_keys:
            raise RuntimeError('Model state keys changed')

        result = {
            name: full[name]
            for name in sorted(selected)
        }

        if any(
            tensor.dtype != torch.float32
            for tensor in result.values()
        ):
            raise RuntimeError('Compact trainables are not FP32')

        return result

    def compact_load(self, state_dict, strict=True, **kwargs):
        if set(state_dict) != selected:
            raise RuntimeError('Partial checkpoint key mismatch')

        for name, tensor in state_dict.items():
            if (
                tensor.dtype != torch.float32
                or tensor.shape != params[name].shape
            ):
                raise RuntimeError(
                    'Partial checkpoint tensor mismatch: ' + name
                )

        result = raw_load(
            state_dict, strict=False, **kwargs
        )

        if (
            set(result.missing_keys) != all_keys - selected
            or result.unexpected_keys
        ):
            raise RuntimeError(
                'Partial checkpoint freeze mismatch'
            )

        return result

    runner.state_dict = types.MethodType(
        compact_state, runner
    )
    runner.load_state_dict = types.MethodType(
        compact_load, runner
    )
    runner.strict_loading = False


def cosine_factor(step, warmup, total, floor=0.25):
    """Relative multiplier, preserving different Muon/AdamW base LRs."""
    if not (0 < warmup < total and 0 < floor <= 1):
        raise ValueError('Bad LR schedule configuration')

    if step < warmup:
        return 0.1 + 0.9 * max(0, step) / warmup

    progress = min(
        1.0,
        max(0.0, (step - warmup) / (total - warmup))
    )

    return floor + (
        (1.0 - floor)
        * 0.5
        * (1 + math.cos(math.pi * progress))
    )


@MIMODEL.register_module()
class DoosanFreshMotionRunner(DoosanCompactRunner):
    """Train task adapters from the released base with fresh optimizer state."""

    def __init__(self, params):
        super().__init__(params)

        cfg = params.get('motion_experiment')

        if not isinstance(cfg, dict):
            raise RuntimeError(
                'Missing motion_experiment configuration'
            )

        self._motion = dict(cfg)

        if (
            self._last_n not in (1, 2)
            or self._motion.get('initialization') != 'released_pretrained_only'
        ):
            raise RuntimeError('Unsupported source model/step')

        for key in (
            'pretrained_sha256',
            'language_sha256',
            'schedule_sha256',
            'trial_id',
            'optimizer_kind',
        ):
            if not self._motion.get(key):
                raise RuntimeError(
                    'Missing motion experiment field: ' + key
                )

        if self._motion['optimizer_kind'] not in (
            'adamw_control', 'hybrid_muon'
        ):
            raise RuntimeError('Invalid optimizer kind')

        if (
            self._replay_contract['data_horizon'] != 2000
            or self._replay_contract['batch_size'] != 24
        ):
            raise RuntimeError(
                'Unexpected sampling horizon/batch'
            )

        self._motion_digest = _hash_config(self._motion)
        self._motion_meta = None
        self._route_report = None

    def configure_model(self):
        # Load the immutable original XR-1 model first.
        BaseRunner.configure_model(self)

        report = apply_last_n_policy(
            self.model, self._last_n
        )

        named = dict(self.named_parameters())

        selected = {
            'model.' + name
            for name in report['trainable_names']
        }

        with torch.no_grad():
            for name in selected:
                param = named[name]

                if param.dtype not in (
                    torch.bfloat16, torch.float32
                ):
                    raise RuntimeError(
                        'Unexpected trainable dtype'
                    )

                param.data = param.detach().to(
                    dtype=torch.float32
                ).contiguous()

        if any(
            p.dtype != torch.bfloat16
            for p in self.parameters()
            if not p.requires_grad
        ):
            raise RuntimeError(
                'Frozen model must stay BF16'
            )

        # BaseRunner.configure_model above loads the released pretrained XR-1
        # weights. No 48,129-step trainable tensor restoration is permitted.
        provenance = {
            'initialization': 'released_pretrained_only',
            'pretrained_sha256': self._compact_cfg['pretrained_sha256'],
            'language_sha256': self._compact_cfg['language_sha256'],
            'last_n': self._last_n,
            'trainable_tensor_count': len(selected),
            'trainable_parameter_count': report['trainable_parameters'],
            'selected_keys_sha256': _hash_keys(selected),
        }

        _, routing = route_last_n(
            self.named_parameters(), last_n=self._last_n
        )

        if (
            sum(routing['element_counts'].values())
            != report['trainable_parameters']
        ):
            raise RuntimeError(
                'Routing differs from freeze policy'
            )

        self._route_report = routing

        self._motion_meta = {
            'schema': 'xr1-motion-compact-lightning-v1',
            'experiment_config_sha256': self._motion_digest,
            'routing_sha256': routing['sha256'],
            'initialization': provenance,
            'selected_keys_sha256': _hash_keys(selected),
            'optimizer_kind': self._motion['optimizer_kind'],
        }

        install_motion_partial_state(
            self, selected
        )

        print(
            'XR1_FRESH_PRETRAINED_INIT=PASS '
            'pretrained_only=YES '
            'tensors=' + str(len(selected)),
            flush=True,
        )

    def configure_optimizers(self):
        if self._route_report is None:
            raise RuntimeError(
                'Configure model before optimizer'
            )

        optimizer, routing = create_motion_optimizer(
            self.named_parameters(),
            last_n=self._last_n,
            kind=self._motion['optimizer_kind'],
            adamw_lr=float(self._motion['adamw_lr']),
            muon_lr=(
                float(self._motion['muon_lr'])
                if self._motion['optimizer_kind'] == 'hybrid_muon'
                else None
            ),
        )

        if routing['sha256'] != self._route_report['sha256']:
            raise RuntimeError(
                'Optimizer parameter routing changed'
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_factor(
                step,
                int(self._motion['warmup_steps']),
                int(self._motion['schedule_total_steps']),
            ),
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

    def on_save_checkpoint(self, checkpoint):
        step = int(self.global_step)

        if not 1 <= step <= self._replay_contract['data_horizon']:
            raise RuntimeError(
                'Checkpoint outside approved step horizon'
            )

        if (
            not checkpoint.get('optimizer_states')
            or not checkpoint.get('lr_schedulers')
        ):
            raise RuntimeError(
                'Optimizer/scheduler state missing'
            )

        if not checkpoint['optimizer_states'][0].get('state'):
            raise RuntimeError(
                'Empty optimizer state in checkpoint'
            )

        checkpoint['xr1_motion_compact'] = dict(
            self._motion_meta
        )

        checkpoint['xr1_motion_replay_contract'] = dict(
            self._replay_contract
        )

        checkpoint['xr1_motion_replay'] = checkpoint_cursor(
            self._replay_contract, step
        )

        if int(checkpoint.get('global_step', -1)) != step:
            raise RuntimeError(
                'Step inconsistent with Lightning checkpoint'
            )

    def on_load_checkpoint(self, checkpoint):
        if (
            checkpoint.get('xr1_motion_compact')
            != self._motion_meta
        ):
            raise RuntimeError(
                'Experimental checkpoint provenance mismatch'
            )

        if (
            checkpoint.get('xr1_motion_replay_contract')
            != self._replay_contract
        ):
            raise RuntimeError(
                'Sampler replay contract mismatch'
            )

        step = int(checkpoint['global_step'])

        if step != self._resume_step:
            raise RuntimeError(
                'Resume must configure dataloader offset first'
            )

        validate_cursor(
            checkpoint['xr1_motion_replay'],
            self._replay_contract,
            checkpoint_step=step,
        )

        if (
            not checkpoint.get('optimizer_states')
            or not checkpoint.get('lr_schedulers')
        ):
            raise RuntimeError(
                'Experimental optimizer/scheduler state missing'
            )
