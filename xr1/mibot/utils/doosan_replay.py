"""Fail-closed single-epoch deterministic DDP replay for Doosan XR1.

Supported production contract: constant world size and batch size, no accumulation,
ordered dataloader workers, no epoch wrap, fixed data horizon and sampler seed.
Prefetch never mutates the completed-step checkpoint cursor.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

SCHEMA = 'xr1-doosan-fixed-horizon-replay-v2'


def make_contract(*, data_horizon: int, batch_size: int, world_size: int,
                  sampler_seed: int, accumulate_grad_batches: int = 1,
                  num_workers: int = 0) -> dict[str, Any]:
    values = dict(data_horizon=data_horizon, batch_size=batch_size,
                  world_size=world_size, sampler_seed=sampler_seed,
                  accumulate_grad_batches=accumulate_grad_batches,
                  num_workers=num_workers)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values.values()):
        raise ValueError('Replay contract fields must be integers')
    if min(data_horizon, batch_size, world_size) < 1 or num_workers not in (0, 4, 8):
        raise ValueError('Replay needs positive horizon/batch/world and workers in (0,4,8)')
    if accumulate_grad_batches != 1:
        raise ValueError('Gradient accumulation not yet supported for deterministic replay')
    if sampler_seed < 0:
        raise ValueError('Invalid negative sampler seed')
    return {'schema': SCHEMA, **values}


def checkpoint_cursor(contract: dict[str, Any], step: int) -> dict[str, Any]:
    if contract.get('schema') != SCHEMA:
        raise ValueError('Unrecognized replay schema')
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step <= contract['data_horizon']:
        raise ValueError('Invalid checkpoint global step for fixed data horizon')
    return {**contract, 'global_step': step, 'rank_sample_offset': step * contract['batch_size']}


def validate_cursor(cursor: dict[str, Any], contract: dict[str, Any], *,
                    checkpoint_step: int, resumable: bool = True) -> int:
    if not isinstance(cursor, dict):
        raise RuntimeError('Missing resume replay cursor')
    expected = checkpoint_cursor(contract, checkpoint_step)
    if cursor != expected:
        raise RuntimeError('Replay contract mismatch: world size, seed, horizon, batch size or step')
    if resumable and checkpoint_step >= contract['data_horizon']:
        raise RuntimeError('Cannot resume beyond fixed data horizon')
    return expected['rank_sample_offset']


class RankOffsetSampler(Sampler[int]):
    """Advance each rank's original DistributedSampler by its consumed samples.

    `use_distributed_sampler=False` must be used in Lightning: otherwise it
    may replace this sampler and silently start from offset zero.
    """
    def __init__(self, base: DistributedSampler, offset: int, *, batch_size: int = 1):
        if not isinstance(base, DistributedSampler):
            raise TypeError('Replay needs a PyTorch DistributedSampler')
        if not isinstance(offset, int) or not 0 <= offset < len(base):
            raise ValueError('Resume offset is outside the rank-local sampler stream')
        if batch_size < 1 or offset % batch_size or len(base) % batch_size:
            raise ValueError('Replay requires whole completed batches, including at resume')
        if (not base.shuffle or base.drop_last or base.epoch != 0
                or len(base.dataset) % base.num_replicas):
            raise ValueError('Replay requires shuffled epoch zero without padding or dropping')
        self.base = base
        self.offset = offset
        self._epoch = 0

    def __iter__(self) -> Iterable[int]:
        if self._epoch != 0:
            raise RuntimeError('Epoch wrap not supported: increase the fixed data horizon')
        return itertools.islice(iter(self.base), self.offset, None)

    def __len__(self) -> int:
        return len(self.base) - self.offset

    def set_epoch(self, epoch: int) -> None:
        if epoch != 0:
            raise RuntimeError('Multi-epoch sampler replay is not yet implemented')
        self._epoch = epoch
        self.base.set_epoch(epoch)


def stable_sample_seed(path: str, frame: int, root: str, base_seed: int) -> int:
    from pathlib import Path
    import os
    p = os.path.relpath(str(Path(path).resolve()), str(Path(root).resolve()))
    if p == os.pardir or p.startswith(os.pardir + os.sep):
        raise RuntimeError('Dataset path outside expected training root')
    ident = json.dumps([int(base_seed), p, int(frame)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(ident.encode('utf-8')).digest()[:8], 'big')


def seed_worker(worker_id: int) -> None:
    """Fallback RNGs; sample augmentation has its own stable path/frame seed."""
    import random
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def validate_full_compact_resume(ck: dict[str, Any], contract: dict[str, Any],
                                 *, expected_metadata: dict[str, Any] | None = None,
                                 resumable: bool = True) -> int:
    """Require weights + AdamW moments + scheduler + Lightning loops + replay.

    Must be called on a trusted locally generated Lightning checkpoint. Loading
    a `.ckpt` with pickle is not safe for arbitrary untrusted inputs.
    """
    needed = ('xr1_compact_fp32', 'xr1_replay', 'state_dict',
              'optimizer_states', 'lr_schedulers', 'loops', 'global_step')
    for key in needed:
        if key not in ck:
            raise RuntimeError('Incomplete compact-resume checkpoint: ' + key)
    meta = ck['xr1_compact_fp32']
    if not isinstance(meta, dict) or meta.get('schema') != 'xr1-doosan-fp32-trainable-compact-lightning-v1':
        raise RuntimeError('Wrong compact checkpoint format')
    if expected_metadata is not None and meta != expected_metadata:
        raise RuntimeError('Compact pretrained/language/selection provenance mismatch')
    if not isinstance(ck['global_step'], int) or isinstance(ck['global_step'], bool):
        raise RuntimeError('Checkpoint step must be an integer')
    step = ck['global_step']
    offset = validate_cursor(ck['xr1_replay'], contract, checkpoint_step=step, resumable=resumable)
    trainables = ck['state_dict']
    if not isinstance(trainables, dict) or len(trainables) != meta['selected_tensor_count']:
        raise RuntimeError('Missing or extra trainable tensor in compact checkpoint')
    dig = hashlib.sha256('\n'.join(sorted(trainables)).encode()).hexdigest()
    if dig != meta['selected_key_sha256']:
        raise RuntimeError('Selected model keys differ from compact manifest')
    for key, value in trainables.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise RuntimeError('Invalid FP32 master weight in ' + key)
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite master weight in ' + key)
    if sum(v.numel() for v in trainables.values()) != meta['trainable_parameter_count']:
        raise RuntimeError('Compact trainable element count mismatch')
    if not ck['optimizer_states'] or len(ck['optimizer_states']) != 1:
        raise RuntimeError('Missing or unexpected optimizer states')
    state = ck['optimizer_states'][0].get('state')
    if not state or len(state) != meta['selected_tensor_count']:
        raise RuntimeError('Missing AdamW state entries')
    groups = ck['optimizer_states'][0].get('param_groups', [])
    ids = [key for group in groups for key in group.get('params', [])]
    if len(ids) != len(set(ids)) or set(ids) != set(state):
        raise RuntimeError('AdamW parameter groups do not match moment states')
    for param_state in state.values():
        for name in ('exp_avg', 'exp_avg_sq'):
            value = param_state.get(name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise RuntimeError('Missing FP32 AdamW moment ' + name)
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite AdamW moment ' + name)
        if param_state['exp_avg'].shape != param_state['exp_avg_sq'].shape:
            raise RuntimeError('AdamW moment shape mismatch')
        t = param_state.get('step', None)
        if t is None or (t.item() if hasattr(t, 'item') else t) != step:
            raise RuntimeError('AdamW step counter does not match global_step')
    schedulers = ck['lr_schedulers']
    if not schedulers or len(schedulers) != 1:
        raise RuntimeError('Missing scheduler state')
    if schedulers[0].get('last_epoch', -1) != step:
        raise RuntimeError('Scheduler step does not match global_step')
    if not isinstance(ck['loops'], dict) or not ck['loops']:
        raise RuntimeError('Missing Lightning loop progress')
    return offset
