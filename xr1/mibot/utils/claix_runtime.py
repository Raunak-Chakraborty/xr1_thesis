"""Step-boundary stopping, persistent telemetry and verified compact publication."""
from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from lightning.fabric.plugins.io import CheckpointIO

from mibot.utils.claix_checkpoint import (
    publish_checkpoint, rotate_checkpoints, rng_state, restore_rng,
)
from mibot.utils.doosan_replay import RankOffsetSampler


class StopRequest:
    """Handlers only set flags. Deadline is an absolute allocation-derived epoch."""
    def __init__(self, deadline):
        self.deadline = float(deadline)
        self.reason = None
        self.handlers = {}

    def handle(self, signum, frame):
        self.reason = f'signal:{signum}'

    def due(self, now=None):
        request_file = os.environ.get('XR1_STOP_REQUEST_FILE')
        if request_file and Path(request_file).is_file():
            self.reason = 'launcher_signal'
        return self.reason is not None or (time.time() if now is None else now) >= self.deadline

    def install(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.handlers[sig] = signal.signal(sig, self.handle)

    def restore(self):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        self.handlers.clear()


class VerifiedCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options:
            raise RuntimeError('Only local POSIX checkpoint storage is supported')
        publish_checkpoint(checkpoint, path)

    def load_checkpoint(self, path, map_location=None):
        # Entrypoint and module validate full receipt/config before this load.
        from mibot.utils.claix_checkpoint import load_verified
        ck = torch.load(path, map_location='cpu', weights_only=False)
        return load_verified(path, ck['xr1_replay_contract'])

    def remove_checkpoint(self, path):
        raise RuntimeError('Use explicit verified checkpoint retention policy')


class ProductionRuntime(Callback):
    def __init__(self, directory, *, deadline, interval=250, stop_after_step=None):
        self.directory = Path(directory)
        self.stop = StopRequest(deadline)
        self.interval = int(interval)
        self.stop_after_step = stop_after_step
        if self.interval < 1:
            raise ValueError('Checkpoint interval must be positive')
        self.last_saved = -1

    def emit(self, trainer, event, **fields):
        if not trainer.is_global_zero:
            return
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), event=event,
                   step=int(trainer.global_step), **fields)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'telemetry.jsonl').open('a', buffering=1) as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        print('CLAIX ' + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    def on_fit_start(self, trainer, pl_module):
        self.fit_started = time.perf_counter()
        self.stop.install()

    def on_train_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        sampler = loader.sampler
        contract = pl_module._replay_contract
        if (not isinstance(sampler, RankOffsetSampler)
                or sampler.offset != trainer.global_step * contract['batch_size']
                or loader.batch_size != contract['batch_size']
                or loader.num_workers != contract['num_workers']
                or sampler.base.seed != contract['sampler_seed']
                or sampler.base.num_replicas != contract['world_size']
                or sampler.base.rank != trainer.global_rank
                or trainer.accumulate_grad_batches != 1
                or trainer.current_epoch != 0):
            raise RuntimeError('Lightning/DataLoader replay sampler contract drift')
        pending = getattr(pl_module, '_pending_rng', None)
        if pending is not None:
            restore_rng(pending[trainer.global_rank])
            pl_module._pending_rng = None
        # Include DataLoader creation and worker startup, not just GPU stage time.
        self.started = self.previous_end = self.fit_started
        self.initial_step = self.previous_step = int(trainer.global_step)
        self.emit(trainer, 'train_start', deadline_epoch=self.stop.deadline,
                  workers=loader.num_workers, batch_size=loader.batch_size)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_started = time.perf_counter()
        self.data_pipeline_seconds = self.batch_started - self.previous_end
        self.retained = int(batch['action_segments'].shape[0])
        self.tokens = int(batch['input_ids'].shape[1])
        self.views = int(batch['image_grid_thw'].shape[0])
        if self.retained != pl_module._replay_contract['batch_size'] or self.views != self.retained * 2:
            raise RuntimeError('Incomplete training batch reached optimizer pipeline')
        if self.tokens > int(os.environ.get('MAX_LENGTH', 0)) or self.tokens > self.retained * 413:
            raise RuntimeError('Packed token certification exceeded')

    def save(self, trainer, pl_module, reason):
        step = int(trainer.global_step)
        if self.last_saved == step:
            return
        local = rng_state()
        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, local)
        else:
            states = [local]
        pl_module._checkpoint_rng = states
        path = self.directory / f'step-{step:06d}.ckpt'
        start = time.perf_counter()
        # All ranks participate; the strategy writes only on rank zero.
        trainer.save_checkpoint(str(path), weights_only=False)
        trainer.strategy.barrier('verified_checkpoint')
        if trainer.is_global_zero:
            from mibot.utils.claix_checkpoint import load_verified
            load_verified(path, pl_module._replay_contract, resumable=False)
            self.emit(trainer, 'checkpoint', path=str(path), bytes=path.stat().st_size,
                      duration_seconds=time.perf_counter() - start, verified=True, reason=reason)
            rotate_checkpoints(self.directory)
        self.last_saved = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step != self.previous_step + 1:
            raise RuntimeError('Optimizer step skipped or repeated')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = now - self.started
        rate = (step - self.initial_step) * self.retained * trainer.world_size / elapsed
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        fields = dict(samples_consumed=step * self.retained * trainer.world_size,
                      retained_examples=self.retained, views=self.views, packed_tokens=self.tokens,
                      pipeline_wait_seconds=self.data_pipeline_seconds,
                      # Includes loader wait, transfers, callbacks and earlier checkpoint I/O.
                      end_to_end_examples_per_second=rate,
                      loss=float(loss.detach()) if loss is not None else None,
                      learning_rate=trainer.optimizers[0].param_groups[0]['lr'],
                      remaining_seconds=(trainer.max_steps-step)*self.retained*trainer.world_size/rate,
                      safe_stop_in_seconds=max(0., self.stop.deadline-time.time()))
        try:
            import psutil
            parent = psutil.Process()
            fields['cpu_rss_bytes'] = parent.memory_info().rss
            fields['worker_rss_bytes'] = sum(p.memory_info().rss for p in parent.children(recursive=True))
        except (ImportError, OSError):
            fields['cpu_rss_bytes'] = fields['worker_rss_bytes'] = None
        fields.update(gpu_utilization_percent=None, gpu_allocated_bytes=None,
                      gpu_reserved_bytes=None, gpu_peak_allocated_bytes=None)
        if torch.cuda.is_available():
            fields.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                          gpu_reserved_bytes=torch.cuda.memory_reserved(),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            try:
                fields['gpu_utilization_percent'] = torch.cuda.utilization()
            except Exception as error:
                fields['gpu_utilization_error'] = type(error).__name__
        self.emit(trainer, 'step', **fields)
        due = self.stop.due() or (self.stop_after_step is not None and step >= self.stop_after_step)
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor(int(due), device=pl_module.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            due = bool(flag.item())
        reason = self.stop.reason or ('qualification_step' if self.stop_after_step is not None and step >= self.stop_after_step
                                     else 'safe_deadline' if due else 'natural_finish' if step >= trainer.max_steps else 'interval')
        if due or step % self.interval == 0 or step == 2000 or step >= trainer.max_steps:
            self.save(trainer, pl_module, reason)
        if due:
            trainer.should_stop = True
            self.emit(trainer, 'graceful_stop', reason=reason)
        # Keep I/O inside the next interval's pipeline overhead and total throughput.
        self.previous_end = now
        self.previous_step = step

    def on_exception(self, trainer, pl_module, exception):
        self.emit(trainer, 'fatal', error_type=type(exception).__name__, error=str(exception))
        self.stop.restore()

    def on_fit_end(self, trainer, pl_module):
        self.emit(trainer, 'fit_end')
        self.stop.restore()
