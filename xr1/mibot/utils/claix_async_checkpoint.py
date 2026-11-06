"""Opt-in, one-snapshot CPU writer; Lightning and DDP stay on the caller thread.

An owned snapshot is immutable by convention: only the writer can reach it after
handoff, and publication only reads it. No live tensor/storage is handed off.
There is no queued second snapshot, retry, coalescing or synchronous fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from mibot.utils import claix_checkpoint as transaction
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO


def owned_cpu_snapshot(value):
    """Blocking copies, including CPU inputs/views, nested metadata and RNG arrays.

    Fail closed on opaque objects rather than trusting arbitrary deepcopy hooks.
    No pinned buffers/nonblocking CUDA copies: return means all D2H copies ended.
    Preserve OrderedDict state_dict metadata. Cyclic/custom state is unsupported.
    """
    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided:
            raise TypeError('Unsupported async checkpoint tensor')
        return value.detach().to(device='cpu', copy=True, non_blocking=False).contiguous()
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError('Object arrays cannot be staged')
        return value.copy()
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError('Object scalars cannot be staged')
        return value.copy()
    if type(value) in (dict, OrderedDict):
        result = type(value)((owned_cpu_snapshot(k), owned_cpu_snapshot(v)) for k, v in value.items())
        if isinstance(value, OrderedDict) and hasattr(value, '_metadata'):
            result._metadata = owned_cpu_snapshot(value._metadata)
        return result
    if type(value) in (list, tuple):
        return type(value)(owned_cpu_snapshot(v) for v in value)
    if value is None or type(value) in (bool, int, float, bytes):
        return value
    if isinstance(value, str):  # torch.__version__ may be a TorchVersion subclass
        return str(value)
    raise TypeError(f'Unsupported async checkpoint state: {type(value).__name__}')


class AsyncVerifiedCheckpointIO(VerifiedCheckpointIO):
    """Single-flight writer. Only the owning training thread calls public methods."""
    def __init__(self, *, expected_config=None, expected_experiment=None):
        super().__init__(expected_config=expected_config, expected_experiment=expected_experiment)
        self._owner = threading.get_ident()
        self._thread = None
        self._result = None
        self._error = None
        self._armed = None
        self._closed = False
        self.last_submitted = -1

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Async checkpoint control must stay on its training thread')

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError('Asynchronous checkpoint writer failed; no retry or fallback') from self._error

    def arm(self, reason, capture_started):
        self._check_owner()
        self._raise_error()
        if self._closed or self._thread is not None or self._armed is not None:
            raise RuntimeError('Drain the single checkpoint slot before capture')
        self._armed = (reason, capture_started)

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self._check_owner()
        self._raise_error()
        if storage_options or self._armed is None or self._closed or self._thread is not None:
            raise RuntimeError('Async save requires an armed completed-boundary capture')
        reason, capture_started = self._armed
        self._armed = None
        stage_started = time.perf_counter()
        snapshot = owned_cpu_snapshot(checkpoint)
        timings = dict(state_capture_seconds=stage_started-capture_started,
                       snapshot_staging_seconds=time.perf_counter()-stage_started)
        validation_started = time.perf_counter()
        transaction.validate_production_checkpoint(snapshot, snapshot['xr1_replay_contract'],
            expected_config=self.expected_config, expected_experiment=self.expected_experiment, resumable=False)
        timings['snapshot_validation_seconds'] = time.perf_counter()-validation_started
        step = snapshot['global_step']
        if step <= self.last_submitted or snapshot['xr1_replay_contract']['world_size'] != 1:
            raise RuntimeError('Async checkpoint requires increasing steps and one rank')
        path = Path(path)
        if path.name != f'step-{step:06d}.ckpt':
            raise RuntimeError('Async checkpoint path does not match completed step')
        # The target owns snapshot exclusively; never store it in callback state.
        # Empty the handoff box in _write: Thread retains its argument tuple until
        # target return, which otherwise keeps staging alive throughout rotation.
        self._thread = threading.Thread(target=self._write, args=([snapshot], path, reason, timings),
                                        name='xr1-checkpoint-writer', daemon=False)
        try:
            self._thread.start()
        except BaseException as error:
            self._thread = None
            self._error = error
            raise
        self.last_submitted = step

    def _write(self, handoff, path, reason, timings):
        started = time.perf_counter()
        try:
            snapshot = handoff.pop()
            receipt = transaction.publish_checkpoint(snapshot, path, timings=timings)
            # Release staging before retention's extra full reads.
            del snapshot
            pointer_started = time.perf_counter()
            if reason != 'interval':
                transaction.atomic_json(path.parent/'safe-stop.json', receipt)
            timings['safe_stop_pointer_seconds'] = time.perf_counter()-pointer_started
            rotation_started = time.perf_counter()
            rotation = transaction.rotate_checkpoints(path.parent, expected_config=self.expected_config)
            timings['rotation_verification_seconds'] = time.perf_counter()-rotation_started
            timings['background_writer_seconds'] = time.perf_counter()-started
            self._result = dict(step=receipt['global_step'], path=str(path), bytes=receipt['bytes'],
                                reason=reason, rotation=rotation, timings=timings, verified=True)
        except BaseException as error:
            # Keep the cause, without traceback frames retaining the large snapshot.
            self._error = error.with_traceback(None)

    def collect(self, *, wait=False):
        self._check_owner()
        if self._thread is not None:
            if not wait and self._thread.is_alive():
                return None
            self._thread.join()
            self._thread = None
        self._raise_error()  # sticky failure, including subsequent close/teardown
        result, self._result = self._result, None
        return result

    def close(self):
        self._check_owner()
        self._closed = True
        self._armed = None
        if self._thread is not None:
            self._thread.join()  # never release the run lifetime with a live writer
            self._thread = None
        self._raise_error()

    def teardown(self):
        self.close()


class AsyncProductionRuntime(ProductionRuntime):
    """Unchanged scheduling/science; distinct requested and committed boundaries."""
    def __init__(self, directory, *, deadline, interval=500, stop_after_step=None):
        if interval != 500:
            raise ValueError('Async experiment requires checkpoint interval 500')
        super().__init__(directory, deadline=deadline, interval=interval, stop_after_step=stop_after_step)
        self._pending_blocking_seconds = 0.

    def _io(self, trainer):
        io = trainer.strategy.checkpoint_io
        if (not isinstance(io, AsyncVerifiedCheckpointIO) or trainer.world_size != 1
                or not trainer.is_global_zero
                or (dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1)):
            raise RuntimeError('Async runtime requires its own IO and single-rank strategy')
        return io

    def _collect(self, trainer, *, wait=False):
        started = time.perf_counter()
        result = self._io(trainer).collect(wait=wait)
        waited = time.perf_counter()-started
        if result is not None:
            step = result.pop('step')
            result['timings']['completion_wait_seconds'] = waited
            result['timings']['main_thread_blocking_seconds'] = self._pending_blocking_seconds + waited
            # Report the saved cursor, never the later trainer cursor at acknowledgement.
            super().emit(SimpleNamespace(is_global_zero=True, global_step=step), 'checkpoint',
                         trainer_step_at_ack=int(trainer.global_step), **result)
            self.last_saved = step
            self._pending_blocking_seconds = 0.
        return waited

    def on_fit_start(self, trainer, pl_module):
        self._io(trainer)
        super().on_fit_start(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._collect(trainer)
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect(trainer)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def save(self, trainer, pl_module, reason):
        io = self._io(trainer)
        # Backpressure BEFORE Lightning capture or any new tensor allocation.
        backpressure = self._collect(trainer, wait=True)
        step = int(trainer.global_step)
        if self.last_saved == step:
            return
        started = time.perf_counter()
        pl_module._checkpoint_rng = [transaction.rng_state()]
        path = self.directory/f'step-{step:06d}.ckpt'
        io.arm(reason, started)
        trainer.save_checkpoint(str(path), weights_only=False)
        if io.last_submitted != step:
            raise RuntimeError('Lightning did not submit the expected async snapshot')
        trainer.strategy.barrier('staged_async_checkpoint')  # caller thread only
        self._pending_blocking_seconds = time.perf_counter()-started
        self.emit(trainer, 'checkpoint_staged', path=str(path), verified=False,
                  backpressure_seconds=backpressure,
                  capture_and_submit_seconds=self._pending_blocking_seconds)
        if reason != 'interval':
            self._collect(trainer, wait=True)

    def on_fit_end(self, trainer, pl_module):
        try:
            self._collect(trainer, wait=True)
            self._io(trainer).close()
            super().on_fit_end(trainer, pl_module)
        finally:
            self.stop.restore()

    def on_exception(self, trainer, pl_module, exception):
        try:
            self._io(trainer).close()
        finally:
            try:
                super().on_exception(trainer, pl_module, exception)
            finally:
                self.stop.restore()  # also when fatal telemetry itself cannot be written
