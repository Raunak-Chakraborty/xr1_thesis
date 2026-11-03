"""Opt-in DataLoader diagnostics. Private APIs require exact-version qualification.

No retries, sample filtering, transport substitution, or ordering changes. Queue
enqueue, serialization, pipe lock/send, receipt and ordered delivery are distinct.
"""
from __future__ import annotations

import faulthandler
import functools
import inspect
import multiprocessing as mp
import multiprocessing.queues as queues
import os
from pathlib import Path
import signal
import socket
import threading
import time
import uuid

import torch
from torch.utils.data import DataLoader
from torch.utils.data import dataloader as dl
from torch.utils.data._utils import worker, pin_memory

from mibot.utils.loader_trace import TraceBuffer, Writer, bind, identify, mark, stage
from mibot.utils.loader_watchdog import identity, watchdog

_ORIGINAL_WORKER = worker._worker_loop
_ORIGINAL_PIN = pin_memory._pin_memory_loop
_startup_lock = threading.Lock()
_active_session = None


def _register(shared, index, stream):
    current = signal.getsignal(signal.SIGUSR2)
    if current not in (signal.SIG_DFL, signal.SIG_IGN):
        raise RuntimeError('Diagnostics require exclusive SIGUSR2; handler already installed')
    stream.setblocking(False)
    faulthandler.register(signal.SIGUSR2, file=stream.fileno(), all_threads=True, chain=False)
    pid, ticks = identity(os.getpid())
    shared.registry[index * 3:index * 3 + 3] = (pid, ticks, 1)


class WorkerInit:
    def __init__(self, original, shared, writers):
        self.original, self.shared, self.writers = original, shared, writers

    def __call__(self, worker_id):
        # Registration happens before Torch's native thread/RNG initialization.
        with stage('worker_init_fn'):
            if self.original is not None:
                self.original(worker_id)


class IndexQueue:
    def __init__(self, queue):
        self.queue = queue

    def get(self, *args, **kwargs):
        with stage('worker_index_wait'):
            value = self.queue.get(*args, **kwargs)
        if isinstance(value, tuple) and isinstance(value[0], int):
            identify(task_sequence=value[0], dataset_index=None, episode=None, frame=None)
            mark('task_received', indices=list(value[1])[:128] if isinstance(value[1], list) else value[1])
        return value

    def __getattr__(self, name):
        return getattr(self.queue, name)


class ResultQueue:
    def __init__(self, queue):
        self.queue = queue

    def put(self, value, *args, **kwargs):
        with stage('worker_result_enqueue'):
            result = self.queue.put(value, *args, **kwargs)
        mark('result_enqueued')  # NOT evidence that the feeder has sent anything
        return result

    def __getattr__(self, name):
        return getattr(self.queue, name)


def _instrument_feeder(queue, shared, worker_id):
    """Use original CPython Queue._feed; observe without copying its algorithm."""
    feeder = Writer(shared, 3 + 2 * worker_id, worker_id=worker_id, role='feeder')
    original_dumps = queues._ForkingPickler.dumps
    original_send, original_error = queue._send_bytes, queue._on_queue_feeder_error
    original_lock = queue._wlock

    def dumps(value, protocol=None):
        # Other reducer/resource-sharer activity must not write this lane.
        if threading.current_thread().name != 'QueueFeederThread':
            return original_dumps(value, protocol)
        bind(feeder)
        identify(task_sequence=value[0] if isinstance(value, tuple) and isinstance(value[0], int) else None)
        with stage('ipc_serialize'):
            result = original_dumps(value, protocol)
        feeder.current = 'ipc_pipe_lock'
        feeder.emit('begin')
        return result

    class PipeLock:
        def acquire(self, *args, **kwargs):
            return original_lock.acquire(*args, **kwargs)

        def release(self):
            return original_lock.release()

    def send(payload):
        bind(feeder)
        with stage('ipc_pipe_send'):
            result = original_send(payload)
        feeder.current = 'feeder_idle'
        feeder.emit('ipc_sent')
        return result

    def error(exception, value):
        bind(feeder)
        feeder.current = 'ipc_error'
        feeder.emit('error', error_type=type(exception).__name__)
        # Original hook prints to stderr and can itself block; record first.
        return original_error(exception, value)

    # Process-local patches, performed before the result queue starts its thread.
    queues._ForkingPickler.dumps = staticmethod(dumps)
    queue._send_bytes, queue._on_queue_feeder_error = send, error
    queue._wlock = PipeLock()


def observed_worker_loop(*args):
    names = tuple(inspect.signature(_ORIGINAL_WORKER).parameters)
    values = dict(zip(names, args))
    init = values['init_fn']
    if not isinstance(init, WorkerInit):
        return _ORIGINAL_WORKER(*args)
    worker_id = values['worker_id']
    bind(Writer(init.shared, 2 + 2 * worker_id, worker_id=worker_id, role='worker'))
    _register(init.shared, worker_id + 1, init.writers[worker_id + 1])
    _instrument_feeder(values['data_queue'], init.shared, worker_id)
    values['index_queue'] = IndexQueue(values['index_queue'])
    values['data_queue'] = ResultQueue(values['data_queue'])
    with stage('worker_bootstrap'):
        return _ORIGINAL_WORKER(*(values[name] for name in names))


def observed_pin_loop(shared, in_queue, out_queue, *args):
    writer = Writer(shared, 1, role='pin_memory')
    bind(writer)

    class Input:
        def get(self, *a, **kw):
            with stage('pin_ipc_receive'):
                value = in_queue.get(*a, **kw)
            identify(task_sequence=value[0] if isinstance(value[0], int) else None)
            writer.current = 'pin_memory'
            mark('begin')
            return value

    class Output:
        def put(self, value, *a, **kw):
            with stage('pin_result_enqueue'):
                result = out_queue.put(value, *a, **kw)
            writer.current = 'pin_idle'
            mark('pin_result_ready')
            return result

    return _ORIGINAL_PIN(Input(), Output(), *args)


class ObservedCollate:
    def __init__(self, original):
        self.original = original

    def __call__(self, items):
        with stage('collate', sample_count=len(items)):
            return self.original(items)


def _local_filesystem(path):
    """Reject Lustre/NFS/BeeGFS for the watchdog spool, including bind mounts."""
    target = str(path.resolve())
    best = ('', '')
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = left.split()[4].replace('\\040', ' ').replace('\\134', '\\')
        if (target == mount or target.startswith(mount.rstrip('/') + '/')) and len(mount) > len(best[0]):
            best = (mount, right.split()[0])
    if best[1] not in ('ext4', 'xfs', 'tmpfs', 'overlay', 'btrfs'):
        raise ValueError('Diagnostics spool requires a verified local filesystem: ' + repr(best))
    return best


class DiagnosticSession:
    def __init__(self, loader, options):
        global _active_session
        if _active_session is not None:
            raise RuntimeError('Only one diagnostic loader may be active in a process')
        if signal.getsignal(signal.SIGUSR2) not in (signal.SIG_DFL, signal.SIG_IGN):
            raise RuntimeError('Diagnostics require exclusive SIGUSR2; handler already installed')
        options = dict(options)
        unknown = set(options) - {'local_dir', 'persistent_dir', 'thresholds', 'poll_seconds', 'reserve_sigusr2'}
        if unknown:
            raise ValueError('Unknown diagnostic options: ' + repr(unknown))
        if options.get('reserve_sigusr2') is not True:
            raise ValueError('Diagnostics require explicit reserve_sigusr2=True')
        if not 1 <= loader.num_workers <= 8 or not loader.in_order:
            raise ValueError('Diagnostics require 1..8 workers and in_order=True')
        thresholds = tuple(float(x) for x in options.get('thresholds', (30, 90, 150)))
        interval = float(options.get('poll_seconds', 1.0))
        if (not thresholds or len(thresholds) > 3 or sorted(set(thresholds)) != list(thresholds)
                or not .02 <= interval <= 5 or thresholds[0] <= interval
                or thresholds[-1] + 2 * interval >= loader.timeout):
            raise ValueError('Diagnostic deadlines must precede DataLoader timeout')
        expected = ('dataset_kind', 'dataset', 'index_queue', 'data_queue', 'done_event',
                    'auto_collation', 'collate_fn', 'drop_last', 'base_seed', 'init_fn',
                    'worker_id', 'num_workers', 'persistent_workers', 'shared_seed')
        if tuple(inspect.signature(_ORIGINAL_WORKER).parameters) != expected:
            raise RuntimeError('Unqualified Torch worker API')
        if not all(hasattr(queues.Queue, name) for name in ('_feed', '_on_queue_feeder_error')):
            raise RuntimeError('Unqualified multiprocessing Queue API')
        context = loader.multiprocessing_context or mp.get_context()
        self.shared = TraceBuffer(context, loader.num_workers)
        self.writer = Writer(self.shared, 0, role='parent')
        self.local = Path(options['local_dir']).resolve(strict=True)
        _local_filesystem(self.local)
        run = 'loader-' + str(os.getpid()) + '-' + uuid.uuid4().hex[:12]
        self.local = self.local / run
        self.local.mkdir(mode=0o700)
        persistent = options.get('persistent_dir')
        persistent = str(Path(persistent).absolute() / run) if persistent else None
        pairs = [socket.socketpair() for _ in range(loader.num_workers + 1)]
        self.readers, self.writers = list(zip(*pairs))
        registered = False
        try:
            for stream in (*self.readers, *self.writers):
                stream.setblocking(False)
                stream.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            _register(self.shared, 0, self.writers[0])
            registered = True
            bind(self.writer)
            self.writer.emit('diagnostics_start', torch_version=torch.__version__,
                             method=context.get_start_method(), spool=str(self.local),
                             sampler_offset=getattr(loader.sampler, 'offset', None),
                             batch_size=loader.batch_size, pin_memory=loader.pin_memory)
            self.process = mp.get_context('spawn').Process(target=watchdog, args=(
                self.shared, self.readers, str(self.local), persistent, thresholds, interval, identity(os.getpid())),
                name='xr1-loader-watchdog', daemon=True)
            self.process.start()
        except BaseException:
            if registered:
                faulthandler.unregister(signal.SIGUSR2)
            for stream in (*self.readers, *self.writers):
                stream.close()
            bind(None)
            raise
        self.closed = False
        _active_session = self

    def close(self):
        global _active_session
        if self.closed:
            return
        self.closed = True
        self.shared.stop.value = 1
        # Bounded grace for final evidence; no training thread waits on disk.
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.2)
        if not self.process.is_alive():
            self.shared.registry[2] = 0
            faulthandler.unregister(signal.SIGUSR2)
            for stream in (*self.readers, *self.writers):
                stream.close()
            _active_session = None
        # If watchdog is stuck in uninterruptible I/O keep handler/FDs valid.
        bind(None)


class DiagnosticIterator(dl._MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.diagnostic = loader._diagnostic_session
        self.diagnostic.shared.wait_since.value = time.monotonic()
        try:
            with stage('parent_worker_startup'):
                super().__init__(loader)
                for index, process in enumerate(self._workers, 1):
                    value = identity(process.pid)
                    if value is not None:
                        self.diagnostic.shared.expected[index * 2:index * 2 + 2] = value
                mark('workers_started', pids=[process.pid for process in self._workers])
        finally:
            self.diagnostic.shared.wait_since.value = 0

    def _order(self):
        info = getattr(self, '_task_info', {})
        return dict(expected_task=getattr(self, '_rcvd_idx', None),
                    sent_tasks=getattr(self, '_send_idx', None),
                    outstanding=getattr(self, '_tasks_outstanding', None),
                    watchdog_heartbeat=self.diagnostic.shared.watchdog_heartbeat.value,
                    buffered_tasks=[key for key, value in info.items() if len(value) == 2][:128],
                    pending_workers=[[key, value[0]] for key, value in info.items()][:128])

    def _get_data(self):
        with stage('parent_ipc_receive', **self._order()):
            value = super()._get_data()
            mark('parent_received', received_task=value[0], **self._order())
            return value

    def _next_data(self):
        bind(self.diagnostic.writer)
        self.diagnostic.shared.wait_since.value = time.monotonic()
        try:
            with stage('parent_ordered_next', **self._order()):
                result = super()._next_data()
                mark('ordered_delivered', **self._order())
                return result
        finally:
            self.diagnostic.shared.wait_since.value = 0


class DiagnosticLoaderMixin:
    def __init__(self, *args, diagnostics=None, **kwargs):
        self.diagnostics = diagnostics
        self._diagnostic_session = None
        super().__init__(*args, **kwargs)

    def _get_iterator(self):
        if not self.diagnostics:
            return super()._get_iterator()
        if self._diagnostic_session is not None:
            self._diagnostic_session.close()
        if not _startup_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent diagnostic loader initialization is unsupported')
        original_init, original_collate = self.worker_init_fn, self.collate_fn
        try:
            self._diagnostic_session = DiagnosticSession(self, self.diagnostics)
            session = self._diagnostic_session
            self.worker_init_fn = WorkerInit(original_init, session.shared, session.writers)
            self.collate_fn = ObservedCollate(original_collate)
            worker._worker_loop = observed_worker_loop
            pin_memory._pin_memory_loop = functools.partial(observed_pin_loop, session.shared)
            return DiagnosticIterator(self)
        except BaseException:
            if self._diagnostic_session is not None:
                self._diagnostic_session.close()
            raise
        finally:
            worker._worker_loop, pin_memory._pin_memory_loop = _ORIGINAL_WORKER, _ORIGINAL_PIN
            self.worker_init_fn, self.collate_fn = original_init, original_collate
            _startup_lock.release()

    def close_diagnostics(self):
        if self._diagnostic_session is not None:
            self._diagnostic_session.close()


class DiagnosticDataLoader(DiagnosticLoaderMixin, DataLoader):
    """Generic CPU fault harness; production uses the same mixin with replay len."""
