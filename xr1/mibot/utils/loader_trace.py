"""Lossy bounded telemetry: no locks, queues, signals or filesystem on the hot path.

One writer per lane. A reader accepts a cell only if its two sequence numbers
agree. An interrupted/concurrent write is discarded, never waited upon. This is
diagnostic telemetry, not a synchronization primitive or scientific input.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import functools
import json
import os
import struct
import threading
import time

CELLS = 64
CELL_BYTES = 4096
_thread = threading.local()


class TraceBuffer:
    def __init__(self, context, workers):
        self.lanes = 2 + 2 * workers  # parent, pin thread, worker/feeder pairs
        self.raw = context.RawArray(ctypes.c_ubyte, self.lanes * CELLS * CELL_BYTES)
        self.registry = context.RawArray(ctypes.c_longlong, (workers + 1) * 3)
        self.expected = context.RawArray(ctypes.c_longlong, (workers + 1) * 2)
        self.stop = context.RawValue(ctypes.c_int, 0)
        self.wait_since = context.RawValue(ctypes.c_double, 0)
        self.watchdog_heartbeat = context.RawValue(ctypes.c_double, 0)

    def read(self, lane):
        view = memoryview(self.raw).cast('B')
        rows = []
        for cell in range(CELLS):
            start = (lane * CELLS + cell) * CELL_BYTES
            before = bytes(view[start:start + 8])
            raw = bytes(view[start:start + CELL_BYTES])
            after = bytes(view[start:start + 8])
            sequence = struct.unpack_from('Q', raw)[0]
            if not sequence or before != after or before != raw[-8:] or before != raw[:8]:
                continue
            length = struct.unpack_from('I', raw, 8)[0]
            if length > CELL_BYTES - 20:
                continue
            try:
                rows.append(json.loads(raw[12:12 + length]))
            except (ValueError, UnicodeError):
                continue
        return sorted(rows, key=lambda row: row['sequence'])


class Writer:
    def __init__(self, shared, lane, **identity):
        self.view = memoryview(shared.raw).cast('B')
        self.lane = lane
        self.sequence = 0
        self.dropped = 0
        self.fields = dict(pid=os.getpid(), lane=lane, **identity)
        self.current = 'ready'

    def emit(self, event, **fields):
        # Diagnostic failures must not replace a training result or exception.
        try:
            self.sequence += 1
            row = dict(self.fields, stage=self.current, event=event,
                       monotonic=time.monotonic(), sequence=self.sequence,
                       dropped=self.dropped, **fields)
            raw = json.dumps(row, separators=(',', ':'), ensure_ascii=True).encode()
            if len(raw) > CELL_BYTES - 20:
                self.dropped += 1
                return
            start = (self.lane * CELLS + self.sequence % CELLS) * CELL_BYTES
            self.view[start:start + 8] = b'\0' * 8
            self.view[start + 8:start + 12] = struct.pack('I', len(raw))
            self.view[start + 12:start + 12 + len(raw)] = raw
            sequence = struct.pack('Q', self.sequence)
            self.view[start + CELL_BYTES - 8:start + CELL_BYTES] = sequence
            self.view[start:start + 8] = sequence
        except Exception:
            self.dropped += 1


def bind(writer):
    _thread.writer = writer


def mark(event, **fields):
    writer = getattr(_thread, 'writer', None)
    if writer is not None:
        writer.emit(event, **fields)


def identify(**fields):
    writer = getattr(_thread, 'writer', None)
    if writer is not None:
        writer.fields.update(fields)


@contextmanager
def stage(name, **fields):
    writer = getattr(_thread, 'writer', None)
    if writer is None:
        yield
        return
    previous, saved = writer.current, writer.fields.copy()
    writer.current = name
    writer.fields.update(fields)
    started = time.monotonic()
    writer.emit('begin')
    try:
        yield
    except BaseException as error:
        writer.emit('error', error_type=type(error).__name__)
        raise
    finally:
        writer.emit('end', seconds=time.monotonic() - started)
        writer.current, writer.fields = previous, saved
        writer.emit('resume')


def traced(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


class TracedPayload:
    """Keep ZIP/PIL streaming semantics, distinguishing lazy decode from reads."""
    def __init__(self, payload):
        self.payload = payload

    def __getattr__(self, name):
        return getattr(self.payload, name)

    def read(self, *args):
        with stage('zip_member_read'):
            return self.payload.read(*args)

    def seek(self, *args):
        with stage('zip_member_seek'):
            return self.payload.seek(*args)
