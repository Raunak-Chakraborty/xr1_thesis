"""Bounded worker-local caches. No tensors, telemetry I/O, or RNG operations."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
import sys

from mibot.utils.loader_trace import mark, stage


def object_bytes(value, limit=None):
    """Conservative decoded JSON size, including container capacity and scalars.

    Count aliased strings/scalars repeatedly (an upper estimate). This avoids a
    potentially large temporary set of object IDs. Excludes allocator overhead,
    parsing temporaries and downstream tensors; it is NOT a process RSS bound.
    With a limit, return the exact estimate if it fits, otherwise stop once the
    partial estimate exceeds the limit. Such a result is only for rejection,
    never retention accounting. Scalar leaves need no recursive Python call.
    """
    size = sys.getsizeof(value)
    if limit is not None and size > limit:
        return size
    if isinstance(value, dict):
        for key, item in value.items():
            size += object_bytes(key)
            size += object_bytes(item, None if limit is None else limit - size)
            if limit is not None and size > limit:
                break
    elif isinstance(value, (list, tuple)):
        for item in value:
            size += (object_bytes(item, None if limit is None else limit - size)
                     if isinstance(item, (dict, list, tuple)) else sys.getsizeof(item))
            if limit is not None and size > limit:
                break
    return size


class CacheMetrics:
    """Fixed-size, saturating counters; optional reports to the existing ring.

    No stdout/file writes. With no bound trace writer, reports are no-ops.
    """
    def __init__(self, names, report_every=0):
        if report_every != 0 and report_every < 128:
            raise ValueError('cache report interval must be zero or at least 128 lookups')
        self.counts = dict.fromkeys(names, 0)
        self.report_every = report_every
        self.lookups = 0

    def add(self, name):
        self.counts[name] = min(self.counts[name] + 1, (1 << 63) - 1)

    def report(self, kind, **occupancy):
        if not self.report_every:
            return
        self.lookups += 1
        if self.lookups >= self.report_every:
            self.lookups = 0
            mark('io_cache_summary', cache=kind, **self.counts, **occupancy)


class JsonEpisodeCache:
    """LRU bounded by episode count AND estimated decoded bytes per dataset.

    Trajectories are read-only to callers (the prompt path already deep-copies).
    Forked children discard inherited values before use; spawn serializes only
    policy. No decoded mutable trajectories travel between workers.
    """
    def __init__(self, max_entries=16, max_bytes=256 * 1024**2, report_every=0):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError('JSON cache bounds must be nonnegative')
        self.policy = (int(max_entries), int(max_bytes), int(report_every))
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.entries = OrderedDict()
        self.bytes = 0
        self.metrics = CacheMetrics(('hits', 'misses', 'evictions', 'oversized'), self.policy[2])

    def _local(self):
        if self.pid != os.getpid():
            self._reset()

    def __getstate__(self):
        return {'policy': self.policy}

    def __setstate__(self, state):
        self.policy = state['policy']
        self._reset()

    def clear(self):
        self._local()
        self.entries.clear()
        self.bytes = 0

    def snapshot(self):
        self._local()
        return dict(self.metrics.counts, pid=self.pid, entries=len(self.entries),
                    estimated_bytes=self.bytes, max_entries=self.policy[0], max_bytes=self.policy[1])

    def read(self, path):
        self._local()
        key = os.fspath(path)
        cached = self.entries.get(key)
        if cached is not None:
            self.metrics.add('hits')
            self.entries.move_to_end(key)
            value = cached[0]
        else:
            self.metrics.add('misses')
            with stage('json_load'):
                with open(path) as stream:
                    value = json.load(stream)
            limit, budget, _ = self.policy
            if limit and budget:
                key_size = object_bytes(key)
                size = object_bytes(value, limit=budget - key_size) + key_size
                if size > budget:
                    self.metrics.add('oversized')
                else:
                    while self.entries and (len(self.entries) >= limit or self.bytes + size > budget):
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.bytes -= removed
                        self.metrics.add('evictions')
                    self.entries[key] = (value, size)
                    self.bytes += size
        self.metrics.report('json', entries=len(self.entries), estimated_bytes=self.bytes)
        return value
