#!/usr/bin/env python3
"""Bounded CPU measurement of the frozen sampler's per-worker JSON LRU stream.

Reads JSON only. Does not fetch camera images, run a model, or change cache policy.
Decoded-memory sizes are Python object graph estimates, not allocator RSS bounds.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import random
import sys
import time


def decoded_bytes(value):
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return total


def profile(root, *, first_step=21501, batches=128, horizon=48284, batch_size=24, workers=4):
    import torch
    if not 1 <= batches <= 128 or first_step < 1 or first_step + batches - 1 > horizon:
        raise ValueError('Only a bounded valid replay window is supported')
    root = Path(root).resolve(strict=True)
    files = sorted(root.rglob('*.json'))
    ends, memory, timings, disk = [], [], [], []
    total = 0
    started = time.monotonic()
    for path in files:
        before = time.monotonic()
        with path.open() as stream:
            value = json.load(stream)
        timings.append(time.monotonic() - before)
        total += int(value['num_frames'])
        ends.append(total)
        memory.append(decoded_bytes(value))
        disk.append(path.stat().st_size)
    if not files or workers < 1:
        raise ValueError('Empty dataset or no workers')
    # Match DoosanJsonDataset._samples, then DistributedSampler(epoch=0, rank=0).
    count = horizon * batch_size
    rng = random.Random(42)
    selected = []
    while len(selected) < count:
        indices = list(range(total))
        rng.shuffle(indices)
        selected.extend(indices[:count - len(selected)])
    order = torch.randperm(count, generator=torch.Generator().manual_seed(42))
    offset = (first_step - 1) * batch_size
    streams = [[] for _ in range(workers)]
    for task in range(batches):
        for index in order[offset + task * batch_size:offset + (task + 1) * batch_size].tolist():
            streams[task % workers].append(bisect_right(ends, selected[index]))
    policies = []
    for capacity in (2, 4, 8, 16, 32, 64, 128, len(files)):
        hits, misses, peak = 0, 0, []
        for stream in streams:
            cache, maximum = OrderedDict(), 0
            for episode in stream:
                if episode in cache:
                    hits += 1
                    cache.move_to_end(episode)
                else:
                    misses += 1
                    cache[episode] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
                maximum = max(maximum, sum(memory[episode] for episode in cache))
            peak.append(maximum)
        policies.append(dict(entries=capacity, hits=hits, misses=misses, hit_rate=hits/(hits+misses),
                             decoded_peak_bytes_by_worker=peak, decoded_peak_bytes_sum=sum(peak)))
    return dict(schema='xr1-json-cache-profile-v1', root=str(root), first_step=first_step, batches=batches,
                horizon=horizon, batch_size=batch_size, workers=workers, episodes=len(files), frames=total,
                elapsed_seconds=time.monotonic()-started, file_bytes_sum=sum(disk),
                decoded_bytes_sum=sum(memory), decoded_bytes_max=max(memory),
                json_load_seconds_sum=sum(timings), json_load_seconds_max=max(timings), policies=policies,
                caveats=['Cold worker caches; production cache warmth unknown at resumed boundary',
                         'Linux ordered round-robin map tasks; no worker loss or skip',
                         'Object graph sizes exclude allocator fragmentation and downstream tensors',
                         'Local read/parse timings do not estimate Lustre latency'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batches', type=int, default=128)
    args = p.parse_args()
    result = profile(args.root, batches=args.batches)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
