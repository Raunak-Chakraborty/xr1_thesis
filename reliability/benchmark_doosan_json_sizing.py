#!/usr/bin/env python3
"""CPU-only sizing comparison; synthetic by default, optional real episode JSON.

Reads one episode at a time. No images, model, checkpoints, GPU or training.
Timings use already-read text/objects and do not measure Lustre performance.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'xr1'))
from mibot.utils.doosan_io_cache import object_bytes


def legacy_bytes(value):
    """Task 01 implementation, kept here only as a measurement oracle."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(legacy_bytes(k) + legacy_bytes(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(legacy_bytes(v) for v in value)
    return size


def synthetic_text(frames):
    # Episode-shaped numeric data; not a claim about frozen episode sizes.
    def rows(width):
        return [[i / 1000 + j for j in range(width)] for i in range(frames)]
    return json.dumps(dict(num_frames=frames,
                           proprios=dict(joint_pos=rows(6), wrench=rows(6)),
                           actions=dict(semantic=rows(7))))


def median_seconds(function, repeats):
    samples = []
    for _ in range(repeats):
        gc.collect()  # Outside measured region; GC otherwise remains enabled.
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
        del result  # Exclude destruction from the next repeat's measured time.
    return statistics.median(samples)


def measure(raw, label, repeats, budget):
    value = json.loads(raw)
    exact = legacy_bytes(value)
    assert object_bytes(value) == exact
    bounded = object_bytes(value, limit=budget)
    if exact <= budget:
        assert bounded == exact
    else:
        assert bounded > budget
    # This second budget deliberately rejects larger objects, independently of
    # the production retention budget. Never use a partial size for retention.
    rejection_budget = min(1024**2, max(0, exact - 1))
    assert object_bytes(value, limit=rejection_budget) > rejection_budget
    return dict(input=label, text_characters=len(raw), exact_estimated_bytes=exact,
                repeats=repeats, budget_bytes=budget, fits_budget=exact <= budget,
                rejection_budget_bytes=rejection_budget,
                median_seconds=dict(
                    parse=median_seconds(lambda: json.loads(raw), repeats),
                    legacy_sizing=median_seconds(lambda: legacy_bytes(value), repeats),
                    bounded_sizing=median_seconds(lambda: object_bytes(value, limit=budget), repeats),
                    bounded_rejection=median_seconds(
                        lambda: object_bytes(value, limit=rejection_budget), repeats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, action='append', default=[],
                        help='Existing frozen JSON; repeat for selected episodes')
    parser.add_argument('--frames', type=int, nargs='+', default=[10000, 100000])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--budget-bytes', type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.repeats < 1 or args.budget_bytes < 0 or any(n < 1 for n in args.frames):
        parser.error('positive repeats/frames and nonnegative budget required')
    print(json.dumps(dict(python=sys.version, clock='perf_counter',
                          caveat='Local CPU medians; no Lustre, RSS or training throughput measurement')))
    if args.episode:
        for path in args.episode:
            print(json.dumps(measure(path.read_text(), str(path), args.repeats, args.budget_bytes)))
    else:
        for frames in args.frames:
            print(json.dumps(measure(synthetic_text(frames), f'synthetic-{frames}-frames',
                                     args.repeats, args.budget_bytes)))


if __name__ == '__main__':
    main()
