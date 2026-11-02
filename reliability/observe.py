#!/usr/bin/env python3
"""Passive, bounded Linux process snapshots. No signals, ptrace, or training hooks."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

from common import require, write_new


def read_proc(path, limit=4096):
    try:
        with Path(path).open() as stream:
            return stream.read(limit)
    except OSError as error:
        return f'UNAVAILABLE:{type(error).__name__}'


def identity(pid):
    raw = read_proc(f'/proc/{pid}/stat')
    if raw.startswith('UNAVAILABLE:'):
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    if fields[0] == 'Z':
        return None
    return (int(pid), int(fields[19]))  # starttime protects against PID reuse


def snapshot(pid):
    root_identity = identity(pid)
    pending, processes = [pid], []
    while pending and len(processes) < 16:
        current = pending.pop()
        before = identity(current)
        if before is None:
            continue
        base = Path(f'/proc/{current}')
        children = read_proc(base / f'task/{current}/children')
        pending.extend(int(x) for x in children.split() if x.isdecimal())
        row = dict(pid=current, start_ticks=before[1],
                   comm=read_proc(base/'comm',128), status=read_proc(base/'status',2048),
                   io=read_proc(base/'io',1024), stat=read_proc(base/'stat',2048), tasks=[])
        try:
            tasks = sorted((base/'task').iterdir(), key=lambda p:int(p.name))[:32]
        except OSError:
            tasks = []
        for task in tasks:
            row['tasks'].append(dict(tid=int(task.name), wchan=read_proc(task/'wchan',128),
                                     syscall=read_proc(task/'syscall',512), kernel_stack=read_proc(task/'stack',1024)))
        if identity(current) == before:
            processes.append(row)
    return dict(timestamp=time.time(), root_identity=root_identity, processes=processes)


def observe(pid, output, *, interval=5., samples=24, duration=85500., stall_seconds=60.):
    """Keep at most two minutes in RAM; publish on inactivity and exit/deadline.

    Does not poll Lustre telemetry and cannot block training if output I/O stalls.
    This captures kernel wait evidence, NOT Python stacks or sample identities.
    """
    require(1 <= samples <= 24 and interval >= .1 and 0 < duration <= 85500,
            'Invalid observer bounds')
    early = Path(str(output) + '.inactivity.json')
    require(not Path(output).exists() and not early.exists(), 'Observer output already exists')
    initial = identity(pid)
    require(initial is not None, 'Target process is absent')
    history = deque(maxlen=samples)
    end = time.monotonic() + duration
    previous, changed, emitted = None, time.monotonic(), False
    while identity(pid) == initial and time.monotonic() < end:
        current = snapshot(pid)
        history.append(current)
        counters = []
        for row in current['processes']:
            raw = row['stat']
            fields = raw[raw.rfind(')')+2:].split()
            counters.append((row['pid'], row['start_ticks'], fields[11:13], row['io']))
        if counters != previous:
            previous, changed = counters, time.monotonic()
        elif not emitted and time.monotonic()-changed >= stall_seconds:
            write_new(early, dict(schema='xr1-process-observer-v1', status='INACTIVITY_NOT_ROOT_CAUSE',
                                 snapshots=list(history)))
            emitted = True
        time.sleep(min(interval, max(0., end-time.monotonic())))
    report = dict(schema='xr1-process-observer-v1', target_identity=initial,
                  status='OBSERVED_NOT_DIAGNOSED', snapshots=list(history))
    write_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', required=True, type=int, help='Training rank PID, not Slurm daemon')
    p.add_argument('--output', required=True, help='New persistent report path')
    p.add_argument('--seconds', type=float, default=85500.)
    args = p.parse_args()
    observe(args.pid, args.output, duration=args.seconds)


if __name__ == '__main__':
    main()
