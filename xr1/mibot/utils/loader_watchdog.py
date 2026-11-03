"""Independent Linux evidence collector and isolated persistent exporter.

Only the collector writes local evidence. The exporter may block on persistent
storage without holding up either the collector or any training process.
"""
from __future__ import annotations

import argparse
import ctypes
from collections import deque
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def read_proc(path, limit=4096):
    try:
        with open(path) as stream:
            return stream.read(limit)
    except OSError as error:
        return 'UNAVAILABLE:' + type(error).__name__


def identity(pid):
    raw = read_proc(f'/proc/{pid}/stat')
    try:
        fields = raw[raw.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else (pid, int(fields[19]))
    except (IndexError, ValueError):
        return None


def process_snapshot(pid, ticks):
    if identity(pid) != (pid, ticks):
        return dict(pid=pid, start_ticks=ticks, state='exited_or_reused')
    root = Path('/proc') / str(pid)
    result = dict(pid=pid, start_ticks=ticks, status=read_proc(root/'status'),
                  io=read_proc(root/'io'), stat=read_proc(root/'stat'), tasks=[])
    try:
        # Count without retaining an unbounded list of descriptor paths.
        with os.scandir(root/'fd') as entries:
            result['fd_count'] = sum(1 for _ in entries)
        tasks = sorted((root/'task').iterdir(), key=lambda p: int(p.name))
        result['threads_total'] = len(tasks)
        result['threads_truncated'] = len(tasks) > 64
        for task in tasks[:64]:
            result['tasks'].append(dict(tid=int(task.name),
                status=read_proc(task/'status', 2048), wchan=read_proc(task/'wchan', 128),
                syscall=read_proc(task/'syscall', 512), kernel_stack=read_proc(task/'stack', 2048)))
    except OSError as error:
        result['proc_error'] = type(error).__name__
    result['identity_still_matches'] = identity(pid) == (pid, ticks)
    return result


def request_stack(pid, ticks):
    """Use a pidfd to avoid signalling a reused PID. Never use SIGUSR1."""
    # Some Conda Python builds omit os.pidfd_open despite a supporting kernel.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_open') or not hasattr(libc, 'pidfd_send_signal'):
        return 'unavailable:libc_pidfd'
    try:
        fd = libc.pidfd_open(pid, 0)
        if fd < 0:
            return 'unavailable:pidfd_open_errno_' + str(ctypes.get_errno())
        try:
            if identity(pid) != (pid, ticks):
                return 'exited_or_reused'
            if libc.pidfd_send_signal(fd, signal.SIGUSR2, None, 0) < 0:
                return 'unavailable:pidfd_send_errno_' + str(ctypes.get_errno())
            return 'requested'
        finally:
            os.close(fd)
    except OSError as error:
        return 'unavailable:' + type(error).__name__


def export(local, persistent):
    """One separate process, bounded file set; atomic publication after fsync."""
    source, destination = Path(local), Path(persistent)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in sorted(source.glob('incident-*.json'))[:8]:
        raw = path.read_bytes()
        if len(raw) > 16 * 1024**2:
            raise RuntimeError('Diagnostic report exceeds bound')
        temporary = destination / (path.name + '.partial')
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination/path.name)
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def watchdog(shared, readers, local, persistent, thresholds, interval, parent_identity):
    """Bounded snapshots triggered by parent wait, including worker startup/IPC."""
    local = Path(local)
    history = deque(maxlen=3)
    traces = {index: bytearray() for index in range(len(readers))}
    trace_bytes = {index: 0 for index in range(len(readers))}
    emitted, wait_key, levels = 0, None, set()
    exporter = None
    export_pending = False

    def drain():
        for index, reader in enumerate(readers):
            # A faulty producer cannot monopolize the watchdog.
            for _ in range(16):
                try:
                    data = reader.recv(8192)
                except (BlockingIOError, OSError):
                    break
                if not data:
                    break
                trace_bytes[index] += len(data)
                traces[index].extend(data[:max(0, 65536 - len(traces[index]))])

    def capture(reason):
        nonlocal emitted, export_pending
        if emitted >= 8:
            return
        registered = []
        for index in range(len(readers)):
            pid, ticks, ready = shared.registry[index * 3:index * 3 + 3]
            if not pid:
                pid, ticks = shared.expected[index * 2:index * 2 + 2]
            if pid and ticks:
                registered.append((index, pid, ticks, ready))
        # Save stages before asking for stacks: sending a signal may wake a syscall.
        rows = [shared.read(lane) for lane in range(shared.lanes)]
        processes = [process_snapshot(pid, ticks) for _, pid, ticks, _ in registered]
        requests = {str(index): request_stack(pid, ticks) if ready else 'handler_not_ready'
                    for index, pid, ticks, ready in registered}
        deadline = time.monotonic() + .1
        while time.monotonic() < deadline:
            drain()
            time.sleep(.01)
        report = dict(schema='xr1-loader-diagnostic-v1', reason=reason,
                      timestamp=time.time(), monotonic=time.monotonic(),
                      parent_wait_since=shared.wait_since.value,
                      ring_capacity=64, rings=rows, processes=processes,
                      overwritten_events_by_lane=[max(0, ring[-1]['sequence'] - 64) if ring else 0 for ring in rows],
                      recent_waits=list(history), stack_requests=requests,
                      python_stacks={str(k): v.decode('ascii', errors='replace') for k, v in traces.items()},
                      trace_bytes=trace_bytes.copy(),
                      trace_truncated={str(k): n >= 65536 for k, n in trace_bytes.items()},
                      native_evidence='proc task syscall and kernel stack; permission failures recorded',
                      persistent_export='separate process; local capture is not proof of export')
        temporary = local / f'incident-{emitted:02d}.partial'
        with temporary.open('x') as stream:
            json.dump(report, stream, separators=(',', ':'))
        temporary.replace(local / f'incident-{emitted:02d}.json')
        emitted += 1
        export_pending = True

    while True:
        now = time.monotonic()
        shared.watchdog_heartbeat.value = now
        drain()
        waiting = shared.wait_since.value
        if waiting != wait_key:
            wait_key, levels = waiting, set()
        history.append(dict(monotonic=now, wait_since=waiting))
        if waiting:
            for threshold in thresholds:
                if now - waiting >= threshold and threshold not in levels:
                    capture('parent_wait_' + str(threshold))
                    levels.add(threshold)
        stopped = bool(shared.stop.value) or identity(parent_identity[0]) != parent_identity
        if stopped:
            capture('shutdown_or_parent_exit')
        if persistent and export_pending and (exporter is None or exporter.poll() is not None):
            exporter = subprocess.Popen([sys.executable, __file__, '--export', str(local), persistent],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, start_new_session=True)
            export_pending = False
        if stopped:
            break
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', nargs=2, required=True)
    args = parser.parse_args()
    export(*args.export)
