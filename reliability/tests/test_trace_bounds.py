import multiprocessing as mp
import struct
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'xr1'))
from mibot.utils.loader_trace import TraceBuffer, Writer, bind, stage


class TraceBoundsTests(unittest.TestCase):
    def test_ring_overwrite_torn_record_and_oversize_are_lossy(self):
        shared = TraceBuffer(mp.get_context(), 1)
        writer = Writer(shared, 0, role='test')
        bind(writer)
        start = time.monotonic()
        for _ in range(10000):
            with stage('fixture'):
                pass
        seconds = time.monotonic()-start
        rows = shared.read(0)
        self.assertEqual(len(rows), 64)
        self.assertEqual(rows[-1]['sequence'], 30000)
        writer.emit('oversize', payload='x' * 10000)
        self.assertEqual(writer.dropped, 1)
        writer.emit('normal')
        self.assertEqual(shared.read(0)[-1]['dropped'], 1)
        # An incomplete publication cannot be mistaken for a valid last event.
        offset = (writer.sequence % 64) * 4096
        writer.view[offset:offset + 8] = struct.pack('Q', 0)
        self.assertEqual(len(shared.read(0)), 63)
        bind(None)
        (ROOT/'reliability/evidence/20261103/trace-microbenchmark.json').write_text(json.dumps(dict(
            stages=10000, events=30000, seconds=seconds, microseconds_per_stage=seconds*100,
            allocated_ring_bytes=len(shared.raw), note='Synthetic telemetry cost, not end-to-end loader overhead'), indent=2)+'\n')

    def test_persistent_exporter_does_not_serialize_collector_progress(self):
        import tempfile
        import os
        from mibot.utils.loader_watchdog import watchdog, identity
        shared = TraceBuffer(mp.get_context(), 0)
        shared.wait_since.value = time.monotonic() - 10
        # The mocked exporter never finishes. Collector still emits all deadlines
        # and final evidence without joining it or starting unbounded exporters.
        shared.stop.value = 1
        with tempfile.TemporaryDirectory(dir=ROOT/'reliability/evidence/tmp') as folder:
            with patch('mibot.utils.loader_watchdog.subprocess.Popen') as popen:
                popen.return_value.poll.return_value = None
                watchdog(shared, [], folder, str(Path(folder)/'persistent'), (.1, .2, .3), .02, identity(os.getpid()))
                self.assertEqual(popen.call_count, 1)
            self.assertEqual(len(list(Path(folder).glob('incident-*.json'))), 4)


if __name__ == '__main__':
    unittest.main()
