"""Real spawned PyTorch workers. NumPy transport is explicit where IPC is restricted."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile
import json

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import bounded_replay


class HangDataset:
    def __len__(self):
        return 192

    def __getitem__(self,index):
        if index == 0:
            time.sleep(30)
        return np.array([index],dtype=np.int64)


def arrays(items):
    return np.stack(items)


class FastDataset(HangDataset):
    def __getitem__(self,index):
        return np.array([index],dtype=np.int64)


class ZipDataset(FastDataset):
    def __init__(self,path):
        self.path = str(path)
        self.samples = [dict(file='/fixture/episode.json',frame_index=i) for i in range(8)]

    def __getitem__(self,index):
        with zipfile.ZipFile(self.path) as archive:
            value = int(archive.read(f'frame_{index:09d}.png'))
        return np.array([value],dtype=np.int64)


class WorkerTests(unittest.TestCase):
    def test_hanging_worker_has_pid_status_and_python_stack_before_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)/'slots'
            report = bounded_replay(HangDataset(),arrays,[[i] for i in range(8)],local,
                                    timeout=12,alarm_seconds=2,transport='numpy-test-only')
            self.assertEqual(report['status'],'FAIL')
            self.assertIn('timed out',report['error'])
            self.assertEqual(len(report['worker_pids']),4)
            traces = '\n'.join(p.read_text() for p in local.glob('worker-*.trace'))
            self.assertIn('__getitem__',traces)
            self.assertIn('Timeout',traces)
            self.assertIn('sample_start','\n'.join(p.read_text() for p in local.glob('worker-*.json')))
            self.assertIn('Timeout',(local/'parent.trace').read_text())

    def test_real_four_worker_ordered_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            batches = [[i] for i in range(8)]
            archive = Path(folder)/'camera.zip'
            with zipfile.ZipFile(archive,'w') as zipped:
                for i in range(8):
                    zipped.writestr(f'frame_{i:09d}.png',str(i))
            seen = []
            report = bounded_replay(ZipDataset(archive),arrays,batches,Path(folder)/'slots',
                                    timeout=20,alarm_seconds=8,transport='numpy-test-only',
                                    on_batch=lambda batch:seen.append(batch.reshape(-1).tolist()))
            self.assertEqual(report['status'],'PASS',report)
            self.assertEqual(seen,batches)
            states = [json.loads(p.read_text()) for p in (Path(folder)/'slots').glob('worker-*.json')]
            self.assertEqual(len(states),4)
            for state in states:
                self.assertEqual(state['archive'],str(archive))
                self.assertEqual(state['episode'],'/fixture/episode.json')
                self.assertEqual(state['phase'],'collate_done_awaiting_queue')
                self.assertTrue(state['member'].startswith('frame_'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
