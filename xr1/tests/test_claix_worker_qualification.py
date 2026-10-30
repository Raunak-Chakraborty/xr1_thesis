"""Probe control-flow fixtures; real tensor IPC remains a CLAIX gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mibot.data.datamodule.doosan_datamodule import shared_memory_budget, shutdown_loader
from mibot.utils.claix_runtime import StopRequest

spec = importlib.util.spec_from_file_location('worker_probe',Path(__file__).resolve().parents[1]/'tools/claix_b24_worker_qualification.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTest(unittest.TestCase):
    def fixture(self, workers, *, drop=False, fail=False):
        class Iterator:
            closed = False
            def __init__(self, step):
                self.step = step
            def __next__(self):
                if fail:
                    raise ValueError('worker failed')
                self.step += 1
                return dict(action_segments=torch.full((23 if drop else 24,2),self.step),
                            image_grid_thw=torch.ones(48,3),input_ids=torch.ones(1,100))
            def _shutdown_workers(self):
                self.closed = True
        class Loader:
            def __iter__(self):
                self._iterator = Iterator(dm.resume_step)
                seen.append(self._iterator)
                return self._iterator
        seen=[]
        dm=SimpleNamespace(num_workers=workers,prefetch_factor=2,resume_step=0,train_dataloader=Loader)
        return dm,seen

    def test_all_worker_cases_keep_every_example_and_close_each_iterator(self):
        baseline=None
        for workers in (0,4,8):
            dm,seen=self.fixture(workers)
            full=probe.fetch_batches(dm,0,4,StopRequest(float('inf')))
            suffix=probe.fetch_batches(dm,2,2,StopRequest(float('inf')))
            self.assertEqual(full['examples'],96)
            self.assertEqual(suffix['examples'],48)
            self.assertEqual(len(full['hashes']),4)
            self.assertEqual(full['hashes'][2:],suffix['hashes'])
            if baseline is not None:
                self.assertEqual(baseline,full['hashes'])
            baseline=full['hashes']
            self.assertTrue(all(iterator.closed for iterator in seen))

    def test_drop_fault_and_deadline_fail_without_partial_success(self):
        for option in ('drop','fail','deadline'):
            dm,seen=self.fixture(4,drop=option=='drop',fail=option=='fail')
            stop=StopRequest(0 if option=='deadline' else float('inf'))
            with self.subTest(option=option), self.assertRaises((ValueError,RuntimeError)):
                probe.fetch_batches(dm,0,4,stop)
            self.assertTrue(all(iterator.closed for iterator in seen))

    def test_shared_memory_floor_and_prefetch_estimate(self):
        with patch('mibot.data.datamodule.doosan_datamodule.os.statvfs',
                   return_value=SimpleNamespace(f_bavail=2*1024**3,f_frsize=1)):
            result=shared_memory_budget(1024**2,workers=8,prefetch_factor=2)
            self.assertEqual(result['required_bytes'],1024**3)
            with self.assertRaisesRegex(RuntimeError,'/dev/shm'):
                shared_memory_budget(128*1024**2,workers=8,prefetch_factor=2)
        self.assertEqual(shared_memory_budget(workers=0,prefetch_factor=2)['required_bytes'],0)
        shutdown_loader(None,None)
