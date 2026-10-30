import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import default_collate

from mibot.data.datamodule.doosan_datamodule import shutdown_loader
from mibot.data.datasets.doosan_json_dataset import DoosanJsonDataset, FrameSampleTable
from mibot.utils.doosan_replay import (
    RankOffsetSampler, checkpoint_cursor, make_contract, seed_worker, stable_sample_seed, validate_cursor,
)


class AugmentedFixture(DoosanJsonDataset):
    """Synthetic images through the actual per-sample RNG scope and augmentation."""
    def __init__(self, length=120):
        self.samples = FrameSampleTable(['/fixture/ep.json'], [length], range(length))
        self._replay_root = '/fixture'
        self.sampler_seed = 42
        self.deterministic_replay = True
        self.pixels = np.arange(32*32*3, dtype=np.uint8).reshape(32,32,3)
    def _getitem_unseeded(self, index):
        image = Image.fromarray(self.pixels)
        augmented = self._augment([image, image])
        return dict(index=index, prompt=random.randrange(5),
                    images=torch.from_numpy(np.stack([np.asarray(x) for x in augmented]).copy()))


class FailingFixture(AugmentedFixture):
    def _getitem_unseeded(self, index):
        raise ValueError('synthetic corrupt camera frame')


def numpy_collate(items):
    # Restricted laptop sandbox forbids AF_UNIX descriptor passing used by
    # torch tensor IPC. Keep real worker processes/augmentation/order, transport
    # CPU arrays through ordinary pickle. Production tensor IPC stays UNVERIFIED.
    return {key: value.numpy() for key,value in default_collate(items).items()}


def fixture_loader(dataset, workers, step=0, batch=6):
    base = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
    return DataLoader(dataset, batch_size=batch,
                      sampler=RankOffsetSampler(base, step*batch, batch_size=batch),
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(42),
                      collate_fn=numpy_collate,
                      **(dict(prefetch_factor=2, persistent_workers=True, in_order=True, timeout=10)
                         if workers else {}))


class ReplayTest(unittest.TestCase):
    def test_workers_and_prefetch_resume(self):
        baseline = None
        for workers in (0, 4, 8):
            with self.subTest(workers=workers):
                dataset = AugmentedFixture()
                loader = fixture_loader(dataset, workers)
                iterator = iter(loader)
                full = [next(iterator) for _ in range(4)]
                # Prefetched work may cover most of the stream, but only 2 batches
                # are committed. Drop a partly processed third batch on restart.
                contract = make_contract(data_horizon=20, batch_size=6, world_size=1,
                                         sampler_seed=42, num_workers=workers)
                self.assertEqual(checkpoint_cursor(contract, 2)['rank_sample_offset'], 12)
                processes = list(getattr(iterator, '_workers', []))
                shutdown_loader(loader, iterator)
                self.assertTrue(all(not p.is_alive() for p in processes))
                del iterator, loader
                resumed_loader = fixture_loader(dataset, workers, step=2)
                resumed_iterator = iter(resumed_loader)
                resumed = [next(resumed_iterator) for _ in range(2)]
                for expected, actual in zip(full[2:], resumed):
                    for key in expected:
                        np.testing.assert_array_equal(expected[key], actual[key])
                if baseline is None:
                    baseline = full
                else:
                    for expected, actual in zip(baseline, full):
                        for key in expected:
                            np.testing.assert_array_equal(expected[key], actual[key])
                processes = list(getattr(resumed_iterator, '_workers', []))
                shutdown_loader(resumed_loader, resumed_iterator)
                self.assertTrue(all(not p.is_alive() for p in processes))
                del resumed_iterator, resumed_loader

    def test_partial_batch_and_sampler_failures(self):
        base = DistributedSampler(range(24), num_replicas=1, rank=0, shuffle=True)
        for offset in (-1, 1, 24):
            with self.assertRaises(ValueError):
                RankOffsetSampler(base, offset, batch_size=6)
        with self.assertRaises(ValueError):
            RankOffsetSampler(DistributedSampler(range(25), num_replicas=2, rank=0), 0)
        with self.assertRaises(ValueError):
            RankOffsetSampler(DistributedSampler(range(24), num_replicas=1, rank=0, shuffle=False), 0)
        sampler = RankOffsetSampler(base, 0)
        with self.assertRaises(RuntimeError):
            sampler.set_epoch(1)
        for workers in (-1, 2):
            with self.assertRaises(ValueError):
                make_contract(data_horizon=3,batch_size=4,world_size=1,sampler_seed=42,num_workers=workers)

    def test_rank_streams_and_contract_changes(self):
        streams = []
        for rank in range(2):
            base = DistributedSampler(range(48), num_replicas=2, rank=rank, seed=42)
            full = list(base)
            self.assertEqual(list(RankOffsetSampler(base, 12, batch_size=6)), full[12:])
            streams.append(set(full))
        self.assertFalse(streams[0] & streams[1])
        contract = make_contract(data_horizon=4,batch_size=6,world_size=2,sampler_seed=42)
        cursor = checkpoint_cursor(contract, 2)
        for key in ('batch_size','world_size','sampler_seed','data_horizon','num_workers'):
            changed = dict(contract); changed[key] += 1
            with self.assertRaises(RuntimeError):
                validate_cursor(cursor, changed, checkpoint_step=2)

    def test_worker_error_surfaces(self):
        loader = fixture_loader(FailingFixture(), 4)
        iterator = iter(loader)
        try:
            with self.assertRaisesRegex(ValueError, 'synthetic corrupt'):
                next(iterator)
        finally:
            del iterator, loader

    def test_augmentation_restores_parent_rng(self):
        dataset = AugmentedFixture()
        random.seed(99)
        before = random.getstate()
        dataset[0]
        self.assertEqual(before, random.getstate())
        with self.assertRaises(RuntimeError):
            stable_sample_seed('/outside/file', 0, '/fixture', 42)

    def test_compact_table_preserves_original_cycles(self):
        dataset = DoosanJsonDataset.__new__(DoosanJsonDataset)
        dataset.files = ['/fixture/a', '/fixture/zero', '/fixture/b']
        dataset.sampler_seed, dataset.max_samples = 42, 21
        counts = [4, 0, 3]
        population = [dict(file=p, frame_index=i) for p,c in zip(dataset.files,counts) for i in range(c)]
        expected, rng = [], random.Random(42)
        while len(expected) < dataset.max_samples:
            ids = list(range(len(population))); rng.shuffle(ids)
            expected.extend(population[i] for i in ids[:dataset.max_samples-len(expected)])
        with patch.object(DoosanJsonDataset, '_read_json', side_effect=lambda p: {'num_frames': counts[dataset.files.index(p)]}):
            table = dataset._samples()
        self.assertEqual(table[:], expected)
        self.assertEqual(table[-1], expected[-1])
