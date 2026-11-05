"""Synthetic CPU regressions for Lustre I/O policy; never loads a processor."""
import io
from contextlib import redirect_stdout
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from mmengine import Config

from mibot.data.datamodule.doosan_datamodule import DoosanDataModule, shutdown_loader
from mibot.data.datasets.doosan_json_dataset import DoosanJsonDataset
from mibot.data.datasets import doosan_zip_images as zipped
from mibot.utils.doosan_io_cache import JsonEpisodeCache, object_bytes
from mibot.utils.doosan_replay import seed_worker
import test_doosan_dataset


ROOT = Path(__file__).resolve().parents[1]


def make_archive(root, name, frames=5, color=0, names=None):
    row = dict(archive=name + '.zip', original_source_path='/original/' + name + '.mp4',
               camera='external_camera_2', frames=frames, crop_bbox=None)
    path = root / row['archive']
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for i, member in enumerate(names or [f'frame_{i:09d}.png' for i in range(frames)]):
            pixels = (np.arange(32 * 32 * 3).reshape(32, 32, 3) + color + i).astype(np.uint8)
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, 'PNG')
            archive.writestr(member, stream.getvalue())
    meta = dict(schema=zipped._SCHEMA, original_source_path=row['original_source_path'],
                frames=frames, crop_bbox=None, archive_bytes=path.stat().st_size)
    path.with_suffix('.metadata.json').write_text(json.dumps(meta))
    return row


def canonical(value):
    if isinstance(value, Image.Image):
        return (value.mode, value.size, value.tobytes())
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.numpy().tobytes())
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def cpu_collate(samples):
    # Avoid tensor FD-passing requirements in the restricted laptop sandbox.
    return [canonical(sample) for sample in samples]


def fork_check(cache, path, root, row, connection):
    try:
        # Production module's fork hook must already have closed parent handles.
        empty_before_use = len(zipped._cache.handles) == 0
        value = cache.read(path)
        archive = zipped._get_archive(Path(root), row)
        connection.send((empty_before_use, value, cache.snapshot(), zipped.cache_info(),
                         archive.read('frame_000000000.png')[:8]))
        zipped.close_cache()
    finally:
        connection.close()


class JsonCacheTest(unittest.TestCase):
    def test_sizing_matches_legacy_and_limit_never_underestimates_retention(self):
        def legacy(value):
            size = sys.getsizeof(value)
            if isinstance(value, dict):
                size += sum(legacy(k) + legacy(v) for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                size += sum(legacy(v) for v in value)
            return size

        rng = random.Random(42)
        def tree(depth):
            if not depth:
                return rng.choice([None, True, False, 1 << 80, -1, 0.25, 'ü' * 100])
            children = [tree(depth - 1) for _ in range(rng.randrange(5))]
            return {str(i): v for i, v in enumerate(children)} if rng.randrange(2) else children

        for value in [tree(4) for _ in range(30)] + [([1., 2.] * 1000,), {'alias': [None] * 1000}]:
            exact = legacy(value)
            self.assertEqual(object_bytes(value), exact)
            for limit in (0, exact // 2, exact - 1, exact, exact + 1):
                result = object_bytes(value, limit=limit)
                if exact <= limit:
                    self.assertEqual(result, exact)
                else:
                    self.assertGreater(result, limit)
        # Early rejection must avoid traversing a large oversized scalar array.
        value = [float(i) for i in range(10000)]
        with patch('mibot.utils.doosan_io_cache.sys.getsizeof', wraps=sys.getsizeof) as sizing:
            self.assertGreater(object_bytes(value, limit=100), 100)
            self.assertEqual(sizing.call_count, 1)

    def test_exact_content_lru_and_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f'{i}.json' for i in range(3)]
            values = [dict(num_frames=i, nested=[{'x': [0.2, 'a', None, True]}]) for i in range(3)]
            for path, value in zip(paths, values):
                path.write_text(json.dumps(value))
            cache = JsonEpisodeCache(2)
            for index in (0, 1, 0, 2, 0, 1):
                self.assertEqual(cache.read(paths[index]), values[index])
            self.assertEqual(cache.snapshot()['hits'], 2)
            self.assertEqual(cache.snapshot()['misses'], 4)
            self.assertEqual(cache.snapshot()['evictions'], 2)
            self.assertEqual(len(cache.entries), 2)
            cache.clear()
            self.assertEqual(cache.snapshot()['estimated_bytes'], 0)

    def test_byte_budget_oversized_and_pickle_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f'{i}.json' for i in range(2)]
            value = {'rows': [[float(i)] * 12 for i in range(50)]}
            for path in paths:
                path.write_text(json.dumps(value))
            size = object_bytes(json.loads(paths[0].read_text())) + object_bytes(str(paths[0]))
            self.assertGreater(size, paths[0].stat().st_size)
            cache = JsonEpisodeCache(16, size + 1)
            cache.read(paths[0]); cache.read(paths[1])
            self.assertEqual(len(cache.entries), 1)
            self.assertLessEqual(cache.bytes, size + 1)
            restored = pickle.loads(pickle.dumps(cache))
            self.assertEqual(restored.snapshot()['entries'], 0)
            cache.read(paths[1])['rows'][0][0] = -99
            self.assertEqual(restored.read(paths[1]), value)
            tiny = JsonEpisodeCache(16, 1)
            self.assertEqual(tiny.read(paths[0]), value)
            self.assertEqual(tiny.snapshot()['entries'], 0)
            self.assertEqual(tiny.snapshot()['oversized'], 1)
            disabled = JsonEpisodeCache(0)
            disabled.read(paths[0]); disabled.read(paths[0])
            self.assertEqual(disabled.snapshot()['misses'], 2)

    def test_failed_parse_never_cached_and_reporting_is_optional_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.json'
            path.write_text('{')
            cache = JsonEpisodeCache(report_every=128)
            with self.assertRaises(json.JSONDecodeError):
                cache.read(path)
            self.assertFalse(cache.entries)
            path.write_text('{"x": 1}')
            with patch('mibot.utils.doosan_io_cache.mark') as mark:
                for _ in range(257):
                    cache.read(path)
                self.assertEqual(mark.call_count, 2)
            with patch('mibot.utils.doosan_io_cache.mark') as mark:
                silent = JsonEpisodeCache()
                for _ in range(5):
                    silent.read(path)
                mark.assert_not_called()

    def test_sizing_preserves_exact_byte_retention_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.json'
            path.write_text(json.dumps({'rows': [[float(i)] * 7 for i in range(1000)]}))
            value = json.loads(path.read_text())
            size = object_bytes(value) + object_bytes(str(path))
            for budget in (size - 1, size):
                with self.subTest(budget=budget):
                    cache = JsonEpisodeCache(16, budget)
                    self.assertEqual(cache.read(path), value)
                    self.assertEqual(cache.snapshot()['entries'], int(budget == size))
                    self.assertEqual(cache.bytes, size if budget == size else 0)
                    self.assertLessEqual(cache.bytes, budget)


class ZipCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = [make_archive(self.root, str(i)) for i in range(3)]
        self.cache = zipped.ZipArchiveCache(max_open=2)

    def tearDown(self):
        self.cache.close()
        zipped.close_cache()
        zipped._manifest.cache_clear()
        self.tmp.cleanup()

    def test_real_filesystem_operations_initial_hit_and_reopen_for_each_policy(self):
        for policy in ('strict', 'immutable'):
            with self.subTest(policy=policy):
                cache = zipped.ZipArchiveCache(max_open=1, integrity_policy=policy)
                try:
                    original_open = Path.open
                    with patch.object(os, 'stat', wraps=os.stat) as stat, \
                         patch.object(os, 'fstat', wraps=os.fstat) as fstat, \
                         patch.object(zipped.zipfile, 'ZipFile', wraps=zipfile.ZipFile) as opens, \
                         patch.object(Path, 'open', autospec=True, side_effect=original_open) as reads:
                        first = cache.get(self.root, self.rows[0])
                        self.assertEqual((stat.call_count, fstat.call_count, opens.call_count,
                                          reads.call_count), (4, 2, 1, 1))
                        for mock in (stat, fstat, opens, reads):
                            mock.reset_mock()
                        for _ in range(48):
                            self.assertIs(cache.get(self.root, self.rows[0]), first)
                        self.assertEqual((stat.call_count, fstat.call_count, opens.call_count,
                                          reads.call_count), (96 if policy == 'strict' else 0, 0, 0, 0))
                        cache.get(self.root, self.rows[1])
                        self.assertIsNone(first.fp)
                        for mock in (stat, fstat, opens, reads):
                            mock.reset_mock()
                        cache.get(self.root, self.rows[0])
                        self.assertEqual((stat.call_count, fstat.call_count, opens.call_count,
                                          reads.call_count), (2, 1, 1, 0))
                    self.assertEqual(cache.snapshot()['hits'], 48)
                    self.assertEqual(cache.snapshot()['sidecar_reads'], 2)
                    self.assertEqual(cache.snapshot()['opens'], 3)
                finally:
                    cache.close()

    def test_explicit_policy_provenance_pickle_and_reconfiguration(self):
        with self.assertRaisesRegex(ValueError, 'integrity policy'):
            zipped.ZipArchiveCache(integrity_policy='automatic')
        self.assertEqual(self.cache.snapshot()['integrity_policy'], 'strict')
        output = io.StringIO()
        with redirect_stdout(output):
            cache = zipped.configure_cache(integrity_policy='immutable')
        record = json.loads(output.getvalue().strip().split(' ', 1)[1])
        self.assertEqual(record['integrity_policy'], 'immutable')
        self.assertTrue(record['immutable_dataset_required'])
        self.assertEqual(record['hit_path_stats'], 0)
        self.assertEqual(record['pid'], os.getpid())
        archive = cache.get(self.root, self.rows[0])
        restored = pickle.loads(pickle.dumps(cache))
        try:
            self.assertEqual(restored.snapshot()['integrity_policy'], 'immutable')
            self.assertFalse(restored.handles)
            strict = zipped.configure_cache(integrity_policy='strict')
            self.assertIsNone(archive.fp)
            self.assertEqual(strict.snapshot()['integrity_policy'], 'strict')
        finally:
            restored.close()

    def test_replacement_mutation_and_deletion_detection_depends_on_policy(self):
        for policy in ('strict', 'immutable'):
            for mutation in ('archive_replace', 'archive_rewrite', 'sidecar_replace',
                             'sidecar_rewrite', 'archive_delete', 'sidecar_delete'):
                with self.subTest(policy=policy, mutation=mutation), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    row = make_archive(root, 'first')
                    other = make_archive(root, 'other')
                    cache = zipped.ZipArchiveCache(max_open=1, integrity_policy=policy)
                    try:
                        held = cache.get(root, row)
                        path = root / row['archive']
                        if mutation.startswith('sidecar'):
                            path = path.with_suffix('.metadata.json')
                        if mutation.endswith('delete'):
                            path.unlink()
                        elif mutation.endswith('replace'):
                            replacement = root / 'replacement'
                            replacement.write_bytes(path.read_bytes())
                            replacement.replace(path)
                        else:
                            before = path.stat()
                            path.write_bytes(path.read_bytes())
                            # Lustre can report identical ctime after rapid writes.
                            # Deliberately change observable mtime by two seconds,
                            # beyond coarse timestamp resolution, without sleeping.
                            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns - 2_000_000_000))
                            self.assertNotEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
                        if policy == 'immutable':
                            # This is deliberate deferred detection, not proof
                            # that a previously verified file stayed unchanged.
                            self.assertIs(cache.get(root, row), held)
                            self.assertEqual(cache.snapshot()['invalidations'], 0)
                            cache.get(root, other)  # Eviction forces revalidation.
                        if mutation.endswith('delete'):
                            with self.assertRaises(FileNotFoundError):
                                cache.get(root, row)
                        else:
                            self.assertIsNot(cache.get(root, row), held)
                            self.assertEqual(cache.snapshot()['invalidations'], 1)
                            self.assertEqual(cache.snapshot()['sidecar_reads'], 3 if policy == 'immutable' else 2)
                        self.assertIsNone(held.fp)
                    finally:
                        cache.close()

    def test_immutable_hit_still_compares_row_provenance(self):
        cache = zipped.ZipArchiveCache(integrity_policy='immutable')
        try:
            held = cache.get(self.root, self.rows[0])
            for field, value in [('original_source_path', '/wrong'), ('frames', 999),
                                 ('crop_bbox', [1, 2, 3, 4])]:
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeError, 'metadata mismatch'):
                        cache.get(self.root, dict(self.rows[0], **{field: value}))
                    self.assertIsNone(held.fp)
                    held = cache.get(self.root, self.rows[0])
        finally:
            cache.close()

    def test_immutable_atomic_replacement_can_serve_old_valid_pixels_until_eviction(self):
        row = self.rows[0]
        path = self.root / row['archive']
        cache = zipped.ZipArchiveCache(max_open=1, integrity_policy='immutable')
        try:
            held = cache.get(self.root, row)
            old_png = held.read('frame_000000000.png')
            replacement = make_archive(self.root, 'new-content', color=91)
            (self.root / replacement['archive']).replace(path)
            sidecar = path.with_suffix('.metadata.json')
            meta = json.loads(sidecar.read_text())
            sidecar.write_text(json.dumps(dict(meta, archive_bytes=path.stat().st_size)))
            self.assertEqual(cache.get(self.root, row).read('frame_000000000.png'), old_png)
            cache.get(self.root, self.rows[1])
            self.assertNotEqual(cache.get(self.root, row).read('frame_000000000.png'), old_png)
            self.assertEqual(cache.snapshot()['invalidations'], 1)
        finally:
            cache.close()

    def test_both_policies_bound_retention_and_release_all_descriptors(self):
        before = len(list(Path('/proc/self/fd').iterdir()))
        for policy in ('strict', 'immutable'):
            for budget in (1, 20000):
                with self.subTest(policy=policy, budget=budget):
                    cache = zipped.ZipArchiveCache(max_open=1, max_bytes=budget,
                                                   metadata_entries=1, metadata_bytes=1500,
                                                   integrity_policy=policy)
                    try:
                        for _ in range(5):
                            for row in self.rows:
                                archive = cache.get(self.root, row)
                                archive.read('frame_000000000.png')
                                cache.release(archive)
                                info = cache.snapshot()
                                self.assertLessEqual(info['handles'], 1)
                                self.assertLessEqual(info['estimated_bytes'], budget)
                                self.assertLessEqual(info['metadata_entries'], 1)
                                self.assertLessEqual(info['metadata_estimated_bytes'], 1500)
                                self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())), before + 1)
                    finally:
                        cache.close()
                    self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_both_policies_preserve_initial_validation_and_failure_cleanup(self):
        path = self.root / self.rows[0]['archive']
        sidecar = path.with_suffix('.metadata.json')
        valid = sidecar.read_text()
        bad_directory = make_archive(self.root, 'bad-initial', frames=3,
                                     names=['frame_000000000.png', 'wrong.png', 'frame_000000002.png'])
        before = len(list(Path('/proc/self/fd').iterdir()))
        for policy in ('strict', 'immutable'):
            cache = zipped.ZipArchiveCache(integrity_policy=policy)
            try:
                for field, value in [('schema', 'wrong'), ('original_source_path', '/wrong'),
                                     ('frames', 999), ('crop_bbox', [1, 2, 3, 4]), ('archive_bytes', 1)]:
                    with self.subTest(policy=policy, field=field):
                        sidecar.write_text(json.dumps(dict(json.loads(valid), **{field: value})))
                        with self.assertRaisesRegex(RuntimeError, 'metadata mismatch'):
                            cache.get(self.root, self.rows[0])
                        self.assertFalse(cache.handles)
                        self.assertFalse(cache.metadata)
                sidecar.write_text('x' * 65537)
                with self.assertRaisesRegex(RuntimeError, 'oversized'):
                    cache.get(self.root, self.rows[0])
                with self.assertRaisesRegex(RuntimeError, 'directory mismatch'):
                    cache.get(self.root, bad_directory)
                self.assertFalse(cache.handles)
                self.assertFalse(cache.metadata)
            finally:
                cache.close()
                sidecar.write_text(valid)
            self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_lru_closes_eviction_and_reuses_validated_sidecar_on_reopen(self):
        a = self.cache.get(self.root, self.rows[0])
        b = self.cache.get(self.root, self.rows[1])
        self.assertIs(a, self.cache.get(self.root, self.rows[0]))
        c = self.cache.get(self.root, self.rows[2])
        self.assertIsNone(b.fp)
        with patch.object(zipped.json, 'loads', side_effect=AssertionError('unexpected sidecar parse')), \
             patch.object(zipped, '_directory_bytes', side_effect=AssertionError('unexpected remeasurement')):
            reopened = self.cache.get(self.root, self.rows[1])
        self.assertIsNot(b, reopened)
        self.assertIsNone(a.fp)
        info = self.cache.snapshot()
        self.assertEqual((info['hits'], info['misses'], info['opens'], info['reopens']), (1, 4, 4, 1))
        self.assertEqual(info['sidecar_reads'], 3)
        self.cache.close()
        self.assertIsNone(c.fp)
        self.assertIsNone(reopened.fp)

    def test_metadata_count_and_byte_eviction(self):
        for options in (dict(metadata_entries=1), dict(metadata_bytes=1)):
            with self.subTest(options=options):
                cache = zipped.ZipArchiveCache(max_open=1, **options)
                try:
                    for row in (self.rows[0], self.rows[1], self.rows[0]):
                        cache.get(self.root, row)
                    self.assertEqual(cache.snapshot()['sidecar_reads'], 3)
                    self.assertLessEqual(len(cache.metadata), 1)
                finally:
                    cache.close()

    def test_directory_memory_budget_and_uncached_cleanup(self):
        archive = self.cache.get(self.root, self.rows[0])
        budget = zipped._directory_bytes(archive)
        cache = zipped.ZipArchiveCache(max_open=32, max_bytes=budget)
        try:
            first = cache.get(self.root, self.rows[0])
            cache.get(self.root, self.rows[1])
            self.assertIsNone(first.fp)
            self.assertLessEqual(cache.bytes, budget)
        finally:
            cache.close()
        cache = zipped.ZipArchiveCache(max_bytes=1)
        try:
            archive = cache.get(self.root, self.rows[0])
            self.assertFalse(cache.handles)
            self.assertEqual(cache.snapshot()['oversized'], 1)
            cache.release(archive)
            self.assertIsNone(archive.fp)
        finally:
            cache.close()

    def test_sidecar_changes_missing_corrupt_and_provenance_rejected(self):
        path = self.root / self.rows[0]['archive']
        sidecar = path.with_suffix('.metadata.json')
        valid = sidecar.read_text()
        for corrupt in ('{', 'null', json.dumps(dict(json.loads(valid), frames=999)),
                        json.dumps(dict(json.loads(valid), crop_bbox=[1, 2, 3, 4])),
                        json.dumps(dict(json.loads(valid), schema='wrong')), 'x' * 65537):
            with self.subTest(corrupt=corrupt[:40]):
                sidecar.write_text(valid)
                previous = self.cache.get(self.root, self.rows[0])
                sidecar.write_text(corrupt)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.cache.get(self.root, self.rows[0])
                self.assertIsNone(previous.fp)
        sidecar.write_text(valid)
        archive = self.cache.get(self.root, self.rows[0])
        sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache.get(self.root, self.rows[0])
        self.assertIsNone(archive.fp)

    def test_atomic_same_size_archive_replacement_revalidates(self):
        row = self.rows[0]
        old = self.cache.get(self.root, row)
        path = self.root / row['archive']
        before = path.stat()
        replacement = self.root / 'replacement'
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
        new = self.cache.get(self.root, row)
        self.assertIsNone(old.fp)
        self.assertIsNot(old, new)
        self.assertEqual(self.cache.snapshot()['sidecar_reads'], 2)
        self.assertEqual(self.cache.snapshot()['invalidations'], 1)

    def test_partial_directory_and_read_crc_corruption_rejected(self):
        bad = make_archive(self.root, 'bad', frames=3,
                           names=['frame_000000000.png', 'wrong.png', 'frame_000000002.png'])
        with self.assertRaisesRegex(RuntimeError, 'directory'):
            self.cache.get(self.root, bad)
        row = self.rows[0]
        path = self.root / row['archive']
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('frame_000000000.png')
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        raw = bytearray(path.read_bytes())
        raw[offset + 10] ^= 1
        path.write_bytes(raw)
        archive = self.cache.get(self.root, row)
        with self.assertRaises(zipfile.BadZipFile):
            archive.read('frame_000000000.png')
        path.write_bytes(raw[:40])
        with self.assertRaises(RuntimeError):
            self.cache.get(self.root, row)
        self.assertIsNone(archive.fp)

    def test_same_size_in_place_rewrite_with_restored_mtime_invalidates(self):
        row = self.rows[0]
        path = self.root / row['archive']
        archive = self.cache.get(self.root, row)
        before = path.stat()
        raw = path.read_bytes()
        path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.cache.get(self.root, row)
        self.assertIsNone(archive.fp)
        self.assertEqual(self.cache.snapshot()['sidecar_reads'], 2)

    def test_reopen_uses_two_path_stats_and_hit_never_opens_sidecar(self):
        self.cache.get(self.root, self.rows[0])
        original_stat = Path.stat
        with patch.object(Path, 'stat', autospec=True, side_effect=original_stat) as stat, \
             patch.object(Path, 'open', side_effect=AssertionError('unexpected sidecar open')):
            self.cache.get(self.root, self.rows[0])
            self.assertEqual(stat.call_count, 2)
        self.cache.get(self.root, self.rows[1])
        self.cache.get(self.root, self.rows[2])
        with patch.object(Path, 'stat', autospec=True, side_effect=original_stat) as stat, \
             patch.object(Path, 'open', side_effect=AssertionError('unexpected sidecar open')):
            self.cache.get(self.root, self.rows[0])
            self.assertEqual(stat.call_count, 2)

    def test_no_fd_leak_on_success_validation_failure_or_reconfiguration(self):
        if not Path('/proc/self/fd').exists():
            self.skipTest('Linux descriptor accounting required')
        before = len(list(Path('/proc/self/fd').iterdir()))
        for _ in range(10):
            for row in self.rows:
                self.cache.get(self.root, row)
            self.cache.close()
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)
        bad = make_archive(self.root, 'bad', frames=3, names=['bad0.png', 'bad1.png', 'bad2.png'])
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                self.cache.get(self.root, bad)
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)
        cache = zipped.configure_cache(max_open=2)
        opened = cache.get(self.root, self.rows[0])
        zipped.configure_cache(max_open=3)
        self.assertIsNone(opened.fp)

    def test_rlimit_cap_and_spawn_pickle_drops_handles(self):
        with patch.object(zipped.resource, 'getrlimit', return_value=(80, 80)):
            cache = zipped.ZipArchiveCache(max_open=32)
            self.assertEqual(cache.max_open, 16)
        with patch.object(zipped.resource, 'getrlimit', return_value=(64, 64)):
            with self.assertRaisesRegex(RuntimeError, 'RLIMIT_NOFILE'):
                zipped.ZipArchiveCache()
        self.cache.get(self.root, self.rows[0])
        restored = pickle.loads(pickle.dumps(self.cache))
        self.assertFalse(restored.handles)
        self.assertFalse(restored.metadata)
        restored.close()

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'fork unavailable')
    def test_actual_fork_discards_parent_handles_and_mutable_json(self):
        path = self.root / 'episode.json'
        path.write_text('{"num_frames": 5}')
        cache = JsonEpisodeCache()
        cache.read(path)['num_frames'] = 999
        zip_cache = zipped.configure_cache(max_open=2)
        parent_handle = zip_cache.get(self.root, self.rows[0])
        context = mp.get_context('fork')
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=fork_check, args=(cache, path, self.root, self.rows[0], child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(15))
            empty, value, json_stats, zip_stats, png = parent.recv()
            self.assertTrue(empty)
            self.assertEqual(value, {'num_frames': 5})
            self.assertEqual(json_stats['misses'], 1)
            self.assertEqual(zip_stats['opens'], 1)
            self.assertEqual(json_stats['pid'], process.pid)
            self.assertEqual(zip_stats['pid'], process.pid)
            self.assertEqual(png, b'\x89PNG\r\n\x1a\n')
            self.assertIsNotNone(parent_handle.fp)
            self.assertEqual(cache.read(path)['num_frames'], 999)
        finally:
            process.join(5)
            if process.is_alive():
                process.kill(); process.join()
            parent.close()
        self.assertEqual(process.exitcode, 0)


class DatasetCacheIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'train'
        self.data.mkdir()
        self.rows = []
        self.original = {}
        for ep in range(3):
            trajectory = test_doosan_dataset.DoosanDatasetContractTest.trajectory()
            trajectory['cameras'] = {}
            for camera in ('external_camera_2', 'tcp_camera'):
                row = make_archive(self.root, f'{ep}-{camera}', color=ep * 31)
                row['camera'] = camera
                self.rows.append(row)
                trajectory['cameras'][camera] = [{'path': row['original_source_path'], 'start': 0}]
            trajectory['instruction'] = {'general': [dict(
                conversations=[{'from': 'human', 'value': f'Task {option}: <image><image>'},
                               {'from': 'gpt', 'value': 'do task'}],
                images=['cameras.external_camera_2', 'cameras.tcp_camera']) for option in range(2)]}
            path = self.data / f'{ep}.json'
            path.write_text(json.dumps(trajectory))
            self.original[str(path)] = trajectory
        # Exercise the real frozen 646-row manifest guard, without making 646 ZIPs.
        manifest = list(self.rows)
        for i in range(646 - len(manifest)):
            manifest.append(dict(self.rows[0], archive=f'unused-{i}.zip', original_source_path=f'/unused/{i}'))
        (self.root / 'media_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in manifest))
        self.env = patch.dict(os.environ, XR1_DOOSAN_ZIP_CACHE_ROOT=str(self.root),
                              XR1_DOOSAN_OFFLINE_FRAME_ROOT='', WORLD_SIZE='1')
        self.env.start()
        self.params = dict(max_steps=4, train_datasets=dict(
            paths=[str(self.data)], batch_size=24, sampler_seed=42, deterministic_replay=True,
            mean=[[0.] * 60 for _ in range(30)], std=[[1.] * 60 for _ in range(30)],
            q01=[[-1.] * 60], q99=[[1.] * 60]))

    def tearDown(self):
        zipped.close_cache()
        zipped._manifest.cache_clear()
        self.env.stop()
        self.tmp.cleanup()

    def test_raw_rgb_matches_uncached_zip_and_invalid_member_fails(self):
        trajectory = self.original[str(self.data / '0.json')]
        keys = ['cameras.external_camera_2', 'cameras.tcp_camera']
        for policy, capacity in ((p, c) for p in ('strict', 'immutable') for c in (1, 32)):
            zipped.configure_cache(max_open=capacity, integrity_policy=policy)
            for frame in (0, 4, 0):
                actual = zipped.decode_images(trajectory, keys, frame)
                for row, image in zip(self.rows[:2], actual):
                    with zipfile.ZipFile(self.root / row['archive']) as archive:
                        with Image.open(io.BytesIO(archive.read(f'frame_{frame:09d}.png'))) as reference:
                            self.assertEqual(canonical(image), canonical(reference.convert('RGB')))
        with self.assertRaises(IndexError):
            zipped.decode_images(trajectory, keys, 5)

    def test_decode_crc_failure_closes_oversized_uncached_handle(self):
        trajectory = self.original[str(self.data / '0.json')]
        row = self.rows[0]
        path = self.root / row['archive']
        # PNG remains readable; corrupt trailing ZIP member bytes must still
        # fail CRC even if PIL stops before consuming those bytes.
        raw = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('frame_000000000.png')
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
            raw[offset + info.file_size - 1] ^= 1
        path.write_bytes(raw)
        for policy in ('strict', 'immutable'):
            for budget in (1, 128 * 1024**2):
                with self.subTest(policy=policy, budget=budget):
                    cache = zipped.configure_cache(max_bytes=budget, integrity_policy=policy)
                    before = len(list(Path('/proc/self/fd').iterdir()))
                    try:
                        with self.assertRaises(zipfile.BadZipFile):
                            zipped.decode_images(trajectory, ['cameras.external_camera_2'], 0)
                        self.assertEqual(len(cache.handles), 0 if budget == 1 else 1)
                    finally:
                        cache.close()
                    self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_sample_identities_order_and_all_training_outputs_equal(self):
        dataset = DoosanJsonDataset(self.params)
        self.assertEqual(dataset._json_cache.snapshot()['entries'], 0)
        population = [dict(file=path, frame_index=i) for path in sorted(self.original) for i in range(5)]
        rng = random.Random(42)
        expected = []
        while len(expected) < len(dataset):
            cycle = list(range(len(population))); rng.shuffle(cycle)
            expected.extend(population[i] for i in cycle[:len(dataset) - len(expected)])
        self.assertEqual(dataset.samples[:], expected)
        baseline = []
        # Original two-entry/12-handle policy and a no-cache JSON oracle.
        dataset._json_cache = JsonEpisodeCache(2)
        dataset._zip_cache_options['max_open'] = 12
        indices = [0, 3, 12, 20, 45, 0]
        for index in indices:
            baseline.append(canonical(dataset[index]))
        dataset._json_cache = JsonEpisodeCache(16)
        dataset._zip_cache_options['max_open'] = 32
        dataset._zip_cache_options['integrity_policy'] = 'immutable'
        rng_state = random.getstate()
        self.assertEqual([canonical(dataset[i]) for i in indices], baseline)
        self.assertEqual(random.getstate(), rng_state)
        with patch.object(dataset, '_read_json', side_effect=lambda path: json.loads(Path(path).read_text())):
            self.assertEqual([canonical(dataset[i]) for i in indices], baseline)
        for path, original in self.original.items():
            self.assertEqual(dataset._read_json(path), original)

    def test_real_four_worker_ordered_loading_matches_zero_worker_outputs(self):
        dataset = DoosanJsonDataset(self.params)
        baseline = None
        for workers, entries in ((0, 2), (4, 2), (4, 16)):
            dataset._json_cache = JsonEpisodeCache(entries)
            dataset._zip_cache_options['max_open'] = 12 if entries == 2 else 32
            dataset._zip_cache_options['integrity_policy'] = 'strict' if entries == 2 else 'immutable'
            sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
            loader = DataLoader(dataset, batch_size=24, sampler=sampler, num_workers=workers,
                                collate_fn=cpu_collate, worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(42),
                                **(dict(prefetch_factor=2, persistent_workers=True, in_order=True,
                                        timeout=600) if workers else {}))
            iterator = None
            try:
                iterator = iter(loader)
                outputs = list(iterator)
                if baseline is None:
                    baseline = outputs
                else:
                    self.assertEqual(outputs, baseline)
            finally:
                shutdown_loader(loader, iterator)

    def test_effective_production_loader_settings(self):
        recipe = yaml.safe_load((ROOT / 'configs/claix_b24_production.yaml').read_text())
        params = Config(dict(max_steps=4, train_datasets=recipe['data']['params']['train_datasets']))
        with patch('mibot.data.datamodule.base_datamodule.CustomCollate'):
            module = DoosanDataModule(params)
        module.train_set = range(4 * 24)
        # Exercise the actual production DDP/sampler branch without networking.
        with patch('mibot.data.datamodule.doosan_datamodule.dist.is_available', return_value=True), \
             patch('mibot.data.datamodule.doosan_datamodule.dist.is_initialized', return_value=True), \
             patch('mibot.data.datamodule.doosan_datamodule.dist.get_world_size', return_value=1), \
             patch('mibot.data.datamodule.doosan_datamodule.dist.get_rank', return_value=0), \
             patch('mibot.data.datamodule.doosan_datamodule.shared_memory_budget'):
            loader = module.train_dataloader()
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.timeout, 600)
        self.assertEqual(loader.num_workers, 4)
        self.assertEqual(loader.prefetch_factor, 2)
        self.assertTrue(loader.persistent_workers)
        self.assertTrue(loader.in_order)
        self.assertEqual(loader.batch_size, 24)
        self.assertEqual(loader.sampler.offset, 0)
        # Configured cache policy reaches the dataset, not just a YAML declaration.
        merged = dict(self.params['train_datasets'], **dict(params.train_datasets))
        dataset = DoosanJsonDataset(dict(max_steps=4, train_datasets=merged))
        self.assertEqual(dataset._json_cache.policy[:2], (16, 268435456))
        self.assertEqual(dataset._zip_cache_options['max_open'], 32)
        self.assertEqual(dataset._zip_cache_options['integrity_policy'], 'immutable')
        dataset[0]
        self.assertEqual(zipped.cache_info()['max_open'], 32)
        self.assertEqual(zipped.cache_info()['integrity_policy'], 'immutable')


if __name__ == '__main__':
    unittest.main(verbosity=2)
