#!/usr/bin/env python3
"""Bounded CPU replay diagnostics; never import this into production training.

Only the CPU probe installs ZIP progress wrappers. No source checks are patched.
Each worker writes a fixed current-status slot and pre-timeout Python traces on
local storage; the parent retains a bounded diagnostic bundle on completion.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

from common import (ROOT, COMPACT_CONFIG, MEDIA_MANIFEST, INVENTORY, SOURCE,
                    read_json, require, source_check, sha, write_new)

_state = {}
_directory = None
_trace = None
_alarm_seconds = 60.


def mark(phase, **fields):
    if _directory is None:
        return
    _state.update(fields, phase=phase, pid=os.getpid(), monotonic=time.monotonic())
    raw = json.dumps(_state, sort_keys=True)
    require(len(raw) < 8192, 'Diagnostic status overflow')
    target = _directory / f'worker-{os.getpid()}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(raw + '\n')
    os.replace(temporary, target)


def instrument_zip():
    """CPU process-local observation around the original calls, with identical returns."""
    original_open = zipfile.ZipFile.open
    original_read = zipfile.ZipExtFile.read
    def opened(archive, name, *args, **kwargs):
        mark('zip_member_open', archive=str(archive.filename), member=str(name))
        result = original_open(archive, name, *args, **kwargs)
        mark('zip_member_open_done')
        return result
    def read(payload, *args, **kwargs):
        mark('zip_read', member=str(payload.name))
        result = original_read(payload, *args, **kwargs)
        mark('zip_read_done')
        return result
    zipfile.ZipFile.open, zipfile.ZipExtFile.read = opened, read
    # Import may be unavailable for generic worker hang tests; dataset CPU CLI requires it.
    if 'mibot.data.datasets.doosan_zip_images' in sys.modules:
        module = sys.modules['mibot.data.datasets.doosan_zip_images']
        original = module._get_archive
        def archive(root, row):
            mark('archive_open_or_cache_lookup', archive=str(root / row['archive']),
                 camera=row['camera'], source=row['original_source_path'])
            result = original(root, row)
            mark('archive_ready')
            return result
        module._get_archive = archive


class WorkerInit:
    def __init__(self, directory, alarm_seconds=60.):
        self.directory, self.alarm_seconds = str(directory), alarm_seconds

    def __call__(self, worker_id):
        global _directory, _trace, _alarm_seconds, _state
        import torch
        import random
        import numpy as np
        _directory = Path(self.directory)
        _alarm_seconds = self.alarm_seconds
        _state = dict(worker_id=worker_id)
        _trace = (_directory / f'worker-{os.getpid()}.trace').open('x')
        faulthandler.enable(file=_trace, all_threads=True)
        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)
        if os.environ.get('XR1_CPU_PROBE_DOOSAN') == '1':
            __import__('mibot.data.datasets.doosan_zip_images')
        instrument_zip()
        mark('worker_ready')


class ObservedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.samples[index] if hasattr(self.dataset, 'samples') else {}
        mark('sample_start', dataset_index=int(index), episode=sample.get('file'),
             frame=sample.get('frame_index'), archive=None, member=None, camera=None, source=None)
        faulthandler.dump_traceback_later(_alarm_seconds, file=_trace, repeat=False)
        try:
            result = self.dataset[index]
            mark('sample_done')
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()


class ObservedCollate:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, items):
        mark('collate_start')
        faulthandler.dump_traceback_later(_alarm_seconds, file=_trace, repeat=False)
        try:
            result = self.collate(items)
            mark('collate_done_awaiting_queue')
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()


def bounded_replay(dataset, collate, batches, local, *, timeout=180., alarm_seconds=60., transport='tensor', on_batch=None):
    import inspect
    import torch
    from torch.utils.data import DataLoader
    from observe import snapshot
    require(0 < alarm_seconds < timeout and 1 <= len(batches) <= 16
            and all(1 <= len(batch) <= 24 for batch in batches), 'Invalid bounded probe window')
    require('in_order' in inspect.signature(DataLoader).parameters, 'Torch lacks explicit ordered delivery')
    local = Path(local)
    local.mkdir(mode=0o700, parents=False)
    # Spawn avoids inheriting tokenizer/thread/ZIP handles. It is a diagnostic mode,
    # not a claim to reproduce production's default Linux fork lifecycle.
    loader = DataLoader(ObservedDataset(dataset), batch_sampler=batches, num_workers=4,
                        prefetch_factor=2, persistent_workers=True, pin_memory=False, in_order=True,
                        multiprocessing_context='spawn', timeout=timeout,
                        generator=torch.Generator().manual_seed(42),
                        worker_init_fn=WorkerInit(local, alarm_seconds), collate_fn=ObservedCollate(collate))
    iterator = None
    delivered = []
    report = dict(status='FAIL', transport=transport, pin_memory=False, multiprocessing_context='spawn')
    main_trace = (local/'parent.trace').open('x')
    try:
        iterator = iter(loader)
        report['worker_pids'] = [worker.pid for worker in iterator._workers]
        write_new(local/'workers.json', report['worker_pids'])
        for index in range(len(batches)):
            faulthandler.dump_traceback_later(alarm_seconds, file=main_trace, repeat=False)
            try:
                batch = next(iterator)
            finally:
                faulthandler.cancel_dump_traceback_later()
            if on_batch is not None:
                on_batch(batch)
            delivered.append(dict(batch=index, planned_indices=batches[index], timestamp=time.time()))
            del batch
        report['status'] = 'PASS'
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), process_snapshot=snapshot(os.getpid()))
    finally:
        report['delivered'] = delivered
        write_new(local/'result.json', report)
        if iterator is not None:
            iterator._shutdown_workers()
        loader._iterator = None
        main_trace.close()
    return report


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='', MAX_LENGTH='12288', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      XR1_CPU_PROBE_DOOSAN='1', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Original resolved_compact_config.json')
    p.add_argument('--local', required=True, help='New directory on allocation-local scratch')
    p.add_argument('--output', required=True, help='New persistent diagnostic directory')
    p.add_argument('--first-step', type=int, default=8751)
    p.add_argument('--batches', type=int, default=16)
    args = p.parse_args()
    require(1 <= args.batches <= 16 and 8751 <= args.first_step <= 8766, 'Only bounded incident/recovery window allowed')
    source_check()
    require(sha(args.config) == COMPACT_CONFIG, 'Wrong original resolved compact configuration')
    local = Path(args.local).resolve()
    require(not str(local).startswith(('/hpcwork/', '/rwthfs/', '/work/', '/home/')), 'Progress slots require local scratch')
    require(not Path(args.output).exists(), 'Output already exists')
    sys.path.insert(0, str(ROOT/'xr1'))
    import torch
    from mmengine import Config, DATASETS
    import mibot.data
    import mibot.data.datasets.doosan_zip_images
    from mibot.utils.claix_contract import split_inventory
    from mibot.utils.doosan_replay import RankOffsetSampler
    from torch.utils.data import DistributedSampler
    conf = read_json(args.config)
    require(split_inventory(conf['data']['params']['train_datasets']['paths'][0])['sha256'] == INVENTORY,
            'CPU probe dataset differs from original inventory')
    cache = Path(os.environ['XR1_DOOSAN_ZIP_CACHE_ROOT']).resolve(strict=True)
    require(sha(cache/'media_manifest.jsonl') == MEDIA_MANIFEST, 'Wrong CPU probe cache manifest')
    require(not os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'), 'Conflicting media backend')
    dm = DATASETS.build(Config(conf['data']))
    dm.setup('fit')
    base = DistributedSampler(dm.train_set, num_replicas=1, rank=0, shuffle=True, seed=42)
    sampler = iter(RankOffsetSampler(base, (args.first_step-1)*24, batch_size=24))
    batches = [[next(sampler) for _ in range(24)] for _ in range(args.batches)]
    report = bounded_replay(dm.train_set, dm.collate_fn, batches, local)
    write_new(local/'execution.json', dict(source_sha256=SOURCE, config_sha256=COMPACT_CONFIG,
              inventory_sha256=INVENTORY, manifest_sha256=MEDIA_MANIFEST, cache=str(cache),
              torch_version=torch.__version__, python_version=sys.version,
              slurm_job_id=os.environ.get('SLURM_JOB_ID'), first_step=args.first_step,
              batches=args.batches, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    shutil.copytree(local, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
