#!/usr/bin/env python3
"""Eight-sample offline equivalence qualification using actual dataset/collator.

The immutable local JSON split and cached Qwen processor are real. Camera PNGs
are conspicuously synthetic fixtures encoded in the production ZIP schema. This
tests code semantics/serialization, not actual camera integrity or throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile


def build_fixture(dataset, split_root, output):
    from PIL import Image
    root = Path(output)
    root.mkdir(parents=True)
    needed = {dataset.samples[i]['file'] for i in range(len(dataset))}
    rows = []
    for path in sorted(Path(split_root).rglob('*.json')):
        trajectory = json.loads(path.read_text())
        for camera, infos in trajectory['videos'].items():
            for info in infos:
                source = info['path']
                name = hashlib.sha256(source.encode()).hexdigest()
                frames = int(trajectory['num_frames']) + int(info.get('start', 0))
                row = dict(original_source_path=source, camera=camera,
                           archive='archives/' + name + '.zip', frames=frames,
                           crop_bbox=info.get('crop_bbox'))
                rows.append(row)
                if str(path) not in needed:
                    continue
                target = root / row['archive']
                target.parent.mkdir(exist_ok=True)
                image = Image.new('RGB', (320, 320), tuple(bytes.fromhex(name[:6])))
                # Spatial texture makes changes in image processing observable.
                for i in range(0, 320, 8):
                    image.paste((i % 255, (i*3) % 255, (i*7) % 255), (i, 0, i+4, 320))
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
                    for frame in range(frames):
                        archive.writestr(f'frame_{frame:09d}.png', buffer.getvalue())
                metadata = dict(schema='doosan-xr1-rgb-png-zip-v1',
                                original_source_path=source, frames=frames,
                                crop_bbox=row['crop_bbox'], archive_bytes=target.stat().st_size,
                                archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                target.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    if len(rows) != 646:
        raise RuntimeError('Fixture manifest does not cover the original 323 episodes/two cameras')
    (root/'media_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (root/'SYNTHETIC_CAMERA_FIXTURE.txt').write_text('Synthetic PNGs. Never use for training or camera provenance.\n')


def tensor_digest(value):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(value.items()):
        h.update(name.encode())
        if isinstance(tensor, (int, float, bool)):
            h.update(json.dumps(tensor).encode())
            continue
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Unexpected non-tensor collator output: ' + name)
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='xr1 Python source directory, baseline or candidate')
    p.add_argument('--config', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--build-fixture', action='store_true')
    p.add_argument('--method', choices=('default', 'fork', 'spawn'), default='default')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--resume-step', type=int, default=0)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.resume_step not in (0, 2):
        raise ValueError('Only the tiny fixture resume boundary is supported')
    sys.path.insert(0, str(Path(args.source).resolve()))
    os.environ.update(CUDA_VISIBLE_DEVICES='', MAX_LENGTH='12288', WORLD_SIZE='1',
                      RANK='0', LOCAL_RANK='0', TOKENIZERS_PARALLELISM='false')
    import torch
    import torch.distributed as dist
    from mmengine import Config
    from mibot.data.datamodule.doosan_datamodule import DoosanDataModule, shutdown_loader
    from mibot.utils.doosan_replay import seed_worker
    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    params = json.loads(Path(args.config).read_text())['data']['params']
    params['max_steps'] = 4
    data = params['train_datasets']
    data.update(paths=[str(Path(args.split_root).resolve()/'train')], batch_size=2, num_workers=4,
                persistent_workers=True, pin_memory=False, prefetch_factor=2, in_order=True,
                deterministic_replay=True, resume_step=args.resume_step, worker_timeout=30)
    if args.method != 'default':
        data['multiprocessing_context'] = args.method
    if args.diagnostics:
        data['diagnostics'] = dict(local_dir=str(output.parent), persistent_dir=str(output.parent/'persistent'),
                                   reserve_sigusr2=True, thresholds=[10, 15, 20], poll_seconds=.25)
    started = time.monotonic()
    dm = DoosanDataModule(Config(params))
    dm.setup('fit')
    if args.build_fixture:
        build_fixture(dm.train_set, args.split_root, args.fixture)
    os.environ['XR1_DOOSAN_ZIP_CACHE_ROOT'] = str(Path(args.fixture).resolve())
    # Real class instances and actual processor, no mocks or transport substitution.
    serialization = {}
    for name, value in (('dataset', dm.train_set), ('collator', dm.collate_fn), ('worker_init', seed_worker)):
        blob = pickle.dumps(value)
        restored = pickle.loads(blob)
        serialization[name] = dict(bytes=len(blob), type=type(restored).__module__+'.'+type(restored).__qualname__)
    dist.init_process_group('gloo', init_method='file://' + str(output.with_suffix('.rendezvous')),
                            rank=0, world_size=1)
    loader, iterator = None, None
    hashes, waits = [], []
    try:
        loader = dm.train_dataloader()
        indices = list(loader.sampler)
        iterator = iter(loader)
        for _ in range(4 - args.resume_step):
            before = time.monotonic()
            batch = next(iterator)
            waits.append(time.monotonic()-before)
            hashes.append(tensor_digest(batch))
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError('Unexpected samples beyond fixture horizon')
    finally:
        shutdown_loader(loader, iterator)
        dist.destroy_process_group()
    result = dict(schema='xr1-local-pipeline-equivalence-v1', source=args.source,
                  torch_version=torch.__version__, method=args.method, diagnostics=args.diagnostics,
                  resume_step=args.resume_step, sample_indices=indices, batch_sha256=hashes,
                  serialization=serialization, batch_wait_seconds=waits,
                  elapsed_seconds=time.monotonic()-started, camera_content='SYNTHETIC_FIXTURE',
                  model_initialized=False, cuda_initialized=torch.cuda.is_initialized())
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
