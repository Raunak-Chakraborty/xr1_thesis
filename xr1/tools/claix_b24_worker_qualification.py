"""Real-data CPU worker qualification, for an allocated CLAIX compute node.

Uses the production DataModule and frozen 48284-step stream. Does not load XR1.
No PASS report is published until all worker/resume comparisons have completed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def batch_digest(batch):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            h.update(str((value.dtype,tuple(value.shape))).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def fetch_batches(dm, step, count, stop):
    """Measure every complete batch, closing workers even on faults/deadline."""
    from mibot.data.datamodule.doosan_datamodule import shutdown_loader, shared_memory_budget
    import torch
    dm.resume_step = step
    loader = iterator = None
    hashes, timings, sizes = [], [], []
    started = time.perf_counter()
    try:
        if stop.due():
            raise RuntimeError('CPU qualification interrupted before completion; no PASS')
        loader = dm.train_dataloader()
        iterator = iter(loader)
        for _ in range(count):
            if stop.due():
                raise RuntimeError('CPU qualification interrupted before completion; no PASS')
            fetched = time.perf_counter()
            batch = next(iterator)
            timings.append(time.perf_counter()-fetched)
            if stop.due():
                raise RuntimeError('CPU qualification interrupted before completion; no PASS')
            if (batch['action_segments'].shape[0] != 24 or batch['image_grid_thw'].shape[0] != 48
                    or batch['input_ids'].shape[1] > 9912):
                raise RuntimeError('Real batch-24 token/view/sample contract failed')
            hashes.append(batch_digest(batch))
            sizes.append(sum(v.numel()*v.element_size() for v in batch.values() if isinstance(v, torch.Tensor)))
        elapsed = time.perf_counter()-started
        memory = shared_memory_budget(max(sizes), workers=dm.num_workers,
                                      prefetch_factor=dm.prefetch_factor)
        try:
            import psutil
            parent = psutil.Process()
            memory['parent_rss_bytes'] = parent.memory_info().rss
            memory['worker_rss_bytes'] = sum(p.memory_info().rss for p in parent.children(recursive=True))
        except (ImportError, OSError):
            memory['parent_rss_bytes'] = memory['worker_rss_bytes'] = None
        return dict(hashes=hashes, fetch_seconds=timings, batches=count, examples=count*24,
                    elapsed_seconds=elapsed, examples_per_second=count*24/elapsed,
                    max_batch_tensor_bytes=max(sizes), memory=memory)
    finally:
        shutdown_loader(loader, iterator)
        del iterator, loader
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overrides',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this real-data probe only inside a compute allocation')
    os.environ.update(MAX_LENGTH='12288',WORLD_SIZE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import torch.distributed as dist
    from hydra import compose, initialize_config_dir
    from mmengine import Config
    from omegaconf import OmegaConf
    from mibot.data.datamodule.doosan_datamodule import DoosanDataModule, shared_memory_budget
    from mibot.utils.claix_runtime import StopRequest
    from mibot.utils.claix_checkpoint import atomic_json
    from mibot.utils.claix_contract import split_inventory
    stop = StopRequest(float(os.environ['XR1_SAFE_STOP_EPOCH']))
    stop.install()
    try:
        torch.set_num_threads(1)
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
            cfg=compose(config_name='claix_b24_production',overrides=json.loads(Path(args.overrides).read_text()))
        params=Config(OmegaConf.to_container(cfg.data.params,resolve=True))
        inventory=split_inventory(params.train_datasets.paths[0])
        if inventory['episodes']!=255 or inventory['frames']!=289703:
            raise RuntimeError('Approved split counts mismatch')
        params.train_datasets.pin_memory=False
        params.train_datasets.worker_timeout=60
        cases={}
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group('gloo',init_method='file://'+folder+'/init',rank=0,world_size=1)
            try:
                for workers in (0,4,8):
                    params.train_datasets.num_workers=workers
                    params.train_datasets.resume_step=0
                    dm=DoosanDataModule(params);dm.setup('fit')
                    if cases:
                        shared_memory_budget(cases['0']['full']['max_batch_tensor_bytes'],
                                             workers=workers, prefetch_factor=dm.prefetch_factor)
                    full=fetch_batches(dm,0,4,stop);resumed=fetch_batches(dm,2,2,stop)
                    if full['hashes'][2:]!=resumed['hashes']:
                        raise RuntimeError(f'Resume mismatch workers={workers}')
                    if cases and full['hashes']!=cases['0']['full']['hashes']:
                        raise RuntimeError(f'Cross-worker mismatch workers={workers}')
                    cases[str(workers)]=dict(full=full,resumed=resumed)
                    del dm
                    gc.collect()
            finally:
                dist.destroy_process_group()
        if stop.due():
            raise RuntimeError('CPU qualification interrupted; no PASS')
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(output,dict(schema='claix-b24-real-worker-qualification-v1',status='PASS',
                                dataset_sha256=inventory['sha256'],cases=cases,
                                torch=torch.__version__,deadline_epoch=stop.deadline,
                                prefetch_factor=params.train_datasets.prefetch_factor,scope='CPU fetch/content only; no GPU throughput claim'))
    finally:
        stop.restore()


if __name__=='__main__':
    main()
