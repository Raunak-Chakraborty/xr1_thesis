"""Explicit opt-in DDP Doosan XR1 compact FP32 training entrypoint."""
from __future__ import annotations

# Configure cuBLAS before any torch CUDA context is initialized. The command
# launcher must also set this before process start (checked below).
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
    raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')

import json
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from mmengine import Config, DATASETS
from omegaconf import DictConfig, OmegaConf

import mibot.data
import mibot.models.runner.doosan_compact_runner  # registration (opt-in)
from mibot.models import MIMODEL
from mibot.utils.doosan_replay import make_contract, validate_full_compact_resume
from mibot.utils.claix_checkpoint import load_verified
from mibot.utils.claix_contract import BASELINE, resume_fingerprint, split_inventory
from mibot.utils.claix_runtime import ProductionRuntime, VerifiedCheckpointIO

def _resume_preflight(path, contract):
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    offset = validate_full_compact_resume(ck, contract)
    return int(ck['global_step']), offset


@hydra.main(config_path='../configs', config_name='doosan_compact_fp32', version_base=None)
def main(cfg: DictConfig):
    conf = Config(OmegaConf.to_container(cfg, resolve=True))
    source_config = OmegaConf.to_container(cfg, resolve=True)
    production = conf.pop('production', None)
    fingerprint = None
    if production:
        max_length = int(production['max_length'])
        if max_length != 12288 or os.environ.get('MAX_LENGTH', '12288') != '12288':
            raise RuntimeError('Production MAX_LENGTH must be 12288 before collator creation')
        os.environ['MAX_LENGTH'] = str(max_length)
        if int(conf.model.params.last_n) != 1 or int(conf.data.params.train_datasets.batch_size) != 24:
            raise RuntimeError('Production requires approved last_n=1 and batch=24')
        if str(conf.model.params.source_commit) != BASELINE:
            raise RuntimeError('Production source baseline identity changed')
        if (int(conf.data.params.max_steps) != 48284
                or int(conf.trainer.scheduler.params.num_training_steps) != 48284
                or int(conf.trainer.devices) != 1):
            raise RuntimeError('Production plan requires one GPU and frozen provisional 48284-step horizon')
        roots = conf.data.params.train_datasets.paths
        if len(roots) != 1:
            raise RuntimeError('Production requires one frozen train split root')
        inventory = split_inventory(roots[0])
        if inventory['episodes'] != 255 or inventory['frames'] != 289703:
            raise RuntimeError('Production frozen training split count mismatch')
        fingerprint = resume_fingerprint(conf.to_dict(), inventory, max_length)
        import time
        deadline = float(os.environ['XR1_SAFE_STOP_EPOCH'])
        if not time.time() < deadline:
            raise RuntimeError('Safe-stop deadline already passed')
    seed_everything(int(conf.trainer.pop('seed', 42)) + int(os.environ.get('RANK', '0')),
                    workers=True)
    if conf.trainer.strategy.type != 'ddp':
        raise RuntimeError('Only explicit DDP strategy supported in production compact path')
    options = dict(conf.trainer.strategy.params)
    if options != {'find_unused_parameters': True}:
        raise RuntimeError('Unexpected DDP strategy options')
    conf.trainer.strategy = DDPStrategy(**options)
    if conf.trainer.precision != 'bf16-mixed':
        raise RuntimeError('Required BF16 mixed compute + FP32 master weights')
    if conf.trainer.get('use_distributed_sampler', True) is not False:
        raise RuntimeError('Lightning must not replace the rank-offset sampler')
    if int(conf.trainer.accumulate_grad_batches) != 1:
        raise RuntimeError('Only gradient accumulation 1 is validated')
    workers = int(conf.data.params.train_datasets.num_workers)
    horizon = int(conf.data.params.max_steps)
    batch_size = int(conf.data.params.train_datasets.batch_size)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    sampler_seed = int(conf.data.params.train_datasets.sampler_seed)
    contract = make_contract(data_horizon=horizon, batch_size=batch_size,
                             world_size=world, sampler_seed=sampler_seed,
                             num_workers=workers)
    if int(conf.trainer.max_steps) > horizon:
        raise RuntimeError('Trainer max_steps exceeds frozen data horizon')
    if world != int(conf.trainer.devices) or int(conf.trainer.num_nodes) != 1:
        raise RuntimeError('World size does not match single-node DDP trainer')
    import hashlib
    def _file_digest(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    base_path = str(conf.model.params.pretrained)
    expected_base = str(conf.model.params.pretrained_sha256)
    if _file_digest(base_path) != expected_base:
        raise RuntimeError('Original XR1 pretrained SHA256 mismatch')
    language_path = str(conf.model.params.language_policy_path)
    expected_language = str(conf.model.params.language_sha256)
    if _file_digest(language_path) != expected_language:
        raise RuntimeError('Frozen Language V2 SHA256 mismatch')
    print('XR1_FP32_IMMUTABLE_BASE_AND_LANGUAGE_SHA256=PASS rank=' +
          str(os.environ.get('RANK')), flush=True)
    conf.model.params.pop('language_policy_path')
    ckpt_path = conf.trainer.pop('ckpt_path', None)
    step = 0
    if ckpt_path:
        if production:
            ck = load_verified(ckpt_path, contract, expected_config=fingerprint)
            step = int(ck['global_step'])
            del ck
        else:
            step, _ = _resume_preflight(ckpt_path, contract)
        if int(conf.trainer.max_steps) <= step:
            raise RuntimeError('New trainer max_steps must exceed checkpoint step')
    conf.data.params.train_datasets.resume_step = step
    conf.data.params.train_datasets.deterministic_replay = True
    conf.model.params.compact_fp32 = {
        'last_n': int(conf.model.params.get('last_n', 1)),
        'data_horizon': horizon, 'batch_size': batch_size, 'world_size': world,
        'sampler_seed': sampler_seed, 'num_workers': workers,
        'resume_step': step,
        'pretrained_sha256': str(conf.model.params.pop('pretrained_sha256')),
        'language_sha256': str(conf.model.params.pop('language_sha256')),
        'source_commit': str(conf.model.params.pop('source_commit')),
    }
    if production:
        conf.model.params.compact_fp32['config_sha256'] = fingerprint
    conf.model.params.pop('last_n', None)
    conf.model.params.optimizer = conf.trainer.pop('optimizer')
    conf.model.params.scheduler = conf.trainer.pop('scheduler')
    root = Path(str(conf.trainer.default_root_dir)).resolve()
    target = root / ('project_' + str(conf.trainer.pop('project'))) / str(conf.trainer.pop('exp_name'))
    target.mkdir(parents=True, exist_ok=True)
    if production:
        from mibot.utils.claix_checkpoint import atomic_json
        import time
        atomic_json(target / f'resolved-{time.time_ns()}.json', source_config)
        atomic_json(target / 'training_inventory.json', inventory)
        atomic_json(target / 'resume_contract.json', {'config_sha256': fingerprint, 'replay': contract})
    with (target / 'resolved_compact_config.json').open('w') as f:
        json.dump(conf.to_dict(), f, indent=2, default=str)
    interval = int(conf.trainer.pop('save_interval', 1000))
    checkpoint = ModelCheckpoint(save_top_k=1, save_last=True,
                                 save_weights_only=False, filename='compact-{step:06d}',
                                 every_n_train_steps=interval,
                                 dirpath=str(target), enable_version_counter=False)
    conf.trainer.default_root_dir = str(target)
    if production:
        checkpoint = ProductionRuntime(target, deadline=deadline, interval=interval,
                                       stop_after_step=production.get('stop_after_step'))
        conf.trainer.plugins = [VerifiedCheckpointIO()]
    conf.trainer.callbacks = [ModelSummary(max_depth=2), checkpoint]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    datamodule = DATASETS.build(conf.data)
    model = MIMODEL.build(conf.model)
    logger = (CSVLogger(save_dir=str(target), name='metrics') if production else
              WandbLogger(project='xr1_doosan', name=target.name, config={'last_n': model._last_n}))
    print(f'XR1_COMPACT_DDP_START rank={os.environ.get("RANK")} resume_step={step} '
          f'world_size={world} max_steps={conf.trainer.max_steps} horizon={horizon}', flush=True)
    trainer = Trainer(**conf.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
