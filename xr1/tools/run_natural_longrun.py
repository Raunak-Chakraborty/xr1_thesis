"""XR-1 Last-1, fresh-pretrained 48,284-step, train-only production launcher.

Reuse pinned 5692780 model/data/optimizer machinery; no inference on CLAIX.
An isolated worktree patch only extends a horizon guard to schedule_total_steps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

from tools import run_fresh_balanced_campaign as base

STEPS=48284
BATCH=24
MODEL='natural_adamw_l1'
RELEASED_SHA='94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7'
base.STEPS=STEPS


def sha256(path):
    return base.sha256(path)


def validation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Must execute inside one Slurm GPU allocation')
    if os.getenv('MAX_LENGTH') != '12288' or os.getenv('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Unqualified deterministic environment')
    if not Path(os.environ['XR1_DOOSAN_ZIP_CACHE_ROOT'],'media_manifest.jsonl').is_file():
        raise RuntimeError('ZIP media cache missing')
    if os.environ.get('XR1_DOOSAN_OFFLINE_FRAME_ROOT'):
        raise RuntimeError('Unexpected PNG override')
    import torch
    if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one H100 GPU is required')
    cfg=base.SOURCE_CONFIG
    if not cfg.is_file():raise RuntimeError('Original resolved model config unavailable')
    conf=json.loads(cfg.read_text())
    p=conf['model']['params'];data=conf['data']['params']['train_datasets']
    if (p['compact_fp32']['last_n'],data['batch_size'])!=(1,BATCH):
        raise RuntimeError('Unexpected original model/data contract')
    if p['compact_fp32']['pretrained_sha256']!=RELEASED_SHA or not Path(p['pretrained']).is_file():
        raise RuntimeError('Pretrained checkpoint mismatch')
    if not Path(data['paths'][0]).is_dir():
        raise RuntimeError('Training dataset not found')
    d=base.ROOT/'schedules'/'natural'
    meta=json.loads((d/'motion_focus_manifest.json').read_text())
    if meta['model']['starting_checkpoint']!='released_XR1_pretrained_only':
        raise RuntimeError('Wrong starting checkpoint identity')
    if (meta['sampling']['steps'],meta['sampling']['samples']) != (STEPS,STEPS*BATCH):
        raise RuntimeError('Natural schedule size mismatch')
    if sha256(d/'sample_global_indices.npy')!=meta['sampling']['sample_schedule_sha256']:
        raise RuntimeError('Natural schedule SHA mismatch')
    print('FRESH_RELEASED_PRETRAINED_CONTRACT=PASS',flush=True)
    return conf


class StopOnRequest:
    @staticmethod
    def build(root):
        from lightning.pytorch.callbacks import Callback
        class StopCallback(Callback):
            def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
                if (root/'STOP_REQUESTED').exists():
                    trainer.should_stop=True
                    if trainer.is_global_zero:
                        print('SAFE_STOP_REQUESTED=YES step='+str(trainer.global_step),flush=True)
        return StopCallback()


def get_resume_step(path, exp):
    import torch
    from mibot.utils.doosan_replay import make_contract, validate_cursor
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if ck.get('xr1_motion_compact',{}).get('experiment_config_sha256')!=exp:
        raise RuntimeError('Resume checkpoint incompatible with current full-run configuration')
    step=int(ck.get('global_step',-1))
    contract=make_contract(data_horizon=STEPS,batch_size=BATCH,world_size=1,
                           sampler_seed=42,accumulate_grad_batches=1,num_workers=4)
    if ck.get('xr1_motion_replay_contract') != contract:
        raise RuntimeError('Resume data contract mismatch')
    validate_cursor(ck['xr1_motion_replay'],contract,checkpoint_step=step)
    if not ck.get('optimizer_states') or not ck.get('lr_schedulers') or not ck.get('loops'):
        raise RuntimeError('Incomplete full-state resume checkpoint')
    del ck
    return step


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['fresh','resume'],required=True)
    ap.add_argument('--resume-ckpt',default='')
    a=ap.parse_args()
    conf=validation()
    import torch
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy
    from mmengine import DATASETS
    import mibot.data
    import mibot.models.runner.doosan_fresh_runner
    from mibot.models import MIMODEL
    from mibot.data.datamodule.doosan_datamodule import shutdown_loader

    root=base.ROOT
    name=MODEL
    target=root/'trials'/name
    profile=base.PROFILE_NAMES[name]
    exp=base.motion_settings(profile)
    digest=hashlib.sha256(json.dumps(exp,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if exp['schedule_total_steps']!=STEPS or exp['initialization']!='released_pretrained_only':
        raise RuntimeError('Wrong experiment identity')
    if a.mode=='fresh':
        if a.resume_ckpt: raise RuntimeError('Cannot supply resume checkpoint with --mode fresh')
        target.mkdir(parents=True,exist_ok=False)
        resume_step=0
        ckpt=None
    else:
        if not a.resume_ckpt or not target.is_dir():
            raise RuntimeError('Existing target and explicit --resume-ckpt required')
        ckpt=Path(a.resume_ckpt).resolve(strict=True)
        if not ckpt.is_relative_to((target/'checkpoints').resolve(strict=True)):
            raise RuntimeError('Resume checkpoint must belong to this isolated run')
        resume_step=get_resume_step(ckpt,digest)
        print('RESUME_CONTRACT=PASS global_step='+str(resume_step),flush=True)
    if (root/'STOP_REQUESTED').exists():
        raise RuntimeError('Old STOP_REQUESTED marker: remove intentionally before resuming')

    model_cfg,data_cfg=base.build_model_and_data(conf,profile,resume_step=resume_step)
    seed_everything(42,workers=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=False)
    checkpoint=ModelCheckpoint(dirpath=str(target/'checkpoints'),
      filename='motion-{step:06d}',auto_insert_metric_name=False,
      save_top_k=-1,save_last=True,every_n_train_steps=2000,save_weights_only=False)
    trainer=Trainer(accelerator='gpu',devices=1,num_nodes=1,
      strategy=DDPStrategy(find_unused_parameters=True),
      precision='bf16-mixed',max_steps=STEPS,max_time=timedelta(hours=22,minutes=55),
      accumulate_grad_batches=1,use_distributed_sampler=False,
      logger=CSVLogger(save_dir=str(target),name='metrics'),
      callbacks=[checkpoint,StopOnRequest.build(root)],
      enable_progress_bar=False,enable_checkpointing=True,num_sanity_val_steps=0,
      gradient_clip_val=1.0,gradient_clip_algorithm='norm',log_every_n_steps=10,
      default_root_dir=str(target))
    datamodule=DATASETS.build(data_cfg)
    model=MIMODEL.build(model_cfg)
    print(f'LONGRUN_TRAIN_START mode={a.mode} horizon={STEPS} resume={resume_step} checkpoints_each=2000',flush=True)
    try:
        trainer.fit(model=model,datamodule=datamodule,ckpt_path=str(ckpt) if ckpt else None)
        step=int(trainer.global_step)
        if not 1<=step<=STEPS:
            raise RuntimeError('Training did not complete one optimizer step')
        # Explicit final checkpoint, including a non-multiple-of-2000 stop.
        final=target/'checkpoints'/f'motion-{step:06d}-graceful.ckpt'
        if final.exists(): raise FileExistsError(final)
        trainer.save_checkpoint(str(final),weights_only=False)
        check=torch.load(final,map_location='cpu',mmap=True,weights_only=False)
        if int(check.get('global_step',-1))!=step or not check.get('optimizer_states') or not check.get('lr_schedulers'):
            raise RuntimeError('Final checkpoint lacks matching optimizer/scheduler state')
        if check.get('xr1_motion_replay',{}).get('rank_sample_offset')!=BATCH*step:
            raise RuntimeError('Final checkpoint replay cursor drift')
        del check
        result=dict(schema='xr1-longrun-natural-last1-v1',global_step=step,
          intended_steps=STEPS,checkpoint=str(final),sha256=sha256(final),
          finished_target=step==STEPS,model=MODEL,source_commit=os.environ.get('XR1_SOURCE_COMMIT'))
        receipt=root/'final_receipt.json'
        base.atomic_json(receipt,result)
        print('LONGRUN_CHECKPOINT_AUDIT=PASS',flush=True)
        print('LONGRUN_FINAL_RECEIPT='+json.dumps(result,sort_keys=True),flush=True)
    finally:
        loader=trainer.train_dataloader
        if loader is not None:
            shutdown_loader(loader,getattr(loader,'_iterator',None))

if __name__=='__main__':main()
