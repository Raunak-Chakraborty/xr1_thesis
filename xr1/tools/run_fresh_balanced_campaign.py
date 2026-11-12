"""Four-arm fresh-XR-1 balanced-sampling screening campaign.

No historical 48,129-step trainable tensors are loaded. This uses a pinned
released base model, separate full optimizer runs and the 34-episode validation.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ['XR1_FRESH_RUN_ROOT']).resolve(strict=True)
WT = Path(__file__).resolve().parents[2]
EXP = WT / 'xr1/experiments/fresh_balanced_v1'
PARENT = Path('/hpcwork/zk443626/robotics_thesis/runs/xr1-fresh-9b04eb66889347cdb121b9be52b8f7d6/project_xr1_doosan/xr1_doosan_b24_experimental_v1/step-048129.ckpt')
SOURCE_CONFIG = PARENT.parent / 'resolved_compact_config.json'
BASE_SHA = '94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7'
BATCH = 24
STEPS = 2000
PROFILES = (
 dict(name='natural_adamw_l1',kind='adamw_control',adamw_lr=1e-5,muon_lr=None,last_n=1,sampler='natural'),
 dict(name='balanced_adamw_l1',kind='adamw_control',adamw_lr=1e-5,muon_lr=None,last_n=1,sampler='balanced'),
 dict(name='balanced_muon_l1',kind='hybrid_muon',adamw_lr=1e-5,muon_lr=7.5e-5,last_n=1,sampler='balanced'),
 dict(name='balanced_muon_l2',kind='hybrid_muon',adamw_lr=1e-5,muon_lr=7.5e-5,last_n=2,sampler='balanced'),
)
PROFILE_NAMES={p['name']:p for p in PROFILES}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp.' + str(os.getpid()))
    if tmp.exists() or path.exists():
        raise FileExistsError(path)
    with open(tmp, 'x') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



def validate_environment():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('This must run in the allocated Slurm GPU job')
    if not Path(os.environ['XR1_DOOSAN_ZIP_CACHE_ROOT'],'media_manifest.jsonl').is_file():
        raise RuntimeError('Approved media cache unavailable')
    if os.getenv('XR1_DOOSAN_OFFLINE_FRAME_ROOT'):
        raise RuntimeError('Conflicting offline PNG override')
    if os.getenv('MAX_LENGTH')!='12288' or os.getenv('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Unqualified deterministic environment')
    import torch
    if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected exactly one H100')
    if not SOURCE_CONFIG.is_file():
        raise RuntimeError('Missing original resolved Doosan config')
    conf=json.loads(SOURCE_CONFIG.read_text())
    p=conf['model']['params']; data=conf['data']['params']['train_datasets']
    if (p['compact_fp32']['last_n'],data['batch_size'])!=(1,24):
        raise RuntimeError('Original Doosan model data contract changed')
    if not Path(p['pretrained']).is_file() or p['compact_fp32']['pretrained_sha256']!=BASE_SHA:
        raise RuntimeError('Released XR-1 pretrained checkpoint identity differs')
    if not Path(data['paths'][0]).is_dir():
        raise RuntimeError('Original dataset path missing')
    for x in ('natural','balanced'):
        d=ROOT/'schedules'/x
        meta=json.loads((d/'motion_focus_manifest.json').read_text())
        if sha256(d/'sample_global_indices.npy') != meta['sampling']['sample_schedule_sha256']:
            raise RuntimeError('New training schedule checksum mismatch: '+x)
        if (meta['sampling']['samples'],meta['sampling']['steps'])!=(STEPS*BATCH,STEPS):
            raise RuntimeError('New training schedule length mismatch')
    print('FRESH_PRETRAINED_XR1_ENV=PASS',flush=True)
    return conf


def motion_settings(profile):
    schedule=ROOT/'schedules'/profile['sampler']/'sample_global_indices.npy'
    return dict(initialization='released_pretrained_only',
        pretrained_sha256=BASE_SHA,
        language_sha256=str(json.loads(SOURCE_CONFIG.read_text())['model']['params']['compact_fp32']['language_sha256']),
        schedule_sha256=sha256(schedule),trial_id=profile['name'],
        optimizer_kind=profile['kind'],adamw_lr=profile['adamw_lr'],muon_lr=profile['muon_lr'],
        warmup_steps=200,schedule_total_steps=STEPS,trainable_last_n=profile['last_n'])


def build_model_and_data(conf, profile, resume_step):
    from mmengine import Config
    conf=json.loads(json.dumps(conf))
    p=conf['model']['params']
    conf['model']['type']='DoosanFreshMotionRunner'
    p.pop('optimizer',None);p.pop('scheduler',None)
    p['model']['motion_focus_loss']=dict(first_5=2.0,next_5=1.25,remaining_20=.75)
    p['motion_experiment']=motion_settings(profile)
    p['compact_fp32'].update(data_horizon=STEPS,batch_size=BATCH,world_size=1,
       sampler_seed=42,num_workers=4,last_n=profile['last_n'],resume_step=resume_step)
    p['compact_fp32'].pop('config_sha256',None)
    p['compact_fp32'].pop('experiment',None)
    data=conf['data']['params'];data['max_steps']=STEPS
    loader=data['train_datasets']
    loader.update(batch_size=BATCH,sampler_seed=42,num_workers=4,
       deterministic_replay=True,resume_step=resume_step,
       motion_focus_plan_dir=str(ROOT/'schedules'/profile['sampler']),
       persistent_workers=True,prefetch_factor=2,in_order=True,
       zip_integrity_policy='immutable',worker_timeout=600)
    return Config(conf['model']),Config(conf['data'])


def checkpoint_path(profile,step):
    return ROOT/'trials'/profile/'checkpoints'/f'motion-{step:06d}.ckpt'

def run_train(conf, profile, steps, resume=None):
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

    name = profile['name']
    target = ROOT/'trials'/name
    if resume is not None:
        raise RuntimeError('Fresh-pretrained trial may not resume prior model states')
    target.mkdir(parents=True, exist_ok=False)
    if steps != 2000:
        raise ValueError('Unexpected optimizer-step target')
    model_cfg, data_cfg = build_model_and_data(conf, profile, resume_step=0)
    seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    checkpoint = ModelCheckpoint(dirpath=str(target/'checkpoints'),
                                 filename='motion-{step:06d}',
                                 auto_insert_metric_name=False,
                                 save_top_k=-1,
                                 save_last=True,
                                 every_n_train_steps=500,
                                 save_weights_only=False)
    trainer = Trainer(accelerator='gpu', devices=1, num_nodes=1,
                      strategy=DDPStrategy(find_unused_parameters=True),
                      precision='bf16-mixed', max_steps=steps,
                      accumulate_grad_batches=1, use_distributed_sampler=False,
                      logger=CSVLogger(save_dir=str(target), name='metrics'),
                      callbacks=[checkpoint], enable_progress_bar=False,
                      enable_checkpointing=True, num_sanity_val_steps=0,
                      gradient_clip_val=1.0, gradient_clip_algorithm='norm',
                      log_every_n_steps=10,
                      default_root_dir=str(target))
    datamodule = DATASETS.build(data_cfg)
    model = MIMODEL.build(model_cfg)
    print(f'MOTION_TRAIN_START profile={name} resume={resume} max_steps={steps}',flush=True)
    try:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
    finally:
        loader = trainer.train_dataloader
        if loader is not None:
            shutdown_loader(loader, getattr(loader, '_iterator', None))
    actual = checkpoint_path(name,steps)
    if not actual.is_file():
        raise RuntimeError('No final Lightning checkpoint: '+str(actual))
    ck = torch.load(actual, map_location='cpu', mmap=True, weights_only=False)
    if int(ck.get('global_step',-1)) != steps:
        raise RuntimeError('Wrong completed step in checkpoint')
    if ck.get('xr1_motion_compact') != model._motion_meta:
        raise RuntimeError('Experimental checkpoint identity mismatch')
    if not ck.get('optimizer_states') or not ck.get('lr_schedulers'):
        raise RuntimeError('Missing optimizer/scheduler state')
    del ck
    receipt = dict(profile=name, step=steps, checkpoint=str(actual.resolve(strict=True)),
                   sha256=sha256(actual), training_valid=True)
    atomic_json(target/f'checkpoint-step{steps}.json',receipt)
    print('MOTION_TRAIN_COMPLETED='+json.dumps(receipt),flush=True)
    return receipt


def evaluate(conf, profile, ckpt_receipt):
    import numpy as np
    import torch
    from mmengine import Config
    from mibot.data.datasets.doosan_json_dataset import DoosanValidationDataset
    from mibot.data.collate.custom_collate import CustomCollate
    from mibot.models.runner.doosan_fresh_runner import DoosanFreshMotionRunner
    from mibot.utils.claix_evaluation import inference_payload, action_errors

    path = Path(ckpt_receipt['checkpoint'])
    if sha256(path) != ckpt_receipt['sha256']:
        raise RuntimeError('Unverified evaluation checkpoint')
    model_cfg, data_cfg = build_model_and_data(conf, profile, resume_step=0)
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    step = int(ck['global_step'])
    if step not in (1000,2000):
        raise RuntimeError('Unexpected evaluation model step')
    runner = DoosanFreshMotionRunner(model_cfg['params'])
    runner.configure_model()
    if ck.get('xr1_motion_compact') != runner._motion_meta:
        raise RuntimeError('Evaluation model provenance mismatch')
    runner.load_state_dict(ck['state_dict'], strict=False)
    del ck
    model = runner.model.eval().to('cuda')

    train = Path(conf['data']['params']['train_datasets']['paths'][0]).resolve(strict=True)
    val = train.parent/'val'
    if not val.is_dir():
        raise RuntimeError('Frozen validation split missing')
    data_cfg['params']['train_datasets']['paths'] = [str(val)]
    data_cfg['params']['train_datasets'].pop('motion_focus_plan_dir', None)
    data_cfg['params']['train_datasets']['deterministic_replay'] = False
    data_cfg['params']['max_steps'] = 2000
    dataset = DoosanValidationDataset(data_cfg['params'])
    if len(dataset.files) != 34:
        raise RuntimeError('Expected 34 validation episodes')
    processor = CustomCollate().processor
    counts = [int(dataset._read_json(p)['num_frames']) for p in dataset.files]
    std = torch.tensor(dataset.std)
    frozen_windows = []
    start = 0
    for path, count in zip(dataset.files, counts):
        if count < 35:
            raise RuntimeError('Too-short validation episode')
        for frac in (0.15,0.35,0.55,0.75):
            idx = start + int((count-31)*frac)
            frozen_windows.append((str(Path(path).relative_to(val)),idx))
        start += count
    if len(frozen_windows) != 136:
        raise RuntimeError('Wrong validation window count')
    h = hashlib.sha256(json.dumps(frozen_windows,sort_keys=True).encode()).hexdigest()
    if h!='66b5d6aa826688a4f733d32d25de3eb12338afeddb08e8f8c7738011ea85519d':
        raise RuntimeError('Frozen validation schedule mismatch')
    total, count = 0.0,0
    physical = {key: [0.0,0] for key in ('translation_m','rotation_rad','gripper')}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,(ep,idx) in enumerate(frozen_windows):
            import random
            frame_id = dataset.samples[idx]['frame_index']
            seed = int.from_bytes(hashlib.sha256(f'73000:{ep}:{frame_id}'.encode()).digest()[:4],'big')
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            item=dataset[idx]
            batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in inference_payload(item,processor).items()}
            pred=model.generate(batch).float().cpu().squeeze(0)
            target=item['action'].float(); mask=item['action_mask'].bool()
            valid=mask[:5,:6]
            if valid.sum().item()!=30:
                raise RuntimeError('Primary score requires all first-five motion labels')
            diff=(pred[:5,:6]-target[:5,:6]).abs()
            if not torch.isfinite(diff).all():
                raise RuntimeError('Nonfinite generated action')
            total+=diff[valid].sum().item();count+=int(valid.sum())
            phys=action_errors(pred,target,mask,std)
            for key,row in phys.items():
                physical[key][0]+=row['absolute_error_sum']
                physical[key][1]+=row['count']
            if (i+1)%34==0:
                print(f'MOTION_VALIDATION_PROGRESS profile={profile["name"]} windows={i+1}/136',flush=True)
    if count != 136*30:
        raise RuntimeError('Invalid validation action count')
    metrics=dict(profile=profile['name'], step=step, windows=136, episodes=34,
                 fixed_validation_windows_sha256=h,
                 first5_normalized_motion_mae=total/count,
                 full30_physical_mae={key: s/n for key,(s,n) in physical.items()},
                 checkpoint_sha256=ckpt_receipt['sha256'],
                 validation_provenance='real_fixed_validation')
    # Event recall: one pre-release 30-action window per held-out episode.
    # Compute no teacher-forced tokens, use the same inference seed for all arms.
    if step == 2000:
        tp=0;pos=0;fp=0;neg=0
        start_offset=0
        for ep_path, num_frames in zip(dataset.files,counts):
            episode=dataset._read_json(ep_path)
            actions=np.asarray(episode['actions']['semantic'],dtype=np.float64)
            state=(actions[:,6]>=0.5).astype(np.int8)
            transitions=np.flatnonzero(np.diff(state)==1)+1
            if len(transitions)!=1 or int(state[0])!=0:
                raise RuntimeError('Held-out release semantics are inconsistent')
            transition=int(transitions[0])
            frame=max(0,min(transition-10,num_frames-30))
            index=start_offset+frame
            ep=str(Path(ep_path).relative_to(val))
            seed=int.from_bytes(hashlib.sha256(f'83000:{ep}:{frame}'.encode()).digest()[:4],'big')
            import random
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            item=dataset[index]
            batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in inference_payload(item,processor).items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model.generate(batch).float().cpu().squeeze(0)
            target=item['action'].float();mask=item['action_mask'].bool()
            valid=mask[:,6].bool();label=(target[:,6]>=.5)&valid;not_open=(target[:,6]<.5)&valid
            predicted=(pred[:,6]>=.5)
            tp+=int((predicted&label).sum());pos+=int(label.sum())
            fp+=int((predicted&not_open).sum());neg+=int(not_open.sum())
            start_offset+=num_frames
        metrics['release_windows']=34
        metrics['release_recall']=tp/pos if pos else None
        metrics['release_false_open_fraction']=fp/neg if neg else None
        metrics['release_positive_frames']=pos
        metrics['release_negative_frames']=neg
        print(f'RELEASE_VALIDATION_COMPLETED profile={profile["name"]} recall={metrics["release_recall"]:.6f}',flush=True)
    atomic_json(ROOT/'trials'/profile['name']/f'validation-step{step}.json',metrics)
    print('MOTION_VALIDATION_COMPLETED='+json.dumps(metrics,sort_keys=True),flush=True)
    return metrics



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=('train','evaluate'))
    ap.add_argument('--profile',required=True,choices=tuple(PROFILE_NAMES))
    ap.add_argument('--steps',required=True,type=int)
    a=ap.parse_args()
    if a.mode=='train' and a.steps!=STEPS:raise RuntimeError('Train exactly 2000 fresh steps per arm')
    if a.mode=='evaluate' and a.steps not in (1000,2000):raise RuntimeError('Only two approved validation points')
    conf=validate_environment()
    p=PROFILE_NAMES[a.profile]
    if a.mode=='train':run_train(conf,p,a.steps)
    else:
        receipt=ROOT/'trials'/p['name']/f'checkpoint-step{a.steps}.json'
        if not receipt.is_file():
            ck=checkpoint_path(p['name'],a.steps)
            if not ck.is_file():raise RuntimeError('Checkpoint missing for validation')
            import torch
            data=torch.load(ck,map_location='cpu',mmap=True,weights_only=False)
            if int(data['global_step'])!=a.steps or not data.get('optimizer_states') or not data.get('lr_schedulers'):
                raise RuntimeError('Untrusted checkpoint state')
            del data
            atomic_json(receipt,dict(profile=p['name'],step=a.steps,
                    checkpoint=str(ck.resolve(strict=True)),sha256=sha256(ck),training_valid=True))
        evaluate(conf,p,json.loads(receipt.read_text()))

if __name__=='__main__':main()
