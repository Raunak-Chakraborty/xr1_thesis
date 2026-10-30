"""Offline held-out generated-action evaluation; never called from training."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--overrides',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--validation-root',required=True)
    p.add_argument('--test-root',required=True,help='Identity/leakage inventory only; no test inference')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0,help='Smoke subset only; 0 evaluates all validation frames')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real XR1 evaluation requires an allocated CLAIX compute node')
    os.environ.update(MAX_LENGTH='12288',WORLD_SIZE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from mmengine import Config
    from mibot.data.datasets.doosan_json_dataset import DoosanValidationDataset
    from mibot.data.collate.custom_collate import CustomCollate
    from mibot.models.runner.doosan_compact_runner import DoosanCompactRunner
    from mibot.utils.claix_checkpoint import atomic_json,file_sha256,load_verified
    from mibot.utils.claix_contract import audit_splits,resume_fingerprint
    from mibot.utils.claix_evaluation import evaluate_items
    from mibot.utils.doosan_replay import make_contract
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg=compose(config_name='claix_b24_production',overrides=json.loads(Path(args.overrides).read_text()))
    conf=Config(OmegaConf.to_container(cfg,resolve=True))
    train,val,test=audit_splits(conf.data.params.train_datasets.paths[0],args.validation_root,args.test_root)
    fingerprint=resume_fingerprint(OmegaConf.to_container(cfg,resolve=True),train,12288)
    data=conf.data.params.train_datasets
    contract=make_contract(data_horizon=int(conf.data.params.max_steps),batch_size=int(data.batch_size),
                           world_size=1,sampler_seed=int(data.sampler_seed),num_workers=int(data.num_workers))
    ck=load_verified(args.checkpoint,contract,expected_config=fingerprint,resumable=False)
    for path,expected in ((conf.model.params.pretrained,conf.model.params.pretrained_sha256),
                          (conf.model.params.language_policy_path,conf.model.params.language_sha256)):
        if file_sha256(path)!=expected:
            raise RuntimeError('Immutable pretrained/language checksum mismatch')
    conf.model.params.compact_fp32=dict(last_n=1,data_horizon=contract['data_horizon'],batch_size=24,
        world_size=1,sampler_seed=contract['sampler_seed'],num_workers=contract['num_workers'],
        resume_step=int(ck['global_step']),pretrained_sha256=conf.model.params.pretrained_sha256,
        language_sha256=conf.model.params.language_sha256,source_commit=conf.model.params.source_commit,
        config_sha256=fingerprint)
    # Loading terminal checkpoints for inference does not invoke resume-only hooks.
    runner=DoosanCompactRunner(conf.model.params);runner.configure_model()
    if ck['xr1_compact_fp32']!=runner._compact_meta:
        raise RuntimeError('Compact model selection/base provenance mismatch')
    runner.load_state_dict(ck['state_dict'])
    model=runner.model.eval().to('cuda')
    conf.data.params.train_datasets.paths=[args.validation_root]
    dataset=DoosanValidationDataset(conf.data.params)
    processor=CustomCollate().processor
    limit=args.limit or len(dataset)
    if limit<1 or limit>len(dataset):
        raise ValueError('Invalid validation frame limit')
    def items():
        for i in range(limit):
            sample=dataset.samples[i]
            yield str(Path(sample['file']).relative_to(args.validation_root)),sample['frame_index'],dataset[i]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        metrics=evaluate_items(model,items(),processor,torch.tensor(data.std),device='cuda')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,dict(schema='claix-b24-validation-v1',checkpoint_sha256=file_sha256(args.checkpoint),
        validation_sha256=val['sha256'],test_identity_sha256=test['sha256'],metrics=metrics,
        scope='FULL_VALIDATION' if limit==len(dataset) else 'SMOKE_SUBSET_NOT_A_VALIDATION_RESULT'))


if __name__=='__main__':
    main()
