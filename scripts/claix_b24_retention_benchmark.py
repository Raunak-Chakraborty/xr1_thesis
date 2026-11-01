#!/usr/bin/env python3
"""CPU-only synthetic compact retention I/O benchmark in a NEW disposable root.

No GPU, dataset, model backbone or network access. Does not qualify H100 behavior.
Default payload ~1.7 GB; --payload-mib 8 provides a cheap script smoke test.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from claix_b24_launch import ROOT, tree_digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, help='Must not exist; benchmark owns only newly created files')
    p.add_argument('--payload-mib', type=int, default=1622)
    p.add_argument('--milestones', type=int, choices=range(7), default=6)
    args = p.parse_args(argv)
    if args.payload_mib < 1:
        p.error('payload-mib must be positive')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(ROOT/'xr1'), str(ROOT/'xr1/tests')]
    import torch
    torch.set_num_threads(1)
    import mibot.utils.claix_checkpoint as checkpoints
    from test_claix_checkpoint import checkpoint_fixture
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    count = args.payload_mib*1024**2//12
    ck = checkpoint_fixture(1, horizon=48284)
    ck['state_dict']['model.weight'] = torch.ones(count)
    ck['optimizer_states'][0]['state'][0]['exp_avg'] = torch.ones(count)
    ck['optimizer_states'][0]['state'][0]['exp_avg_sq'] = torch.ones(count)
    ck['xr1_compact_fp32']['trainable_parameter_count'] = count
    rows = []
    for step in (*sorted(checkpoints.HISTORICAL_STEPS)[:args.milestones], 40250, 40500, 40750):
        ck['global_step'] = step
        ck['optimizer_states'][0]['state'][0]['step'] = torch.tensor(float(step))
        ck['lr_schedulers'][0]['last_epoch'] = step
        ck['xr1_replay'] = __import__('mibot.utils.doosan_replay', fromlist=['checkpoint_cursor']).checkpoint_cursor(ck['xr1_replay_contract'], step)
        progress = dict(ready=step, started=step, processed=step, completed=step-1)
        loop = ck['loops']['fit_loop']
        loop['epoch_loop.batch_progress'] = dict(total=progress, current=dict(progress))
        loop['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed'] = step
        started = time.perf_counter()
        receipt = checkpoints.publish_checkpoint(ck, root/f'step-{step:06d}.ckpt')
        rows.append(dict(step=step, bytes=receipt['bytes'], publication_seconds=time.perf_counter()-started))
    del ck
    started = time.perf_counter()
    with patch.object(torch, 'load', wraps=torch.load) as loads, \
         patch.object(checkpoints, 'file_sha256', wraps=checkpoints.file_sha256) as hashes, \
         patch.object(checkpoints, 'load_verified', wraps=checkpoints.load_verified) as verifications:
        result = checkpoints.rotate_checkpoints(root)
    report = dict(source_tree_sha256=tree_digest(), kind='cpu', synthetic=True,
                  rotation_seconds=time.perf_counter()-started, rotation=result,
                  torch_load_calls=loads.call_count, checkpoint_sha256_reads=hashes.call_count,
                  candidate_verifications=verifications.call_count, publications=rows,
                  filesystem=str(root), cache_policy='No cache flush: report warm OS cache; run on target filesystem')
    (root/'retention-benchmark.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    if (loads.call_count, hashes.call_count, verifications.call_count) != (3,3,3):
        raise RuntimeError('Unexpected periodic retention I/O count')


if __name__ == '__main__':
    main()
