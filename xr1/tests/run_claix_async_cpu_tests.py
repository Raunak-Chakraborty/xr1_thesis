#!/usr/bin/env python3
"""Task 04B CPU checks. Missing framework acceptance is BLOCKED, never a pass."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'xr1'), str(ROOT/'scripts')]
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', OMP_NUM_THREADS='1',
                  USE_TF='0', USE_FLAX='0')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-framework-stubs', action='store_true',
                   help='Run only isolated transaction/adapter tests; does not qualify Lightning')
    args = p.parse_args()
    missing = [name for name in ('lightning', 'hydra', 'mmengine') if importlib.util.find_spec(name) is None]
    if missing:
        print('TASK04B_REAL_FRAMEWORK=BLOCKED missing=' + ','.join(missing), flush=True)
        if not args.allow_framework_stubs:
            sys.exit(2)
    if 'lightning' in missing:
        print('EXPLICIT_TEST_STUB=Callback/CheckpointIO bases only; synthetic Trainer adapter', flush=True)
        for name in ('lightning', 'lightning.pytorch', 'lightning.pytorch.callbacks',
                     'lightning.fabric', 'lightning.fabric.plugins', 'lightning.fabric.plugins.io'):
            module = types.ModuleType(name); module.__path__ = []
            sys.modules[name] = module
        sys.modules['lightning.pytorch.callbacks'].Callback = type('Callback', (), {})
        sys.modules['lightning.fabric.plugins.io'].CheckpointIO = type('CheckpointIO', (), {})
    import torch
    torch.set_num_threads(1)
    print(f'CPU_ENV torch={torch.__version__}; CUDA hidden; real CPU tensors and POSIX transactions', flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'xr1/tests'), pattern='test_claix_async_checkpoint.py')
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'xr1/tests'), pattern='test_claix_two_way.py'))
    if not args.allow_framework_stubs:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'xr1/tests'), pattern='test_claix_async_framework.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
