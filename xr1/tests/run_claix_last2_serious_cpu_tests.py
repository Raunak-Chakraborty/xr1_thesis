#!/usr/bin/env python3
"""Local CPU contracts/transactions; explicitly not real Lightning/Hydra acceptance."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'xr1'), str(ROOT / 'scripts')]
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', OMP_NUM_THREADS='1')

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    if importlib.util.find_spec('lightning') is None:
        for name in ('lightning', 'lightning.pytorch', 'lightning.pytorch.callbacks',
                     'lightning.fabric', 'lightning.fabric.plugins', 'lightning.fabric.plugins.io'):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules['lightning.pytorch.callbacks'].Callback = type('Callback', (), {})
        sys.modules['lightning.fabric.plugins.io'].CheckpointIO = type('CheckpointIO', (), {})
        print('TEST_MOCK=Lightning Callback/CheckpointIO bases; Trainer is not executed', flush=True)
    print(f'CPU_ENV torch={torch.__version__}; CUDA hidden; synthetic inputs only', flush=True)
    suite = unittest.TestSuite()
    for pattern in ('test_claix_last2_serious.py', 'test_claix_serious_retention.py',
                    'test_claix_last2_compact.py', 'test_claix_sync500_last2_contract.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'xr1/tests'), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
