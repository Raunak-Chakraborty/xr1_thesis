#!/usr/bin/env python3
"""Offline synthetic I/O tests; no model, processor, GPU, or dataset downloads.

Pass --allow-framework-stubs only on a laptop missing Lightning/mmengine.
Actual dataset, augmentation, PyTorch DataLoader and subprocesses still execute.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  USE_TF='0', USE_FLAX='0', WANDB_MODE='disabled', OMP_NUM_THREADS='1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-framework-stubs', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.allow_framework_stubs:
        # Reuse the repository's explicit offline registry/backbone bootstrap.
        from run_claix_cpu_tests import bootstrap
        bootstrap()
        if importlib.util.find_spec('lightning') is None:
            module = types.ModuleType('lightning')
            module.LightningDataModule = type('LightningDataModule', (), {})
            sys.modules['lightning'] = module
            print('TEST_MOCK=lightning.LightningDataModule base only', flush=True)
    suite = unittest.TestSuite()
    for pattern in ('test_doosan_io_cache.py', 'test_doosan_dataset.py', 'test_doosan_datamodule.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    print(f'CPU_ENV torch={torch.__version__}', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
