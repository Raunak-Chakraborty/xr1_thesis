#!/usr/bin/env python3
"""Focused offline CPU suite. Framework boundaries are explicitly mocked.

Only Callback/CheckpointIO bases are stubbed when Lightning is absent. The real
runtime hooks, entrypoint function body, torch optimizer/scheduler, checkpoint
serialization/checksums/rotation, and storage/launch guards execute. YAML defaults
are composed by the focused fixture; actual Hydra/Lightning integration is a
CLAIX-only acceptance check. No model or frozen dataset is constructed/scanned.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'xr1'), str(ROOT/'scripts')]
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', USE_TF='0', USE_FLAX='0',
                  OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/xr1-task02-matplotlib')

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    if importlib.util.find_spec('lightning') is None:
        print('TEST_MOCK=Lightning Callback/CheckpointIO base classes only; no Trainer execution', flush=True)
        for name in ('lightning', 'lightning.pytorch', 'lightning.pytorch.callbacks',
                     'lightning.fabric', 'lightning.fabric.plugins', 'lightning.fabric.plugins.io'):
            module = types.ModuleType(name); module.__path__ = []
            sys.modules[name] = module
        sys.modules['lightning.pytorch.callbacks'].Callback = type('Callback', (), {})
        sys.modules['lightning.fabric.plugins.io'].CheckpointIO = type('CheckpointIO', (), {})
    print(f'CPU_ENV torch={torch.__version__}; CUDA hidden; synthetic files only', flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'xr1/tests'), pattern='test_claix_experimental.py')
    from test_claix_checkpoint import CheckpointTest
    from test_claix_final_release import RotationIOTest, ReleasePreparationTest, StoragePreflightTest
    from test_claix_final_corrections import PersistentPathsTest, ResumeDirectorySafetyTest, AllocationStorageGateTest
    for cls in (CheckpointTest, RotationIOTest, ReleasePreparationTest, StoragePreflightTest,
                PersistentPathsTest, ResumeDirectorySafetyTest, AllocationStorageGateTest):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    # The ignored CLAIX staged-path reference is intentionally absent locally.
    # Legacy launcher tests get explicit synthetic overrides, never real staging.
    from test_claix_launch import launch, LaunchTest
    # Hydra composition itself requires the existing CLAIX framework environment.
    # Keep all framework-independent legacy launch tests in this focused suite.
    for name in unittest.defaultTestLoader.getTestCaseNames(LaunchTest):
        if name not in ('test_gpu_command_hydra_composition_with_paths_and_resume',
                        'test_qualification_step_bounds_and_alias_preserve_horizons'):
            suite.addTest(LaunchTest(name))
    original_main = launch.main
    with tempfile.TemporaryDirectory() as folder:
        paths = Path(folder)/'synthetic-paths.json'
        paths.write_text(json.dumps([
            'data.params.train_datasets.paths=["/hpcwork/SYNTHETIC/train"]',
            'model.params.pretrained="/hpcwork/SYNTHETIC/model_states.pt"',
            'model.params.language_policy_path="/hpcwork/SYNTHETIC/language.yaml"',
            'trainer.default_root_dir="/hpcwork/SYNTHETIC/unused"']))
        def explicit_paths(argv):
            return original_main(argv if '--overrides' in argv else [*argv, '--overrides', str(paths)])
        print('TEST_FIXTURE=explicit synthetic path overrides for legacy launcher tests', flush=True)
        with patch.object(launch, 'main', side_effect=explicit_paths):
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
