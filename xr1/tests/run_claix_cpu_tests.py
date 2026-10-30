#!/usr/bin/env python3
"""Explicit offline harness; mocks unavailable registry/backbone imports only.

No model weights or processor assets are loaded. Lightning, torch DataLoader,
optimizers, scheduler, production runner hooks and checkpoint I/O execute for real.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  WANDB_MODE='disabled', USE_TF='0', MAX_LENGTH='12288', MPLCONFIGDIR='/tmp/claix-b24-matplotlib')
import torch
torch.set_num_threads(1)


def bootstrap():
    if importlib.util.find_spec('mmengine') is None:
        print('TEST_MOCK=mmengine.Config/Registry (dependency unavailable)', flush=True)
        class Config(dict):
            def __init__(self, value=None):
                super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in (value or {}).items()})
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name) from None
            __setattr__ = dict.__setitem__
        class Registry:
            def __init__(self, *args):
                self.entries = {}
            def register_module(self):
                def decorate(cls):
                    self.entries[cls.__name__] = cls
                    return cls
                return decorate
            def get(self, name):
                return self.entries[name]
            def build(self, cfg):
                return self.get(cfg['type'])(cfg['params'])
        module = types.ModuleType('mmengine')
        module.Config, module.Registry, module.DATASETS = Config, Registry, Registry()
        sys.modules['mmengine'] = module

    # Keep the heavyweight registry __init__ from importing Qwen3-VL unavailable
    # in laptop transformers. The real runner module is imported by tests.
    from mmengine import Registry
    models = types.ModuleType('mibot.models')
    models.__path__ = [str(ROOT / 'mibot/models')]
    models.MIMODEL = Registry('test_models')
    sys.modules['mibot.models'] = models
    print('TEST_MOCK=XR1 backbone import; existing flow-loss methods extracted verbatim via AST', flush=True)
    source = ast.parse((ROOT / 'mibot/models/VLA/XR1.py').read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'xr1')
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ('compute_flow_loss', 'compute_choice_loss')]
    cls = ast.ClassDef(name='xr1', bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    module = types.ModuleType('mibot.models.VLA.XR1')
    import torch.nn.functional as F
    module.__dict__.update(torch=torch, F=F)
    exec(compile(code, str(ROOT / 'mibot/models/VLA/XR1.py'), 'exec'), module.__dict__)
    sys.modules[module.__name__] = module


if __name__ == '__main__':
    bootstrap()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for pattern in ('test_doosan_*.py', 'test_claix_*.py'):
        suite.addTests(loader.discover(str(ROOT / 'tests'), pattern=pattern))
    import test_training_logic
    for name in sorted(dir(test_training_logic)):
        if name.startswith('test_'):
            suite.addTest(unittest.FunctionTestCase(getattr(test_training_logic, name)))
    print(f'CPU_ENV torch={torch.__version__}', flush=True)
    import lightning
    print(f'CPU_ENV lightning={lightning.__version__}', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
