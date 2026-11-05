"""Synthetic offline paths; never read or write the private CLAIX reference tree."""
import json
from pathlib import Path
import tempfile
import unittest


def write_path_overrides(root):
    path = Path(root) / 'synthetic-paths.json'
    # Exclusive creation also prevents accidental replacement of caller data.
    with path.open('x') as stream:
        json.dump([
            'data.params.train_datasets.paths=["/hpcwork/SYNTHETIC/train"]',
            'model.params.pretrained="/hpcwork/SYNTHETIC/model_states.pt"',
            'model.params.language_policy_path="/hpcwork/SYNTHETIC/language.yaml"',
            'trainer.default_root_dir="/hpcwork/SYNTHETIC/unused"',
        ], stream)
    return path


class SyntheticPathOverridesTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory(prefix='xr1-offline-paths-')
        # unittest runs cleanups even if later setup or the test fails.
        self.addCleanup(folder.cleanup)
        self.path_args = ['--overrides', str(write_path_overrides(folder.name))]
