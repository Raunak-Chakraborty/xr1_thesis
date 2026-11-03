"""Use the repository's documented offline model-import bootstrap, unchanged."""
import unittest
from run_claix_cpu_tests import bootstrap


def load_tests(loader, tests, pattern):
    bootstrap()
    return loader.loadTestsFromNames([
        'test_doosan_dataset', 'test_doosan_datamodule', 'test_claix_replay',
        'test_claix_checkpoint', 'test_claix_runtime', 'test_claix_collate',
    ])
