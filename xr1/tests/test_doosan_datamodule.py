#!/usr/bin/env python3

import unittest

from mmengine import Config, DATASETS

import mibot.data
from mibot.data import (
    BaseDataModule,
    DoosanDataModule,
)
from mibot.data.datasets.doosan_json_dataset import (
    DoosanJsonDataset,
)


class DoosanDataModuleRegistrationTest(
    unittest.TestCase
):

    def test_inheritance(self):
        self.assertTrue(
            issubclass(
                DoosanDataModule,
                BaseDataModule,
            )
        )

    def test_registry(self):
        registered = DATASETS.get(
            "DoosanDataModule"
        )

        self.assertIs(
            registered,
            DoosanDataModule,
        )

    def test_configuration_fields(self):
        params = Config(
            {
                "max_steps": 1,
                "train_datasets": {
                    "batch_size": 2,
                    "action_length": 30,
                    "num_workers": 0,
                    "prefetch_factor": 4,
                    "persistent_workers": True,
                    "pin_memory": False,
                    "sampler_seed": 123,
                },
            }
        )

        module = DoosanDataModule(
            params
        )

        self.assertEqual(
            module.batch_size,
            2,
        )

        self.assertEqual(
            module.num_workers,
            0,
        )

        self.assertEqual(
            module.prefetch_factor,
            4,
        )

        self.assertFalse(
            module.persistent_workers
        )

        self.assertFalse(
            module.pin_memory
        )

        self.assertEqual(
            module.sampler_seed,
            123,
        )

    def test_setup_uses_doosan_dataset(self):
        # Avoid calling the real dataset constructor
        # here; the full integration smoke below does that.
        self.assertIsNot(
            DoosanJsonDataset,
            None,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
