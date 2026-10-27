"""Hydra composition smoke test, requiring only the existing Hydra dependency."""

import importlib.util
from pathlib import Path
import unittest

HYDRA = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(HYDRA, "Hydra unavailable; experiment composition not executed")
class EncoderConfigTests(unittest.TestCase):
    def test_dedicated_experiment_keeps_required_stats_and_checkpoint(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omegaconf.errors import InterpolationToMissingValueError
        root = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(config_dir=str(root), version_base=None):
            config = compose(config_name="force_encoder")
        self.assertEqual(config.data.type, "DoosanDataModule")
        self.assertTrue(config.model.params.model.force_encoder.enabled)
        self.assertFalse(config.model.params.model.force_encoder.dual_path)
        self.assertTrue(config.data.params.train_datasets.use_wrench)
        self.assertTrue(config.data.params.train_datasets.force_encoder_input)
        self.assertEqual(config.data.params.train_datasets.action_length, 30)
        for name in ("q01", "q99", "mean", "std", "paths"):
            self.assertTrue(OmegaConf.is_missing(config.data.params.train_datasets, name))
        self.assertTrue(OmegaConf.is_missing(config.model.params, "pretrained"))
        self.assertTrue(OmegaConf.is_missing(config, "encoder_run_dir"))
        with self.assertRaises(InterpolationToMissingValueError):
            _ = config.trainer.default_root_dir

    def test_explicit_encoder_run_dir_resolves_without_changing_generic_default(self):
        from hydra import compose, initialize_config_dir
        root = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(config_dir=str(root), version_base=None):
            encoder = compose(config_name="force_encoder", overrides=["encoder_run_dir=/synthetic/encoder-run"])
            generic = compose(config_name="config")
        self.assertEqual(encoder.trainer.default_root_dir, "/synthetic/encoder-run")
        self.assertEqual(generic.trainer.default_root_dir, "./test/")


if __name__ == "__main__":
    unittest.main()
