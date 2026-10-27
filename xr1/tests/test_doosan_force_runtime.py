"""CPU regressions using real tensors; skipped if workstation dependencies are absent."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_doosan_force_profile import ROOT, statistics


DEPENDENCIES = ("numpy", "torch", "PIL", "mmengine", "lightning", "transformers", "decord", "torchvision", "tqdm", "liger_kernel")
MISSING = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]


@unittest.skipIf(bool(MISSING), "workstation dependencies missing: " + ", ".join(MISSING))
class ForceRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from mmengine import Config
        from mibot.data.datasets.doosan_json_dataset import DoosanJsonDataset
        from mibot.server.deploy import load_stats
        from mibot.server.runtime.doosan_client import DoosanClient
        from mibot.server.runtime.server import Server
        from mibot.utils.io import normalize_quantile

        cls.np, cls.torch, cls.Config = np, torch, Config
        cls.Dataset, cls.Client, cls.Server = DoosanJsonDataset, DoosanClient, Server
        cls.load_stats = staticmethod(load_stats)
        cls.normalize = staticmethod(normalize_quantile)

    def trajectory(self):
        return {
            "num_frames": 2,
            "proprios": {
                "joint_pos": [[1, 2, 3, 4, 5, 6]] * 2,
                "gripper": [0.5] * 2,
                "tcp_pos": [[0.1, 0.2, 0.3]] * 2,
                "tcp_rotvec": [[0.01, 0.02, 0.03]] * 2,
                "joint_vel": [[6, 5, 4, 3, 2, 1]] * 2,
                "wrench": [[-4, -2, 0, 2, 4, 6], [-3, -1, 1, 3, 5, 7]],
            },
            "actions": {"semantic": [[0.01, 0.02, 0.03, 0, 0, 0, 0.75]] * 2},
        }

    def dataset(self, use_wrench):
        from mibot.utils.doosan_profile import resolve_doosan_data
        dataset = self.Dataset.__new__(self.Dataset)
        dataset.use_wrench = use_wrench
        dataset.action_length = 30
        dataset.semantic_action_key = "actions.semantic"
        data = resolve_doosan_data(statistics(use_wrench))
        dataset.q01 = self.np.asarray(data["q01"], dtype=self.np.float32)
        dataset.q99 = self.np.asarray(data["q99"], dtype=self.np.float32)
        return dataset

    def test_all_six_values_and_exact_state_placement(self):
        trajectory = self.trajectory()
        for mode in (False, True):
            state = self.dataset(mode)._doosan_state(trajectory, 0)
            self.assertEqual(state.shape, (1, 60))
            self.np.testing.assert_array_equal(state[0, :6], [1, 2, 3, 4, 5, 6])
            self.assertEqual(state[0, 6], 0)
            self.np.testing.assert_array_equal(state[0, 20:26], [-4, -2, 0, 2, 4, 6] if mode else [0] * 6)
            self.np.testing.assert_array_equal(state[0, 26:], 0)

    def test_missing_or_invalid_wrench_only_rejected_when_enabled(self):
        trajectory = self.trajectory()
        for value in (None, [[1] * 5] * 2, [[float("nan")] * 6] * 2):
            if value is None:
                trajectory["proprios"].pop("wrench", None)
            else:
                trajectory["proprios"]["wrench"] = value
            self.dataset(False)._doosan_state(trajectory, 0)
            with self.assertRaises(ValueError):
                self.dataset(True)._doosan_state(trajectory, 0)

    def test_actions_masks_and_vlm_targets_never_include_wrench(self):
        from mibot.utils.doosan import build_doosan_action_mask
        action = self.dataset(True)._doosan_action(self.trajectory(), 0, 2)
        mask = build_doosan_action_mask(30, [1, 1] + [0] * 28)
        self.assertEqual(action.shape, (30, 60))
        self.assertEqual(mask.shape, (30, 60))
        self.assertEqual(int(mask.sum()), 14)
        self.np.testing.assert_array_equal(action[:, 7:], 0)
        self.np.testing.assert_array_equal(mask[:, 7:], 0)
        changed = self.trajectory()
        changed["proprios"]["wrench"] = [[100] * 6] * 2
        self.np.testing.assert_array_equal(action, self.dataset(True)._doosan_action(changed, 0, 2))

    def test_dataset_item_and_initialization_without_images_or_processor(self):
        from unittest.mock import patch
        trajectory = self.trajectory()
        for mode in (False, True):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = Path(directory) / "trajectory.json"
                path.write_text(json.dumps(trajectory), encoding="utf-8")
                params = {"max_steps": 1, "train_datasets": dict(statistics(mode), paths=[str(path)])}
                dataset = self.Dataset(params)
                with patch.object(dataset, "_prompt", return_value={"conversations": [], "images": []}), \
                     patch.object(dataset, "_images", return_value=[]), \
                     patch.object(dataset, "_augment", return_value=[]):
                    item = dataset[0]
                self.assertEqual(item["state"].shape, (1, 60))
                self.assertEqual(item["action"].shape, (30, 60))
                self.assertEqual(item["vlm_action_target"].shape, (2, 60))
                self.assertEqual(int(item["action_mask"].sum()), 14)
                self.assertEqual(int(item["vlm_action_mask"].sum()), 14)
                self.assertTrue(self.torch.all(item["action"][2:] == 0))
                self.assertTrue(self.torch.all(item["vlm_action_target"][:, 7:] == 0))

    def test_training_client_and_server_preprocessing_match(self):
        trajectory = self.trajectory()
        robot_state = {name: values[0] for name, values in trajectory["proprios"].items()}
        for mode in (False, True):
            dataset = self.dataset(mode)
            training_raw = dataset._doosan_state(trajectory, 0)
            inference_raw = self.Client.pack_state(robot_state, use_wrench=mode)
            self.np.testing.assert_array_equal(training_raw, inference_raw)
            cfg = self.Config({"data": {"type": "DoosanDataModule", "params": {
                "type": "doosan_json", "train_datasets": statistics(mode),
            }}})
            mean, std, q01, q99, mask = self.load_stats(cfg, "cpu")
            server = self.Server("localhost", 0, None, mean, std, q01, q99, mask, "cpu", doosan_use_wrench=mode)
            actual = server.preprocess_state(self.torch.from_numpy(inference_raw)[None], mode)
            expected = self.normalize(training_raw, dataset.q01, dataset.q99)
            self.np.testing.assert_allclose(actual.numpy()[0], expected, atol=2e-7)
            self.assertEqual(int(mask.sum()), 30 * 7)
            if not mode:
                self.assertTrue(self.torch.all(actual[..., 20:] == 0))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                server.preprocess_state(self.torch.from_numpy(inference_raw)[None], not mode)
            if mode:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    server.preprocess_state(self.torch.from_numpy(inference_raw)[None])

    def test_client_explicit_mode_rejects_missing_wrench(self):
        robot_state = {name: values[0] for name, values in self.trajectory()["proprios"].items()}
        del robot_state["wrench"]
        self.Client.pack_state(robot_state, use_wrench=False)
        with self.assertRaises(ValueError):
            self.Client.pack_state(robot_state, use_wrench=True)

    def test_server_rejects_nonfinite_or_malformed_force_state(self):
        tensor = self.torch.tensor
        dataset = self.dataset(True)
        server = self.Server("localhost", 0, None, self.torch.zeros(30, 60), self.torch.zeros(30, 60),
                             tensor(dataset.q01), tensor(dataset.q99), self.torch.ones(30, 60),
                             "cpu", doosan_use_wrench=True)
        with self.assertRaisesRegex(ValueError, "shape"):
            server.preprocess_state(self.torch.zeros(1, 60), True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            server.preprocess_state(self.torch.full((1, 1, 60), float("nan")), True)

    def test_generic_deployment_action_mask_is_unchanged(self):
        from mibot.utils.io import build_action_mask
        cfg = self.Config({"data": {"type": "BaseDataModule", "params": {
            "train_datasets": statistics(False),
        }}})
        *_, mask = self.load_stats(cfg, "cpu")
        self.np.testing.assert_array_equal(mask.numpy(), build_action_mask(30))

    def test_generic_server_normalization_ignores_doosan_masking(self):
        data = statistics(False)
        tensor = lambda value: self.torch.tensor(value, dtype=self.torch.float32)
        server = self.Server("localhost", 0, None, tensor(data["mean"]), tensor(data["std"]),
                             tensor(data["q01"]), tensor(data["q99"]), self.torch.ones(30, 60), "cpu")
        raw = self.np.zeros((1, 60), dtype=self.np.float32)
        expected = self.normalize(raw, self.np.asarray(data["q01"], dtype=self.np.float32),
                                  self.np.asarray(data["q99"], dtype=self.np.float32))
        self.np.testing.assert_allclose(server.preprocess_state(tensor(raw)[None]).numpy()[0], expected, atol=2e-7)

    def test_original_projectors_accept_all_six_force_channels_and_gradients(self):
        from mibot.models.VLA.XR1 import Projector
        self.torch.manual_seed(42)
        for output_dim in (12, 16):
            projector = Projector(60, output_dim, num_layers=2)
            state = self.torch.zeros(1, 1, 60, requires_grad=True)
            output = projector(state)
            output.square().sum().backward()
            self.assertIsNotNone(state.grad)
            for index in range(20, 26):
                changed = state.detach().clone()
                changed[..., index] = 0.5
                self.assertFalse(self.torch.equal(projector(changed), output.detach()))
            probe = self.torch.randn(1, 1, 60, requires_grad=True)
            projector(probe).square().sum().backward()
            self.assertTrue(self.torch.all(probe.grad[..., 20:26].abs() > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
