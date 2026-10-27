"""Actual dataset/normalization methods with synthetic I/O, never real episodes."""

import ast
from collections.abc import Mapping
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
ROOT = Path(__file__).resolve().parents[1]
if AVAILABLE:
    import torch
    import numpy as np
    from mibot.utils.force_contract import FORCE_NORMALIZATION_ID, ForceEncoderConfig, validate_force_quantiles
    from mibot.utils.force_normalization import normalize_force_state


def definitions(path, namespace, selected=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign))
             and (selected is None or getattr(n, "name", None) in selected)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)


@unittest.skipUnless(AVAILABLE, "PyTorch/NumPy unavailable; data-path numerical tests not executed")
class ForceDataTests(unittest.TestCase):
    def setUp(self):
        ns = dict(np=np, torch=torch, Mapping=Mapping, ACTION_DIM=60, STATE_DIM=60, ACTION_EPS=1e-6)
        definitions("mibot/utils/io.py", ns, {"get_value", "normalize_quantile", "normalize_action"})
        definitions("mibot/utils/doosan.py", ns)
        # Use the actual parent's window/padding methods, avoiding image imports.
        parent = ast.parse((ROOT / "mibot/data/datasets/json_dataset.py").read_text())
        cls = next(n for n in parent.body if isinstance(n, ast.ClassDef))
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_pad", "_mask"}]
        cls.body = methods
        cls.bases = []
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "JsonDataset", "exec"), ns)
        ns.update(FORCE_NORMALIZATION_ID=FORCE_NORMALIZATION_ID, validate_force_quantiles=validate_force_quantiles)
        definitions("mibot/data/datasets/doosan_json_dataset.py", ns)
        self.ns = ns
        self.dataset = ns["DoosanJsonDataset"].__new__(ns["DoosanJsonDataset"])
        ds = self.dataset
        ds.action_length = 30
        ds.use_wrench = ds.force_encoder_input = True
        ds.semantic_action_key = "actions.semantic"
        ds.mean = np.zeros((30, 60), dtype=np.float32)
        ds.std = np.ones((30, 60), dtype=np.float32)
        ds.q01 = np.full((1, 60), -10., dtype=np.float32)
        ds.q99 = np.full((1, 60), 20., dtype=np.float32)
        ds.samples = [{"file": "synthetic-only", "frame_index": 2}]
        self.traj = {"num_frames": 5, "proprios": {
            "joint_pos": [[.1] * 6] * 5, "gripper": [1.] * 5, "tcp_pos": [[.2] * 3] * 5,
            "tcp_rotvec": [[.3] * 3] * 5, "joint_vel": [[.4] * 6] * 5,
            "wrench": [[float(i + j) for j in range(6)] for i in range(5)]},
            "actions": {"semantic": [[i * .001] * 6 + [float(i % 2)] for i in range(5)]}}
        ds._read_json = lambda _: self.traj
        ds._prompt = lambda _: {"conversations": [], "images": []}
        ds._images = lambda *args: []
        ds._augment = lambda images: images
        ds._messages = lambda conversations, images: conversations

    def test_dataset_uses_current_timestep_and_once_normalized_wrench(self):
        result = self.dataset[0]
        ds = self.dataset
        raw = ds._doosan_state(self.traj, 2)
        np.testing.assert_array_equal(raw[0, 20:26], [2, 3, 4, 5, 6, 7])
        expected = self.ns["normalize_quantile"](raw, ds.q01, ds.q99)
        torch.testing.assert_close(result["state"], torch.from_numpy(expected), rtol=0, atol=0)
        self.assertEqual(result["force_normalization"].item(), 1)
        self.assertEqual(result["state"].shape, (1, 60))
        self.assertEqual(result["action"].shape, (30, 60))
        self.assertTrue(torch.all(result["action_mask"][:3, :7] == 1))
        self.assertEqual(torch.count_nonzero(result["action_mask"][3:]).item(), 0)
        self.assertEqual(torch.count_nonzero(result["action_mask"][:, 7:]).item(), 0)
        expected_labels = np.asarray(self.traj["actions"]["semantic"][2:], dtype=np.float32) / (1 + 1e-6)
        torch.testing.assert_close(result["action"][:3, :7], torch.from_numpy(expected_labels))
        # Force perturbations cannot alter semantic labels or masks.
        self.traj["proprios"]["wrench"][2] = [99.] * 6
        second = ds[0]
        torch.testing.assert_close(second["action"], result["action"], rtol=0, atol=0)
        torch.testing.assert_close(second["action_mask"], result["action_mask"], rtol=0, atol=0)

    def test_dataset_rejects_missing_malformed_and_nonfinite_wrench(self):
        for value in ([[1, 2, 3], [4, 5, 6]], [float("nan")] * 6, [float("inf")] * 6):
            self.traj["proprios"]["wrench"][2] = value
            with self.assertRaises(ValueError):
                self.dataset[0]
        del self.traj["proprios"]["wrench"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.dataset[0]

    def test_numpy_and_torch_normalization_match_including_clipping(self):
        raw = self.dataset._doosan_state(self.traj, 2)
        for values in ([0.] * 6, [-1e12, 1e12, 2., 3., 4., 5.]):
            raw[0, 20:26] = values
            expected = self.ns["normalize_quantile"](raw, self.dataset.q01, self.dataset.q99)
            actual = normalize_force_state(torch.from_numpy(raw)[None], torch.from_numpy(self.dataset.q01),
                                           torch.from_numpy(self.dataset.q99))
            torch.testing.assert_close(actual[0], torch.from_numpy(expected), rtol=0, atol=0)

    def test_server_matches_training_and_rejects_missing_and_double_normalized_input(self):
        ns = dict(torch=torch, FORCE_NORMALIZATION_ID=1, ACTION_EPS=1e-6,
                  normalize_force_state=normalize_force_state, validate_force_quantiles=validate_force_quantiles)
        definitions("mibot/server/runtime/server.py", ns, {"Server"})
        mask = torch.zeros(30, 60, dtype=torch.int32)
        mask[:, :7] = 1
        server = ns["Server"]("unused", 0, SimpleNamespace(force_encoder_config=ForceEncoderConfig(enabled=True)),
                              torch.zeros(30, 60), torch.ones(30, 60),
                              torch.from_numpy(self.dataset.q01), torch.from_numpy(self.dataset.q99), mask, "cpu")
        raw = torch.from_numpy(self.dataset._doosan_state(self.traj, 2))[None]
        output = server.normalize_observation({"state": raw, "wrench_present": True})
        torch.testing.assert_close(output["state"][0], self.dataset[0]["state"], rtol=0, atol=0)
        self.assertEqual(output["force_normalization"].dtype, torch.int64)
        for extra in ({}, {"wrench_present": False}, {"wrench_present": True, "force_normalization": torch.tensor([1])}):
            with self.assertRaises(ValueError):
                server.normalize_observation({"state": raw, **extra})


if __name__ == "__main__":
    unittest.main()
