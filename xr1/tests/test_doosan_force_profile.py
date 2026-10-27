"""Dependency-free contract and source regression tests for Stage 4."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mibot.utils.doosan_profile import (
    NORMALIZATION_SCHEMA,
    WRENCH_ORDER,
    is_doosan_config,
    resolve_doosan_data,
)


ROOT = Path(__file__).resolve().parents[1]


def statistics(use_wrench=True):
    return {
        "use_wrench": use_wrench,
        "action_length": 30,
        "mean": [[0.0] * 60 for _ in range(30)],
        "std": [[1.0] * 7 + [0.0] * 53 for _ in range(30)],
        "q01": [[-10.0] * 60],
        "q99": [[20.0] * 60],
        "normalization_metadata": {
            "schema": NORMALIZATION_SCHEMA,
            "source": "dataset",
            "split": "train",
            "use_wrench": True,
            "dataset_id": "synthetic-unit-test-only",
            "num_state_samples": 100,
            "wrench_order": list(WRENCH_ORDER),
            "wrench_frame": "synthetic-test-frame",
            "force_unit": "N",
            "torque_unit": "Nm",
        },
    }


class ForceProfileTest(unittest.TestCase):
    def test_force_retains_each_wrench_quantile_without_mutation(self):
        data = statistics()
        data["q01"][0][20:26] = [-6, -5, -4, -3, -2, -1]
        data["q99"][0][20:26] = [1, 2, 3, 4, 5, 6]
        original = copy.deepcopy(data)
        resolved = resolve_doosan_data(data)
        self.assertEqual(resolved["q01"][0][20:26], [-6, -5, -4, -3, -2, -1])
        self.assertEqual(resolved["q99"][0][20:26], [1, 2, 3, 4, 5, 6])
        self.assertEqual(data, original)
        self.assertEqual(resolved["q01"][0][6], 0)
        self.assertEqual(resolved["q99"][0][26:], [0] * 34)

    def test_baseline_cannot_normalize_zero_wrench_into_nonzero_state(self):
        resolved = resolve_doosan_data(statistics(False))
        self.assertEqual(resolved["q01"][0][20:], [0] * 40)
        self.assertEqual(resolved["q99"][0][20:], [0] * 40)

    def test_legacy_baseline_does_not_require_force_metadata(self):
        data = statistics(False)
        del data["use_wrench"]
        del data["normalization_metadata"]
        self.assertIs(resolve_doosan_data(data)["use_wrench"], False)

    def test_every_missing_zero_constant_reversed_or_tiny_force_span_rejected(self):
        for index in range(20, 26):
            for lower, upper in ((0, 0), (1, 1), (2, 1), (0, 1e-8), (1e10, 1e10 + 1)):
                with self.subTest(index=index, lower=lower, upper=upper):
                    data = statistics()
                    data["q01"][0][index] = lower
                    data["q99"][0][index] = upper
                    with self.assertRaises(ValueError):
                        resolve_doosan_data(data)

    def test_nan_infinity_and_float32_overflow_rejected(self):
        for name in ("q01", "q99", "mean", "std"):
            for value in (float("nan"), float("inf"), -float("inf"), 1e50):
                with self.subTest(name=name, value=value):
                    data = statistics()
                    data[name][0][20] = value
                    with self.assertRaises(ValueError):
                        resolve_doosan_data(data)

    def test_missing_malformed_shapes_and_negative_std_rejected(self):
        for name in ("mean", "std", "q01", "q99"):
            for value in (None, [], [[0] * 59]):
                with self.subTest(name=name, value=value):
                    data = statistics()
                    data[name] = value
                    with self.assertRaises(ValueError):
                        resolve_doosan_data(data)
        data = statistics()
        data["std"][0][0] = -1
        with self.assertRaises(ValueError):
            resolve_doosan_data(data)

    def test_force_action_statistics_and_wrong_horizon_rejected(self):
        for name in ("mean", "std"):
            data = statistics()
            data[name][0][20] = 1
            with self.assertRaisesRegex(ValueError, "action mean/std"):
                resolve_doosan_data(data)
        data = statistics()
        data["action_length"] = 29
        with self.assertRaisesRegex(ValueError, "30"):
            resolve_doosan_data(data)

    def test_force_requires_converter_provenance_and_axis_units(self):
        for name in statistics()["normalization_metadata"]:
            data = statistics()
            del data["normalization_metadata"][name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_doosan_data(data)
        for name, value in (("split", "test"), ("wrench_order", list(reversed(WRENCH_ORDER))),
                            ("num_state_samples", True), ("use_wrench", 1),
                            ("force_unit", "kN"), ("wrench_frame", "")):
            data = statistics()
            data["normalization_metadata"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_doosan_data(data)

    def test_boolean_strings_not_truthy_flags(self):
        for value in ("false", "true", 0, 1, None):
            data = statistics()
            data["use_wrench"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_doosan_data(data)

    def test_json_resolution_matches_inline_and_is_idempotent(self):
        data = statistics()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            resolved = resolve_doosan_data({
                "normalization_path": str(path), "use_wrench": True,
                "mean": None, "std": None, "q01": None, "q99": None,
            })
            self.assertEqual(resolved, resolve_doosan_data(data))
            self.assertEqual(resolve_doosan_data(resolved), resolved)
            self.assertNotIn("normalization_path", resolved)
            with self.assertRaisesRegex(ValueError, "not both"):
                resolve_doosan_data(dict(data, normalization_path=str(path)))

    def test_only_doosan_profiles_selected(self):
        self.assertTrue(is_doosan_config({"type": "DoosanDataModule"}))
        self.assertTrue(is_doosan_config({"params": {"type": "doosan_json"}}))
        self.assertFalse(is_doosan_config({"type": "BaseDataModule"}))

    def test_validator_cli_accepts_measured_contract_and_rejects_zero_force_stats(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "synthetic.json"
            data = statistics()
            path.write_text(json.dumps(data), encoding="utf-8")
            command = [sys.executable, "-B", str(ROOT / "tools/validate_doosan_normalization.py"),
                       str(path), "--use-wrench"]
            valid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertIn("use_wrench=True", valid.stdout)
            data["q01"][0][20:26] = [0] * 6
            data["q99"][0][20:26] = [0] * 6
            path.write_text(json.dumps(data), encoding="utf-8")
            invalid = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("wrench index 20", invalid.stderr)


class ForceSourceTest(unittest.TestCase):
    def source(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_exact_adapter_contract_constants(self):
        tree = ast.parse(self.source("mibot/utils/doosan.py"))
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                constants[node.targets[0].id] = ast.unparse(node.value)
        self.assertEqual(constants["WRENCH_SLICE"], "slice(20, 26)")
        self.assertEqual(constants["DOOSAN_ACTION_SLICE"], "slice(0, 7)")
        self.assertEqual(constants["DOOSAN_PROPRIO_DIM"], "19")
        self.assertEqual(constants["DOOSAN_FORCE_AWARE_STATE_DIM"], "25")

    def test_original_vlm_and_dit_receive_full_state_without_force_encoder(self):
        source = self.source("mibot/models/VLA/XR1.py")
        self.assertIn("self.state_shape = (1, 60)", source)
        self.assertIn("self.action_shape = (30, 60)", source)
        self.assertIn("self.state_projector_choice = Projector(self.state_shape[-1],", source)
        self.assertIn("self.state_projector = Projector(self.state_shape[-1],", source)
        self.assertIn("self.state_projector_choice(state.flatten(0, 1))", source)
        self.assertIn("state_embed = self.state_projector(state)", source)
        self.assertIn("torch.cat([sink, state_embed, noisy_action]", source)
        self.assertNotIn("force_encoder", source)
        self.assertNotIn("wrench", source)
        vlm = self.source("mibot/models/VLM/qwen3vl.py")
        self.assertIn("inputs_embeds.masked_scatter(expanded_state_mask, state_embeds)", vlm)

    def test_training_and_deployment_share_profile_resolution(self):
        for name in ("mibot/utils/cfg_utils.py", "mibot/data/datasets/doosan_json_dataset.py",
                     "mibot/server/deploy.py"):
            with self.subTest(name=name):
                self.assertIn("resolve_doosan_data(", self.source(name))
        self.assertIn("build_doosan_action_mask if doosan else build_action_mask",
                      self.source("mibot/server/deploy.py"))
        self.assertIn("action = action * action_mask",
                      self.source("mibot/data/datasets/doosan_json_dataset.py"))

    def test_force_configuration_inherits_existing_doosan_pipeline(self):
        force = self.source("configs/data/doosan_force_state.yaml")
        baseline = self.source("configs/data/doosan.yaml")
        self.assertIn("- doosan", force)
        self.assertIn("use_wrench: true", force)
        self.assertIn("normalization_path: ???", force)
        self.assertIn("use_wrench: false", baseline)
        self.assertIn("type: DoosanDataModule", baseline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
