"""Dependency-free contract and strict-loader tests; no fabricated neural results."""

import ast
from pathlib import Path
import unittest

from mibot.utils.force_contract import (
    ForceEncoderConfig, WRENCH_CHANNELS, validate_force_experiment, validate_force_quantiles,
)
from mibot.utils.force_checkpoint import load_force_checkpoint, model_weights


class ForceContractTests(unittest.TestCase):
    def test_defaults_preserve_generic_model(self):
        cfg = ForceEncoderConfig.from_mapping()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.dual_path)
        self.assertEqual(cfg.pretrained_policy, "original")
        self.assertEqual(WRENCH_CHANNELS, ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"))

    def test_bad_configs_rejected(self):
        for values in ({"output_dim": 0}, {"hidden_dims": [True]}, {"hidden_dims": 64},
                       {"fusion_type": "moe"}, {"input_normalization": "raw"},
                       {"enabled": "false"}, {"pretrained_policy": "mystery"},
                       {"fusion_init_std": float("nan")}, {"init_seed": -1},
                       {"enabled": True, "fusion_init": "zero", "train_fusion": False}):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                ForceEncoderConfig.from_mapping(values)

    def test_force_stats_all_six_required(self):
        low, high = [[-1.] * 60], [[1.] * 60]
        validate_force_quantiles(low, high)
        for axis in range(20, 26):
            invalid = [high[0].copy()]
            invalid[0][axis] = low[0][axis]
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "wrench"):
                validate_force_quantiles(low, invalid)

    def test_force_stats_missing_padding_nonfinite_and_wrong_layout(self):
        for low, high in ((None, None), ([[0.] * 60], [[0.] * 60]),
                          ([[-1.] * 25], [[1.] * 25]),
                          ([[float("nan")] * 60], [[1.] * 60]),
                          ([[-1.] * 60], [[float("inf")] * 60])):
            with self.subTest(low=low), self.assertRaises(ValueError):
                validate_force_quantiles(low, high)

    def test_experiment_requires_doosan_wrench_and_horizon(self):
        model = {"force_encoder": {"enabled": True}}
        data = {"type": "DoosanDataModule", "params": {"train_datasets": {
            "use_wrench": True, "force_encoder_input": True, "action_length": 30,
            "q01": [[-1.] * 60], "q99": [[1.] * 60],
        }}}
        validate_force_experiment(model, data)
        for key, value in (("use_wrench", False), ("force_encoder_input", False), ("action_length", 29)):
            original = data["params"]["train_datasets"][key]
            data["params"]["train_datasets"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_force_experiment(model, data)
            data["params"]["train_datasets"][key] = original
        with self.assertRaises(ValueError):
            validate_force_experiment(model, {"type": "BaseDataModule"})

    def test_disabled_accepts_generic_data_contract(self):
        validate_force_experiment({}, {"type": "BaseDataModule"})


class Weight:
    def __init__(self, shape):
        self.shape = shape


class WeightsOnlyModule:
    """Only exercises loader validation control flow, not PyTorch numerics."""
    def __init__(self):
        self.weights = {"vlm.weight": Weight((2, 3)), "dit.weight": Weight((4, 5)),
                        "force_conditioner.encoder.weight": Weight((6, 7)),
                        "force_conditioner.fusion.weight": Weight((7, 8))}
        self.loaded = False

    def state_dict(self):
        return self.weights.copy()

    def named_parameters(self):
        return self.weights.items()

    def load_state_dict(self, state, strict):
        assert strict is True
        assert state.keys() == self.weights.keys()
        self.loaded = True
        return "strict load"


class StrictCheckpointControlTests(unittest.TestCase):
    def test_complete_checkpoint_strict(self):
        module = WeightsOnlyModule()
        self.assertEqual(load_force_checkpoint(module, module.weights), "strict load")

    def test_only_explicit_complete_new_module_exception(self):
        module = WeightsOnlyModule()
        base = {k: v for k, v in module.weights.items() if not k.startswith("force_conditioner.")}
        with self.assertRaises(RuntimeError):
            load_force_checkpoint(module, base)
        load_force_checkpoint(module, base, allow_base_checkpoint=True)
        self.assertTrue(module.loaded)

    def test_original_missing_unexpected_shape_and_partial_extension_fail_before_load(self):
        for alteration in ("original_missing", "unexpected", "shape", "partial"):
            module = WeightsOnlyModule()
            values = module.state_dict()
            if alteration == "original_missing":
                del values["vlm.weight"]
            elif alteration == "unexpected":
                values["mystery.weight"] = Weight((1,))
            elif alteration == "shape":
                values["dit.weight"] = Weight((5, 4))
            else:
                del values["force_conditioner.encoder.weight"]
            with self.subTest(alteration=alteration), self.assertRaises(RuntimeError):
                load_force_checkpoint(module, values, allow_base_checkpoint=True)
            self.assertFalse(module.loaded)

    def test_checkpoint_unwrap_does_not_discard_unexpected_keys(self):
        self.assertEqual(model_weights({"module": {"model.a": 1}}), {"a": 1})
        self.assertEqual(model_weights({"state_dict": {"model.a": 1}}), {"a": 1})
        self.assertEqual(model_weights({"module": {"model.a": 1, "other": 2}}),
                         {"model.a": 1, "other": 2})


class SourceSyntaxTests(unittest.TestCase):
    def test_all_xr1_python_parses(self):
        for path in Path(__file__).resolve().parents[1].rglob("*.py"):
            with self.subTest(path=path):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
