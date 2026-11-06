"""E1 compact checkpoint and strict last-N qualification regression tests."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from mibot.utils.claix_checkpoint import (
    load_verified, publish_checkpoint, validate_production_checkpoint,
)
from mibot.utils.doosan_compact_fp32 import install_compact_fp32
from mibot.utils.doosan_last_n import apply_last_n_policy
from test_claix_checkpoint import checkpoint_fixture


class LastTwoCompactTest(unittest.TestCase):
    def fixture(self):
        ck = checkpoint_fixture(step=30, horizon=48284)
        ck["xr1_compact_fp32"]["last_n"] = 2
        ck["xr1_production"]["qualification_last2_vram_probe"] = True
        return ck

    def test_original_last_one_still_valid(self):
        ck = checkpoint_fixture(step=30, horizon=48284)
        limit = validate_production_checkpoint(
            ck, ck["xr1_replay_contract"]
        )
        self.assertGreater(limit, 0)

    def test_last_two_requires_explicit_marker(self):
        ck = self.fixture()
        validate_production_checkpoint(
            ck, ck["xr1_replay_contract"]
        )

        missing = copy.deepcopy(ck)
        del missing["xr1_production"]["qualification_last2_vram_probe"]
        with self.assertRaises(RuntimeError):
            validate_production_checkpoint(
                missing, missing["xr1_replay_contract"]
            )

        foreign = copy.deepcopy(ck)
        foreign["xr1_production"]["experiment"] = {"run_id": "foreign"}
        with self.assertRaises(RuntimeError):
            validate_production_checkpoint(
                foreign, foreign["xr1_replay_contract"]
            )

        third = copy.deepcopy(ck)
        third["xr1_compact_fp32"]["last_n"] = 3
        with self.assertRaises(RuntimeError):
            validate_production_checkpoint(
                third, third["xr1_replay_contract"]
            )

        marked_last_one = checkpoint_fixture(step=30, horizon=48284)
        marked_last_one["xr1_production"]["qualification_last2_vram_probe"] = True
        with self.assertRaises(RuntimeError):
            validate_production_checkpoint(
                marked_last_one,
                marked_last_one["xr1_replay_contract"],
            )

    def test_last_two_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            ck = self.fixture()
            path = Path(folder) / "step-000030.ckpt"

            receipt = publish_checkpoint(ck, path)
            restored = load_verified(
                path,
                ck["xr1_replay_contract"],
                expected_config="a" * 64,
            )

            self.assertEqual(receipt["global_step"], 30)
            self.assertEqual(restored["global_step"], 30)
            self.assertEqual(
                restored["xr1_compact_fp32"]["last_n"], 2
            )
            self.assertTrue(
                restored["xr1_production"]["qualification_last2_vram_probe"]
            )
            self.assertEqual(set(restored["state_dict"]), {"model.weight"})

    def test_only_selected_trainable_parameters_are_serialized(self):
        class XR1Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm = torch.nn.Module()
                self.vlm.model = torch.nn.Module()

                self.vlm.model.visual = torch.nn.Module()
                self.vlm.model.visual.blocks = torch.nn.ModuleList(
                    torch.nn.Linear(2, 2) for _ in range(24)
                )

                self.vlm.model.language_model = torch.nn.Module()
                self.vlm.model.language_model.layers = torch.nn.ModuleList(
                    torch.nn.Linear(2, 2) for _ in range(36)
                )

                self.dit = torch.nn.Module()
                self.dit.layers = torch.nn.ModuleList(
                    torch.nn.Linear(2, 2) for _ in range(36)
                )

                for name in (
                    "state_projector", "action_projector",
                    "action_output_layer", "t_embedder",
                    "t_projector", "sink",
                ):
                    setattr(self, name, torch.nn.Linear(2, 2))

        class TinyRunner(torch.nn.Module):
            strict_loading = True

            def __init__(self):
                super().__init__()
                self.model = XR1Tiny().bfloat16()

            def on_save_checkpoint(self, checkpoint):
                pass

            def on_load_checkpoint(self, checkpoint):
                pass

        runner = TinyRunner()

        selected = apply_last_n_policy(runner.model, 2)
        full_parameter_count = len(dict(runner.named_parameters()))

        expected = {
            "model." + name
            for name in selected["trainable_names"]
        }

        install_compact_fp32(
            runner,
            selected_keys=expected,
            pretrained_sha256="a" * 64,
            language_sha256="b" * 64,
            source_commit="synthetic",
            last_n=2,
        )

        compact = runner.state_dict()

        self.assertEqual(set(compact), expected)
        self.assertLess(len(compact), full_parameter_count)
        self.assertEqual(
            sum(t.numel() for t in compact.values()),
            selected["trainable_parameters"],
        )
        self.assertTrue(
            all(t.dtype == torch.float32 for t in compact.values())
        )

        frozen = [
            p for p in runner.parameters() if not p.requires_grad
        ]
        self.assertTrue(frozen)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in frozen))

        for stack in (
            runner.model.vlm.model.visual.blocks,
            runner.model.vlm.model.language_model.layers,
            runner.model.dit.layers,
        ):
            self.assertTrue(
                all(not p.requires_grad
                    for block in stack[:-2]
                    for p in block.parameters())
            )
            self.assertTrue(
                all(p.requires_grad
                    for block in stack[-2:]
                    for p in block.parameters())
            )


if __name__ == "__main__":
    unittest.main()
