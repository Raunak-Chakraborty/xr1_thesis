"""CPU-only parameter-routing and AdamW-control tests."""
import unittest

import torch

from mibot.utils.doosan_motion_optimizer import (
    route_last_one,
    create_motion_optimizer,
)


class TestMotionOptimizer(unittest.TestCase):

    def example(self):
        params = [
            (
                "model.vlm.model.visual.blocks.23.attn.qkv.weight",
                torch.nn.Parameter(torch.randn(8, 12)),
            ),
            (
                "model.vlm.model.language_model.layers.35.mlp.weight",
                torch.nn.Parameter(torch.randn(12, 8)),
            ),
            (
                "model.dit.layers.35.mlp.fc1.weight",
                torch.nn.Parameter(torch.randn(7, 7)),
            ),
            (
                "model.action_output_layer.weight",
                torch.nn.Parameter(torch.randn(6, 7)),
            ),
            (
                "model.dit.layers.35.norm.weight",
                torch.nn.Parameter(torch.ones(7)),
            ),
            (
                "model.action_output_layer.bias",
                torch.nn.Parameter(torch.zeros(6)),
            ),
        ]

        frozen = torch.nn.Parameter(
            torch.randn(8, 8),
            requires_grad=False,
        )

        return params + [
            (
                "model.vlm.model.visual.blocks.22.attn.weight",
                frozen,
            )
        ]

    def test_partition(self):
        groups, report = route_last_one(
            self.example()
        )

        self.assertEqual(
            report["tensor_counts"],
            {
                "muon_hidden_matrix": 3,
                "adamw_decay": 1,
                "adamw_no_decay": 2,
            },
        )

        self.assertEqual(
            len({
                id(p)
                for values in groups.values()
                for p in values
            }),
            6,
        )

        self.assertEqual(
            len(report["sha256"]),
            64,
        )

    def test_deterministic_receipt(self):
        params = self.example()

        first = route_last_one(params)[1]["sha256"]
        second = route_last_one(
            params[::-1]
        )[1]["sha256"]

        self.assertEqual(first, second)

    def test_adamw_step_and_restore(self):
        params = self.example()

        opt, report = create_motion_optimizer(
            params,
            kind="adamw_control",
            adamw_lr=2e-5,
        )

        self.assertIsInstance(
            opt, torch.optim.AdamW
        )

        self.assertEqual(
            report["param_group_count"], 3
        )

        for _, p in params:
            if p.requires_grad:
                p.grad = torch.ones_like(p)

        opt.step()
        state = opt.state_dict()

        self.assertEqual(
            len(state["state"]), 6
        )

        restored, _ = create_motion_optimizer(
            params,
            kind="adamw_control",
            adamw_lr=2e-5,
        )

        restored.load_state_dict(state)

        self.assertEqual(
            len(restored.state_dict()["state"]),
            6,
        )

    def test_nonfp32_and_duplicate_fail(self):
        p = torch.nn.Parameter(
            torch.ones(8, 8, dtype=torch.float16)
        )

        with self.assertRaises(ValueError):
            route_last_one([
                ("model.dit.layers.35.mlp.weight", p)
            ])

        q = torch.nn.Parameter(
            torch.ones(8, 8)
        )

        with self.assertRaises(ValueError):
            route_last_one([
                ("model.dit.layers.35.mlp.weight", q),
                ("duplicate", q),
            ])

    def test_muon_backend_fail_closed(self):
        import importlib.util

        if importlib.util.find_spec(
            "pytorch_optimizer"
        ) is not None:
            self.skipTest(
                "Backend installed: real Muon test belongs "
                "to experimental dependency qualification"
            )

        with self.assertRaises(RuntimeError):
            create_motion_optimizer(
                self.example(),
                kind="hybrid_muon",
                adamw_lr=1e-5,
                muon_lr=2e-4,
            )

    def test_no_eligible_matrix_fails(self):
        with self.assertRaises(ValueError):
            route_last_one([
                (
                    "model.output.weight",
                    torch.nn.Parameter(torch.ones(4, 4)),
                )
            ])


if __name__ == "__main__":
    unittest.main()
