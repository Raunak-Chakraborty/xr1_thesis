#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from mmengine import Config

from mibot.server.deploy import load_stats
from mibot.server.runtime.server import Server
from mibot.utils.doosan import (
    DOOSAN_ACTION_SLICE,
    DOOSAN_FORCE_AWARE_STATE_DIM,
    DOOSAN_PROPRIO_DIM,
    DOOSAN_SEMANTIC_ACTION_DIM,
    GRIPPER_STATE_INDEX,
    JOINT_POS_SLICE,
    JOINT_VEL_SLICE,
    TCP_POS_SLICE,
    TCP_ROTVEC_SLICE,
    WRENCH_SLICE,
    build_doosan_action_mask,
    compose_doosan_action,
    compose_doosan_state,
    extract_doosan_action,
)
from mibot.utils.io import (
    ACTION_DIM,
    STATE_DIM,
    build_action_mask,
    compose_state,
    denormalize_action,
    normalize_action,
    normalize_quantile,
)


class DoosanRuntimeContractTest(unittest.TestCase):
    @staticmethod
    def config(module_type="DoosanDataModule", dataset_type="doosan_json", use_wrench=False, action_length=30):
        mean = np.linspace(-0.05, 0.05, action_length * ACTION_DIM, dtype=np.float32).reshape(action_length, ACTION_DIM)
        std = np.full_like(mean, 0.25)
        q01 = np.zeros((1, STATE_DIM), dtype=np.float32)
        q99 = np.zeros_like(q01)
        if module_type == "DoosanDataModule":
            std[:, DOOSAN_ACTION_SLICE.stop:] = 0.0
            for state_slice in (JOINT_POS_SLICE, TCP_POS_SLICE, TCP_ROTVEC_SLICE, JOINT_VEL_SLICE):
                q01[:, state_slice] = -1.0
                q99[:, state_slice] = 1.0
            q99[:, GRIPPER_STATE_INDEX] = 1.0
            if use_wrench:
                q01[:, WRENCH_SLICE] = -2.0
                q99[:, WRENCH_SLICE] = 2.0
        else:
            q01[:, :16] = -2.0
            q99[:, :16] = 3.0
            q01[:, [7, 15]] = 0.0
            q99[:, [7, 15]] = 1.0
        return Config({
            "data": {
                "type": module_type,
                "params": {
                    "type": dataset_type,
                    "train_datasets": {
                        "action_length": action_length,
                        "mean": mean,
                        "std": std,
                        "q01": q01,
                        "q99": q99,
                    },
                },
            },
        })

    @staticmethod
    def state(use_wrench=False):
        return compose_doosan_state(
            joint_pos=[-3.0, -0.5, 0.0, 0.5, 1.0, 3.0],
            gripper=0.25,
            tcp_pos=[0.1, -0.2, 0.3],
            tcp_rotvec=[0.01, -0.02, 0.03],
            joint_vel=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            wrench=[-3.0, -1.0, 0.0, 0.5, 1.0, 3.0] if use_wrench else None,
        )

    @classmethod
    def request(cls, batch_size=1, use_wrench=False):
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.int64),
            "state": torch.from_numpy(np.stack([cls.state(use_wrench)] * batch_size)),
            "pixel_values": torch.ones((batch_size, 3, 2, 2), dtype=torch.float32),
            "task": "insert the peg",
        }

    @classmethod
    def generic_request(cls, batch_size=1):
        request = cls.request(batch_size)
        state = compose_state(
            left_gripper=[0.25],
            left_joint=[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
            right_gripper=[0.75],
            right_joint=[3.0, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0],
        )
        request["state"] = torch.from_numpy(np.stack([state] * batch_size))
        return request

    def server(self, cfg=None):
        stats = load_stats(self.config() if cfg is None else cfg, "cpu")
        return Server("127.0.0.1", 0, None, *stats, device="cpu")

    def test_doosan_deployment_mask_matches_training_datamodule(self):
        for dataset_type in ("doosan_json", None):
            with self.subTest(dataset_type=dataset_type):
                cfg = self.config(dataset_type=dataset_type)
                if dataset_type is None:
                    del cfg.data.params["type"]
                *stats, mask = load_stats(cfg, "cpu")
                self.assertEqual(mask.shape, (30, 60))
                self.assertEqual(mask.dtype, torch.int32)
                self.assertEqual(mask.device.type, "cpu")
                np.testing.assert_array_equal(mask.numpy(), build_doosan_action_mask(30))
                np.testing.assert_array_equal(np.flatnonzero(mask[0].numpy()), np.arange(7))
                self.assertTrue(torch.all(mask.sum(dim=-1) == 7))
                self.assertEqual([tensor.shape for tensor in stats], [(30, 60), (30, 60), (1, 60), (1, 60)])
                self.assertTrue(all(tensor.dtype == torch.float32 for tensor in stats))
                self.assertTrue(all(tensor.device.type == "cpu" for tensor in stats))

    def test_generic_deployment_mask_is_unchanged(self):
        for module_type, dataset_type in (
            ("BaseDataModule", "json"),
            ("BaseDataModule", None),
            ("OtherDataModule", "other"),
            (None, None),
        ):
            with self.subTest(module_type=module_type, dataset_type=dataset_type):
                cfg = self.config(module_type, dataset_type)
                if module_type is None:
                    del cfg.data["type"]
                if dataset_type is None:
                    del cfg.data.params["type"]
                mask = load_stats(cfg, "cpu")[-1]
                np.testing.assert_array_equal(mask.numpy(), build_action_mask(30))
                np.testing.assert_array_equal(
                    np.flatnonzero(mask[0].numpy()),
                    list(range(7)) + list(range(8, 15)) + list(range(16, 20)),
                )
                self.assertEqual(mask.shape, (30, 60))
                self.assertEqual(mask.dtype, torch.int32)

    def test_conflicting_deployment_markers_are_rejected(self):
        for module_type, dataset_type in (
            ("DoosanDataModule", "json"),
            ("DoosanDataModule", "other"),
            ("BaseDataModule", "doosan_json"),
            ("OtherDataModule", "doosan_json"),
            (None, "doosan_json"),
        ):
            with self.subTest(module_type=module_type, dataset_type=dataset_type):
                cfg = self.config(module_type, dataset_type)
                if module_type is None:
                    del cfg.data["type"]
                with self.assertRaisesRegex(ValueError, "requires"):
                    load_stats(cfg, "cpu")

    def test_action_horizon_fallback_is_preserved(self):
        for module_type, dataset_type in (("DoosanDataModule", "doosan_json"), ("BaseDataModule", "json")):
            for dataset_length, parent_length, expected_length in ((12, 18, 12), (12, None, 12), (None, 18, 18), (None, None, 30)):
                with self.subTest(module_type=module_type, dataset_length=dataset_length, parent_length=parent_length):
                    cfg = self.config(module_type, dataset_type, action_length=expected_length)
                    if dataset_length is None:
                        del cfg.data.params.train_datasets["action_length"]
                    if parent_length is not None:
                        cfg.data.params.action_length = parent_length
                    server = self.server(cfg)
                    batch = server.prepare_batch(self.request(batch_size=2))
                    self.assertEqual(server.mean.shape, (expected_length, 60))
                    self.assertEqual(server.std.shape, (expected_length, 60))
                    self.assertEqual(server.action_mask.shape, (expected_length, 60))
                    self.assertEqual(batch["action"].shape, (2, expected_length, 60))
                    self.assertEqual(batch["action_mask"].shape, (2, expected_length, 60))

    def test_action_horizon_must_match_statistics(self):
        cfg = self.config(action_length=12)
        cfg.data.params.train_datasets.action_length = 18
        with self.assertRaisesRegex(ValueError, "mean expected shape"):
            load_stats(cfg, "cpu")

    def test_state_normalization_matches_existing_preprocessing_contract(self):
        for use_wrench in (False, True):
            for batch_size in (1, 2):
                with self.subTest(use_wrench=use_wrench, batch_size=batch_size):
                    server = self.server(self.config(use_wrench=use_wrench))
                    request = self.request(batch_size, use_wrench)
                    originals = {key: value.clone() for key, value in request.items() if isinstance(value, torch.Tensor)}
                    batch = server.prepare_batch(request)
                    expected = normalize_quantile(self.state(use_wrench), server.q01.numpy(), server.q99.numpy())
                    np.testing.assert_array_equal(batch["state"].numpy(), np.stack([expected] * batch_size))
                    for key, original in originals.items():
                        torch.testing.assert_close(request[key], original, rtol=0, atol=0)
                    self.assertEqual(batch["state"].shape, (batch_size, 1, 60))
                    self.assertEqual(batch["state"].dtype, torch.float32)
                    self.assertTrue(torch.isfinite(batch["state"]).all())
                    self.assertTrue(torch.all(batch["state"].abs() <= 1.0))
                    self.assertTrue(torch.all(batch["state"][..., server.q99[0] == server.q01[0]] == 0))
                    self.assertIs(batch["pixel_values"], request["pixel_values"])
                    self.assertEqual(batch["input_ids"].dtype, torch.int64)
                    self.assertEqual(batch["task"], request["task"])
                    self.assertNotIn("action", request)
                    self.assertNotIn("action_mask", request)

    def test_state_normalization_matches_training_for_offset_ranges(self):
        for module_type, dataset_type, use_wrench in (
            ("DoosanDataModule", "doosan_json", False),
            ("DoosanDataModule", "doosan_json", True),
            ("BaseDataModule", "json", False),
        ):
            for batch_size in (1, 2):
                for lower, upper, value in ((1.0, 1.0 + 2**-20, 1.0 + 2**-21), (100.0, 100.01, 100.005)):
                    with self.subTest(module_type=module_type, use_wrench=use_wrench, batch_size=batch_size, lower=lower):
                        cfg = self.config(module_type, dataset_type, use_wrench)
                        cfg.data.params.train_datasets.q01[0, 0] = lower
                        cfg.data.params.train_datasets.q99[0, 0] = upper
                        server = self.server(cfg)
                        request = self.request(batch_size, use_wrench)
                        request["state"][..., 0] = value
                        batch = server.prepare_batch(request)
                        expected = np.stack([
                            normalize_quantile(state.numpy(), server.q01.numpy(), server.q99.numpy())
                            for state in request["state"]
                        ])
                        np.testing.assert_array_equal(batch["state"].numpy(), expected)
                        if lower == 1.0:
                            self.assertTrue(torch.all(batch["state"][..., 0] == -0.5))

    def test_state_normalization_clipping_boundaries(self):
        for use_wrench in (False, True):
            for batch_size in (1, 2):
                for fraction in (-1.0, 0.0, 0.5, 1.0, 2.0):
                    with self.subTest(use_wrench=use_wrench, batch_size=batch_size, fraction=fraction):
                        server = self.server(self.config(use_wrench=use_wrench))
                        request = self.request(batch_size, use_wrench)
                        valid = server.q99[0] > server.q01[0]
                        request["state"].fill_(99.0)
                        request["state"][..., valid] = (
                            server.q01[..., valid]
                            + fraction * (server.q99[..., valid] - server.q01[..., valid])
                        )
                        batch = server.prepare_batch(request)
                        expected = np.stack([
                            normalize_quantile(state.numpy(), server.q01.numpy(), server.q99.numpy())
                            for state in request["state"]
                        ])
                        np.testing.assert_array_equal(batch["state"].numpy(), expected)
                        self.assertTrue(torch.all(batch["state"][..., ~valid] == 0))
                        if fraction < 0.0:
                            self.assertTrue(torch.all(batch["state"][..., valid] == -1.0))
                        elif fraction > 1.0:
                            self.assertTrue(torch.all(batch["state"][..., valid] == 1.0))

    def test_generic_state_normalization_preserves_contract(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                server = self.server(self.config("BaseDataModule", "json"))
                request = self.generic_request(batch_size)
                original = request["state"].clone()
                batch = server.prepare_batch(request)
                expected = np.stack([
                    normalize_quantile(state.numpy(), server.q01.numpy(), server.q99.numpy())
                    for state in original
                ])
                np.testing.assert_array_equal(batch["state"].numpy(), expected)
                self.assertEqual(batch["state"].shape, (batch_size, 1, STATE_DIM))
                self.assertEqual(batch["state"].dtype, torch.float32)
                self.assertTrue(torch.all(batch["state"][..., 16:] == 0))
                torch.testing.assert_close(request["state"], original, rtol=0, atol=0)

    def test_zero_quantiles_still_suppress_force_channels(self):
        server = self.server()
        request = self.request(use_wrench=True)
        batch = server.prepare_batch(request)
        self.assertTrue(torch.all(batch["state"][..., WRENCH_SLICE] == 0))

    def test_missing_action_preserves_bfloat16_zero_initialization(self):
        for module_type, dataset_type in (("DoosanDataModule", "doosan_json"), ("BaseDataModule", "json")):
            for batch_size in (1, 2):
                with self.subTest(module_type=module_type, batch_size=batch_size):
                    server = self.server(self.config(module_type, dataset_type))
                    batch = server.prepare_batch(self.request(batch_size))
                    self.assertEqual(batch["action"].shape, (batch_size, 30, 60))
                    self.assertEqual(batch["action"].dtype, torch.bfloat16)
                    self.assertEqual(batch["action"].device.type, "cpu")
                    self.assertEqual(torch.count_nonzero(batch["action"]).item(), 0)
                    self.assertEqual(batch["action_mask"].dtype, torch.int32)
                    torch.testing.assert_close(batch["action_mask"], server.action_mask.unsqueeze(0).expand(batch_size, -1, -1))

    def test_semantic_action_normalization_and_roundtrip(self):
        server = self.server()
        translation = np.arange(90, dtype=np.float32).reshape(30, 3) / 1000.0
        rotation = -translation / 10.0
        gripper = np.linspace(0.0, 1.0, 30, dtype=np.float32)
        action = compose_doosan_action(translation, rotation, gripper)
        semantic = np.concatenate((translation, rotation, gripper[:, None]), axis=-1)
        np.testing.assert_array_equal(extract_doosan_action(action), semantic)
        self.assertEqual(action.shape, (30, 60))
        self.assertEqual(action.dtype, np.float32)
        request = self.request(batch_size=2)
        request["action"] = torch.from_numpy(np.stack([action] * 2))
        original = request["action"].clone()
        batch = server.prepare_batch(request)
        expected = normalize_action(action, server.mean.numpy(), server.std.numpy()) * server.action_mask.numpy()
        np.testing.assert_allclose(batch["action"].numpy(), np.stack([expected] * 2), rtol=1e-6, atol=1e-6)
        self.assertEqual(batch["action"].shape, (2, 30, 60))
        self.assertEqual(batch["action"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch["action"]).all())
        self.assertTrue(torch.all(batch["action"][..., DOOSAN_ACTION_SLICE.stop:] == 0))
        torch.testing.assert_close(request["action"], original)
        restored = denormalize_action(batch["action"], server.mean, server.std) * batch["action_mask"]
        np.testing.assert_allclose(extract_doosan_action(restored.numpy()), np.stack([semantic] * 2), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.all(restored[..., DOOSAN_ACTION_SLICE.stop:] == 0))

    def test_action_prefix_normalization_preserves_request(self):
        for module_type, dataset_type in (("DoosanDataModule", "doosan_json"), ("BaseDataModule", "json")):
            for batch_size in (1, 2):
                with self.subTest(module_type=module_type, batch_size=batch_size):
                    server = self.server(self.config(module_type, dataset_type))
                    request = self.request(batch_size) if module_type == "DoosanDataModule" else self.generic_request(batch_size)
                    prefix_length = 4
                    action = np.zeros((30, ACTION_DIM), dtype=np.float32)
                    if module_type == "DoosanDataModule":
                        motion = np.linspace(-0.02, 0.03, prefix_length * 6, dtype=np.float32).reshape(prefix_length, 6)
                        action[:prefix_length] = compose_doosan_action(
                            motion[:, :3], motion[:, 3:], np.linspace(0.0, 1.0, prefix_length, dtype=np.float32)
                        )
                    else:
                        action[:prefix_length] = np.linspace(-0.2, 0.3, prefix_length * ACTION_DIM, dtype=np.float32).reshape(prefix_length, ACTION_DIM)
                    action *= server.action_mask.numpy()
                    request["action"] = torch.from_numpy(np.stack([action] * batch_size))
                    request["prefix_length"] = prefix_length
                    originals = {key: value.clone() for key, value in request.items() if isinstance(value, torch.Tensor)}
                    original_keys = set(request)
                    batch = server.prepare_batch(request)
                    expected = normalize_action(action, server.mean.numpy(), server.std.numpy()) * server.action_mask.numpy()
                    np.testing.assert_array_equal(batch["action"].numpy(), np.stack([expected] * batch_size))
                    self.assertEqual(batch["action"].dtype, torch.float32)
                    self.assertEqual(batch["prefix_length"], prefix_length)
                    self.assertEqual(batch["task"], request["task"])
                    self.assertEqual(set(request), original_keys)
                    for key, original in originals.items():
                        torch.testing.assert_close(request[key], original, rtol=0, atol=0)
                    restored = denormalize_action(batch["action"], server.mean, server.std) * batch["action_mask"]
                    np.testing.assert_allclose(restored.numpy(), np.stack([action] * batch_size), rtol=1e-6, atol=1e-7)
                    self.assertTrue(torch.all(restored[batch["action_mask"] == 0] == 0))
                    if module_type == "BaseDataModule":
                        self.assertTrue(torch.any(restored[:, :prefix_length, 8:15] != 0))
                        self.assertTrue(torch.any(restored[:, :prefix_length, 16:20] != 0))

    def test_run_masks_output_when_model_mutates_batch_without_network(self):
        for module_type, dataset_type in (("DoosanDataModule", "doosan_json"), ("BaseDataModule", "json")):
            for batch_size in (1, 2):
                with self.subTest(module_type=module_type, batch_size=batch_size):
                    server = self.server(self.config(module_type, dataset_type))
                    request = self.request(batch_size) if module_type == "DoosanDataModule" else self.generic_request(batch_size)
                    originals = {key: value.clone() for key, value in request.items() if isinstance(value, torch.Tensor)}
                    original_keys = set(request)

                    def generate(batch):
                        action = batch.pop("action")
                        action_mask = batch.pop("action_mask")
                        state = batch.pop("state")
                        self.assertEqual(action_mask.shape, (batch_size, 30, ACTION_DIM))
                        self.assertEqual(state.shape, (batch_size, 1, STATE_DIM))
                        return torch.ones_like(action, dtype=torch.float32)

                    server.model = Mock()
                    server.model.generate.side_effect = generate
                    connection = Mock()
                    with (
                        patch("mibot.server.runtime.server.socket.socket") as socket_factory,
                        patch("mibot.server.runtime.server.tqdm"),
                        patch.object(server, "_recv", side_effect=[request, None]),
                        patch.object(server, "_send") as send,
                        patch.object(server, "prepare_batch", wraps=server.prepare_batch) as prepare,
                    ):
                        socket_factory.return_value.__enter__.return_value.accept.side_effect = [(connection, None), KeyboardInterrupt]
                        with self.assertRaises(KeyboardInterrupt):
                            server.run()
                    prepare.assert_called_once_with(request)
                    server.model.generate.assert_called_once()
                    self.assertNotIn("action_mask", server.model.generate.call_args.args[0])
                    send.assert_called_once()
                    sent_connection, action = send.call_args.args
                    self.assertIs(sent_connection, connection)
                    mask = server.action_mask.unsqueeze(0).expand(batch_size, -1, -1)
                    expected = denormalize_action(torch.ones((batch_size, 30, ACTION_DIM)), server.mean, server.std) * mask
                    torch.testing.assert_close(action, expected, rtol=0, atol=0)
                    self.assertTrue(torch.all(action[mask == 0] == 0))
                    self.assertEqual(action.dtype, torch.float32)
                    self.assertEqual(action.device.type, "cpu")
                    self.assertEqual(set(request), original_keys)
                    for key, original in originals.items():
                        torch.testing.assert_close(request[key], original, rtol=0, atol=0)
                    if module_type == "BaseDataModule":
                        self.assertTrue(torch.all(action[..., 8:15] != 0))
                        self.assertTrue(torch.all(action[..., 16:20] != 0))
                    connection.close.assert_called_once()

    def test_adapter_dimensions_are_unchanged(self):
        self.assertEqual(DOOSAN_PROPRIO_DIM, 19)
        self.assertEqual(DOOSAN_FORCE_AWARE_STATE_DIM, 25)
        self.assertEqual(DOOSAN_SEMANTIC_ACTION_DIM, 7)
        self.assertEqual(DOOSAN_ACTION_SLICE, slice(0, 7))
        self.assertEqual(STATE_DIM, 60)
        self.assertEqual(ACTION_DIM, 60)


if __name__ == "__main__":
    unittest.main()
