#!/usr/bin/env python3

import unittest

import numpy as np

from mibot.server.runtime.doosan_client import (
    DOOSAN_ACTION_DIM,
    XR1_ACTION_DIM,
    XR1_ACTION_HORIZON,
    DoosanClient,
)


class FakeImage:
    pass


class DoosanClientContractTest(unittest.TestCase):

    def test_exactly_two_images_in_prompt(self):
        external = FakeImage()
        tcp = FakeImage()

        messages = DoosanClient._messages(
            "insert the peg",
            external,
            tcp,
        )

        self.assertEqual(
            len(messages),
            2,
        )

        user_content = messages[0]["content"]

        image_items = [
            item
            for item in user_content
            if item["type"] == "image"
        ]

        self.assertEqual(
            len(image_items),
            2,
        )

        self.assertIs(
            image_items[0]["image"],
            external,
        )

        self.assertIs(
            image_items[1]["image"],
            tcp,
        )

        text = "".join(
            item["text"]
            for item in user_content
            if item["type"] == "text"
        )

        self.assertIn(
            "External Workspace View",
            text,
        )

        self.assertIn(
            "TCP / Wrist View",
            text,
        )

        self.assertIn(
            "insert the peg",
            text,
        )

        self.assertNotIn(
            "Left-Wrist",
            text,
        )

        self.assertNotIn(
            "Right-Wrist",
            text,
        )

    def test_state_packing(self):
        robot_state = {
            "joint_pos": np.arange(
                1,
                7,
                dtype=np.float32,
            ),
            "gripper": 0.25,
            "tcp_pos": np.array(
                [0.1, 0.2, 0.3],
                dtype=np.float32,
            ),
            "tcp_rotvec": np.array(
                [0.01, 0.02, 0.03],
                dtype=np.float32,
            ),
            "joint_vel": np.arange(
                11,
                17,
                dtype=np.float32,
            ),
        }

        state = DoosanClient.pack_state(
            robot_state
        )

        self.assertEqual(
            state.shape,
            (1, 60),
        )

        np.testing.assert_allclose(
            state[0, :6],
            robot_state["joint_pos"],
        )

        self.assertEqual(
            state[0, 6],
            0.0,
        )

        self.assertAlmostEqual(
            float(state[0, 7]),
            0.25,
        )

        np.testing.assert_allclose(
            state[0, 8:11],
            robot_state["tcp_pos"],
        )

        np.testing.assert_allclose(
            state[0, 11:14],
            robot_state["tcp_rotvec"],
        )

        np.testing.assert_allclose(
            state[0, 14:20],
            robot_state["joint_vel"],
        )

        np.testing.assert_allclose(
            state[0, 20:],
            0.0,
        )

    def test_optional_wrench(self):
        robot_state = {
            "joint_pos": np.zeros(
                6,
                dtype=np.float32,
            ),
            "gripper": 0.5,
            "tcp_pos": np.zeros(
                3,
                dtype=np.float32,
            ),
            "tcp_rotvec": np.zeros(
                3,
                dtype=np.float32,
            ),
            "joint_vel": np.zeros(
                6,
                dtype=np.float32,
            ),
            "wrench": np.array(
                [1, 2, 3, 4, 5, 6],
                dtype=np.float32,
            ),
        }

        state = DoosanClient.pack_state(
            robot_state
        )

        np.testing.assert_allclose(
            state[0, 20:26],
            robot_state["wrench"],
        )

    def test_semantic_prefix_padding(self):
        prefix = np.zeros(
            (4, DOOSAN_ACTION_DIM),
            dtype=np.float32,
        )

        prefix[:, 6] = 0.7

        packed = DoosanClient.pack_action_prefix(
            prefix
        )

        self.assertEqual(
            packed.shape,
            (
                XR1_ACTION_HORIZON,
                XR1_ACTION_DIM,
            ),
        )

        np.testing.assert_allclose(
            packed[:4, :7],
            prefix,
        )

        np.testing.assert_allclose(
            packed[:4, 7:],
            0.0,
        )

        np.testing.assert_allclose(
            packed[4:],
            0.0,
        )

    def test_output_extraction(self):
        native = np.zeros(
            (
                1,
                XR1_ACTION_HORIZON,
                XR1_ACTION_DIM,
            ),
            dtype=np.float32,
        )

        expected = np.arange(
            XR1_ACTION_HORIZON * 7,
            dtype=np.float32,
        ).reshape(
            XR1_ACTION_HORIZON,
            7,
        )

        native[0, :, :7] = expected
        native[0, :, 7:] = 999.0

        action = DoosanClient.extract_action(
            native
        )

        self.assertEqual(
            action.shape,
            (
                XR1_ACTION_HORIZON,
                DOOSAN_ACTION_DIM,
            ),
        )

        np.testing.assert_allclose(
            action,
            expected,
        )

    def test_missing_state_field_rejected(self):
        robot_state = {
            "joint_pos": np.zeros(6),
            "gripper": 0.5,
            "tcp_pos": np.zeros(3),
            "tcp_rotvec": np.zeros(3),
        }

        with self.assertRaises(KeyError):
            DoosanClient.pack_state(
                robot_state
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
