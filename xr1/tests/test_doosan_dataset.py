#!/usr/bin/env python3

import unittest

import numpy as np

from mibot.data.datasets.doosan_json_dataset import (
    DoosanJsonDataset,
)


class DoosanDatasetContractTest(unittest.TestCase):

    @staticmethod
    def dataset_stub(
        use_wrench=False,
    ):
        dataset = object.__new__(
            DoosanJsonDataset
        )

        dataset.action_length = 30
        dataset.use_wrench = use_wrench
        dataset.semantic_action_key = (
            "actions.semantic"
        )

        dataset.mean = np.zeros(
            (30, 60),
            dtype=np.float32,
        )

        dataset.std = np.ones(
            (30, 60),
            dtype=np.float32,
        )

        return dataset

    @staticmethod
    def trajectory():
        num_frames = 5

        return {
            "num_frames": num_frames,
            "proprios": {
                "joint_pos": [
                    [
                        0.1 + 0.01 * i,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                    ]
                    for i in range(num_frames)
                ],
                "gripper": [
                    0.5
                    for _ in range(num_frames)
                ],
                "tcp_pos": [
                    [
                        0.4 + 0.001 * i,
                        -0.1,
                        0.3,
                    ]
                    for i in range(num_frames)
                ],
                "tcp_rotvec": [
                    [
                        0.01,
                        0.02,
                        0.03,
                    ]
                    for _ in range(num_frames)
                ],
                "joint_vel": [
                    [
                        0.01,
                        0.02,
                        0.03,
                        0.04,
                        0.05,
                        0.06,
                    ]
                    for _ in range(num_frames)
                ],
                "wrench": [
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                    ]
                    for _ in range(num_frames)
                ],
            },
            "actions": {
                "semantic": [
                    [
                        0.001,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.01,
                        0.75,
                    ]
                    for _ in range(num_frames)
                ],
            },
        }

    def test_force_free_state(self):
        dataset = self.dataset_stub(
            use_wrench=False
        )

        state = dataset._doosan_state(
            self.trajectory(),
            0,
        )

        self.assertEqual(
            state.shape,
            (1, 60),
        )

        np.testing.assert_allclose(
            state[0, 20:26],
            0.0,
        )

    def test_force_aware_state(self):
        dataset = self.dataset_stub(
            use_wrench=True
        )

        state = dataset._doosan_state(
            self.trajectory(),
            0,
        )

        np.testing.assert_allclose(
            state[0, 20:26],
            [1, 2, 3, 4, 5, 6],
        )

    def test_action_chunk(self):
        dataset = self.dataset_stub()

        action = dataset._doosan_action(
            self.trajectory(),
            frame=0,
            steps=5,
        )

        self.assertEqual(
            action.shape,
            (30, 60),
        )

        np.testing.assert_allclose(
            action[:5, 0],
            0.001,
        )

        np.testing.assert_allclose(
            action[:5, 5],
            0.01,
        )

        np.testing.assert_allclose(
            action[:5, 6],
            0.75,
        )

        np.testing.assert_allclose(
            action[:, 7:],
            0.0,
        )

        # Xiaomi's native loader pads the remaining horizon
        # by repeating the final valid action. The temporal
        # mask excludes these padded steps from the loss.
        expected_tail = np.repeat(
            action[4:5, :7],
            action.shape[0] - 5,
            axis=0,
        )

        np.testing.assert_allclose(
            action[5:, :7],
            expected_tail,
        )

    def test_precomputed_semantic_action_is_not_differenced_again(self):
        """XR-1 must consume converter-produced deltas exactly once."""

        dataset = self.dataset_stub()

        traj = self.trajectory()

        semantic = np.asarray(
            [
                [0.001, 0.002, -0.003, 0.010, 0.020, -0.030, 0.0],
                [0.004, -0.001, 0.002, -0.020, 0.010, 0.005, 0.0],
                [-0.002, 0.005, 0.001, 0.030, -0.010, 0.020, 1.0],
                [0.006, -0.004, 0.003, -0.015, 0.025, -0.005, 1.0],
                [0.000, 0.003, -0.006, 0.005, -0.030, 0.040, 1.0],
            ],
            dtype=np.float32,
        )

        traj["actions"]["semantic"] = (
            semantic.tolist()
        )

        action = dataset._doosan_action(
            traj,
            frame=0,
            steps=5,
        )

        # The first five XR-1 semantic rows must equal the
        # converter-produced labels byte-for-value after
        # float32 conversion. No second TCP/action difference
        # is permitted here.
        np.testing.assert_array_equal(
            action[:5, :7],
            semantic,
        )

        # This explicitly catches an accidental second
        # finite difference: the varying translation labels
        # must survive unchanged.
        self.assertAlmostEqual(
            float(action[1, 0]),
            0.004,
            places=7,
        )

        self.assertAlmostEqual(
            float(action[2, 0]),
            -0.002,
            places=7,
        )

        # Padding is allowed, but only after the valid
        # precomputed semantic labels have been consumed.
        expected_tail = np.repeat(
            semantic[-1:],
            25,
            axis=0,
        )

        np.testing.assert_array_equal(
            action[5:, :7],
            expected_tail,
        )

    def test_missing_wrench_rejected_when_enabled(self):
        dataset = self.dataset_stub(
            use_wrench=True
        )

        traj = self.trajectory()
        del traj["proprios"]["wrench"]

        with self.assertRaises(ValueError):
            dataset._doosan_state(
                traj,
                0,
            )

    def test_bad_semantic_action_width_rejected(self):
        dataset = self.dataset_stub()

        traj = self.trajectory()

        traj["actions"]["semantic"] = [
            [0.0] * 6
            for _ in range(5)
        ]

        with self.assertRaises(ValueError):
            dataset._doosan_action(
                traj,
                frame=0,
                steps=5,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
