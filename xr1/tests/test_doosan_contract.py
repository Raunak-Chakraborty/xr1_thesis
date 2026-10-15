#!/usr/bin/env python3

import unittest

import numpy as np

from mibot.utils.doosan import (
    ACTION_GRIPPER_INDEX,
    ACTION_ROTATION_SLICE,
    ACTION_TRANSLATION_SLICE,
    DOOSAN_ACTION_SLICE,
    GRIPPER_STATE_INDEX,
    JOINT_POS_SLICE,
    JOINT_VEL_SLICE,
    SEVENTH_JOINT_PAD_INDEX,
    TCP_POS_SLICE,
    TCP_ROTVEC_SLICE,
    WRENCH_SLICE,
    build_doosan_action_mask,
    compose_doosan_action,
    compose_doosan_state,
    extract_doosan_action,
)


class DoosanStateContractTest(unittest.TestCase):

    def test_force_free_state(self):
        joint_pos = np.array(
            [1, 2, 3, 4, 5, 6],
            dtype=np.float32,
        )

        gripper = 0.25

        tcp_pos = np.array(
            [0.10, -0.20, 0.30],
            dtype=np.float32,
        )

        tcp_rotvec = np.array(
            [0.01, -0.02, 0.03],
            dtype=np.float32,
        )

        joint_vel = np.array(
            [11, 12, 13, 14, 15, 16],
            dtype=np.float32,
        )

        state = compose_doosan_state(
            joint_pos=joint_pos,
            gripper=gripper,
            tcp_pos=tcp_pos,
            tcp_rotvec=tcp_rotvec,
            joint_vel=joint_vel,
        )

        self.assertEqual(state.shape, (1, 60))

        np.testing.assert_allclose(
            state[0, JOINT_POS_SLICE],
            joint_pos,
        )

        self.assertEqual(
            state[0, SEVENTH_JOINT_PAD_INDEX],
            0.0,
        )

        self.assertAlmostEqual(
            float(state[0, GRIPPER_STATE_INDEX]),
            gripper,
        )

        np.testing.assert_allclose(
            state[0, TCP_POS_SLICE],
            tcp_pos,
        )

        np.testing.assert_allclose(
            state[0, TCP_ROTVEC_SLICE],
            tcp_rotvec,
        )

        np.testing.assert_allclose(
            state[0, JOINT_VEL_SLICE],
            joint_vel,
        )

        np.testing.assert_allclose(
            state[0, WRENCH_SLICE],
            0.0,
        )

        np.testing.assert_allclose(
            state[0, 26:],
            0.0,
        )

    def test_force_aware_reserved_state(self):
        wrench = np.array(
            [1, 2, 3, 4, 5, 6],
            dtype=np.float32,
        )

        state = compose_doosan_state(
            joint_pos=np.zeros(6),
            gripper=0.5,
            tcp_pos=np.zeros(3),
            tcp_rotvec=np.zeros(3),
            joint_vel=np.zeros(6),
            wrench=wrench,
        )

        np.testing.assert_allclose(
            state[0, WRENCH_SLICE],
            wrench,
        )

        np.testing.assert_allclose(
            state[0, 26:],
            0.0,
        )

    def test_bad_joint_shape_rejected(self):
        with self.assertRaises(ValueError):
            compose_doosan_state(
                joint_pos=np.zeros(7),
                gripper=0.5,
                tcp_pos=np.zeros(3),
                tcp_rotvec=np.zeros(3),
                joint_vel=np.zeros(6),
            )

    def test_bad_gripper_rejected(self):
        with self.assertRaises(ValueError):
            compose_doosan_state(
                joint_pos=np.zeros(6),
                gripper=1.5,
                tcp_pos=np.zeros(3),
                tcp_rotvec=np.zeros(3),
                joint_vel=np.zeros(6),
            )


class DoosanActionContractTest(unittest.TestCase):

    def test_action_packing(self):
        length = 30

        delta_pos = np.arange(
            length * 3,
            dtype=np.float32,
        ).reshape(length, 3) / 1000.0

        delta_rot = np.arange(
            length * 3,
            dtype=np.float32,
        ).reshape(length, 3) / 10000.0

        gripper = np.linspace(
            0.0,
            1.0,
            length,
            dtype=np.float32,
        )

        action = compose_doosan_action(
            delta_tcp_pos=delta_pos,
            delta_tcp_rotvec=delta_rot,
            gripper_target=gripper,
        )

        self.assertEqual(action.shape, (30, 60))

        np.testing.assert_allclose(
            action[:, ACTION_TRANSLATION_SLICE],
            delta_pos,
        )

        np.testing.assert_allclose(
            action[:, ACTION_ROTATION_SLICE],
            delta_rot,
        )

        np.testing.assert_allclose(
            action[:, ACTION_GRIPPER_INDEX],
            gripper,
        )

        np.testing.assert_allclose(
            action[:, 7:],
            0.0,
        )

        extracted = extract_doosan_action(action)

        self.assertEqual(extracted.shape, (30, 7))

        np.testing.assert_allclose(
            extracted,
            action[:, DOOSAN_ACTION_SLICE],
        )

    def test_action_mask(self):
        temporal = np.ones(
            30,
            dtype=np.int32,
        )

        temporal[-5:] = 0

        mask = build_doosan_action_mask(
            30,
            temporal,
        )

        self.assertEqual(mask.shape, (30, 60))

        np.testing.assert_array_equal(
            mask[:25, :7],
            1,
        )

        np.testing.assert_array_equal(
            mask[25:, :],
            0,
        )

        np.testing.assert_array_equal(
            mask[:, 7:],
            0,
        )

    def test_bad_horizon_rejected(self):
        with self.assertRaises(ValueError):
            compose_doosan_action(
                delta_tcp_pos=np.zeros((30, 3)),
                delta_tcp_rotvec=np.zeros((29, 3)),
                gripper_target=np.zeros(30),
            )

    def test_bad_action_gripper_rejected(self):
        gripper = np.zeros(30)
        gripper[4] = -0.1

        with self.assertRaises(ValueError):
            compose_doosan_action(
                delta_tcp_pos=np.zeros((30, 3)),
                delta_tcp_rotvec=np.zeros((30, 3)),
                gripper_target=gripper,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
