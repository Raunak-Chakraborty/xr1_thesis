"""Doosan M1013 compatibility adapter for XR-1.

This module deliberately preserves XR-1's native 60D state and action
interfaces. It contains only embodiment-specific packing and validation.

Force/torque is optional observation data and is never part of the action.
"""

from __future__ import annotations

import numpy as np

from mibot.utils.io import ACTION_DIM, STATE_DIM


DOOSAN_PROPRIO_DIM = 19
DOOSAN_FORCE_AWARE_STATE_DIM = 25
DOOSAN_SEMANTIC_ACTION_DIM = 7

JOINT_POS_SLICE = slice(0, 6)
SEVENTH_JOINT_PAD_INDEX = 6
GRIPPER_STATE_INDEX = 7

TCP_POS_SLICE = slice(8, 11)
TCP_ROTVEC_SLICE = slice(11, 14)
JOINT_VEL_SLICE = slice(14, 20)

WRENCH_SLICE = slice(20, 26)

DOOSAN_ACTION_SLICE = slice(0, 7)
ACTION_TRANSLATION_SLICE = slice(0, 3)
ACTION_ROTATION_SLICE = slice(3, 6)
ACTION_GRIPPER_INDEX = 6


def _vector(name: str, value, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)

    if array.shape != (width,):
        raise ValueError(
            f"{name} expected shape ({width},), got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")

    return array


def _matrix(name: str, value, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)

    if array.ndim == 1:
        if array.shape != (width,):
            raise ValueError(
                f"{name} expected shape ({width},) or (T,{width}), "
                f"got {array.shape}"
            )
        array = array[None, :]

    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} expected shape (T,{width}), got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")

    return array


def _normalized_gripper_scalar(value) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)

    if array.shape != (1,):
        raise ValueError(
            f"gripper expected one scalar value, got {array.shape}"
        )

    result = float(array[0])

    if not np.isfinite(result):
        raise ValueError("gripper contains a non-finite value")

    if not 0.0 <= result <= 1.0:
        raise ValueError(
            f"gripper must be normalized to [0,1], got {result}"
        )

    return result


def compose_doosan_state(
    joint_pos,
    gripper,
    tcp_pos,
    tcp_rotvec,
    joint_vel,
    wrench=None,
) -> np.ndarray:
    """Pack the Doosan semantic observation into XR-1's native 60D state.

    Force-free layout:
        0:6   joint positions
        6     zero seventh-joint compatibility slot
        7     normalized measured gripper opening
        8:11  TCP position in robot-base frame
        11:14 absolute base-to-TCP rotation vector
        14:20 joint velocities
        20:60 zero

    Force-aware extension:
        20:26 [Fx,Fy,Fz,Tx,Ty,Tz]
    """

    if STATE_DIM != 60:
        raise RuntimeError(
            f"XR-1 STATE_DIM changed unexpectedly: {STATE_DIM}"
        )

    state = np.zeros((1, STATE_DIM), dtype=np.float32)

    state[0, JOINT_POS_SLICE] = _vector(
        "joint_pos", joint_pos, 6
    )

    # state[0, 6] intentionally stays zero.

    state[0, GRIPPER_STATE_INDEX] = _normalized_gripper_scalar(
        gripper
    )

    state[0, TCP_POS_SLICE] = _vector(
        "tcp_pos", tcp_pos, 3
    )

    state[0, TCP_ROTVEC_SLICE] = _vector(
        "tcp_rotvec", tcp_rotvec, 3
    )

    state[0, JOINT_VEL_SLICE] = _vector(
        "joint_vel", joint_vel, 6
    )

    if wrench is not None:
        state[0, WRENCH_SLICE] = _vector(
            "wrench", wrench, 6
        )

    return state


def compose_doosan_action(
    delta_tcp_pos,
    delta_tcp_rotvec,
    gripper_target,
) -> np.ndarray:
    """Pack the 7D Doosan semantic action into XR-1's 60D action.

    Semantic action:
        0:3 translation delta in robot-base coordinates
        3:6 relative rotation vector
        6   absolute normalized gripper target

    All remaining XR-1 action dimensions stay zero.
    """

    if ACTION_DIM != 60:
        raise RuntimeError(
            f"XR-1 ACTION_DIM changed unexpectedly: {ACTION_DIM}"
        )

    delta_pos = _matrix(
        "delta_tcp_pos", delta_tcp_pos, 3
    )

    delta_rot = _matrix(
        "delta_tcp_rotvec", delta_tcp_rotvec, 3
    )

    gripper = np.asarray(
        gripper_target,
        dtype=np.float32,
    )

    if gripper.ndim == 1:
        gripper = gripper[:, None]

    if gripper.ndim != 2 or gripper.shape[1] != 1:
        raise ValueError(
            "gripper_target expected shape (T,) or (T,1), "
            f"got {gripper.shape}"
        )

    if not np.isfinite(gripper).all():
        raise ValueError(
            "gripper_target contains non-finite values"
        )

    if np.any((gripper < 0.0) | (gripper > 1.0)):
        raise ValueError(
            "gripper_target must be normalized to [0,1]"
        )

    length = delta_pos.shape[0]

    if delta_rot.shape[0] != length:
        raise ValueError(
            "translation and rotation horizons differ: "
            f"{length} vs {delta_rot.shape[0]}"
        )

    if gripper.shape[0] != length:
        raise ValueError(
            "translation and gripper horizons differ: "
            f"{length} vs {gripper.shape[0]}"
        )

    action = np.zeros(
        (length, ACTION_DIM),
        dtype=np.float32,
    )

    action[:, ACTION_TRANSLATION_SLICE] = delta_pos
    action[:, ACTION_ROTATION_SLICE] = delta_rot
    action[:, ACTION_GRIPPER_INDEX] = gripper[:, 0]

    return action


def build_doosan_action_mask(
    action_length: int,
    temporal_mask=None,
) -> np.ndarray:
    """Mask only the seven semantically meaningful Doosan dimensions."""

    if action_length <= 0:
        raise ValueError("action_length must be positive")

    if temporal_mask is None:
        temporal = np.ones(
            action_length,
            dtype=np.int32,
        )
    else:
        temporal = np.asarray(
            temporal_mask,
            dtype=np.int32,
        ).reshape(-1)

        if temporal.shape != (action_length,):
            raise ValueError(
                "temporal_mask expected shape "
                f"({action_length},), got {temporal.shape}"
            )

        if not np.all(
            (temporal == 0) | (temporal == 1)
        ):
            raise ValueError(
                "temporal_mask must contain only 0 or 1"
            )

    mask = np.zeros(
        (action_length, ACTION_DIM),
        dtype=np.int32,
    )

    mask[:, DOOSAN_ACTION_SLICE] = temporal[:, None]

    return mask


def extract_doosan_action(action) -> np.ndarray:
    """Extract the seven physical Doosan channels from XR-1 output."""

    array = np.asarray(
        action,
        dtype=np.float32,
    )

    if array.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"XR-1 action last dimension must be {ACTION_DIM}, "
            f"got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "XR-1 action contains non-finite values"
        )

    return array[..., DOOSAN_ACTION_SLICE].copy()
