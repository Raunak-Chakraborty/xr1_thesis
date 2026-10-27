"""Two-camera Doosan runtime client for XR-1.

Embodiment-specific behavior:
- external workspace camera
- TCP/wrist camera
- Doosan state -> XR-1 native 60D state
- XR-1 native 60D action -> Doosan semantic 7D action

The original Xiaomi runtime client is intentionally unchanged.
"""

from __future__ import annotations

from io import BytesIO
import socket
import struct

import numpy as np
import torch
from transformers import AutoProcessor

from mibot.utils.doosan import (
    ACTION_GRIPPER_INDEX,
    ACTION_ROTATION_SLICE,
    ACTION_TRANSLATION_SLICE,
    compose_doosan_state,
    extract_doosan_action,
)
from mibot.utils.io import resize_image
from mibot.utils.doosan_profile import validate_use_wrench


QWEN_PROCESSOR = "Qwen/Qwen3-VL-4B-Instruct"

XR1_ACTION_HORIZON = 30
XR1_ACTION_DIM = 60
DOOSAN_ACTION_DIM = 7


class DoosanClient:
    """XR-1 runtime client for the single-arm Doosan M1013."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10086,
        use_wrench: bool = False,
    ) -> None:
        self.use_wrench = validate_use_wrench(use_wrench)
        self.socket = socket.create_connection((host, port))

        self.processor = AutoProcessor.from_pretrained(
            QWEN_PROCESSOR
        )
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _recv_all(sock, length):
        data = b""

        while len(data) < length:
            packet = sock.recv(length - len(data))

            if not packet:
                raise ConnectionError(
                    "connection closed while receiving response"
                )

            data += packet

        return data

    def _send(self, payload):
        arrays = {}

        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()

                arrays[key] = (
                    value.float().numpy()
                    if value.dtype == torch.bfloat16
                    else value.numpy()
                )
            else:
                arrays[key] = np.asarray(value)

        buffer = BytesIO()
        np.savez(buffer, **arrays)

        data = buffer.getvalue()

        self.socket.sendall(
            struct.pack(">I", len(data)) + data
        )

    def _recv(self):
        size = struct.unpack(
            ">I",
            self._recv_all(self.socket, 4),
        )[0]

        with np.load(
            BytesIO(
                self._recv_all(
                    self.socket,
                    size,
                )
            ),
            allow_pickle=False,
        ) as payload:
            return torch.from_numpy(
                payload["action"].copy()
            )

    @staticmethod
    def _messages(
        instruction,
        external_obs,
        tcp_obs,
    ):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The following observations are captured "
                            "from two synchronized robot-camera views."
                            "\n# External Workspace View\n"
                        ),
                    },
                    {
                        "type": "image",
                        "image": external_obs,
                    },
                    {
                        "type": "text",
                        "text": "\n# TCP / Wrist View\n",
                    },
                    {
                        "type": "image",
                        "image": tcp_obs,
                    },
                    {
                        "type": "text",
                        "text": (
                            "\nGenerate robot actions for the task:\n"
                            f"{instruction} /no_cot"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<cot></cot>",
                    }
                ],
            },
        ]

    @staticmethod
    def pack_state(robot_state, use_wrench=None) -> np.ndarray:
        if use_wrench is not None:
            validate_use_wrench(use_wrench)
        if use_wrench is True and robot_state.get("wrench") is None:
            raise ValueError("force-state inference requires robot_state.wrench")
        required = (
            "joint_pos",
            "gripper",
            "tcp_pos",
            "tcp_rotvec",
            "joint_vel",
        )

        missing = [
            key
            for key in required
            if key not in robot_state
        ]

        if missing:
            raise KeyError(
                f"robot_state missing required keys: {missing}"
            )

        return compose_doosan_state(
            joint_pos=robot_state["joint_pos"],
            gripper=robot_state["gripper"],
            tcp_pos=robot_state["tcp_pos"],
            tcp_rotvec=robot_state["tcp_rotvec"],
            joint_vel=robot_state["joint_vel"],
            wrench=robot_state.get("wrench") if use_wrench is not False else None,
        )

    @staticmethod
    def pack_action_prefix(action_prefix):
        if action_prefix is None:
            return None

        prefix = np.asarray(
            action_prefix,
            dtype=np.float32,
        )

        if prefix.ndim != 2 or prefix.shape[1] != DOOSAN_ACTION_DIM:
            raise ValueError(
                "Doosan action_prefix expected shape (T,7), "
                f"got {prefix.shape}"
            )

        if prefix.shape[0] > XR1_ACTION_HORIZON:
            raise ValueError(
                "Doosan action_prefix exceeds XR-1 horizon: "
                f"{prefix.shape[0]} > {XR1_ACTION_HORIZON}"
            )

        if not np.isfinite(prefix).all():
            raise ValueError(
                "Doosan action_prefix contains non-finite values"
            )

        gripper = prefix[:, ACTION_GRIPPER_INDEX]

        if np.any((gripper < 0.0) | (gripper > 1.0)):
            raise ValueError(
                "Doosan prefix gripper must be normalized to [0,1]"
            )

        padded = np.zeros(
            (XR1_ACTION_HORIZON, XR1_ACTION_DIM),
            dtype=np.float32,
        )

        padded[
            : prefix.shape[0],
            :DOOSAN_ACTION_DIM,
        ] = prefix

        return padded

    @staticmethod
    def extract_action(action) -> np.ndarray:
        array = np.asarray(
            action,
            dtype=np.float32,
        )

        if array.shape == (
            1,
            XR1_ACTION_HORIZON,
            XR1_ACTION_DIM,
        ):
            array = array[0]

        if array.shape != (
            XR1_ACTION_HORIZON,
            XR1_ACTION_DIM,
        ):
            raise ValueError(
                "expected XR-1 server output shape "
                f"({XR1_ACTION_HORIZON},{XR1_ACTION_DIM}), "
                f"got {array.shape}"
            )

        return extract_doosan_action(array)

    @staticmethod
    def semantic_slices():
        return {
            "delta_tcp_pos": ACTION_TRANSLATION_SLICE,
            "delta_tcp_rotvec": ACTION_ROTATION_SLICE,
            "gripper_target": slice(
                ACTION_GRIPPER_INDEX,
                ACTION_GRIPPER_INDEX + 1,
            ),
        }

    def __call__(
        self,
        robot_state,
        external_obs,
        tcp_obs,
        instruction,
        action_prefix=None,
        crop_bboxes=None,
    ):
        """Run one inference request against an XR-1 server.

        This method does not directly command the robot.
        """

        crop_bboxes = crop_bboxes or (
            None,
            None,
        )

        if len(crop_bboxes) != 2:
            raise ValueError(
                "crop_bboxes must contain exactly two entries"
            )

        images = (
            external_obs,
            tcp_obs,
        )

        processed_images = []

        for image, bbox in zip(
            images,
            crop_bboxes,
        ):
            if bbox is not None:
                top, left, height, width = (
                    int(value)
                    for value in bbox
                )

                image = image.crop(
                    (
                        left,
                        top,
                        left + width,
                        top + height,
                    )
                )

            image = resize_image(
                image,
                factor=32,
                max_pixels=160000,
            )

            processed_images.append(image)

        external_obs, tcp_obs = processed_images

        payload = self.processor.apply_chat_template(
            [
                self._messages(
                    instruction,
                    external_obs,
                    tcp_obs,
                )
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={
                "do_resize": False,
            },
        )

        state = self.pack_state(
            robot_state, use_wrench=self.use_wrench,
        )
        payload["use_wrench"] = self.use_wrench

        payload["state"] = torch.from_numpy(
            state
        )[None]

        padded_prefix = self.pack_action_prefix(
            action_prefix
        )

        if padded_prefix is not None:
            payload["action"] = torch.from_numpy(
                padded_prefix
            )[None]

            payload["prefix_length"] = len(
                action_prefix
            )

        self._send(payload)

        action = self._recv()

        if isinstance(
            action,
            torch.Tensor,
        ):
            action = (
                action.detach()
                .cpu()
                .numpy()
            )

        return self.extract_action(
            action
        )
