"""Doosan-specific XR-1 JSON training dataset.

The adapter reuses Xiaomi's image/prompt/video handling while replacing
the embodiment-specific state/action construction.

Canonical per-frame JSON fields:

proprios.joint_pos      -> 6
proprios.gripper        -> scalar in [0,1]
proprios.tcp_pos        -> 3
proprios.tcp_rotvec     -> 3
proprios.joint_vel      -> 6
proprios.wrench         -> optional 6

actions.semantic        -> 7
    [delta_tcp_pos_3,
     delta_tcp_rotvec_3,
     absolute_gripper_target_1]

`actions.semantic` is already the measured per-transition action
computed exactly once by doosan-forcevla-data-tools from consecutive
synchronized TCP poses. This dataset only windows, pads, normalizes,
masks, and packs those labels. It must never finite-difference them
again.

The resulting tensors preserve XR-1's native:
- state:  (1,60)
- action: (30,60)
"""

from __future__ import annotations

import numpy as np
import torch

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.doosan import (
    DOOSAN_SEMANTIC_ACTION_DIM,
    build_doosan_action_mask,
    compose_doosan_action,
    compose_doosan_state,
)
from mibot.utils.io import (
    get_value,
    normalize_action,
    normalize_quantile,
)


class DoosanJsonDataset(JsonDataset):
    """XR-1 post-training dataset for the Doosan M1013."""

    def __init__(self, params):
        # Base JsonDataset.__init__ calls self._samples(), so set seed first.
        self.sampler_seed = int(params["train_datasets"].get("sampler_seed", 42))
        super().__init__(params)

        data = params["train_datasets"]

        self.use_wrench = bool(
            data.get("use_wrench", False)
        )

        self.semantic_action_key = str(
            data.get(
                "semantic_action_key",
                "actions.semantic",
            )
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        traj = self._read_json(
            sample["file"]
        )

        frame = sample["frame_index"]

        steps = min(
            self.action_length,
            int(traj["num_frames"]) - frame,
        )

        if steps <= 0:
            raise IndexError(
                f"invalid sample at index={index}: "
                f"frame_index={frame}"
            )

        prompt = self._prompt(traj)

        raw_action = self._doosan_action(
            traj,
            frame,
            steps,
        )

        action = normalize_action(
            raw_action,
            self.mean,
            self.std,
        )

        raw_state = self._doosan_state(
            traj,
            frame,
        )

        state = normalize_quantile(
            raw_state,
            self.q01,
            self.q99,
        )

        temporal_mask = self._mask(
            traj,
            steps,
        )

        action_mask = build_doosan_action_mask(
            self.action_length,
            temporal_mask,
        )

        prompt["conversations"] += [
            {
                "from": "human",
                "value": "Robot state: <state>",
            },
            {
                "from": "gpt",
                "value": (
                    "".join(
                        f"<a_{i}>"
                        for i in range(steps)
                    )
                    + "<score>"
                ),
            },
        ]

        return {
            "messages": self._messages(
                prompt["conversations"],
                self._augment(
                    self._images(
                        traj,
                        prompt["images"],
                        frame,
                    )
                ),
            ),
            "action": torch.from_numpy(
                action
            ),
            "action_mask": torch.from_numpy(
                action_mask
            ),
            "state": torch.from_numpy(
                state
            ),
            "vlm_action_target": torch.from_numpy(
                action[:steps]
            ),
            "vlm_action_mask": torch.from_numpy(
                action_mask[:steps]
            ),
            "vlm_action_actual_length": steps,
        }

    def _samples(self):
        """Uniformly sample frames from all training files, reproducibly.

        Select without replacement until the full split has been used;
        reshuffle for further cycles when max_samples exceeds its length.
        This prevents the sorted-prefix selection in Xiaomi JsonDataset.
        """
        import random

        population = []
        for path in self.files:
            num_frames = int(self._read_json(path)["num_frames"])
            if num_frames <= 0:
                continue
            population.extend(
                {"file": path, "frame_index": index}
                for index in range(num_frames)
            )

        if not population:
            raise ValueError("Doosan training paths contain no frames")

        rng = random.Random(self.sampler_seed)
        selected = []
        remaining = self.max_samples
        while remaining > 0:
            indices = list(range(len(population)))
            rng.shuffle(indices)
            take = min(remaining, len(population))
            selected.extend(population[i] for i in indices[:take])
            remaining -= take

        unique_files = len({sample["file"] for sample in selected})
        print(
            f"XR1_DOOSAN_SAMPLER=PASS seed={self.sampler_seed} "
            f"raw_frames={len(population)} samples={len(selected)} "
            f"unique_episodes={unique_files}",
            flush=True,
        )
        return selected

    @staticmethod
    def _required_frame(
        traj,
        key,
        frame,
    ):
        values = get_value(
            traj,
            key,
        )

        if values is None:
            raise ValueError(
                f"{key} is missing from trajectory"
            )

        if frame >= len(values):
            raise ValueError(
                f"{key} has only {len(values)} frames; "
                f"requested frame {frame}"
            )

        return np.asarray(
            values[frame],
            dtype=np.float32,
        )

    def _doosan_state(
        self,
        traj,
        frame,
    ):
        wrench = None

        if self.use_wrench:
            wrench = self._required_frame(
                traj,
                "proprios.wrench",
                frame,
            )

        return compose_doosan_state(
            joint_pos=self._required_frame(
                traj,
                "proprios.joint_pos",
                frame,
            ),
            gripper=self._required_frame(
                traj,
                "proprios.gripper",
                frame,
            ),
            tcp_pos=self._required_frame(
                traj,
                "proprios.tcp_pos",
                frame,
            ),
            tcp_rotvec=self._required_frame(
                traj,
                "proprios.tcp_rotvec",
                frame,
            ),
            joint_vel=self._required_frame(
                traj,
                "proprios.joint_vel",
                frame,
            ),
            wrench=wrench,
        )

    def _doosan_action(
        self,
        traj,
        frame,
        steps,
    ):
        values = get_value(
            traj,
            self.semantic_action_key,
        )

        if values is None:
            raise ValueError(
                f"{self.semantic_action_key} "
                "is missing from trajectory"
            )

        semantic = np.asarray(
            values[
                frame : frame + steps
            ],
            dtype=np.float32,
        )

        if (
            semantic.ndim != 2
            or semantic.shape[1]
            != DOOSAN_SEMANTIC_ACTION_DIM
        ):
            raise ValueError(
                f"{self.semantic_action_key} expected "
                f"shape (N,{DOOSAN_SEMANTIC_ACTION_DIM}), "
                f"got {semantic.shape}"
            )

        semantic = self._pad(
            semantic,
            steps,
        )

        return compose_doosan_action(
            delta_tcp_pos=semantic[:, 0:3],
            delta_tcp_rotvec=semantic[:, 3:6],
            gripper_target=semantic[:, 6],
        )

    # XR1_DOOSAN_OFFLINE_OPTIN_V1: fail-closed PNG cache.
    @staticmethod
    def _images(traj, keys, frame):
        import os
        if os.environ.get("XR1_DOOSAN_OFFLINE_FRAME_ROOT"):
            from mibot.data.datasets.doosan_cached_images import decode_images
        else:
            from mibot.data.datasets.doosan_video_decoder import decode_images
        return decode_images(traj, keys, frame)
