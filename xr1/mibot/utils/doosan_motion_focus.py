"""Opt-in deterministic Doosan motion-focus sampling and action masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class MotionFocusPlan:
    def __init__(self, directory):
        directory = Path(directory).expanduser().resolve(strict=True)

        meta_path = directory / "motion_focus_manifest.json"
        schedule_path = directory / "sample_global_indices.npy"

        self.meta = json.loads(meta_path.read_text())
        cfg = self.meta["sampling"]

        if self.meta.get("schema") != "xr1_motion_focus_v1_preflight":
            raise ValueError("Unknown motion-focus plan schema")

        if self.meta.get("status") != "DESIGN_ONLY_NOT_CONNECTED_TO_DATALOADER":
            raise ValueError("Unexpected plan identity/status")

        self.indices = np.load(
            schedule_path,
            allow_pickle=False,
        )

        actual = hashlib.sha256(
            schedule_path.read_bytes()
        ).hexdigest()

        if actual != cfg["sample_schedule_sha256"]:
            raise ValueError("Frozen sample schedule checksum mismatch")

        if (
            self.indices.ndim != 1
            or not np.issubdtype(self.indices.dtype, np.integer)
            or len(self.indices) != int(cfg["samples"])
        ):
            raise ValueError("Invalid frozen sample array")

        rows = self.meta["train_episode_records"]

        self.records = {
            row["relative_episode"]: row
            for row in rows
        }

        if len(self.records) != len(rows):
            raise ValueError("Duplicate episode records")

        self._root = None

    def sampling_indices(self, files, counts, root, expected_samples):
        root = Path(root).expanduser().resolve(strict=True)

        relative = [
            Path(p).resolve(strict=True).relative_to(root).as_posix()
            for p in files
        ]

        rows = self.meta["train_episode_records"]

        if len(relative) != len(rows) or len(counts) != len(rows):
            raise ValueError(
                "Dataset file population does not match motion-focus plan"
            )

        offset = 0

        for name, count, record in zip(
            relative, counts, rows, strict=True
        ):
            if (
                name != record["relative_episode"]
                or count != record["num_frames"]
                or offset != record["global_offset"]
            ):
                raise ValueError(
                    f"Frozen episode/offset mismatch: {name}"
                )

            offset += int(count)

        if offset != self.meta["counts"]["frames"]:
            raise ValueError("Frame total mismatch")

        if int(expected_samples) != len(self.indices):
            raise ValueError(
                "Training horizon/batch size inconsistent with frozen schedule"
            )

        if len(self.indices) and int(self.indices.max()) >= offset:
            raise ValueError("Sample schedule index outside dataset")

        terminals = np.array([
            int(r["global_offset"]) + int(r["num_frames"]) - 1
            for r in rows
        ], dtype=np.int64)

        if np.isin(self.indices, terminals).any():
            raise ValueError(
                "Frozen sampler selects an invalid terminal frame"
            )

        self._root = root

        return self.indices.astype(
            np.int64, copy=False
        ).tolist()

    def apply_action_mask(self, path, frame, steps, mask):
        if self._root is None:
            raise RuntimeError(
                "Call sampling_indices before action-mask adjustment"
            )

        rel = (
            Path(path)
            .resolve(strict=True)
            .relative_to(self._root)
            .as_posix()
        )

        record = self.records[rel]
        result = np.asarray(mask, dtype=np.int32).copy()

        if (
            result.ndim != 2
            or result.shape[1] != 60
            or steps > len(result)
        ):
            raise ValueError("Unexpected 60D action mask")

        terminal = int(record["num_frames"]) - 1

        for source_idx in record["invalid_action_indices"]:
            pos = int(source_idx) - int(frame)

            if 0 <= pos < int(steps):
                # Terminal row: mask all six motion channels.
                # Isolated translation discrepancy: mask XYZ only.
                # Preserve the absolute gripper target.
                result[
                    pos,
                    :6 if int(source_idx) == terminal else 3
                ] = 0

        return result
