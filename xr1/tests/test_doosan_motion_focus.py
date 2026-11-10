"""CPU-only motion-focus dataset contract tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mibot.utils.doosan_motion_focus import MotionFocusPlan


class MotionFocusContractTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)
        self.data = self.root / "train"
        self.data.mkdir()

        files = []
        rows = []
        offset = 0

        for name, n, bad in [
            ("episode_a.json", 4, [1, 3]),
            ("episode_b.json", 5, [4]),
        ]:
            f = self.data / name
            f.write_text("{}")

            files.append(str(f))

            rows.append(dict(
                relative_episode=name,
                num_frames=n,
                global_offset=offset,
                invalid_action_indices=bad,
            ))

            offset += n

        self.files = files
        self.counts = [4, 5]

        plan = self.root / "plan"
        plan.mkdir()

        schedule = np.array(
            [0, 1, 4, 5], dtype=np.uint32
        )

        file = plan / "sample_global_indices.npy"
        np.save(file, schedule, allow_pickle=False)

        meta = dict(
            schema="xr1_motion_focus_v1_preflight",
            status="DESIGN_ONLY_NOT_CONNECTED_TO_DATALOADER",
            sampling=dict(
                samples=4,
                sample_schedule_sha256=hashlib.sha256(
                    file.read_bytes()
                ).hexdigest(),
            ),
            counts=dict(frames=9),
            train_episode_records=rows,
        )

        (
            plan / "motion_focus_manifest.json"
        ).write_text(json.dumps(meta))

        self.plan_dir = plan

    def test_sampling_and_mask(self):
        plan = MotionFocusPlan(self.plan_dir)

        self.assertEqual(
            plan.sampling_indices(
                self.files, self.counts, self.data, 4
            ),
            [0, 1, 4, 5],
        )

        mask = np.zeros((3, 60), dtype=np.int32)
        mask[:, :7] = 1

        out = plan.apply_action_mask(
            self.files[0], 1, 3, mask
        )

        self.assertTrue(np.all(out[0, :3] == 0))
        self.assertTrue(np.all(out[0, 3:7] == 1))
        self.assertTrue(np.all(out[1, :7] == 1))

        self.assertTrue(np.all(out[2, :6] == 0))
        self.assertEqual(int(out[2, 6]), 1)

        np.testing.assert_array_equal(
            mask[:, :7], 1
        )

    def test_bad_schedule_checksum_fails(self):
        path = (
            self.plan_dir / "sample_global_indices.npy"
        )

        path.write_bytes(
            path.read_bytes() + b"corrupt"
        )

        with self.assertRaises(ValueError):
            MotionFocusPlan(self.plan_dir)

    def test_terminal_sample_fails(self):
        path = (
            self.plan_dir / "sample_global_indices.npy"
        )

        np.save(
            path,
            np.array([0, 3, 4, 5], dtype=np.uint32),
            allow_pickle=False,
        )

        meta_path = (
            self.plan_dir / "motion_focus_manifest.json"
        )

        meta = json.loads(meta_path.read_text())

        meta["sampling"]["sample_schedule_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

        meta_path.write_text(json.dumps(meta))

        plan = MotionFocusPlan(self.plan_dir)

        with self.assertRaises(ValueError):
            plan.sampling_indices(
                self.files, self.counts, self.data, 4
            )


if __name__ == "__main__":
    unittest.main()
