"""Synthetic retention tests; not a substitute for real checkpoint qualification."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mibot.utils import claix_checkpoint as ck


class SeriousRetentionTest(unittest.TestCase):

    @staticmethod
    def populate(root, steps, stop=None):
        receipts = {}

        for step in steps:
            name = f"step-{step:06d}.ckpt"
            path = root / name
            path.write_bytes(f"synthetic-{step}".encode())

            receipt = {
                "schema": "claix-b24-verified-v1",
                "global_step": step,
                "path": name,
                "config_sha256": "1" * 64,
                "sha256": "2" * 64,
                "bytes": path.stat().st_size,
            }

            (root / f"step-{step:06d}.verified.json").write_text(
                json.dumps(receipt)
            )

            receipts[step] = receipt

        (root / "latest.json").write_text(
            json.dumps(receipts[max(steps)])
        )

        if stop is not None:
            (root / "safe-stop.json").write_text(
                json.dumps(receipts[stop])
            )

    def test_all_nine_milestones_survive(self):
        milestones = ck.SERIOUS_LAST2_MILESTONES
        steps = sorted(
            set(milestones) | {44500, 45500, 46000, 46500, 47000}
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate(root, steps, stop=44500)

            checked = []

            def fake_load(path, **kwargs):
                checked.append(str(path))
                return {}

            with patch.object(
                ck, "load_verified", side_effect=fake_load
            ):
                result = ck.rotate_checkpoints(
                    root,
                    expected_config="1" * 64,
                    retention_milestones=milestones,
                )

            self.assertLessEqual(
                len(checked), ck.ROTATION_VERIFY_BUDGET
            )

            self.assertLessEqual(
                result["verified"], ck.ROTATION_VERIFY_BUDGET
            )

            for step in milestones:
                self.assertTrue(
                    (root / f"step-{step:06d}.ckpt").is_file(),
                    step,
                )

            # Safe stop, latest and previous recovery payload.
            for step in (44500, 46500, 47000):
                self.assertTrue(
                    (root / f"step-{step:06d}.ckpt").is_file()
                )

            self.assertGreaterEqual(result["deleted"], 1)

    def test_missing_milestone_blocks_rotation(self):
        milestones = ck.SERIOUS_LAST2_MILESTONES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate(
                root, [5000, 10000, 15000, 15500, 16000]
            )

            (root / "step-010000.ckpt").unlink()

            before = {p.name for p in root.iterdir()}

            with patch.object(
                ck, "load_verified", return_value={}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "milestone"
                ):
                    ck.rotate_checkpoints(
                        root,
                        retention_milestones=milestones,
                    )

            after = {p.name for p in root.iterdir()}
            self.assertEqual(before, after)

    def test_unapproved_milestones_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "Unreviewed"
        ):
            ck.rotate_checkpoints(
                "/not/created",
                retention_milestones=[5000, 10000],
            )

    def test_historical_policy_unchanged(self):
        self.assertEqual(
            ck.HISTORICAL_STEPS,
            frozenset(
                (2000, 5000, 10000, 20000, 30000, 40000)
            ),
        )

        self.assertEqual(
            ck.SERIOUS_LAST2_MILESTONES,
            tuple(range(5000, 50000, 5000)),
        )


if __name__ == "__main__":
    unittest.main()
