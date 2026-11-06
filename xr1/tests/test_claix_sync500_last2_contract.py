"""CPU-only tests for the exact Sync-500 + E1 Slurm allocation contract."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/claix_b24_launch.py"

spec = importlib.util.spec_from_file_location(
    "sync500_last2_tested_launcher",
    LAUNCHER,
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

BATCH = (
    ROOT / "scripts/claix_b24_sync500_last2.sbatch"
).read_text()


class CombinedAllocationContractTest(unittest.TestCase):

    def allocation(
        self,
        mode,
        interval,
        leg,
        *,
        limit="08:00:00",
        cpus=12,
        gpu="NVIDIA H100",
        batch=BATCH,
    ):
        job_id = "999123"
        end_epoch = int(time.time()) + 8 * 3600

        fields = {
            "JobId": job_id,
            "JobState": "RUNNING",
            "Account": "thes2452",
            "Partition": "c23g",
            "NumNodes": "1",
            "NumTasks": "1",
            "NumCPUs": str(cpus),
            "CPUs/Task": str(cpus),
            "AllocTRES": f"cpu={cpus},mem=96G,gres/gpu=1",
            "TimeLimit": limit,
            "EndTime": "2026-11-08T08:00:00",
            "StdOut": "/hpcwork/SYNTHETIC/job.out",
            "StdErr": "/hpcwork/SYNTHETIC/job.err",
            "WorkDir": "/hpcwork/SYNTHETIC",
        }

        payload = " ".join(
            f"{key}={value}" for key, value in fields.items()
        )

        def fake_run(cmd, *args, **kwargs):
            if cmd[:3] != ["scontrol", "show", "job"]:
                raise AssertionError(f"Unexpected subprocess.run: {cmd}")

            return subprocess.CompletedProcess(
                cmd, 0, stdout=payload, stderr=""
            )

        def fake_output(cmd, *args, **kwargs):
            if cmd[0] == "nvidia-smi":
                return gpu + "\n"

            if cmd[:3] == [
                "scontrol", "write", "batch_script"
            ]:
                return batch

            if cmd[0] == "date":
                return str(end_epoch) + "\n"

            raise AssertionError(
                f"Unexpected subprocess.check_output: {cmd}"
            )

        filesystem_stub = types.SimpleNamespace(
            inspect_directory=lambda path: {
                "synthetic": True,
                "path": str(path),
            }
        )

        with (
            patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": job_id,
                    "CUDA_VISIBLE_DEVICES": "0",
                },
            ),
            patch.dict(
                sys.modules,
                {"claix_b24_filesystem": filesystem_stub},
            ),
            patch.object(
                launcher.subprocess,
                "run",
                side_effect=fake_run,
            ),
            patch.object(
                launcher.subprocess,
                "check_output",
                side_effect=fake_output,
            ),
        ):
            return launcher.allocation_deadline(
                mode,
                checkpoint_interval=interval,
                combined_leg=leg,
            )

    def test_sync500_combined_allocation_accepted(self):
        deadline = self.allocation(
            "experimental", 500, "B"
        )
        self.assertGreater(deadline, time.time())

    def test_last2_combined_allocation_accepted(self):
        deadline = self.allocation(
            "h100-qualification", 250, "E1"
        )
        self.assertGreater(deadline, time.time())

    def test_original_qualification_rejects_eight_hours(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "h100-qualification", 250, None
            )

    def test_original_experiment_rejects_eight_hours(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "experimental", 500, None
            )

    def test_wrong_combined_leg_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "h100-qualification", 250, "B"
            )

        with self.assertRaises(RuntimeError):
            self.allocation(
                "experimental", 500, "E1"
            )

    def test_wrong_walltime_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "experimental", 500, "B",
                limit="07:00:00",
            )

        with self.assertRaises(RuntimeError):
            self.allocation(
                "h100-qualification", 250, "E1",
                limit="07:00:00",
            )

    def test_wrong_batch_script_rejected(self):
        for mode, interval, leg in (
            ("experimental", 500, "B"),
            ("h100-qualification", 250, "E1"),
        ):
            with self.subTest(leg=leg):
                with self.assertRaises(RuntimeError):
                    self.allocation(
                        mode, interval, leg,
                        batch="#!/bin/bash\n",
                    )

    def test_insufficient_cpu_allocation_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "h100-qualification", 250, "E1",
                cpus=8,
            )

    def test_non_h100_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(
                "h100-qualification", 250, "E1",
                gpu="NVIDIA A100",
            )

    def test_batch_wrapper_resources(self):
        required = (
            "#SBATCH --account=thes2452",
            "#SBATCH --partition=c23g",
            "#SBATCH --gres=gpu:hopper:1",
            "#SBATCH --cpus-per-task=12",
            "#SBATCH --mem=96G",
            "#SBATCH --time=08:00:00",
            "#SBATCH --signal=B:USR1@1800",
        )

        for line in required:
            self.assertIn(line, BATCH)


if __name__ == "__main__":
    unittest.main()
