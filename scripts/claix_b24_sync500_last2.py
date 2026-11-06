#!/usr/bin/env python3
"""Fail-closed sequential Sync-500/1500 + E1-last2/30 H100 experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SYNC_NAME = "xr1_doosan_b24_checkpoint500_v1"
E1_NAME = "qualification_last2_vram_w4"
STOP = {"requested": False, "signal": None, "child": None}

def stamp():
    return datetime.now(timezone.utc).isoformat()

def atomic_report(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def handle(signum, frame):
    STOP["requested"] = True
    STOP["signal"] = signum
    child = STOP["child"]
    if child is not None and child.poll() is None:
        child.send_signal(signum)

def command(args, leg):
    launcher = str(ROOT / "scripts/claix_b24_launch.py")
    base = [args.python, launcher]
    root = str(args.run_root / ("B_sync500" if leg == "B" else "E1_last2"))
    if leg == "B":
        return base + [
            "--mode", "experimental",
            "--experimental-unqualified",
            "--experimental-checkpoint-interval", "500",
            "--combined-leg", "B",
            "--stop-after-step", "1500",
            "--workers", "4",
            "--run-root", root,
            "--overrides", args.overrides,
            "--filesystem-source", args.filesystem_source,
            "--project-id", str(args.project_id),
            "--python", args.python,
        ]
    return base + [
        "--mode", "h100-qualification",
        "--qualification-last2-vram-probe",
        "--qualification-max-steps", "30",
        "--qualification-gpu-monitor",
        "--combined-leg", "E1",
        "--workers", "4",
        "--run-root", root,
        "--overrides", args.overrides,
        "--python", args.python,
    ]

def verify_source(expected):
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    )
    if commit != expected or status:
        raise RuntimeError("Source commit or clean-tree attestation failed")

def run_leg(args, leg, report):
    if STOP["requested"]:
        raise RuntimeError("Stopping before next leg after signal")

    label = "B_sync500" if leg == "B" else "E1_last2"
    cmd = command(args, leg) + ["--execute"]

    logfile = args.run_root / f"{label}.launcher.log"
    report["active_leg"] = label
    report["legs"][label] = {"status": "RUNNING", "command": cmd, "started": stamp()}
    atomic_report(args.run_root / "combined-status.json", report)

    print(f"COMBINED_START={label} LOG={logfile}", flush=True)

    with logfile.open("x") as output:
        child = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT
        )
        STOP["child"] = child
        try:
            while child.poll() is None:
                time.sleep(2)
            rc = child.wait()
        finally:
            STOP["child"] = None

    report["legs"][label].update(exit_code=rc, completed=stamp())
    report["legs"][label]["status"] = "CHILD_EXITED" if rc == 0 else "FAILED"
    atomic_report(args.run_root / "combined-status.json", report)

    if STOP["requested"] or rc != 0:
        raise RuntimeError(f"{label} stopped or failed: exit={rc}")

    verify_source(os.environ["XR1_EXPECTED_SHA"])
    return label

def verify_leg(args, leg, report):
    label = "B_sync500" if leg == "B" else "E1_last2"
    path = args.run_root / label / "project_xr1_doosan" / (
        SYNC_NAME if leg == "B" else E1_NAME
    )

    if leg == "B":
        script = ROOT / "scripts/claix_b24_sync500_last2_verify.py"
        cmd = [
            args.python, "-B", str(script),
            "--target", str(path), "--mode", "sync500",
        ]
    else:
        script = ROOT / "scripts/claix_b24_last2_checkpoint_audit.py"
        cmd = [
            args.python, "-B", str(script),
            "--checkpoint", str(path / "step-000030.ckpt"),
            "--expected-step", "30",
        ]

    audit = args.run_root / f"{label}.verification.log"
    with audit.open("x") as output:
        completed = subprocess.run(
            cmd, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT
        )

    if completed.returncode:
        report["legs"][label]["status"] = "VERIFICATION_FAILED"
        atomic_report(args.run_root / "combined-status.json", report)
        raise RuntimeError(f"{label} verification failed; see {audit}")

    if leg == "E1":
        rows = [
            json.loads(line) for line in (path / "telemetry.jsonl").read_text().splitlines()
        ]
        steps = [r for r in rows if r.get("event") == "step"]
        peaks = [
            r.get("gpu_peak_reserved_bytes")
            for r in steps if isinstance(r.get("gpu_peak_reserved_bytes"), int)
        ]
        if len(steps) != 30 or not peaks or max(peaks) <= 0:
            raise RuntimeError("E1 did not produce 30 steps and GPU peak measurements")
        report["legs"][label]["gpu_peak_reserved_bytes"] = max(peaks)
        report["legs"][label]["gpu_peak_allocated_bytes"] = max(
            int(r.get("gpu_peak_allocated_bytes") or 0) for r in steps
        )

    report["legs"][label]["status"] = "PASS"
    report["legs"][label]["verification_log"] = str(audit)
    atomic_report(args.run_root / "combined-status.json", report)
    print(f"COMBINED_VERIFIED={label}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--overrides", required=True)
    p.add_argument("--filesystem-source", required=True)
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--python", required=True)
    args = p.parse_args()

    if not args.run_root.is_absolute() or args.run_root.is_symlink():
        raise RuntimeError("Combined run root must be an absolute non-symlink path")

    if args.project_id != 44885 or not args.filesystem_source:
        raise RuntimeError("Unexpected CLAIX storage project or filesystem")

    if not Path(args.python).is_file():
        raise RuntimeError("Approved Python executable not found")

    if not Path(args.overrides).is_file():
        raise RuntimeError("Frozen path override file missing")

    cmds = [command(args, leg) for leg in ("B", "E1")]
    if not args.execute:
        print(json.dumps({
            "source": str(ROOT),
            "new_gpu_training_runs": 2,
            "historical_sync250_rerun": False,
            "order": ["B_sync500", "E1_last2"],
            "commands": cmds,
        }, indent=2))
        return

    expected = os.environ.get("XR1_EXPECTED_SHA")
    if not expected or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Pinned source and Slurm allocation required")

    verify_source(expected)

    if args.run_root.exists():
        raise RuntimeError("Combined run root already exists")

    # Persistent Lustre only.
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(args.run_root.parent)],
        text=True,
    ).strip()
    if filesystem != "lustre":
        raise RuntimeError("Combined output is not on persistent Lustre")

    args.run_root.mkdir(mode=0o700)

    report = {
        "schema": "xr1-sync500-last2-combined-v1",
        "status": "RUNNING",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source_commit": expected,
        "started": stamp(),
        "legs": {},
    }
    atomic_report(args.run_root / "combined-status.json", report)

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle)

    try:
        run_leg(args, "B", report)
        verify_leg(args, "B", report)

        if STOP["requested"]:
            raise RuntimeError("Signal received after Sync-500")

        run_leg(args, "E1", report)
        verify_leg(args, "E1", report)

        report["status"] = "PASS"
        report["completed"] = stamp()
        atomic_report(args.run_root / "combined-status.json", report)
        print("COMBINED_RESULT=PASS", flush=True)

    except BaseException as error:
        report["status"] = "BLOCKED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["completed"] = stamp()
        atomic_report(args.run_root / "combined-status.json", report)
        print("COMBINED_RESULT=BLOCKED", flush=True)
        raise

if __name__ == "__main__":
    main()
