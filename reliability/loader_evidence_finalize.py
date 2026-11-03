#!/usr/bin/env python3
"""Finalize XR-1 diagnostic incidents without touching training state.

Run after the instrumented training process exits. This utility copies
completed incident reports from node-local scratch to persistent storage,
verifies content hashes, and writes an atomic receipt.

It does not delete source incidents, alter checkpoints, or signal processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_LOADER_DIRECTORIES = 8
MAX_REPORTS_PER_LOADER = 8


def digest(data):
    return hashlib.sha256(data).hexdigest()


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".",
        suffix=".partial",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

        fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )

        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--receipt", required=True)

    args = parser.parse_args()

    local = Path(args.local_root).resolve(strict=True)
    persistent = Path(args.persistent_root)
    receipt = Path(args.receipt)

    if not local.is_dir():
        raise RuntimeError("Local diagnostic spool is not a directory")

    if persistent.is_symlink() or receipt.is_symlink():
        raise RuntimeError("Refusing symlink destination")

    directories = sorted(local.glob("loader-*"))

    if len(directories) > MAX_LOADER_DIRECTORIES:
        raise RuntimeError("Unexpected number of loader directories")

    results = []

    for directory in directories:

        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError("Invalid loader directory")

        reports = sorted(directory.glob("incident-*.json"))

        if len(reports) > MAX_REPORTS_PER_LOADER:
            raise RuntimeError("Incident-report limit exceeded")

        for source in reports:

            if source.is_symlink() or not source.is_file():
                raise RuntimeError("Invalid incident report")

            if source.stat().st_size > MAX_REPORT_BYTES:
                raise RuntimeError("Incident report exceeds size limit")

            raw = source.read_bytes()

            # Reject incomplete or malformed reports before publication.
            row = json.loads(raw)

            if row.get("schema") != "xr1-loader-diagnostic-v1":
                raise RuntimeError("Unknown incident-report schema")

            target = persistent / directory.name / source.name

            if target.is_symlink():
                raise RuntimeError("Refusing symlink incident destination")

            if target.exists():
                if digest(target.read_bytes()) != digest(raw):
                    raise RuntimeError(
                        "Existing persistent incident differs from local"
                    )
            else:
                publish(target, raw)

            saved = target.read_bytes()

            if digest(saved) != digest(raw):
                raise RuntimeError("Persistent incident verification failed")

            results.append({
                "loader": directory.name,
                "incident": source.name,
                "bytes": len(raw),
                "sha256": digest(raw),
                "persistent_path": str(target),
            })

    manifest = {
        "schema": "xr1-loader-evidence-finalization-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "local_root": str(local),
        "persistent_root": str(persistent),
        "incidents_verified": len(results),
        "incidents": results,
        "status": "PASS",
    }

    publish(
        receipt,
        (
            json.dumps(manifest, indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    )

    print("EVIDENCE_FINALIZATION=PASS")
    print("INCIDENTS_VERIFIED=", len(results))
    print("RECEIPT=", receipt)


if __name__ == "__main__":
    main()
