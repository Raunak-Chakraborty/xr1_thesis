#!/usr/bin/env python3
"""Read-only post-training verification for the Sync-500 leg."""
import argparse
import json
from pathlib import Path

import torch

from mibot.utils.claix_checkpoint import load_verified, file_sha256

p = argparse.ArgumentParser()
p.add_argument("--target", required=True)
p.add_argument("--mode", choices=("sync500",), required=True)
args = p.parse_args()

target = Path(args.target)

if not target.is_dir() or target.is_symlink():
    raise RuntimeError("Invalid Sync-500 checkpoint directory")

for step in (500, 1000, 1500):
    name = f"step-{step:06d}"
    receipt_path = target / (name + ".verified.json")
    checkpoint = target / (name + ".ckpt")

    receipt = json.loads(receipt_path.read_text())

    if receipt["global_step"] != step or receipt["path"] != checkpoint.name:
        raise RuntimeError(f"Bad checkpoint receipt at step {step}")

    # Retention may delete an old payload but must preserve its receipt.
    if checkpoint.is_file():
        if checkpoint.is_symlink():
            raise RuntimeError("Symlink checkpoint forbidden")
        if checkpoint.stat().st_size != receipt["bytes"]:
            raise RuntimeError("Checkpoint size mismatch")
        if file_sha256(checkpoint) != receipt["sha256"]:
            raise RuntimeError("Checkpoint SHA256 mismatch")
    elif step != 500:
        raise RuntimeError(f"Required recovery checkpoint missing at {step}")

for step in (1000, 1500):
    ck = load_verified(target / f"step-{step:06d}.ckpt", resumable=False)
    meta = ck["xr1_compact_fp32"]

    if ck["global_step"] != step or meta["last_n"] != 1:
        raise RuntimeError("Wrong Sync-500 checkpoint architecture")

    if len(ck["state_dict"]) != meta["selected_tensor_count"]:
        raise RuntimeError("Noncompact Sync-500 model checkpoint")

    if not ck["optimizer_states"] or not ck["lr_schedulers"]:
        raise RuntimeError("Missing resumable optimizer/scheduler")

for pointer in ("latest.json", "safe-stop.json"):
    row = json.loads((target / pointer).read_text())
    if row["global_step"] != 1500:
        raise RuntimeError(f"{pointer} does not point to step 1500")

rows = [json.loads(x) for x in (target / "telemetry.jsonl").read_text().splitlines()]
steps = [r["step"] for r in rows if r.get("event") == "step"]
if steps != list(range(1, 1501)):
    raise RuntimeError("Sync-500 optimizer-step telemetry incomplete")

if not any(
    r.get("event") == "fit_end" and r.get("final_checkpoint_verified") is True
    for r in rows
):
    raise RuntimeError("Sync-500 final checkpoint not verified at fit end")

print("SYNC500_CHECKPOINT_RECEIPTS=PASS")
print("SYNC500_TRAINABLE_ONLY_MODEL_STATE=PASS")
print("SYNC500_FINAL_STEP=1500")
print("SYNC500_VERIFICATION=PASS")
