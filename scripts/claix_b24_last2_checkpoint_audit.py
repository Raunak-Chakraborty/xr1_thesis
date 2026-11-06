#!/usr/bin/env python3
"""Read-only audit of an E1 last-two-block verified compact checkpoint.

Run on an allocated compute node, not the CLAIX login node.
The file must originate from the trusted local training run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from mibot.utils.claix_checkpoint import load_verified


STACKS = {
    "model.vlm.model.visual.blocks.": {22, 23},
    "model.vlm.model.language_model.layers.": {34, 35},
    "model.dit.layers.": {34, 35},
}

INTERFACES = (
    "state_projector",
    "action_projector",
    "action_output_layer",
    "t_embedder",
    "t_projector",
    "sink",
)

PRETRAINED_SHA256 = (
    "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7"
)

LANGUAGE_SHA256 = (
    "21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, default=30)
    args = parser.parse_args()

    path = Path(args.checkpoint)

    if not path.is_absolute():
        raise RuntimeError("Checkpoint path must be absolute")

    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Invalid checkpoint path")

    # Full SHA256, receipt and production-checkpoint validation.
    ck = load_verified(path, resumable=False)

    step = ck["global_step"]
    meta = ck["xr1_compact_fp32"]

    assert step == args.expected_step
    assert meta["last_n"] == 2
    assert meta["trainable_storage_dtype"] == "torch.float32"
    assert meta["frozen_weights_source"] == "external_original_pretrained_checkpoint"

    assert meta["pretrained_sha256"] == PRETRAINED_SHA256
    assert meta["language_sha256"] == LANGUAGE_SHA256

    assert ck["xr1_production"]["qualification_last2_vram_probe"] is True
    assert ck["xr1_production"].get("experiment") is None

    state = ck["state_dict"]

    assert isinstance(state, dict)
    assert len(state) == meta["selected_tensor_count"]

    selected = set(state)

    digest = hashlib.sha256(
        "\n".join(sorted(selected)).encode()
    ).hexdigest()

    assert digest == meta["selected_key_sha256"]

    assert all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        for value in state.values()
    )

    parameter_count = sum(
        value.numel() for value in state.values()
    )

    assert parameter_count == meta["trainable_parameter_count"]

    seen_stacks = {
        name: set() for name in STACKS
    }

    seen_interfaces = set()

    for key in selected:
        matched = False

        for prefix, allowed_indices in STACKS.items():
            if key.startswith(prefix):
                index_text = key[len(prefix):].split(".", 1)[0]

                if not index_text.isdigit():
                    raise RuntimeError(
                        f"Unrecognized transformer key: {key}"
                    )

                index = int(index_text)

                if index not in allowed_indices:
                    raise RuntimeError(
                        f"Frozen transformer parameter was saved: {key}"
                    )

                seen_stacks[prefix].add(index)
                matched = True
                break

        if matched:
            continue

        for interface in INTERFACES:
            if key.startswith(f"model.{interface}."):
                seen_interfaces.add(interface)
                matched = True
                break

        if not matched:
            raise RuntimeError(
                f"Unexpected saved model parameter: {key}"
            )

    for prefix, expected in STACKS.items():
        if seen_stacks[prefix] != expected:
            raise RuntimeError(
                f"Missing last-two transformer blocks: {prefix}"
            )

    if seen_interfaces != set(INTERFACES):
        raise RuntimeError(
            f"Missing trainable interfaces: {set(INTERFACES)-seen_interfaces}"
        )

    optimizers = ck["optimizer_states"]

    assert len(optimizers) == 1

    optimizer = optimizers[0]

    assert optimizer["state"]

    for values in optimizer["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            assert name in values
            assert isinstance(values[name], torch.Tensor)
            assert values[name].dtype == torch.float32

    replay = ck["xr1_replay"]

    assert replay["rank_sample_offset"] == step * 24

    size = path.stat().st_size

    ceiling = 12 * parameter_count + 64 * 1024**2

    if size > ceiling:
        raise RuntimeError(
            "Checkpoint unexpectedly exceeds compact-state size ceiling"
        )

    report = {
        "status": "PASS",
        "checkpoint": str(path),
        "global_step": step,
        "last_n": 2,
        "trainable_parameter_count": parameter_count,
        "selected_tensor_count": len(state),
        "saved_model_parameters_are_trainable_only": True,
        "frozen_model_parameters_in_state_dict": 0,
        "optimizer_state_present": True,
        "scheduler_state_present": bool(ck["lr_schedulers"]),
        "replay_offset": replay["rank_sample_offset"],
        "checkpoint_bytes": size,
        "checkpoint_size_ceiling_bytes": ceiling,
        "checkpoint_receipt_sha256_verified": True,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    print("E1_LAST2_COMPACT_CHECKPOINT=PASS")


if __name__ == "__main__":
    main()
