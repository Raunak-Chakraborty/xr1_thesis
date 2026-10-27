"""Dependency-free Doosan training/deployment normalization contract."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path


WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
NORMALIZATION_SCHEMA = "doosan_state_v1"


def is_doosan_config(data):
    return (
        data.get("type") == "DoosanDataModule"
        or data.get("params", {}).get("type") == "doosan_json"
    )


def validate_use_wrench(value):
    if not isinstance(value, bool):
        raise ValueError("use_wrench must be an explicit boolean")
    return value


def _matrix(data, name, rows):
    value = data.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{name} must have shape ({rows},60)")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 60:
            raise ValueError(f"{name} must have shape ({rows},60)")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or abs(item) > 3.4028234663852886e38
            for item in row
        ):
            raise ValueError(f"{name} must contain finite float32-compatible numbers")
        result.append([struct.unpack("f", struct.pack("f", item))[0] for item in row])
    return result


def _validate_wrench_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("force-state training requires normalization_metadata from the converter")
    required = {
        "schema": NORMALIZATION_SCHEMA,
        "source": "dataset",
        "split": "train",
        "use_wrench": True,
        "force_unit": "N",
        "torque_unit": "Nm",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"normalization_metadata.{name} must be {expected!r}")
    validate_use_wrench(metadata["use_wrench"])
    if metadata.get("wrench_order") != list(WRENCH_ORDER):
        raise ValueError(f"normalization_metadata.wrench_order must be {list(WRENCH_ORDER)}")
    for name in ("dataset_id", "wrench_frame"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"normalization_metadata.{name} must be a nonempty string")
    count = metadata.get("num_state_samples")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("normalization_metadata.num_state_samples must be an integer >= 2")


def resolve_doosan_data(data):
    """Load optional statistics JSON, validate, and return a fresh model-facing profile.

    File statistics and inline statistics are mutually exclusive. Resolved profiles
    omit normalization_path so saved experiment configs are self-contained.
    Metadata is a provenance declaration, not proof that statistics were measured.
    """
    result = dict(data)
    use_wrench = validate_use_wrench(result.get("use_wrench", False))
    result["use_wrench"] = use_wrench
    if result.get("action_length", 30) != 30:
        raise ValueError("Doosan action_length must remain 30")
    result["action_length"] = 30
    path = result.pop("normalization_path", None)
    if path is not None:
        if not isinstance(path, str) or not path.strip() or path == "???":
            raise ValueError("normalization_path must name a real dataset-derived statistics JSON")
        keys = ("mean", "std", "q01", "q99", "normalization_metadata")
        if any(result.get(name) is not None for name in keys):
            raise ValueError("choose normalization_path OR inline statistics, not both")
        for name in keys:
            result.pop(name, None)
        with Path(path).open(encoding="utf-8") as source:
            statistics = json.load(source)
        if not isinstance(statistics, Mapping):
            raise ValueError("normalization JSON must be an object")
        result.update({name: statistics[name] for name in keys if name in statistics})

    mean = _matrix(result, "mean", 30)
    std = _matrix(result, "std", 30)
    q01 = _matrix(result, "q01", 1)
    q99 = _matrix(result, "q99", 1)
    if any(value < 0 for row in std for value in row):
        raise ValueError("std cannot be negative")
    if any(value != 0 for matrix in (mean, std) for row in matrix for value in row[7:]):
        raise ValueError("Doosan action mean/std must be zero outside indices 0:7")
    for index, (lower, upper) in enumerate(zip(q01[0], q99[0])):
        if upper <= lower and (lower != 0 or upper != 0):
            raise ValueError(f"q99 must exceed q01 outside padding (state index {index})")
    if use_wrench:
        _validate_wrench_metadata(result.get("normalization_metadata"))
        for index in range(20, 26):
            span = q99[0][index] - q01[0][index]
            if not 1e-6 < span <= 3.4028234663852886e38:
                raise ValueError(
                    f"wrench index {index} requires dataset-derived q99 - q01 > 1e-6; "
                    "missing, constant, or zero-padded force statistics are not usable"
                )
    active_end = 26 if use_wrench else 20
    for matrix in (q01, q99):
        matrix[0][6] = 0.0
        matrix[0][active_end:] = [0.0] * (60 - active_end)
    result.update(mean=mean, std=std, q01=q01, q99=q99)
    return result
