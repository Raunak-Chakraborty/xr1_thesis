"""Deterministic, standard-library-only statistics for converted Doosan episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mibot.utils.doosan_profile import NORMALIZATION_SCHEMA, WRENCH_ORDER, resolve_doosan_data


HORIZON = 30
WIDTH = 60
SPLITS = ("train", "validation", "test")
ACTION_SEMANTICS = "precomputed_delta_tcp_pos_rotvec_absolute_gripper"
STATE_FIELDS = (("joint_pos", 6, 0), ("tcp_pos", 3, 8),
                ("tcp_rotvec", 3, 11), ("joint_vel", 6, 14))


class PreflightError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _read_json(path):
    payload = path.read_bytes()
    value = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: JSON must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    try:
        result = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        raise ValueError(f"{name}: number is outside float32 range") from None
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite float32 value")
    return result


def _vector(value, width, name):
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name}: expected {width} values")
    return [_number(item, name) for item in value]


def _series(mapping, name, count):
    value = mapping.get(name)
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name}: expected exactly num_frames={count} observations")
    return value


def _wrench_declaration(manifest, use_wrench):
    declaration = manifest.get("wrench")
    if declaration is None and not use_wrench:
        return None
    if not isinstance(declaration, dict):
        raise ValueError("force data requires a manifest.wrench declaration")
    for name, expected in (("order", list(WRENCH_ORDER)), ("force_unit", "N"), ("torque_unit", "Nm")):
        if declaration.get(name) != expected:
            raise ValueError(f"manifest.wrench.{name} must be {expected!r}")
    for name in ("frame", "frame_provenance", "compensation", "calibration_reference"):
        _text(declaration.get(name), f"manifest.wrench.{name}")
    if type(declaration.get("calibration_verified")) is not bool:
        raise ValueError("manifest.wrench.calibration_verified must be a boolean declaration")
    return declaration


def _membership(manifest, manifest_path, report):
    if manifest.get("schema") != "doosan_split_v1":
        raise ValueError("manifest.schema must be 'doosan_split_v1'")
    _text(manifest.get("dataset_id"), "manifest.dataset_id")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("manifest.provenance is required")
    for name in ("converter_revision", "dataset_revision"):
        _text(provenance.get(name), f"manifest.provenance.{name}")
    if provenance.get("action_semantics") != ACTION_SEMANTICS:
        raise ValueError(f"manifest.provenance.action_semantics must be {ACTION_SEMANTICS!r}")
    root = (manifest_path.parent / _text(manifest.get("root"), "manifest.root")).resolve()
    if not root.is_dir():
        raise ValueError("manifest.root must be an existing episode directory")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("manifest.splits must explicitly declare train, validation, and test lists")
    records, seen_ids, seen_paths = [], {}, {}
    for split in SPLITS:
        entries = splits[split]
        if not isinstance(entries, list) or (split == "train" and not entries):
            raise ValueError(f"manifest.splits.{split} must be a list; train must be nonempty")
        report["split_episode_counts"][split] = len(entries)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{split} entries must be objects")
            identity = _text(entry.get("episode_id"), "episode_id")
            relative = Path(_text(entry.get("path"), "episode path"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{identity}: episode path must be relative and stay inside root")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.suffix != ".json":
                raise ValueError(f"{identity}: episode must be an existing .json file inside root")
            checksum = entry.get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError(f"{identity}: sha256 must be 64 lowercase hexadecimal characters")
            for seen, key, label in ((seen_ids, identity, "episode ID"), (seen_paths, path, "episode path")):
                if key in seen:
                    raise ValueError(f"duplicate {label} / split leakage: {identity} ({seen[key]}, {split})")
                seen[key] = split
            records.append(dict(entry, split=split, resolved_path=path))
    discovered = {path.resolve() for path in root.rglob("*.json") if path.resolve() != manifest_path}
    if discovered != set(seen_paths):
        unexpected = sorted(str(path.relative_to(root)) for path in discovered - set(seen_paths))
        raise ValueError(f"manifest membership mismatch; unlisted episode JSON files: {unexpected}")
    report["membership"] = "exact partition of episode JSON root"
    return sorted(records, key=lambda record: (SPLITS.index(record["split"]), record["episode_id"], record["path"]))


def _episode(trajectory, record, use_wrench, declaration):
    identity, split = record["episode_id"], record["split"]
    count = trajectory.get("num_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{identity}: num_frames must be a positive integer (empty episodes are not skipped)")
    for name, expected in (("episode_id", identity), ("split", split)):
        if name in trajectory and trajectory[name] != expected:
            raise ValueError(f"{identity}: trajectory {name} disagrees with manifest")
    kind = trajectory.get("trajectory_type", "success")
    if kind not in ("success", "ongoing", "invalid"):
        raise ValueError(f"{identity}: unsupported trajectory_type")
    proprio, actions = trajectory.get("proprios"), trajectory.get("actions")
    if not isinstance(proprio, dict) or not isinstance(actions, dict):
        raise ValueError(f"{identity}: proprios and actions objects are required")
    if "wrench_metadata" in trajectory and trajectory["wrench_metadata"] != declaration:
        raise ValueError(f"{identity}: wrench_metadata disagrees with manifest declaration")
    states = [[0.0] * WIDTH for _ in range(count)]
    for name, width, offset in STATE_FIELDS:
        for index, value in enumerate(_series(proprio, name, count)):
            states[index][offset:offset + width] = _vector(value, width, name)
    for index, value in enumerate(_series(proprio, "gripper", count)):
        gripper = _number(value[0] if isinstance(value, list) and len(value) == 1 else value, "gripper")
        if not 0 <= gripper <= 1:
            raise ValueError(f"{identity}: gripper must be normalized to [0,1]")
        states[index][7] = gripper
    wrench = None
    if "wrench" in proprio:
        if declaration is None:
            raise ValueError(f"{identity}: recorded wrench requires a manifest.wrench declaration")
        wrench = [_vector(value, 6, "wrench") for value in _series(proprio, "wrench", count)]
        if use_wrench:
            for state, value in zip(states, wrench):
                state[20:26] = value
    elif use_wrench:
        raise ValueError(f"{identity}: missing proprios.wrench observations")
    semantic = [_vector(value, 7, "actions.semantic") for value in _series(actions, "semantic", count)]
    if any(not 0 <= value[6] <= 1 for value in semantic):
        raise ValueError(f"{identity}: action gripper target must be normalized to [0,1]")
    numeric_digest = hashlib.sha256(json_bytes({
        "states": states, "actions": semantic, "wrench": wrench,
    })).hexdigest()
    return dict(states=states, actions=semantic, wrench=wrench, count=count, kind=kind,
                numeric_sha256=numeric_digest, episode_id=identity, split=split)


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.squared_deviation = 0.0

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_deviation += delta * (value - self.mean)

    def merge(self, other):
        if not other.count:
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.squared_deviation += (
            other.squared_deviation + delta * delta * self.count * other.count / total
        )
        self.mean += delta * other.count / total
        self.count = total

    def std(self):
        return math.sqrt(max(0.0, self.squared_deviation / self.count))


def quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _summary(values):
    ordered = sorted(values)
    moments = Moments()
    for value in values:
        moments.add(value)
    return dict(count=len(values), min=ordered[0], max=ordered[-1], mean=moments.mean,
                std=moments.std(), **{name: quantile(ordered, probability)
                                     for name, probability in (("q01", .01), ("q25", .25),
                                                               ("median", .5), ("q75", .75), ("q99", .99))})


def _force_quality(episodes, declaration, force_limit, torque_limit):
    recorded = [episode for episode in episodes if episode["wrench"] is not None]
    if not recorded:
        return {"status": "no recorded training wrench", "channels": {}, "review_flags": []}
    channels, flags = {}, [{"code": "physical_interpretation_unverified", "message": (
        "Wrench may include cable load, gravity, bias, drift, or genuine contact. "
        "Statistics and declarations do not verify calibration or contact causality."
    )}]
    if not declaration["calibration_verified"]:
        flags.append({"code": "calibration_declared_unverified"})
    for index, name in enumerate(WRENCH_ORDER):
        values = [value[index] for episode in recorded for value in episode["wrench"]]
        summary = _summary(values)
        spread = summary["q75"] - summary["q25"]
        lower, upper = summary["q25"] - 3 * spread, summary["q75"] + 3 * spread
        outliers = sum(value < lower or value > upper for value in values)
        examples = []
        for episode in recorded:
            for frame_index, value in enumerate(episode["wrench"]):
                if (value[index] < lower or value[index] > upper) and len(examples) < 20:
                    examples.append(dict(episode_id=episode["episode_id"], frame_index=frame_index, value=value[index]))
        steps = [abs(current[index] - previous[index]) for episode in recorded
                 for previous, current in zip(episode["wrench"], episode["wrench"][1:])]
        summary.update(iqr_fence=[lower, upper], iqr_outlier_count=outliers,
                       iqr_outlier_examples=examples, iqr_outlier_examples_truncated_count=outliers - len(examples),
                       outside_q01_q99_count=sum(value < summary["q01"] or value > summary["q99"] for value in values),
                       max_adjacent_step=max(steps, default=0.0),
                       adjacent_equal_count=sum(value == 0 for value in steps), adjacent_pair_count=len(steps))
        channels[name] = summary
        if outliers:
            flags.append(dict(code="potential_outliers", channel=name, count=outliers))
    norms = {}
    for name, offset, limit in (("force_norm_N", 0, force_limit), ("torque_norm_Nm", 3, torque_limit)):
        values = [math.hypot(*value[offset:offset + 3]) for episode in recorded for value in episode["wrench"]]
        norms[name] = _summary(values)
        if limit is not None:
            count = sum(value > limit for value in values)
            norms[name].update(operator_review_limit=limit, above_limit_count=count)
            if count:
                flags.append(dict(code="operator_review_limit_exceeded", quantity=name, count=count))
    return dict(status="diagnostics only; no filtering or correction applied", channels=channels,
                norms=norms, review_flags=flags)


def _statistics(episodes, manifest, manifest_digest, use_wrench, declaration, report):
    total = sum(episode["count"] for episode in episodes)
    active = list(range(6)) + list(range(7, 26 if use_wrench else 20))
    q01, q99 = [[0.0] * WIDTH], [[0.0] * WIDTH]
    report["state_sample_counts"] = [[total if index in active else 0 for index in range(WIDTH)]]
    report["num_training_windows"] = total
    for index in active:
        values = sorted(state[index] for episode in episodes for state in episode["states"])
        q01[0][index], q99[0][index] = quantile(values, .01), quantile(values, .99)
    report["degenerate_state_quantile_indices"] = [
        index for index in active if _number(q99[0][index], "q99") <= _number(q01[0][index], "q01")
    ]
    moments = [[Moments() for _ in range(7)] for _ in range(HORIZON)]
    for episode in episodes:
        suffix = [Moments() for _ in range(7)]
        for index in range(episode["count"] - 1, -1, -1):
            for channel, value in enumerate(episode["actions"][index]):
                suffix[channel].add(value)
                if index < HORIZON:
                    moments[index][channel].merge(suffix[channel])
    counts = [row[0].count for row in moments]
    report["action_sample_counts"] = [[count] * 7 + [0] * 53 for count in counts]
    report["temporally_padded_windows_excluded_per_horizon"] = [total - count for count in counts]
    if not all(counts):
        raise ValueError("no unmasked action samples at one or more horizons; need an episode with >= 30 frames")
    mean = [[moment.mean for moment in row] + [0.0] * 53 for row in moments]
    std = [[moment.std() for moment in row] + [0.0] * 53 for row in moments]
    metadata = dict(
        schema=NORMALIZATION_SCHEMA, source="dataset", split="train", use_wrench=use_wrench,
        dataset_id=manifest["dataset_id"], num_state_samples=total, manifest_sha256=manifest_digest,
        exporter="standalone_doosan_v1", provenance=manifest["provenance"],
        quantile_method="linear: (n-1)*p; recorded float32 values",
        action_window_policy="every frame once; horizon h uses actions[start+h] only if start+h < num_frames",
        action_std_ddof=0,
    )
    if declaration is not None:
        metadata.update(wrench_order=list(WRENCH_ORDER), wrench_frame=declaration["frame"],
                        force_unit="N", torque_unit="Nm", wrench_provenance=declaration)
    payload = dict(mean=mean, std=std, q01=q01, q99=q99, normalization_metadata=metadata)
    resolved = resolve_doosan_data(dict(payload, use_wrench=use_wrench, action_length=HORIZON))
    return {name: resolved[name] for name in payload}


def build_normalization(manifest_path, *, use_wrench, force_limit_n=None, torque_limit_nm=None):
    """Return (statistics, preflight); raise PreflightError with a report on failure.

    Every listed split is integrity-checked, but only train contributes numbers.
    Quantiles are exact and retain training states in memory; no sampling occurs.
    """
    report = dict(schema="doosan_preflight_v1", status="failed", errors=[], warnings=[],
                  split_episode_counts={}, files=[], use_wrench=use_wrench)
    try:
        if type(use_wrench) is not bool:
            raise ValueError("use_wrench must be an explicit boolean")
        for name, value in (("force_limit_n", force_limit_n), ("torque_limit_nm", torque_limit_nm)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        path = Path(manifest_path).resolve()
        manifest, manifest_digest = _read_json(path)
        report["manifest_sha256"] = manifest_digest
        records = _membership(manifest, path, report)
        declaration = _wrench_declaration(manifest, use_wrench)
        report["wrench_declaration"] = declaration
        seen_hashes, seen_numeric, training = {}, {}, []
        frame_counts, kinds, missing_wrench = Counter(), Counter(), Counter()
        for record in records:
            try:
                trajectory, digest = _read_json(record["resolved_path"])
                file_record = {name: record[name] for name in ("episode_id", "path", "split")}
                file_record.update(sha256=digest, checksum_matches=digest == record["sha256"])
                report["files"].append(file_record)
                if digest != record["sha256"]:
                    raise ValueError(f"{record['episode_id']}: checksum mismatch")
                declared_count = trajectory.get("num_frames")
                if type(declared_count) is int and declared_count > 0:
                    proprio = trajectory.get("proprios")
                    observations = proprio.get("wrench") if isinstance(proprio, dict) else None
                    absent = declared_count if not isinstance(observations, list) else (
                        max(0, declared_count - len(observations))
                        + sum(value is None for value in observations[:declared_count])
                    )
                    missing_wrench[record["split"]] += absent
                    file_record["missing_wrench_observations"] = absent
                episode = _episode(trajectory, record, use_wrench, declaration)
                file_record.update(num_frames=episode["count"], numeric_sha256=episode["numeric_sha256"])
                for seen, key, label in ((seen_hashes, digest, "JSON content"),
                                         (seen_numeric, episode["numeric_sha256"], "numeric episode content")):
                    if key in seen:
                        raise ValueError(f"duplicate {label} / split leakage: {seen[key]}, {record['episode_id']}")
                    seen[key] = record["episode_id"]
                frame_counts[record["split"]] += episode["count"]
                if record["split"] == "train":
                    training.append(episode)
                    kinds[episode["kind"]] += 1
            except (ValueError, OSError, OverflowError) as error:
                report["errors"].append(f"{record['episode_id']}: {error}")
        report["split_frame_counts"] = dict(frame_counts)
        report["missing_wrench_observations_per_split"] = {split: missing_wrench[split] for split in SPLITS}
        report["training_trajectory_types"] = dict(kinds)
        if report["errors"]:
            raise PreflightError(report)
        report["force_quality"] = _force_quality(training, declaration, force_limit_n, torque_limit_nm)
        if any(not manifest["splits"][split] for split in ("validation", "test")):
            report["warnings"].append("empty held-out lists: separation checked only against declared files, not external datasets")
        report["leakage_check"] = "IDs, canonical paths, raw checksums, and canonical numeric-content hashes are disjoint"
        report["scope"] = "episode JSON and declared numeric data only; no video decoding, calibration, or physical-contact verification"
        payload = _statistics(training, manifest, manifest_digest, use_wrench, declaration, report)
        report["statistics_sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
        report["status"] = "passed"
        return payload, report
    except PreflightError:
        raise
    except (ValueError, OSError, OverflowError) as error:
        report["errors"].append(str(error))
        raise PreflightError(report) from error


def _write_new(path, payload):
    with Path(path).open("xb") as destination:
        destination.write(json_bytes(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="new statistics JSON, outside the episode root")
    parser.add_argument("--report", required=True, help="new preflight JSON, including on structural failure")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--use-wrench", action="store_true")
    mode.add_argument("--force-free", action="store_true")
    parser.add_argument("--force-limit-n", type=float, help="operator-supplied force-norm review threshold, not a safety limit")
    parser.add_argument("--torque-limit-nm", type=float, help="operator-supplied torque-norm review threshold, not a safety limit")
    args = parser.parse_args()
    destinations = [Path(value).resolve() for value in (args.output, args.report)]
    if len(set(destinations)) != 2 or any(path.exists() for path in destinations):
        parser.error("output/report must be different, new files; existing files are never overwritten")
    if any(not path.parent.is_dir() for path in destinations):
        parser.error("output/report parent directories must already exist")
    try:
        manifest_path = Path(args.manifest).resolve()
        manifest, _ = _read_json(manifest_path)
        root = (manifest_path.parent / _text(manifest.get("root"), "manifest.root")).resolve()
        if any(path.is_relative_to(root) for path in destinations):
            parser.error("output/report must be outside the recursively scanned episode root")
        payload, report = build_normalization(
            manifest_path, use_wrench=args.use_wrench,
            force_limit_n=args.force_limit_n, torque_limit_nm=args.torque_limit_nm,
        )
    except (PreflightError, ValueError, OSError) as error:
        report = error.report if isinstance(error, PreflightError) else dict(
            schema="doosan_preflight_v1", status="failed", errors=[str(error)],
        )
        _write_new(args.report, report)
        parser.exit(2, "preflight failed; see report (no statistics written)\n")
    _write_new(args.report, report)
    _write_new(args.output, payload)
    print(f"wrote statistics and preflight: {report['num_training_windows']} training states/windows; no demonstrations modified")


if __name__ == "__main__":
    main()
