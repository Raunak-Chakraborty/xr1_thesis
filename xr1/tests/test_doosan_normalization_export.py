"""Genuine numeric regression tests; only Python's standard library is required."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from mibot.utils.doosan_profile import resolve_doosan_data
from tools.build_doosan_normalization import (
    ACTION_SEMANTICS,
    PreflightError,
    build_normalization,
    json_bytes,
    quantile,
)


ROOT = Path(__file__).resolve().parents[1]


def episode(count=32, offset=0, wrench=True):
    proprio = {
        "joint_pos": [[offset + index + (channel + 1) * .25 for channel in range(6)] for index in range(count)],
        "gripper": [(index % 3) / 2 for index in range(count)],
        "tcp_pos": [[offset + index * .01 + channel * .001 for channel in range(3)] for index in range(count)],
        "tcp_rotvec": [[offset + index * .02 + channel * .002 for channel in range(3)] for index in range(count)],
        "joint_vel": [[offset + index * .1 + channel * .05 for channel in range(6)] for index in range(count)],
    }
    if wrench:
        proprio["wrench"] = [[offset + (index - count / 2) * (channel + 1) * .2 for channel in range(6)]
                              for index in range(count)]
    return dict(num_frames=count, proprios=proprio, actions={"semantic": [
        [offset + index, 2 * index + offset, -index - offset, .01 * index, -.02 * index, .03 * index, index % 2]
        for index in range(count)
    ]})


class NormalizationExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "episodes"
        self.root.mkdir()
        self.manifest_path = self.directory / "split.json"
        self.manifest = dict(
            schema="doosan_split_v1", dataset_id="synthetic-test-training-split", root="episodes",
            provenance=dict(converter_revision="synthetic-fixture-v1", dataset_revision="synthetic-only",
                            action_semantics=ACTION_SEMANTICS),
            wrench=dict(order=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], frame="synthetic-sensor-frame",
                        force_unit="N", torque_unit="Nm", frame_provenance="synthetic-test-no-robot",
                        compensation="none: synthetic values", calibration_reference="synthetic-unverified",
                        calibration_verified=False),
            splits=dict(train=[], validation=[], test=[]),
        )
        self.primary = self.add_episode("train", "one", episode())

    def add_episode(self, split, identity, payload):
        relative = f"{split}/{identity}.json"
        path = self.root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(json_bytes(payload))
        record = dict(episode_id=identity, path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.manifest["splits"][split].append(record)
        return record

    def replace_episode(self, record, payload):
        path = self.root / record["path"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self):
        self.manifest_path.write_bytes(json_bytes(self.manifest))

    def build(self, use_wrench=True, **kwargs):
        self.save()
        return build_normalization(self.manifest_path, use_wrench=use_wrench, **kwargs)

    def reject(self, pattern, **kwargs):
        with self.assertRaisesRegex(PreflightError, pattern) as caught:
            self.build(**kwargs)
        self.assertEqual(caught.exception.report["status"], "failed")
        self.assertTrue(caught.exception.report["errors"])
        return caught.exception.report

    def cli(self, *extra, mode="--use-wrench", suffix=""):
        self.save()
        output = self.directory / f"statistics{suffix}.json"
        report = self.directory / f"preflight{suffix}.json"
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "tools/build_doosan_normalization.py"),
            "--manifest", str(self.manifest_path), "--output", str(output),
            "--report", str(report), mode, *extra,
        ], capture_output=True, text=True)
        return result, output, report

    def test_correct_60d_state_and_action_statistics(self):
        payload, report = self.build()
        self.assertEqual(len(payload["mean"]), 30)
        self.assertTrue(all(len(row) == 60 for row in payload["std"]))
        self.assertEqual(len(payload["q01"]), 1)
        self.assertEqual(len(payload["q01"][0]), 60)
        self.assertAlmostEqual(payload["mean"][0][0], 15.5)
        self.assertAlmostEqual(payload["std"][0][0], statistics.pstdev(range(32)), places=6)
        self.assertAlmostEqual(payload["mean"][0][1], 31)
        self.assertAlmostEqual(payload["mean"][0][2], -15.5)
        self.assertAlmostEqual(payload["mean"][0][6], .5)
        self.assertAlmostEqual(payload["std"][0][6], .5)
        self.assertAlmostEqual(payload["q01"][0][0], .25 + .31, places=6)
        self.assertEqual(report["status"], "passed")
        for name in ("mean", "std"):
            self.assertTrue(all(row[7:] == [0] * 53 for row in payload[name]))
        for name in ("q01", "q99"):
            self.assertEqual(payload[name][0][6], 0)
            self.assertEqual(payload[name][0][26:], [0] * 34)

    def test_exact_six_wrench_quantiles_at_indices_20_through_25(self):
        payload, report = self.build()
        for channel in range(6):
            expected_lower = (-16 + .31) * (channel + 1) * .2
            expected_upper = (-16 + 30.69) * (channel + 1) * .2
            self.assertAlmostEqual(payload["q01"][0][20 + channel], expected_lower, places=5)
            self.assertAlmostEqual(payload["q99"][0][20 + channel], expected_upper, places=5)
            self.assertEqual(report["state_sample_counts"][0][20 + channel], 32)

    def test_horizon_statistics_exclude_repeated_padding(self):
        self.add_episode("train", "short", episode(3, 100))
        payload, report = self.build()
        for horizon in range(30):
            observations = list(range(horizon, 32)) + [100 + index for index in range(horizon, 3)]
            self.assertAlmostEqual(payload["mean"][horizon][0], statistics.mean(observations), places=5)
            self.assertAlmostEqual(payload["std"][horizon][0], statistics.pstdev(observations), places=5)
            self.assertEqual(report["action_sample_counts"][horizon], [len(observations)] * 7 + [0] * 53)
            self.assertEqual(report["temporally_padded_windows_excluded_per_horizon"][horizon], 35 - len(observations))
        self.assertEqual(report["num_training_windows"], 35)
        self.assertEqual(payload["mean"][29][0], 30)
        padded_biased_mean = statistics.mean([31] * 29 + [29, 30, 31] + [102] * 3)
        self.assertNotAlmostEqual(payload["mean"][29][0], padded_biased_mean)

    def test_action_labels_are_not_rederived_from_tcp_poses(self):
        recorded = episode()
        for index, value in enumerate(recorded["actions"]["semantic"]):
            value[0] = index * 17 - 100
        self.replace_episode(self.primary, recorded)
        payload, _ = self.build()
        self.assertAlmostEqual(payload["mean"][0][0], statistics.mean(index * 17 - 100 for index in range(32)))
        self.assertAlmostEqual(payload["mean"][29][0], statistics.mean(index * 17 - 100 for index in range(29, 32)))

    def test_exact_linear_quantile_definition(self):
        self.assertEqual(quantile([0, 10], .25), 2.5)
        self.assertEqual(quantile([0, 10, 20, 30], .5), 15)
        self.assertEqual(quantile([4], .99), 4)

    def test_metadata_contains_real_counts_checksums_and_declarations(self):
        payload, report = self.build()
        metadata = payload["normalization_metadata"]
        self.assertEqual(metadata["schema"], "doosan_state_v1")
        self.assertEqual(metadata["split"], "train")
        self.assertEqual(metadata["source"], "dataset")
        self.assertIs(metadata["use_wrench"], True)
        self.assertEqual(metadata["num_state_samples"], 32)
        self.assertEqual(metadata["dataset_id"], self.manifest["dataset_id"])
        self.assertEqual(metadata["wrench_frame"], self.manifest["wrench"]["frame"])
        self.assertEqual(metadata["wrench_order"], self.manifest["wrench"]["order"])
        self.assertEqual(metadata["manifest_sha256"], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertEqual(report["files"][0]["sha256"], self.primary["sha256"])
        self.assertEqual(report["statistics_sha256"], hashlib.sha256(json_bytes(payload)).hexdigest())

    def test_deterministic_statistics_and_preflight(self):
        self.add_episode("train", "two", episode(31, 2))
        first = self.build()
        second = self.build()
        self.assertEqual(json_bytes(first), json_bytes(second))

    def test_force_free_without_wrench_and_without_guessed_declarations(self):
        self.replace_episode(self.primary, episode(wrench=False))
        del self.manifest["wrench"]
        payload, report = self.build(use_wrench=False)
        self.assertEqual(payload["q01"][0][20:], [0] * 40)
        self.assertEqual(payload["q99"][0][20:], [0] * 40)
        self.assertEqual(report["missing_wrench_observations_per_split"]["train"], 32)
        self.assertEqual(report["state_sample_counts"][0][20:], [0] * 40)
        self.assertIs(payload["normalization_metadata"]["use_wrench"], False)
        self.assertNotIn("wrench_frame", payload["normalization_metadata"])
        resolved = resolve_doosan_data(dict(payload, use_wrench=False))
        self.assertEqual(resolved["q01"], payload["q01"])

    def test_force_free_still_reports_present_force_without_conditioning_on_it(self):
        payload, report = self.build(use_wrench=False)
        self.assertEqual(payload["q01"][0][20:26], [0] * 6)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["count"], 32)

    def test_per_channel_counts_distinguish_semantic_channels_and_padding(self):
        _, report = self.build()
        self.assertEqual(sum(count > 0 for count in report["state_sample_counts"][0]), 25)
        self.assertEqual(report["state_sample_counts"][0][6], 0)
        self.assertEqual(report["action_sample_counts"][0], [32] * 7 + [0] * 53)
        _, baseline = self.build(use_wrench=False)
        self.assertEqual(sum(count > 0 for count in baseline["state_sample_counts"][0]), 19)

    def test_duplicate_episode_id_rejected(self):
        self.add_episode("train", "other", episode(31, 3))["episode_id"] = "one"
        self.reject("duplicate episode ID")

    def test_duplicate_canonical_path_rejected(self):
        self.manifest["splits"]["train"].append(dict(self.primary, episode_id="alias"))
        self.reject("duplicate episode path")

    def test_duplicate_json_bytes_rejected(self):
        self.add_episode("train", "copy", episode())
        self.reject("duplicate JSON content")

    def test_train_heldout_episode_id_leakage_rejected(self):
        self.add_episode("validation", "heldout", episode(32, 100))["episode_id"] = "one"
        self.reject("split leakage")

    def test_train_heldout_numeric_leakage_despite_different_json_metadata(self):
        copied = episode()
        copied["instruction"] = {"different": "metadata does not make a new numeric episode"}
        self.add_episode("test", "renamed", copied)
        self.reject("duplicate numeric episode content / split leakage")

    def test_heldout_values_do_not_contribute_to_statistics(self):
        reference, _ = self.build()
        self.add_episode("validation", "heldout", episode(40, 1000))
        payload, report = self.build()
        for name in ("mean", "std", "q01", "q99"):
            self.assertEqual(payload[name], reference[name])
        self.assertEqual(payload["normalization_metadata"]["num_state_samples"], 32)
        self.assertEqual(report["split_frame_counts"]["validation"], 40)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["count"], 32)

    def test_checksum_mismatch_rejected_without_omitting_episode(self):
        (self.root / self.primary["path"]).write_bytes(json_bytes(episode(32, 1)))
        report = self.reject("checksum mismatch")
        self.assertFalse(report["files"][0]["checksum_matches"])

    def test_unlisted_json_rejected(self):
        (self.root / "unlisted.json").write_bytes(json_bytes(episode()))
        self.reject("manifest membership mismatch")

    def test_missing_or_incomplete_wrench_rejected(self):
        for value in (None, [], [[1] * 6] * 31):
            with self.subTest(value=value):
                recorded = episode(wrench=False)
                if value is not None:
                    recorded["proprios"]["wrench"] = value
                self.replace_episode(self.primary, recorded)
                report = self.reject("wrench")
                expected_missing = 32 if value is None else 32 - len(value)
                self.assertEqual(report["missing_wrench_observations_per_split"]["train"], expected_missing)

    def test_invalid_wrench_width_nonfinite_and_overflow_rejected(self):
        for value in ([1] * 5, [0, 0, 0, 0, 0, float("nan")], [float("inf")] * 6, [1e100] * 6, [True] * 6):
            with self.subTest(value=value):
                recorded = episode()
                recorded["proprios"]["wrench"][0] = value
                self.replace_episode(self.primary, recorded)
                self.reject("wrench")

    def test_zero_constant_tiny_and_float32_collapsed_force_quantiles_rejected(self):
        for transform in (lambda index: 0, lambda index: 3, lambda index: index * 1e-10,
                          lambda index: 1e10 + index * 1e-3):
            with self.subTest(transform=transform):
                recorded = episode()
                for index, value in enumerate(recorded["proprios"]["wrench"]):
                    value[0] = transform(index)
                self.replace_episode(self.primary, recorded)
                self.reject("q99|wrench index")

    def test_nonzero_constant_proprioception_is_not_fabricated_into_a_range(self):
        recorded = episode()
        recorded["proprios"]["gripper"] = [.5] * 32
        self.replace_episode(self.primary, recorded)
        self.reject("q99 must")

    def test_short_only_dataset_rejected_for_unobserved_horizons(self):
        self.replace_episode(self.primary, episode(29))
        report = self.reject("need an episode with >= 30 frames")
        self.assertEqual(report["action_sample_counts"][29], [0] * 60)

    def test_exactly_30_frames_yield_one_real_sample_at_last_horizon(self):
        self.replace_episode(self.primary, episode(30))
        payload, report = self.build()
        self.assertEqual(payload["mean"][29][0], 29)
        self.assertEqual(payload["std"][29][0], 0)
        self.assertEqual(report["action_sample_counts"][29], [1] * 7 + [0] * 53)

    def test_large_action_offsets_use_stable_population_variance(self):
        recorded = episode()
        values = [100000000 + index * 16 for index in range(32)]
        for action, value in zip(recorded["actions"]["semantic"], values):
            action[0] = value
        self.replace_episode(self.primary, recorded)
        payload, _ = self.build()
        self.assertEqual(payload["mean"][0][0], statistics.mean(values))
        self.assertAlmostEqual(payload["std"][0][0], statistics.pstdev(values), places=4)

    def test_constant_unused_wrench_is_permitted_for_force_free_statistics(self):
        recorded = episode()
        recorded["proprios"]["wrench"] = [[0] * 6 for _ in range(32)]
        self.replace_episode(self.primary, recorded)
        payload, report = self.build(use_wrench=False)
        self.assertEqual(payload["q99"][0][20:26], [0] * 6)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["std"], 0)

    def test_wrench_order_units_frame_and_provenance_are_mandatory(self):
        original = copy.deepcopy(self.manifest["wrench"])
        for name in original:
            with self.subTest(name=name):
                self.manifest["wrench"] = copy.deepcopy(original)
                del self.manifest["wrench"][name]
                self.reject("manifest.wrench")
        for name, value in (("order", ["Fz", "Fy", "Fx", "Tx", "Ty", "Tz"]),
                            ("force_unit", "kN"), ("torque_unit", "Nmm"), ("frame", "")):
            self.manifest["wrench"] = dict(original, **{name: value})
            self.reject("manifest.wrench")

    def test_explicit_all_split_membership_and_converter_provenance_required(self):
        del self.manifest["splits"]["test"]
        self.reject("explicitly declare")
        self.manifest["splits"]["test"] = []
        del self.manifest["provenance"]["converter_revision"]
        self.reject("converter_revision")

    def test_paths_may_not_escape_dataset_root(self):
        self.primary["path"] = "../escape.json"
        self.reject("relative and stay inside")

    def test_state_action_shapes_lengths_finiteness_and_gripper_contract(self):
        variations = [
            lambda value: value.update(num_frames=0),
            lambda value: value.update(num_frames=True),
            lambda value: value["proprios"].update(joint_pos=[[0] * 5] * 32),
            lambda value: value["proprios"].update(joint_vel=[[0] * 6] * 31),
            lambda value: value["proprios"].update(tcp_pos=[[float("nan")] * 3] * 32),
            lambda value: value["proprios"].update(gripper=[2] * 32),
            lambda value: value["actions"].update(semantic=[[0] * 6] * 32),
            lambda value: value["actions"].update(semantic=[[0] * 6 + [2]] * 32),
            lambda value: value["actions"].update(semantic=[[float("inf")] * 7] * 32),
        ]
        for variation in variations:
            with self.subTest(variation=variation):
                recorded = episode()
                variation(recorded)
                self.replace_episode(self.primary, recorded)
                self.reject("num_frames|values|observations|finite|gripper")

    def test_invalid_trajectory_type_is_included_as_training_reader_does(self):
        recorded = episode()
        recorded["trajectory_type"] = "invalid"
        self.replace_episode(self.primary, recorded)
        payload, report = self.build()
        self.assertEqual(report["training_trajectory_types"], {"invalid": 1})
        self.assertEqual(payload["mean"][0][0], 15.5)

    def test_contamination_warnings_do_not_filter_or_claim_contact(self):
        recorded = episode()
        recorded["proprios"]["wrench"][-1][0] = 100000
        self.replace_episode(self.primary, recorded)
        payload, report = self.build(force_limit_n=100, torque_limit_nm=100)
        flags = {flag["code"] for flag in report["force_quality"]["review_flags"]}
        self.assertIn("potential_outliers", flags)
        self.assertIn("physical_interpretation_unverified", flags)
        self.assertIn("calibration_declared_unverified", flags)
        self.assertIn("operator_review_limit_exceeded", flags)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["max"], 100000)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["iqr_outlier_examples"][0],
                         dict(episode_id="one", frame_index=31, value=100000))
        self.assertGreater(payload["q99"][0][20], 1000)
        self.assertEqual(report["force_quality"]["channels"]["Fx"]["count"], 32)
        self.assertEqual(report["errors"], [])

    def test_cli_and_existing_stage4_validator_accept_real_computed_fixture_statistics(self):
        before = (self.root / self.primary["path"]).read_bytes()
        result, output, report_path = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(output.read_text())
        resolved = resolve_doosan_data(dict(payload, use_wrench=True))
        self.assertEqual(resolved["q99"], payload["q99"])
        validation = subprocess.run([
            sys.executable, "-B", str(ROOT / "tools/validate_doosan_normalization.py"), str(output), "--use-wrench",
        ], capture_output=True, text=True)
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertEqual(json.loads(report_path.read_text())["status"], "passed")
        self.assertEqual((self.root / self.primary["path"]).read_bytes(), before)
        repeated, second, second_report = self.cli(suffix="_second")
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(output.read_bytes(), second.read_bytes())
        self.assertEqual(report_path.read_bytes(), second_report.read_bytes())

    def test_cli_failure_writes_preflight_but_no_statistics(self):
        self.replace_episode(self.primary, episode(wrench=False))
        result, output, report = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(report.read_text())["status"], "failed")

    def test_force_free_cli_and_existing_baseline_validator(self):
        self.replace_episode(self.primary, episode(wrench=False))
        del self.manifest["wrench"]
        result, output, report = self.cli(mode="--force-free")
        self.assertEqual(result.returncode, 0, result.stderr)
        validation = subprocess.run([
            sys.executable, "-B", str(ROOT / "tools/validate_doosan_normalization.py"), str(output),
        ], capture_output=True, text=True)
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertIs(json.loads(output.read_text())["normalization_metadata"]["use_wrench"], False)
        self.assertEqual(json.loads(report.read_text())["state_sample_counts"][0][20:], [0] * 40)

    def test_duplicate_json_keys_and_episode_declarations_rejected(self):
        recorded = episode()
        recorded["split"] = "test"
        self.replace_episode(self.primary, recorded)
        self.reject("disagrees with manifest")
        recorded = episode()
        recorded["wrench_metadata"] = dict(self.manifest["wrench"], frame="different-sensor-frame")
        self.replace_episode(self.primary, recorded)
        self.reject("wrench_metadata disagrees")
        path = self.root / self.primary["path"]
        path.write_text('{"num_frames": 32, "num_frames": 31}', encoding="utf-8")
        self.primary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reject("duplicate JSON key")

    def test_multiple_structural_errors_are_reported_without_exporting_partial_data(self):
        self.replace_episode(self.primary, episode(wrench=False))
        self.add_episode("train", "other_bad", episode(31, 3, wrench=False))
        report = self.reject("missing proprios.wrench")
        self.assertEqual(len(report["errors"]), 2)
        self.assertEqual(report["missing_wrench_observations_per_split"]["train"], 63)

    def test_undeclared_present_wrench_is_not_given_an_assumed_frame(self):
        del self.manifest["wrench"]
        self.reject("recorded wrench requires", use_wrench=False)

    def test_non_boolean_modes_and_invalid_review_limits_rejected(self):
        for value in ("false", 1, None):
            with self.subTest(value=value):
                self.reject("explicit boolean", use_wrench=value)
        for value in (-1, 0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.reject("positive and finite", force_limit_n=value)

    def test_cli_refuses_existing_outputs(self):
        existing = self.directory / "statistics.json"
        existing.write_text("preserve-existing-result", encoding="utf-8")
        result, _, report = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(existing.read_text(), "preserve-existing-result")
        self.assertFalse(report.exists())

    def test_cli_refuses_outputs_inside_episode_root(self):
        result, output, report = self.cli("--output", str(self.root / "statistics.json"))
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())
        self.assertFalse((self.root / "statistics.json").exists())

    def test_adapter_layout_constants_match_exporter_mapping(self):
        from tools.build_doosan_normalization import STATE_FIELDS
        source = ast.parse((ROOT / "mibot/utils/doosan.py").read_text())
        assignments = {node.targets[0].id: ast.unparse(node.value) for node in source.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        for name, expected in (("JOINT_POS_SLICE", "slice(0, 6)"), ("TCP_POS_SLICE", "slice(8, 11)"),
                               ("TCP_ROTVEC_SLICE", "slice(11, 14)"), ("JOINT_VEL_SLICE", "slice(14, 20)"),
                               ("WRENCH_SLICE", "slice(20, 26)"), ("DOOSAN_ACTION_SLICE", "slice(0, 7)")):
            self.assertEqual(assignments[name], expected)
        self.assertEqual(STATE_FIELDS, (("joint_pos", 6, 0), ("tcp_pos", 3, 8),
                                        ("tcp_rotvec", 3, 11), ("joint_vel", 6, 14)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
