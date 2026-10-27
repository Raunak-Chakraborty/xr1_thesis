# Standalone Doosan normalization exporter (Stage 4.1)

`tools/build_doosan_normalization.py` computes actual statistics from an explicit,
checksummed training split. It is an interim implementation inside the force-state
worktree; it does not modify the authoritative converter or any XR-1 model,
dataset, datamodule, or generic training code. Python's standard library is the
only dependency: **no NumPy, PyTorch, weights, CUDA, or model imports** are needed.

## Existing converted episode format

Episodes are the JSON objects already consumed by `DoosanJsonDataset`, unchanged:

| Field | Required values |
| --- | --- |
| `num_frames` | Positive integer N |
| `proprios.joint_pos` | N flat vectors of width 6 |
| `proprios.gripper` | N measured scalars in [0,1]; singleton lists also accepted |
| `proprios.tcp_pos` | N flat vectors of width 3 |
| `proprios.tcp_rotvec` | N flat vectors of width 3 |
| `proprios.joint_vel` | N flat vectors of width 6 |
| `proprios.wrench` | N flat vectors `[Fx,Fy,Fz,Tx,Ty,Tz]`; required in force mode |
| `actions.semantic` | N flat vectors of width 7, already computed by conversion |

The seven action values are translation delta, relative rotation vector, and
absolute normalized gripper target. The exporter never differences poses or
actions again. It accepts extra prompt/image/instruction fields without changing
them, but does not decode videos or validate prompt availability. All lengths
must equal N; an N-1 action sequence with N states is rejected rather than
silently relabeled/padded. Empty or invalid-structure episodes are not skipped.

`trajectory_type` may be success, ongoing, or invalid; all three contribute
exactly as the existing dataset's temporal mask does. An unknown type fails.
If optional episode-level `episode_id` or `split` fields exist, they must agree
with the manifest. Optional `wrench_metadata`, when present, must exactly match
the manifest's wrench declaration. No such additional episode fields are required.

State packing mirrors the existing adapter, without importing its ML dependencies:
joint position 0:6, zero at 6, gripper at 7, TCP position 8:11, TCP rotation
11:14, joint velocity 14:20, wrench 20:26, zeros thereafter. Baseline mode has
19 active semantic dimensions and zero wrench; force mode has 25. Action storage
is always 60D, with only 0:7 active and horizon 30. Source-layout regression tests
tie the standalone packing offsets to the unchanged adapter.

## Explicit split manifest

Recommended layout (manifest and output files are outside the episode root):

```text
dataset/
  splits.json
  episodes/
    train/episode_0001.json
    validation/episode_0200.json
    test/episode_0300.json
  videos/...
project/reports/stage4_1/<fresh-run>/
```

The episode root must contain only the partition's episode JSON files, including
held-out episodes. Nested episode directories are supported. Any unlisted JSON
file under that root is an error, as it would also be picked up by the dataset's
recursive JSON loader. The exporter consumes explicit members only, never an
implicitly discovered training split. Empty validation/test lists must be declared
explicitly and produce a limited-coverage warning, not an assertion that no
external held-out dataset exists.

Manifest schema (replace every placeholder with actual verified input information):

```json
{
  "schema": "doosan_split_v1",
  "dataset_id": "<exact dataset and training-split identifier>",
  "root": "episodes",
  "provenance": {
    "converter_revision": "<actual converter revision>",
    "dataset_revision": "<actual dataset manifest or revision>",
    "action_semantics": "precomputed_delta_tcp_pos_rotvec_absolute_gripper"
  },
  "wrench": {
    "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
    "frame": "<actual frame; never guess sensor/base/TCP>",
    "frame_provenance": "<source or transform-chain record establishing the frame>",
    "force_unit": "N",
    "torque_unit": "Nm",
    "compensation": "<actual bias/gravity/tool compensation, or explicitly none>",
    "calibration_reference": "<actual record, or explicit unverified status/reference>",
    "calibration_verified": false
  },
  "splits": {
    "train": [
      {"episode_id": "episode_0001", "path": "train/episode_0001.json", "sha256": "<actual 64-character lowercase SHA256>"}
    ],
    "validation": [
      {"episode_id": "episode_0200", "path": "validation/episode_0200.json", "sha256": "<actual SHA256>"}
    ],
    "test": [
      {"episode_id": "episode_0300", "path": "test/episode_0300.json", "sha256": "<actual SHA256>"}
    ]
  }
}
```

`root` resolves relative to the manifest; member paths resolve relative to root
and cannot escape it, including through symlinks. Every member requires its actual
byte checksum, obtainable with `sha256sum`. IDs, canonical paths, raw JSON hashes,
and float32 numeric-content fingerprints must be disjoint across all episodes.
Numeric fingerprints detect copied observations/actions hidden behind different
instructions, video paths, or other JSON metadata. This is deliberately conservative:
even independently recorded identical numeric streams require review rather than
an automatic duplicate override. Near-duplicates cannot be proven absent.

Every listed held-out episode is integrity-checked, but its values never enter
normalization statistics or training force summaries. Episode JSON checksums and
the manifest checksum are recorded. Video bytes are not hashed or decoded by
this tool; verify media checksums separately using the authoritative dataset
archive manifest. Declared frame, ordering, units, calibration, and compensation
are audited for consistency, not physically established from numeric data.

In force-free mode, wrench may be absent and the wrench declaration may then
be omitted. Present optional wrench is still validated, requires its real
declaration, and is summarized for review, but never enters state quantiles.

## Exact statistics and padding policy

Every training frame is one state observation and one action-window start. For
an episode with N frames and horizon position h in 0..29:

```text
window starting at s contributes actions.semantic[s+h] iff s+h < N
count at horizon h = sum_over_training_episodes(max(N-h, 0))
values for one episode at horizon h = semantic[h], ..., semantic[N-1]
```

The loader repeats the last action to store a complete 30-step chunk, then masks
the artificial tail. Those repeated padded rows contribute **zero** samples to
statistics. Short episodes still contribute at their observable horizons; they
are not dropped. For N=3, h=0 sees actions 0,1,2; h=1 sees 1,2; h=2 sees 2;
h>=3 sees nothing from that episode. Counts differ by horizon, not by one of the
seven semantic action channels. Padded action channels 7:60 have mean/std/count
zero. At least one training episode must have N>=30: otherwise some horizons
have no measurements and the exporter refuses to invent their statistics.

Population standard deviation (`ddof=0`) and mean are accumulated using stable
Welford moments over reversed episode suffixes, then deterministically merged.
This avoids expanding N windows into N*30 stored action arrays. States enter
quantiles once each; semantic state counts are N and padding counts are zero.
This is a corpus statistic over all frame starts, not a statistic weighted by a
particular `max_steps`, distributed sampler, repeated/truncated sample list,
packed-token rejection, or random async-prefix choice in one future training run.

Input numeric values are rounded to float32, matching the adapter. q01/q99 use
exact sorted samples and linear interpolation at `(sample_count-1)*p`, followed
by the existing resolver's float32 output conversion. No random sampling,
clipping, filtering, centering, debiasing, or synthetic variation is applied.
The force quantiles must have span >1e-6 after conversion. Constant/nonzero
proprioceptive quantiles also fail the existing Stage 4 contract; all-zero
constant proprioception is permitted as in that resolver and appears in
`degenerate_state_quantile_indices`. Do not add fake ranges to work around a
failure. Review/re-export data, or obtain approval for a different contract.

The statistics payload contains mean/std `(30,60)`, q01/q99 `(1,60)`, and
`normalization_metadata` accepted by the unchanged `resolve_doosan_data()`.
Metadata includes doosan_state_v1, source=dataset, split=train, actual state
count, declared dataset ID/provenance/frame/units, manifest SHA256, and the
quantile/window/std definitions. Byte-for-byte results are deterministic for
unchanged manifest and episode bytes on the same Python arithmetic platform.
No clock, random seed, or output pathname is included in generated payloads.
Manifest-byte changes intentionally change the provenance digest.

## CLI and reports

Workstation/laptop CPU statistics generation (not training):

```bash
export PROJECT=/home/horus/robotics_thesis/xr1_force_exp
export REPO="$PROJECT/worktrees/xr1_force_state"
export MANIFEST=/absolute/path/to/dataset/splits.json
export RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
export OUTPUT_DIR="$PROJECT/reports/stage4_1/$RUN_ID"
mkdir -p "$OUTPUT_DIR"
cd "$REPO"
python3 -B xr1/tools/build_doosan_normalization.py \
  --manifest "$MANIFEST" --use-wrench \
  --output "$OUTPUT_DIR/doosan_force_statistics.json" \
  --report "$OUTPUT_DIR/doosan_force_preflight.json"
python3 -B xr1/tools/validate_doosan_normalization.py \
  "$OUTPUT_DIR/doosan_force_statistics.json" --use-wrench
```

For a baseline export, replace `--use-wrench` with `--force-free`, choose new
output/report filenames, and omit `--use-wrench` from the validator. Both modes
may use the same manifest when its optional wrench is valid and declared.

Both output paths must be new, distinct files with existing parent directories,
outside the episode root. Existing files are never overwritten. On structural
or normalization failure, exit status is 2 and only a failed preflight report is
written; no partial statistics are exported. Argument/output-location errors
fail before creating files. Successful preflight produces both files and exit 0.
The report records split/valid-frame counts, per-file checksums/fingerprints,
missing wrench counts, `(1,60)` state counts, `(30,60)` action counts, excluded
padding counts, degeneration indices, force summaries, and separate review flags.
Early structural failures may leave count fields partial; only a passed report
claims a complete integrity check. Structural episode errors are accumulated
where possible, and no success payload is computed from a silently reduced set.

Optional operator-supplied `--force-limit-n` and `--torque-limit-nm` annotate
force/torque vector-norm exceedances. They are review thresholds, not robot safety
limits, and do not filter samples or alter training statistics. No physical
threshold is guessed. Every force channel reports min/max/mean/std, median,
q01/q25/q75/q99, 3-IQR-fence outlier counts, outside-quantile counts, maximum
within-episode step, and adjacent equal-value counts. Up to 20 outlier examples
per channel include episode ID, zero-based frame index, and value; the report
explicitly counts any additional examples not listed. All measurements still
contribute to statistics. Force/torque norms are also
summarized. Quantile clipping in later model preprocessing can hide extreme
values: review these reports before accepting the statistics.

## Reviewing rogue forces versus contact

Treat quality flags separately from structural corruption. A finite outlier or
nonzero offset can be genuine contact, cable load, gravity, tool preload, bias,
drift, acceleration, or synchronization error; numerical summaries cannot decide.
Calibration status in the manifest is a declaration, never proof.

Review recorded force traces against synchronized RGB, TCP pose/velocity, robot
motion and task phases. Compare known free-space and contact segments; inspect
cable routing/tool orientation changes, bias resets, saturation/flatlines,
isolated spikes, timestamp alignment, and repeated episode-specific offsets.
Check the sensor's actual frame transform, force/torque units, gravity/tool
compensation, and calibration record with the experiment owner. A cable-tension
signal correlated with robot pose alone must not be labeled contact automatically.
Preserve raw data and flags. Any exclusion, compensation, resynchronization, or
relabeling should be a separately reviewed, versioned conversion change with new
checksums and split provenance; this exporter never modifies demonstrations.

## Verification gates before workstation training

Dependency-free tests, runnable now:

```bash
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/xr1" \
  python3 -B -m unittest discover -s xr1/tests -p test_doosan_normalization_export.py -v
bash xr1/scripts/test_force_state_offline.sh
```

These exercise real standard-library floating-point calculations, independently
check horizon moments using `statistics.mean/pstdev`, and run both exported modes
through the existing validator. There are no numeric mocks or NumPy requirements.

On the lab workstation, also run `bash xr1/scripts/test_doosan_offline.sh`; require
all existing Stage 4 tensor tests to run without dependency skips. Preserve the
environment requirements and checkpoint/one-batch gates in
`doosan_force_state_stage4.md`. The old unrelated lowercase import in
`test_training_logic.py` remains outside these unittest commands and unchanged.
No checkpoint loading or training is started by this exporter or its tests.

For actual fine-tuning, ensure `train_datasets.paths` contains exactly manifest
training members, never the all-split root. A dedicated `episodes/train` tree can
be supplied as a directory; otherwise supply the explicit train file list from
the manifest. Set `data=doosan_force_state` and its `normalization_path` to the
generated statistics. Reconfirm file checksums and membership if data changes.

## Upstream integration and remaining evidence

Future authorized converter work should emit the same per-frame numeric JSON,
stable episode identities, complete train/validation/test manifest with actual
JSON/media checksums, actual converter/dataset revisions, and explicit sensor
frame/order/units/compensation/calibration provenance. Integrate these masked
suffix moments and exact quantile rules into conversion, retain raw/processed
identities, and add fixture parity tests against this standalone exporter and
the actual XR-1 adapter. Add checksummed video/synchronization integrity tests
there rather than importing model dependencies into this tool. No upstream
converter repository was accessed or modified here.

Only synthetic converted episodes have been tested in this sandbox. The real
lab dataset, physical sensor semantics/calibration, images/videos, full GPU stack,
checkpoint compatibility, and learning behavior remain unverified. The existing
Git administrative directory is unavailable here, so the requested state branch
and actual HEAD must be verified on the workstation before patch integration.
