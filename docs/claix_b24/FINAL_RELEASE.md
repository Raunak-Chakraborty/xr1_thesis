# Batch-24 final engineering review

This is the earlier review snapshot. [FINAL_CORRECTIONS.md](FINAL_CORRECTIONS.md)
supersedes its targeted H100, persistent-path, quota and final-source instructions.

Reviewed base: `4339a0fcd7a9c18c81356d371360bbf629656c0c`, isolated worktree
`b24_final_release`. No commit, push, GPU training, Slurm submission, original
worktree write, dataset change, or earlier experiment change was performed.
This is an engineering preparation, **not a production release approval**.

Job **5357168** remains historical H100 evidence from an earlier source revision:
real batch-24 training, exact restart equivalence, 251 sustained steps, compact
publication, SIGUSR1 stop and validation inference smoke. Its measured
**14.44 examples/second** is historical, not a measurement of this source.
Do not change its original job ID/source commit, relabel its source hash, or
resume its checkpoint with this source.

## Checkpoint retention

`rotate_checkpoints` enumerates small receipts, filters namespace/configuration,
and keeps a heap of at most eight ordinary candidates, six milestones and one
safe-stop pin. Receipts for deleted payloads remain immutable audit evidence.
It tracks the newest committed receipt independently of usable payloads, so a
stale latest pointer or damaged latest payload cannot authorize deletion.

It verifies SHA256, deserializes **once**, and runs the full existing compact
validator for each selected candidate. The temporary checkpoint is released
before considering the next candidate; no payload trust cache persists between
calls or resumes. File size only filters candidates. It never proves integrity.

Before any unlink it verifies the newest checkpoint, most recent safe-stop pin
and newest two usable recovery checkpoints. The very first publication is the
only allowed one-checkpoint bootstrap; if earlier owned receipts exist and two
usable checkpoints cannot be established, rotation fails closed. Corruption of
a recovery candidate causes fallback to an older usable checkpoint. Invalid or
corrupt candidates stay on disk. Milestones 2000, 5000, 10000, 20000, 30000,
40000 stay pinned. A superseded safe-stop payload becomes eligible for full
verification and reclamation, while its receipt remains.

There are at most **eight distinct verification attempts per call**, including
failed attempts. Insufficient recovery within that budget raises before any
deletion. A deletion backlog is reclaimed incrementally. Filesystem failures
propagate; partial cleanup can leave extra payloads, but protected recovery and
safe-stop payloads are never cleanup targets. All writers must honor the
existing exclusive run-root lock; concurrent mutation outside that protocol is
unsupported. An unreadable directory/pointer fails closed.

Initial publication still fsyncs, fully reads back/validates, computes SHA256,
uses atomic no-clobber payload publication, then publishes receipt and pointer.
Receipt publication now also uses no-clobber hard linking. Dangling symlinks
count as existing destinations. Existing symlink payloads, receipts and pointers
are refused. Resume still performs SHA256 and full validation against the
expected replay/configuration contract. The redundant preliminary load in
`VerifiedCheckpointIO` was removed; the entrypoint/module checks remain.
The compact payload schema and scientific validation functions are unchanged.

For N receipt files, payload size P, and verification budget B=8:

* Receipt enumeration is O(N log B), with O(B + 7) retained receipt rows.
* Payload reads and validation are O(B P) worst case; healthy steady rotation
  uses three candidates regardless of historical milestone count.
* Host memory holds one loaded candidate at a time, O(P), plus bounded metadata
  and the validator's temporary tensors. It never accumulates loaded payloads.

Measured **rotation-only** call counts in CPU tests:

| Scenario | torch.load | checkpoint SHA256 reads | candidate verification attempts |
|---|---:|---:|---:|
| Two recovery + one deletion, zero milestones | 3 | 3 | 3 |
| Same, all six milestones + 200 malformed receipts | 3 | 3 | 3 |
| Separate safe-stop pin, healthy normal rotation | 4 | 4 | 4 |
| Supersede old safe-stop pointer | 4 | 4 | 4 |
| Corrupt previous recovery, older fallback + deletion | 3 | 4 | 4 |
| Large healthy cleanup backlog | 8 | 8 | 8 |
| Recovery cannot be established within budget | at most 8 | at most 8 | 8; no deletions |

Publication adds one full readback load and one SHA256 pass. Thus a healthy
periodic publication + rotation makes four loads and four SHA256 reads total,
plus one write. Each payload is loaded at most once **within rotation**;
publication's mandatory initial readback is deliberately retained. For a 1.7 GB
payload this is approximately **13.6 GB logical reads + 1.7 GB write** per
periodic checkpoint; a separate safe-stop pin raises reads to 17 GB. These are
logical byte passes, not necessarily physical device traffic: OS cache, SHA256,
validation and shared filesystem bandwidth determine elapsed overhead. Final
fit-end verification adds one final SHA256/load pair, not one per interval.

`checkpoint.duration_seconds` now includes rotation; `rotation.verified` and
`rotation.deleted` are persistent telemetry fields. Measure these on CLAIX;
no final-source checkpoint latency or throughput is claimed here. There are
193 regular 250-step publications over 48284 steps, plus final/safe-stop writes.
The historical rate corresponds to roughly 416 seconds per 6000 examples; it
is only a planning reference for interpreting measured checkpoint duration.

## Metrics and monitoring

`trainer.log_every_n_steps=1` and all loss/training mathematics are unchanged.
The CPU interrupted/resumed run reconstructs all steps 1..8 from two separate
CSV logger versions and append-only telemetry. It checks `train/loss`,
`train/loss_freq`, `train/loss_l1`, `train/loss_mse`, `train/loss_score`,
`train/token` and `lr`, and compares final FP32 weights, AdamW moments/step and
scheduler state exactly with uninterrupted training. The synthetic component
metrics exercise logging without changing the loss passed to backward.

Lightning CSV `step` is zero-based; completed optimizer step is `step + 1` in
this approved accumulation=1, automatic optimization path. Telemetry `step` is
the completed optimizer step. Keep every `metrics/version_*/metrics.csv` and
`telemetry.jsonl`; do not copy a resumed logger onto an older version or discard
pre-resume rows. Training telemetry also carries learning rate, consumed
examples, token/view counts, pipeline wait and GPU memory.

The existing controller `NvidiaSmiMonitor` uses a 3-second subprocess timeout,
reaps timed-out queries, writes timestamped persistent samples with device
UUIDs, and terminates its thread on close. Missing executable, driver failure,
timeout, empty output or log failure disable only monitoring. Tests include a
real local executable that hangs: it is killed/reaped, and the monitor thread
terminates. Signal handlers remain unchanged. Production monitoring remains on
by default, with its existing environment opt-out.

`--qualification-gpu-monitor` opts into **the same implementation** only in
H100 qualification. CPU qualification rejects that flag and does not start the
monitor, even if the production environment opt-in variable is present. No
network service or package download is used.

## Slurm and approved environment

`scripts/claix_b24_production.sbatch` requests account `thes2452`, partition
`c23g`, one task, **`--gres=gpu:1`**, 12 CPUs, 96G RAM, exactly `23:45:00` and
`--signal=B:USR1@1800`. The unchanged training entrypoint still requires exactly
one visible GPU whose model name contains H100. Production workers are fixed
to four; qualification worker comparisons remain separate.

Both prepared production and targeted scripts source
`scripts/claix_b24_environment.sh`, which runs the established
`module load Python/3.11.5` and sources
`$HOME/robotics_thesis/configs/xr1_hf_offline.env`. Set `XR1_PYTHON` to the
existing approved environment's executable; no environment is created.
The known existing example is
`/home/zk443626/robotics_thesis/envs/xr1_fp32/bin/python`.
Set `XR1_WORKTREE`, `XR1_RUN_ROOT`, and the existing approved
`XR1_DOOSAN_ZIP_CACHE_ROOT`. Production additionally requires
`XR1_RELEASE_RECORD` referring to a reviewed **read-only** record. Submit from
an existing persistent submission directory so relative Slurm output/error
files persist; production checks its submission directory is under `/hpcwork`
or `/work`. Launcher stdout/stderr, GPU samples and provenance additionally go
beneath the persistent run root. No job was submitted.

The four previously staged path overrides were absent in the new worktree.
For local validation only, the existing approved
`.claix_b24_reference/configs/hydra_overrides.json` was copied from
`b24_metrics_release`; no path values changed. Stage this existing reference
file on CLAIX or pass `--overrides` explicitly. Its exact contents are bound in
the production plan. It is an ignored local input, not a new training recipe.

## Read-only storage/quota preflight

Run separately at the actual existing persistent parent, before a fresh run:

```bash
"$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_storage_preflight.py" \
  --run-root "$XR1_RUN_ROOT" --filesystem-source '<reviewed findmnt SOURCE>' \
  > /existing/persistent/review/storage-initial.json
```

This does not create the run root or write/delete probe files. `findmnt` must
match the operator's confirmed persistent source; tmpfs/overlay/ramfs and
temporary run paths fail. `statvfs.f_bavail` must show at least **30 GiB** usable
headroom and `f_favail` at least **10000** inodes. The conservative peak budget
is six milestones + two recovery + one safe-stop + one new 2 GiB compact
publication (20 GiB), plus 4 GiB for logs and temporary metadata; 30 GiB leaves
additional headroom. Existing retained/orphan/unrelated files already consume
space and must be included in operator review. Resume should also retain at
least this free headroom; nothing is deleted by preflight.

It queries `lfs quota -u` and `quota -s -u`, with a ten-second bound per query.
If the Lustre project ID is known, add `--project-id ID` to query that quota;
GID is **not** assumed to be a project ID. Raw results are retained in the
report. Automatic interpretation across quota systems is intentionally not
trusted: quota remains **UNVERIFIED**, exit 1, until an explicit operator
attestation establishes both applicable user and project limits (including
soft/hard limits, grace and inode quotas), or establishes non-applicability.
There is no automatic quota PASS and no CLAIX quota claim from this local run.

After actual site quota review, create a separate attestation JSON with:

```json
{
  "reviewed_by": "REAL_OPERATOR",
  "timestamp": "CURRENT_UTC_ISO_TIMESTAMP",
  "run_root": "/EXACT/PERSISTENT/RUN/ROOT",
  "filesystem_source": "EXACT_SOURCE_FROM_REPORT",
  "uid": 0,
  "user_and_project_reviewed": true,
  "usable_bytes": 0,
  "usable_inodes": 0,
  "quota_query_sha256": "EXACT_QUOTA_QUERY_SHA256_FROM_REPORT"
}
```

The UID, usable limits and timestamp above are placeholders, not approval.
Use effective remaining capacity after **both** applicable quota limits and
retention/log/publication reservations. Repeat preflight with the same arguments
and `--quota-review /absolute/review.json`. The attestation must be no older
than one hour and match run root, filesystem, UID and current raw quota-query
hash. Changed quota output requires fresh review. Invalid/unknown quota,
filesystem or inode status exits nonzero. Preserve and hash the resulting
report as release evidence; repeat close to submission.

## Release record preparation and evidence

Save the exact `production_plan` object from a production launcher dry-run to
`plan.json`, using the real proposed run root and four approved path overrides.
Then prepare an exclusive-create draft:

```bash
"$XR1_PYTHON" scripts/claix_b24_release.py --plan /absolute/plan.json \
  --output /absolute/release.DRAFT.json
```

The builder binds the current source-tree SHA256, checkout identity, exact plan
and run root. It sets **all gates UNVERIFIED**, with blank reviewers. It never
marks a gate PASS. The sample draft under `final_evidence` uses a REVIEW_ONLY
root; replace it by running the builder with the actual proposed plan.

There are exactly nine gates:

1. `frozen_splits`
2. `tokens_24_examples_48_views`
3. `workers_0_4_8`
4. `h100_b24_memory`
5. `end_to_end_throughput`
6. `real_compact_resume`
7. `safe_stop_and_storage`
8. `held_out_validation`
9. `lr_and_horizon_approval`

Each gate row points to a SHA256-bound JSON evidence envelope with this shape:

```json
{
  "gate": "EXACT_GATE_NAME",
  "kind": "historical_h100 OR current_h100 OR cpu OR operator",
  "source_commit": "ORIGINAL_SOURCE_COMMIT",
  "source_tree_sha256": "ORIGINAL_SOURCE_TREE_SHA256",
  "job_id": "ORIGINAL_H100_JOB_ID_IF_APPLICABLE",
  "artifacts": [
    {"path": "/absolute/original-report.json", "sha256": "REAL_FILE_SHA256"},
    {"path": "/absolute/original-provenance.json", "sha256": "REAL_FILE_SHA256"}
  ],
  "provenance_bindings": {
    "source_commit": {"path": "/absolute/original-provenance.json", "key": "git_commit"},
    "source_tree_sha256": {"path": "/absolute/original-report.json", "key": "source_tree_sha256"},
    "job_id": {"path": "/absolute/original-report.json", "key": "job_id"}
  }
}
```

Dot-separated keys support nested original provenance (for example
`slurm.SLURM_JOB_ID`). Bind fields to their actual original JSON locations;
do not rewrite historical results to fit an envelope. Every provenance binding
must reference an artifact whose actual bytes hash correctly. CPU/operator
kinds do not require a job ID, but still require original source provenance.
Keep separate envelopes for old H100 and new CPU evidence, and include their
original artifacts/envelopes when several sources support one gate.

Reviewers must explicitly fill each gate's status, applicability reviewer and
rationale. Missing/mismatched files, provenance, hashes, extra/missing gates,
unapproved scientific parameters, changed source/plan or missing reviewers are
rejected. Hardware-sensitive gates `h100_b24_memory`, `end_to_end_throughput`,
`real_compact_resume`, and `safe_stop_and_storage` require a current-H100 envelope
bound to this exact final source; supporting CPU/storage/history artifacts can
be included as additional hashed artifacts. A validation inference smoke alone
does not establish held-out scientific validation. Throughput acceptance and
LR/horizon approval remain real operator/scientific decisions.

After review, copy/finalize the record as a read-only artifact (`chmod a-w`),
then check:

```bash
"$XR1_PYTHON" scripts/claix_b24_release.py --plan /absolute/plan.json \
  --check /absolute/release.REVIEWED.json
```

The existing production launch gate still runs, with additional evidence and
review checks. The launcher binds the validated record bytes, checks they have
not changed before child launch, and preserves the full record in provenance.
There is no bypass option, submission, or automatic scientific approval.

## Optional single-allocation final H100 check

Prepared, **not submitted**: `scripts/claix_b24_targeted_h100.sbatch` and its
controller. One H100 via generic GPU request, 12 CPUs, 96G, maximum 45 minutes.
Use a new persistent root and the approved environment/cache. A local dry-run:

```bash
"$XR1_PYTHON" scripts/claix_b24_targeted_h100.py --run-root /NEW/PERSISTENT/ROOT
```

Inside its future allocation it runs at most 260 total training steps, sends
USR1 to the launcher after telemetry reaches 250, verifies a safe completed-step
stop before 260, then resumes the same source/configuration to 260. It checks
all CSV losses and telemetry steps across both versions, both monitor logs and
UUID samples, source provenance, retained receipt checksums, latest/safe-stop
pointers, actual rotation deletion and checkpoint overhead events. Failure or
partial execution never creates a release PASS. Its targeted report is narrow
engineering evidence, **not nine automatically passed gates**.

It checks the exact resumed **step sequence**. It does not repeat an
uninterrupted H100 reference run or claim exact H100 tensor equivalence; the
CPU interrupted/uninterrupted comparison supplies that focused engineering
check. It does not repeat workers 0/4/8, frozen split comparisons, or validation
inference. A 45-minute request is expected to be ample based on historical
throughput, but final-source timing is not yet measured.

**One final H100 allocation is recommended and necessary for the prepared
checker to accept final-source hardware-sensitive release evidence.** It is
not needed to test the retention algorithm's operation counts. A CPU-only
storage-I/O alternative on the target filesystem is prepared:

```bash
CUDA_VISIBLE_DEVICES='' "$XR1_PYTHON" scripts/claix_b24_retention_benchmark.py \
  --run-root /NEW/DISPOSABLE/PERSISTENT/BENCHMARK/ROOT --payload-mib 1622
```

This creates synthetic full-size compact payloads in a new root, exercises all
six milestones and periodic deletion, measures publication/rotation durations,
and asserts three loads, three SHA256 reads and three verification operations.
It uses no dataset/backbone/GPU/network. It requires roughly 15 GB of synthetic
writes before rotation, so check its storage budget first. It never attaches to
an existing root. `--payload-mib 8` is only a cheap local smoke test. Cache is
not flushed; reports explicitly describe warm OS cache. This benchmark can
validate retention I/O on CLAIX without GPU allocation, but cannot qualify
H100 memory, throughput, mixed-precision training or production signals there.

## Validation and source implications

See `final_evidence` for actual CPU logs, syntax/scientific-source checks,
production and targeted dry-runs, a negative local storage preflight,
synthetic I/O smoke results, source hashes and an UNVERIFIED draft record.
The initial baseline's missing staged overrides were a local prerequisite
failure, not a claim that its 98 tests passed; the final existing-suite rerun
is recorded separately. Tests use real CPU Lightning/AdamW/checkpoint I/O with
explicit backbone/registry mocks from the existing offline harness; no hardware
results are inferred from them.

`engineering_source_digest` is unchanged. Its input includes Python/config and
`claix_b24*` scripts, including the new tools/tests, so this work necessarily
changes both the source-tree SHA256 and compact resume configuration fingerprint.
The Git HEAD remains the reviewed base with uncommitted engineering changes.
Record both HEAD and exact tree digest; a commit ID alone does not identify
this worktree. Any later included-source edit invalidates the prepared source
binding and requires refreshed evidence/review. Old 5357168 checkpoints remain
preserved evidence and cannot be resumed under the new fingerprint.

All scientific configuration files, model/loss code, data/replay semantics,
AdamW/scheduler code and training entrypoint are byte-for-byte unchanged from
the reviewed base. In particular: LR 2e-5 / 5e-6 / 5e-7, 500 warmup steps,
48284-step horizon, batch 24, workers 4, last_n=1, two cameras, BF16 compute,
FP32 selected trainable weights and 250-step checkpoint interval remain intact.

No production gate is certified PASS by this review. Remaining work is actual
persistent storage/quota review, final-source H100 evidence, applicable held-out
validation and scientific LR/horizon approval, nine explicit gate decisions,
and final review of the exact production record before any submission.

## Changed files

Runtime/launch changes:

* `xr1/mibot/utils/claix_checkpoint.py`: bounded receipt selection and verification,
  no duplicate candidate loads, immutable receipts, regular-file/symlink checks
  and a 16 KiB receipt/pointer read limit.
* `xr1/mibot/utils/claix_runtime.py`: remove redundant loads, bind rotation to the
  expected configuration, include rotation counts/time in checkpoint telemetry.
* `scripts/claix_b24_launch.py`: qualification monitor flag, four-worker production
  enforcement, reviewed evidence checks and immutable release binding.
* `scripts/claix_b24_production.sbatch`: proven generic GRES, approved bootstrap,
  persistent Slurm submission directory check.

New preparation tools:

* `scripts/claix_b24_environment.sh`
* `scripts/claix_b24_storage_preflight.py`
* `scripts/claix_b24_release.py`
* `scripts/claix_b24_targeted_h100.py`
* `scripts/claix_b24_targeted_h100.sbatch`
* `scripts/claix_b24_retention_benchmark.py`

Tests/review artifacts:

* `xr1/tests/test_claix_final_release.py`: focused retention operation counts,
  corruption/pointer/memory bounds, publication/resume checks, write failures,
  large/FIFO/symlink metadata, release provenance/approval, quota, real monitor
  timeout/reaping, Slurm/bootstrap and synthetic targeted verifier checks.
* `xr1/tests/test_claix_launch.py`: use the stricter reviewed release fixture in
  the existing nine-gate/checksum test.
* `xr1/tests/test_claix_metrics_release.py`: update the mocked release checker
  return value; the interrupted/resumed metric and optimizer comparisons remain.
* `docs/claix_b24/release_record.template.json`: review/provenance/science fields;
  every gate remains UNVERIFIED.
* `docs/claix_b24/FINAL_RELEASE.md`: this review, commands and limitations.
* `docs/claix_b24/final_evidence/*`: actual local evidence, exact manifest and
  unapproved draft. No historical evidence file was modified.

The local ignored staged overrides file described above was copied unchanged.

Final local validation results:

| Check actually run | Result |
|---|---|
| Existing CPU suite, isolated runner | 98/98 PASS, 16.718 seconds |
| Full offline CPU harness, including focused additions | 119/119 PASS, 23.181 seconds |
| Python AST syntax checks | 72 files PASS |
| Bash syntax checks | All five batch scripts plus the environment helper PASS |
| Scientific/model/data/replay/entrypoint diff against reviewed base | Unchanged |
| Production launcher dry-run | PASS, no child/allocation |
| Targeted H100 controller dry-run | PASS, no child/allocation |
| Unverified draft release check | Expected rejection |
| Local read-only storage check with deliberately wrong CLAIX source | Expected FAIL; quota UNVERIFIED; no run root created |
| Synthetic 8 MiB CPU retention benchmark, all six milestones | 3 loads / 3 SHA256 reads / 3 candidate verifications; one payload reclaimed |
| Real local monitor timeout test | Query killed/reaped; thread ended; training-independent error |
| Whitespace/diff check | PASS |

Final engineering source-tree SHA256:
`a6154eeef8bff0668d8c8c9e7d5693f5018d2805b1ddc0b529dd905a8af53280`.
The synthetic local benchmark measured approximately 0.111 seconds for its
small rotation; this is explicitly **not** a 1.7 GB or CLAIX timing result.
