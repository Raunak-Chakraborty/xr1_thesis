# Final targeted correctness corrections

Reviewed 2026-11-01 in `b24_final_release`, HEAD
`4339a0fcd7a9c18c81356d371360bbf629656c0c`. Existing uncommitted work was
preserved. **Final-source H100 qualification remains UNVERIFIED.** No GPU
training, Slurm submission, commit, push, installation, download, data deletion
or browser launch occurred. Original job **5357168** and its evidence remain
untouched; its results must retain their original source identity.

## 1. Exact changes in this correction pass

- `scripts/claix_b24_targeted_h100.py`: add isolated uninterrupted 6 / stop 3 /
  resume 6 comparison before the existing 260-step experiment. Reuse the original
  combined controller's process ownership, deadlines, cleanup, checkpoint and
  training-state comparison helpers; add exact saved RNG comparison. Signal
  only after verified interval-250 publication/rotation telemetry, so a separate
  later signal checkpoint exercises actual deletion at step 260. Reuse the
  sustained memory/throughput validator. Require identical configuration across
  both experiments and exact-source per-invocation provenance. Partial phases,
  mismatches, signals and insufficient time leave the overall report FAIL.
- New `scripts/claix_b24_filesystem.py`: canonical directory and mount checks.
  `claix_b24_environment.sh` exposes this check; production batch checks both
  submission directory and current directory for persistent Slurm logs.
- `scripts/claix_b24_storage_preflight.py`: distinguish submission review from
  allocation-time measurements; implement the conservative unattended refresh
  policy described below, with explicit reasons and machine-readable outputs.
- `scripts/claix_b24_release.py` and `release_record.template.json`: bind actual
  pre-submission storage report and quota attestation bytes to the immutable
  reviewed record. Drafts still approve nothing.
- `scripts/claix_b24_launch.py`: fresh storage check before any training child;
  persist `launch-*.storage.json`, abort on FAIL/UNVERIFIED, and keep immutable
  release-record checks. Add an optional narrower allocation cap for the
  targeted controller; production's lease and margin are unchanged.
- One demonstrated additional defect was fixed in the launcher: an explicit
  older/foreign resume into existing output could append duplicate steps before
  no-clobber checkpoint publication failed. Under the existing run lock, a
  metadata-only check now requires that directory's latest receipt and a
  contiguous telemetry stream ending at exactly that checkpoint. It rejects
  rollback, foreign checkpoint, incomplete logs and uncheckpointed tails before
  loading the model. Preserve such a directory and recover into a **new** root
  with a newly reviewed production plan/record. No logs or checkpoints are edited.
- `xr1/tests/test_claix_final_corrections.py`, storage cases in
  `test_claix_final_release.py`, and the existing monitor test fixture cover these
  paths. Existing checkpoint/runtime engineering changes were retained without
  further edits in this pass. `FINAL_RELEASE.md` is the earlier review snapshot;
  this document supersedes its targeted-run, path and quota instructions.

## 2. CPU and static validation

Actual command:

```bash
/home/horus/anaconda3/bin/python xr1/tests/run_claix_cpu_tests.py
```

**134/134 PASS, 24.284 seconds.** This is the complete existing offline harness,
including its real CPU Lightning/AdamW/checkpoint tests and the focused additions
(119 previously prepared cases plus 15 added cases). Environment: torch
`2.6.0+cu124`, Lightning `2.5.1.post0`; CUDA hidden. The existing explicit
registry/backbone mocks remain; no 5B assets were loaded. CPU synthetic H100 and
filesystem fixtures are test inputs, not hardware evidence.

Additional actual checks:

| Check | Result |
|---|---|
| Python AST syntax | 74 Python files PASS |
| Bash syntax | All five `claix_b24*.sbatch` files and environment helper PASS |
| Production launch dry-run | PASS; no child or Slurm call |
| Targeted qualification dry-run | PASS; all five phases, 272 steps, one allocation |
| Current-source read-only UNVERIFIED release draft | Expected rejection at first unqualified gate |
| Invalid filesystem storage preflight | Nonzero, machine-readable reason, no run root created |
| Original provenance/checksum/science/release negative tests | PASS in complete harness |
| Scientific source diff against reviewed base | Empty for configs, model/loss/runner, data, replay, compact format, scheduler, contract and training entrypoint |
| `git diff --check` | PASS |

Negative tests include a one-element model/moment/counter/scheduler/cursor/RNG
change, missing CUDA RNG, corrupt three-step checkpoint before resume, source
and replay drift, aliases and escaping paths, tmpfs/ramfs/overlay/local mounts,
stale/future/expired reviews, wrong UID/project, changed/rounded/grace/failed
quota output, fresh low space/inodes, modified release-bound files, rollback
and duplicate telemetry, and storage failure that must never launch training.
No per-step gradient logging was added.

Full logs, dry-runs, syntax/science results and source manifest are under
[`corrections_evidence/`](corrections_evidence/). Earlier evidence files remain
historical snapshots, including their original source digests and test counts.

## 3. Exact H100 comparison design

Keep `claix_b24_targeted_h100.sbatch`: one H100, account `thes2452`, partition
`c23g`, generic `--gres=gpu:1`, 12 CPUs, 96G, maximum **00:45:00**, USR1 at
600 seconds remaining. The controller also rejects allocations longer than
45 minutes. Use one **new** persistent root:

```text
ROOT/exact-restart/uninterrupted/   # 6 real optimizer steps
ROOT/exact-restart/restarted/      # stop at 3, verify receipt/checksum/state, resume to 6
ROOT/exact-restart/restart-comparison.json
ROOT/logging-retention/            # distinct 250 -> SIGUSR1 -> resume to 260 experiment
ROOT/logs/                         # phase commands, stdout/stderr, starts/exits
ROOT/targeted-report.json
ROOT/targeted-artifacts.json
```

The small runs use the same approved real dataset/model, batch 24, workers 4,
seed/replay contract, optimizer, scheduler and precision. Only qualification
trainer stopping boundaries differ; the scheduler/data horizon remains 48284.
The stopped run must finish at step 3 with `qualification_step` and a fully
verified compact checkpoint. Actual resumed start must be step 3 / cursor 72;
resumed telemetry must contain exactly steps 1..6 across invocations.

Reuse `compare_states` from the original combined qualification. Compare **all
selected FP32 trainable tensors**, all AdamW `exp_avg` / `exp_avg_sq` tensors
and step counters, complete optimizer parameter groups, scheduler state, global
step, replay cursor/contract and compact immutable metadata/configuration hash.
Compare all saved rank RNG states exactly: Python, NumPy array/scalars, Torch
CPU bytes and each CUDA RNG byte tensor. Differences use zero tolerance;
model/optimizer tensor diagnostics remain chunked. Unrelated log/timestamp
metadata is excluded. The resolved config fingerprint includes the engineering
source digest; three launch provenance files must also match the current exact
source digest, and the retention experiment must have the same config hash.

A genuine comparison failure is saved as FAIL and stops further phases. Overall
`exact_tensor_equivalence=PASS` is written only after both experiments and final
source/signal/deadline checks pass. Missing/partial evidence never creates an
overall PASS. The targeted run repeats neither workers 0/4/8 nor validation
inference nor the entire old combined qualification.

## 4. Extra startup overhead and 45-minute constraint

There are **three additional complete model-starting processes**, 12 additional
optimizer updates (6 + 3 + 3), three additional compact publications, two final
states loaded for comparison, and additional resume verification/checksum work.
There are five training invocations total. Each extra startup repeats immutable
base hashing/model construction and worker setup; the resumed one also loads
compact optimizer state. Do not count this as just twelve steps of overhead.

At historical 14.44 examples/s, twelve batch-24 steps correspond to about
**20 seconds of training**. This is a planning calculation, not final-source
throughput. Added startup cost is approximately `3 * measured_startup_latency`,
plus checkpoint I/O/comparison/cleanup. No reliable startup latency for this
final source is available; the old combined runbook itself estimated a wide
20–90-minute overall range. Therefore **completion in 45 minutes cannot be
guaranteed from current evidence**. This constraint remains explicit; steps,
comparisons and safety margins have not been reduced to force a PASS.

The outer deadline is Slurm EndTime minus 600 seconds. Reused process handling
reserves another 60 seconds before that deadline and asks training to stop
180 seconds before the phase hard deadline; each phase needs at least 240
seconds of remaining phase budget to start. A phase stopped early fails its
required step/checkpoint checks. Controller stop signals prevent subsequent
phases; bounded cooperative stop and owned-descendant cleanup replace the old
30-second unconditional termination path. Phase command/exit timestamps and
training telemetry provide actual startup/fit/I/O timing after execution.
If this budget is insufficient, retain partial evidence and report FAIL; any
longer future request requires explicit operator review, not an automatic retry.

## 5. Persistent path handling

Approved namespaces are exactly `/hpcwork`, `/work`, and the known canonical
alias `/rwthfs/rz/cluster/hpcwork`. Paths are compared by components, resolved
through actual symlinks, checked for existing directory parents and inspected
with bounded read-only `findmnt`. The actual mount must be an approved persistent
network/parallel filesystem type and contain the resolved path. Storage checks
also require the exact operator-confirmed mount SOURCE.

A valid `/hpcwork/...` resolving to `/rwthfs/rz/cluster/hpcwork/...` passes.
Arbitrary `/rwthfs/...`, prefix lookalikes, temporary paths, symlink escapes,
missing submission directories, files and tmpfs/ramfs/overlay mounts fail.
An unknown site alias/type fails closed until separately established; the whole
`/rwthfs` tree is never approved. The helper can also be run **before submission**:

```bash
"$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_filesystem.py" --path "$PWD"
```

Relative Slurm stdout/stderr directives remain persistent in the verified
submission/current directory; launch logs also remain beneath the run root.
Approved production account/partition/GRES, **23:45:00** and **USR1@1800** are
unchanged, as is read-only release-record validation.

## 6. Quota and space after long queue waits

Before submission, run the read-only preflight at the real proposed run root
with the confirmed filesystem SOURCE and, when applicable, actual Lustre
project ID. Initial output is UNVERIFIED until an operator reviews both user
and project limits/non-applicability, grace/soft/hard limits, effective remaining
bytes/inodes, existing retention/orphans and publication/log reserve. GID is not
substituted for project ID. No tool failure or empty quota output implies PASS.

The attestation fields from the earlier review remain, plus `project_id` if used.
`run_root` is the canonical path. To authorize the conservative unattended
Lustre refresh, the operator may additionally supply:

```json
{
  "project_id": "REPLACE_WITH_ACTUAL_INTEGER",
  "allocation_refresh_policy": "identical_successful_lustre_queries_v1",
  "allocation_valid_until": "OPERATOR_CHOSEN_TIMEZONE_AWARE_ISO_TIMESTAMP"
}
```

This is a schema illustration, not an approval or a valid attestation. Expiry
must be after review and no later than **seven days** after it. Re-run preflight
with the actual reviewed attestation **within one hour of its timestamp**.
Submission stage never accepts an older review. Preserve the resulting PASS
report and attestation as hashed artifacts. Fill the release record binding:

```json
"storage_preflight": {
  "report": "/absolute/actual-submission-report.json",
  "report_sha256": "ACTUAL_SHA256",
  "quota_review": "/absolute/actual-quota-review.json",
  "quota_review_sha256": "ACTUAL_SHA256",
  "filesystem_source": "ACTUAL_REVIEWED_FINDMNT_SOURCE",
  "project_id": "ACTUAL_INTEGER_OR_NULL"
}
```

The record checker verifies these hashes, report status/stage, matching root,
filesystem/project, query fingerprint, reviewer and review-to-measurement age.
Do not edit these artifacts while a job is queued. Finalize the reviewed release
record read-only as before. The stale submission measurement is retained solely
as provenance; it is **never** the allocation's free-space measurement.

At allocation, before any model child starts, the launcher independently repeats
canonical filesystem/SOURCE verification, `statvfs.f_bavail` and `f_favail`, and
all quota queries. Require current **30 GiB** and **10000 inodes**. Queries use
the stable mount target, rather than whichever ancestor existed before the run
root was created. `findmnt` and each quota command have ten-second timeouts.
The machine-readable result is saved as `launch-*.storage.json` and also emitted
as `ALLOCATION_STORAGE=` JSON to persistent Slurm stdout if that report cannot
be written. Explicit policy expiry is honored even for a review under one hour
old. Failures have
explicit `abort_reasons` and stop before loading the 5B model.

A review under one hour old still requires identical **fresh** query output.
A review older than one hour requires the explicit unexpired allocation policy,
Lustre, an explicit project ID, successful complete unrounded user **and**
project quota rows with matching identities, no warnings/grace indicators, and
byte-identical freshly queried usage/limits versus the operator-reviewed
snapshot. This method carries forward the human's capacity interpretation
only when its underlying current measurements are unchanged. It does not infer
limits from an unsupported command or extend an old free-space measurement.

Changed shared-project usage, unknown formats, grace output, quota query errors,
missing/expired policy or queue waits beyond expiry abort unattended. Generic
`quota -s` output alone cannot renew an old review. This intentionally strict
policy may reject a still-safe filesystem; it cannot reliably authorize every
long-queued CLAIX job. Site quota output must first be inspected to establish
that this format is supported. If it is not, there is **no automatic quota PASS**:
use an operator-arranged bounded start/reservation window with fresh review,
or obtain a separately reviewed reliable site quota adapter. If the check
aborts, re-review on the login node and prepare a new immutable record/job.
No human is asked to approve anything at the instant of allocation. No fallback
silently extends validity. No dataset/checkpoint deletion or production-data
writing probe occurs. Quotas remain shared and can change after the check;
this is a start preflight, not a reservation of disk capacity.

## 7. Checkpoint verification cost and error-path review

Rotation/publication implementation is unchanged in this correction pass.
The complete harness again verifies no-clobber payload/receipt publication,
corrupt-newest/safe-stop rejection before deletion, older verified fallback,
symlink/FIFO/oversized receipt rejection, publication/receipt/unlink failures,
separate safe-stop pins and all historical milestones. Payloads are loaded
one at a time; metadata is bounded, and each rotation attempts at most **eight**
distinct checkpoint verifications, including failures. No trusted payload cache
or per-step verification was introduced.

Healthy steady rotation costs **3 loads + 3 SHA256 passes**; publication adds
one readback load/hash. Thus ordinary publication+rotation costs four full
logical reads through each mechanism (8 payload-byte passes), plus one write:
about **13.6 GB logical reads + 1.7 GB write** for a 1.7 GB checkpoint. A separate
safe-stop pin increases rotation's count to four, making about 17 GB logical
reads including publication. Fit-end adds one verification pair per invocation.
These are operation counts/byte estimates, not physical device traffic or
CLAIX latency. The extra small comparison uses ordinary verification without
weakening any of these checks; its two-state exact comparison also uses CPU RAM
and bounded FP64 difference chunks. Measure actual durations in H100 telemetry.

Safe stops remain at completed optimizer boundaries. A signal arriving during
an interval save is handled at a subsequent boundary. Final fit-end requires a
verified checkpoint; corruption/publication failures propagate. CPU tests retain
all CSV version directories, full loss components each step, exact scheduler
and optimizer state, append-only telemetry and GPU monitor timeout/reaping/UUID
checks. The new resume guard covers multiple clean resume boundaries without
rewriting earlier versions. Monitoring remains best effort and independent of
training; qualification requires actual samples for its PASS.

A catastrophic failure before the first verified checkpoint, or a filesystem
that becomes unwritable after preflight, cannot be promised a recoverable new
checkpoint. Do not claim otherwise or save an uncertain mid-step state on an
exception. Prior verified checkpoints remain protected; an uncheckpointed log
tail requires recovery into a new root. Checkpoint interval stays 250.

## 8. Remaining production blockers

- Actual final-source H100 execution and exact comparison, sustained memory,
  throughput, rotation and safe-stop evidence have **not** been obtained.
- Completion within the prepared 45-minute bound is not established.
- Actual CLAIX filesystem/SOURCE, user/project quota support and operator review
  are still needed; unsupported stale quota refresh blocks unattended launch.
- Nine evidence/applicability decisions, scientific LR/horizon approval and any
  required held-out validation evidence remain operator/scientific work.
- A new actual-root plan and immutable release record must bind this exact tree
  and genuine artifacts. Prior draft records remain unapproved and obsolete.

No scientific change was made: LR max **2e-5**, min **5e-6**, warmup start
**5e-7**, warmup **500**, horizon **48284**, batch **24**, workers **4**,
last_n **1**, BF16 compute/FP32 trainables, interval **250**, and historical
milestones **2000/5000/10000/20000/30000/40000** remain intact. Production remains
23h45m with its 30-minute safe-stop margin. Scientific validation milestones
and model/loss/dataset/sampler/optimizer/scheduler/action semantics are unchanged.

## 9. Source-fingerprint implications

Final engineering source-tree SHA256:
`533f8071446d89585d862400929aa9ddadea03ea37c2630c338e3b2436abbc39`.
The identical launcher/contract digest includes every `xr1/**/*.py`, configuration
YAML and `scripts/claix_b24*` file, including tests/new helpers. The hashing
algorithm itself is unchanged. Consequently the resume configuration fingerprint
also changes. Its exact resolved value depends on the real approved data/model
paths and frozen inventory and is recorded during qualification; it is not
fabricated locally. Documentation/evidence artifacts are outside this source
hash, so preserving this report does not create a fingerprint cycle.

HEAD alone does not identify these uncommitted sources. Stage/copy the exact
manifested tree on CLAIX and verify the digest **before** qualification and release
review. Any later included-source edit, even a test/script edit, invalidates the
current binding and requires refreshed evidence. Original 5357168 checkpoints
must not be relabelled or resumed under this fingerprint. New small comparison
runs all use the same new source; no fingerprint bypass was added.

## 10. Can all nine gates use genuine evidence?

The corrected tooling supports that outcome **conditionally**, without another
workers experiment or the full old combined qualification:

| Gate | Genuine evidence required |
|---|---|
| `frozen_splits` | Original frozen inventories/hashes and applicability review |
| `tokens_24_examples_48_views` | Existing certificates plus current complete-batch telemetry |
| `workers_0_4_8` | Preserved original worker evidence with its real source/job identity |
| `h100_b24_memory` | Current-source targeted sustained memory telemetry/report |
| `end_to_end_throughput` | Current-source sustained/fit timings and explicit throughput acceptance |
| `real_compact_resume` | Current-source exact 6/3/6 comparison, verified receipts and launch provenance |
| `safe_stop_and_storage` | Current-source signal/resume/rotation artifacts plus genuine submission/allocation storage evidence |
| `held_out_validation` | Genuine existing applicable scientific validation evidence and preserved milestone procedure; an inference smoke alone is insufficient |
| `lr_and_horizon_approval` | Explicit scientific approval of unchanged LR/horizon |

All four hardware-sensitive gates still require `current_h100` evidence matching
this final source, with original hashed provenance bindings; old 5357168 results
cannot fill those requirements by themselves. The targeted run can supply the
missing engineering evidence if it actually passes. Operator signatures alone
cannot manufacture missing held-out validation or quota results. If existing
scientific validation is insufficient, that remains an additional blocker after
the targeted run. No gate has been automatically populated or declared PASS by
this CPU review, and no final source is claimed H100-qualified.
