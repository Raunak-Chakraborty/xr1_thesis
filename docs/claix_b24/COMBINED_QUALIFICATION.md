# One-allocation H100 batch-24 qualification

This is the targeted follow-up to reviewed commit
`5e2dbb8d184899901ba4162d5e6bbb208d281a8c`, on
`thesis/claix-b24-combined-qualification-v1`. Development uses only this isolated
worktree. The new combined hardware qualification is **UNVERIFIED** until its
compute-node report and raw evidence have been reviewed. No job was submitted.

## Existing hardware evidence

The operator supplied these actual CLAIX results; this development session did
not rerun or independently fetch the cluster artifacts:

- CPU job **5350462**, COMPLETED / exit 0: workers 0/4/8 qualified, deterministic
  real-batch hashes matched; four workers measured **28.53 examples/s** for the CPU
  probe. This is not GPU training throughput.
- H100 job **5350463**, COMPLETED / exit 0: XR1 5.1B, batch 24, 48 camera views,
  six optimizer steps, torch **2.8.0+cu128**, Lightning **2.5.3**,
  **141255168** trainable FP32 parameters, approximately **64.91 GiB** peak
  reserved GPU memory, verified step-6 checkpoint, about **8.43 s** checkpoint I/O.

Original checkpoint (read-only historical evidence):

```text
/hpcwork/zk443626/robotics_thesis/runs/xr1_b24_h100_qualification/job_5350463/project_xr1_doosan/qualification_b24_w4/step-000006.ckpt
```

Its adjacent `step-000006.verified.json` is also preserved. The combined job does
not load, modify, resume, or compare this original experiment. Its engineering
source/config fingerprint can legitimately differ from the modified source tree;
never weaken resume validation to force a historical fingerprint match.

## Entry points and fixed contract

- Slurm: `scripts/claix_b24_combined_qualification.sbatch`.
- Orchestrator: `scripts/claix_b24_combined_qualification.py`.
- Training controller: existing `scripts/claix_b24_launch.py`, H100 qualification
  mode, **four workers** throughout. No nested sbatch/salloc/srun calls.

The request is account `thes2452`, partition `c23g`, one node/task/GPU with generic
CLAIX-compatible `--gres=gpu:1`, 12 CPU cores, 96 GiB host RAM, **02:00:00**,
`--signal=B:USR1@600`. The entrypoint additionally verifies one visible H100 and
requires the demonstrated torch/Lightning versions. The existing environment,
model, processor cache, ZIP cache and dataset paths are used offline. No installs
or downloads happen inside the job. Keep Slurm stdout/stderr in a persistent
submission directory.

The launcher adds `--qualification-max-steps INTEGER`, accepted only for H100
qualification (including its old `qualification` alias), range **6..300**, default
**6**. `--stop-after-step` must be within that trainer limit. These arguments
cannot override the replay horizon or LR horizon: both remain **48284**, warmup
**500**. The existing YAML recipe is unchanged. Batch 24, two RGB cameras, all
state/action/normalization/prompt/split/pretrained/last_n=1 semantics, BF16 compute,
FP32 selected weights and accumulation=1 (no accumulation) remain fixed.

`--qualification-deadline-epoch` can only shorten the authoritative H100 safe
allocation deadline. Both new options are rejected for production and CPU mode.
Production still requires the entire source/plan/evidence release record and
**exactly 23:45:00**. The combined report is not a production release record.

## Single job sequence and fail-closed gates

1. Preflight: check environment/H100, offline ZIP mode, frozen split identities,
   255/34/34 episode counts, 289703 training frames and cross-split episode/media
   isolation. Bind the current source/configuration, normalization, model and data
   fingerprints; preserve full split inventories and per-file source/reference
   SHA256s. Test episodes are inventoried only; no test predictions are generated.
2. Safe CPU clock probe: check the real callback `StopRequest.due` immediately
   before and at a simulated deadline. Local tests separately exercise the real
   CPU Lightning deadline checkpoint and mocked launcher controller clock. This
   does not claim hardware allocation-expiry qualification.
3. Fresh `uninterrupted/`: six real optimizer steps, verified step-6 receipt,
   pointer and clean `fit_end`.
4. Fresh `restarted/`: same recipe/max_steps=6, stop at completed step 3; verify
   step-3 payload, receipt, pointer and reason before proceeding.
5. Restart **that same root** from its verified step-3 checkpoint, finish step 6,
   verify receipt/pointer, complete step sequence 1..6 and actual resume offset 72.
6. Compare new uninterrupted and restarted step-6 states with **zero tolerance**:
   every FP32 trainable tensor, every optimizer moment/counter/parameter group,
   scheduler state, global optimizer step, replay contract/cursor, selected model,
   pretrained/language/source metadata and resolved data/config fingerprint.
   `restart-comparison.json` records equality, differing element counts, maximum
   absolute and relative differences per tensor, plus unequal scalar/identity
   values. It is written before rejecting a mismatch. Any required mismatch fails
   the job and prevents the sustained run. Checkpoint validation stays enabled.
7. Fresh `sustained/`: actual real-data max_steps=260. Preserve every step's loss,
   LR, retained examples, camera grids, packed tokens, pipeline wait/overhead,
   allocator allocated/reserved/peak allocated/**peak reserved**, parent/worker
   RSS, cumulative and per-step time, end-to-end rate and FP32/BF16 invariants.
   Verify actual immutable interval checkpoint **250** and its receipt.
8. Only after that verification, send **SIGUSR1 to the Popen-owned launcher
   controller PID**. No signals go to Slurm processes or torchrun during the
   normal stop test. The controller handler sets a flag. Ordinary controller code
   atomically publishes the request; the runtime reads it at a completed optimizer
   boundary, writes/verifies a **new** checkpoint at **251..259**, and exits 0.
   Require `launcher_signal:10` (platform SIGUSR1 value), controller request record,
   checkpoint event with that reason, matching final latest pointer, clean
   `fit_end`, no orphan payload and terminated child/worker processes. Finishing
   naturally at 260, stopping at 250, missing signal/request evidence, or a
   timeout fails the gate. The bounded 10-step window is intentional; the test
   fails if verification is too slow to signal within it.
9. Standalone generated-action inference on **four canonical validation frames**
   using the new sustained final checkpoint. Require matching train/validation/test
   inventories, checkpoint SHA/config provenance, processor/model execution,
   30x60 finite generated outputs, semantic masks and finite physical-unit micro
   and episode-macro metrics. Preserve per-frame output SHA256, the seven semantic
   action channels, masked metrics and result JSON. This is explicitly
   `SMOKE_SUBSET_NOT_A_VALIDATION_RESULT`, not full scientific validation.
10. SHA256 all persistent experiment evidence and checkpoint payloads; publish the
    final machine-readable `qualification-report.json`. Only all required gates
    passing permits `status=PASS`. Later phases stay UNVERIFIED after a failure.
    Existing evidence is retained on failure and a nonzero exit is propagated.

The sustained summary measures steps **11..250** (5760 real examples), and also
reports the fit-end rate including final checkpoint publication/verification.
It imposes no invented scientific throughput threshold. The full raw telemetry
is the evidence for capacity/performance review. Pipeline wait includes callback
and preceding checkpoint overhead, not purely DataLoader latency. RSS includes
shared pages; memory is sampled at step boundaries, with CUDA allocator peaks
covering the run. Optional GPU-utilization counters may be unavailable; required
allocator/CPU/worker measurements may not be null for a PASS.

## Deadline and process ownership

The authoritative Slurm EndTime minus **600 s** is the outer safety deadline.
Each training phase receives a further shortened controller deadline **180 s**
before its hard phase budget. The sustained phase ends at least **660 s** before
the outer deadline, reserving time for validation/evidence. At least 240 s must
remain to launch any phase. The controller makes a normal stop request on its
clock deadline; a hung H100 qualification child gets at most 90 s cooperative
shutdown beyond that deadline, then TERM/KILL of its own new-session child group
with bounded waits. Emergency cleanup is a failure, never a successful stop test.

The orchestrator polls only children it starts, records commands/exits/timestamps,
tracks descendants with psutil Process objects (PID reuse checks), and rejects
live workers after controller shutdown. Abort requests USR1 of its own controller,
then bounded cleanup of owned descendants. Signal handlers perform no I/O or saves.
The batch controller's own Slurm warning stops the sequence; it is not counted as
the intentionally injected sustained-run signal. No test waits for allocation
expiry. SIGKILL/node failure/stuck filesystem operations cannot guarantee a final
report; a missing or RUNNING report is never PASS.

## Required inputs and proposed submission (not executed)

Use the already staged existing environment/cache paths; no preprocessing on a
login node. The submission directory must already exist and be persistent.

```bash
export XR1_WORKTREE=/path/on/CLAIX/to/reviewed/b24_combined_qualification
export XR1_PYTHON=/home/zk443626/robotics_thesis/envs/xr1_fp32/bin/python
export XR1_DOOSAN_ZIP_CACHE_ROOT=/hpcwork/zk443626/robotics_thesis/cache/frames_zip/v1
export XR1_RUN_ROOT=/hpcwork/zk443626/robotics_thesis/runs/xr1_b24_combined_qualification_v1_unique
export XR1_SPLIT_IDENTITIES=/path/to/existing/reviewed/frozen-split-identities.json
```

The new run root **must not exist**. Never use any root from jobs 5350462/5350463.
`XR1_SPLIT_IDENTITIES` must refer to reviewed inventory hashes from frozen split
artifacts, in this exact format:

```json
{"train": "<64-character inventory SHA256>", "validation": "<64-character inventory SHA256>", "test": "<64-character inventory SHA256>"}
```

These are `split_inventory(...)["sha256"]` values (SHA256 of canonical inventory
rows), not tar/ZIP hashes or hashes of inventory JSON formatting. The repository's
reference plan has counts and paths, **not all three authoritative inventory
hashes**. Supply them from existing reviewed artifacts; do not fill placeholders,
use arbitrary values, or mint the expected values from this job's current scan.
Preflight fails closed if these identities are missing or disagree. The prior
run's `training_inventory.json` can supply the train identity; validation/test
identities must also be reviewed existing inventories. A missing identity record
is an input still needed before execution, not a successful qualification gate.

Optional `XR1_VALIDATION_ROOT`/`XR1_TEST_ROOT` override only the staged held-out root
locations if their directory spelling differs. Defaults are transfer
`dataset_v2/splits/val` and `.../test`; all hashes must still match. Workers and
smoke length are fixed by the sbatch entrypoint. The Python orchestrator supports
`--smoke-frames 1..16` for explicit reviewed smoke changes, default 4.

Dry plan, safe on the development CPU (no directories, data scans or Slurm calls):

```bash
python3 scripts/claix_b24_combined_qualification.py --run-root /tmp/combined-plan-only
```

Exact proposed later command, from the persistent submission directory, after
setting the required variables above:

```bash
sbatch --export=ALL "$XR1_WORKTREE/scripts/claix_b24_combined_qualification.sbatch"
```

No job has been submitted in this development task. Source staging/review and
required existing identity inputs must be complete before that future command.

## Persistent evidence layout

```text
<new root>/qualification-report.json     # RUNNING / FAIL / PASS, all phase states
           restart-comparison.json      # every required numeric/identity comparison
           signal-test.json             # owned controller + verified step-250 prerequisite
           validation-smoke.json         # provenance, generated outputs, masked metrics
           {train,validation,test}-inventory.json
           frozen-split-identities.json
           path-overrides.json
           source-and-reference-sha256.json
           evidence-sha256.json           # relative paths, sizes, SHA256s
           logs/<phase>.{command.json,out,err,exit.json}
           uninterrupted/{launch-*,hydra/,project_xr1_doosan/qualification_b24_w4/}
           restarted/{launch-*,hydra/,project_xr1_doosan/qualification_b24_w4/}
           sustained/{launch-*,hydra/,project_xr1_doosan/qualification_b24_w4/}
```

Each training target retains telemetry, configs/inventory/contracts, CSV logs,
checkpoint/receipt pairs and latest pointer. Existing newest-two retention leaves
steps 3/6 in restarted and 250/final in sustained. Old runs are never overwritten.
The SHA manifest excludes itself and the changing final report; the final report
contains the SHA of that manifest. Partial evidence is hashed on normal failure.

## Risks and work remaining after the job

Historical six-step peak reserved memory is **64.91 GiB**; this gives roughly
15 GiB nominal headroom on an 80-GiB H100, not proof of sustained safety. Different
packed lengths, workspace/allocator fragmentation and worker RSS may increase
memory. Host RAM is 96 GiB; compact optimizer state and CPU equivalence/checksum
work also consume RAM. Four workers keep the already measured CPU choice.

Runtime is not established by the provided six-step evidence. Four model-starting
training invocations plus a separate evaluator, 250+ real optimizer steps, immutable
base checksums and filesystem verification must fit the bounded budget. A planning
range of **20–90 minutes** is only an estimate, not a measured result or guarantee;
slow data/filesystem/model startup can produce a clean FAIL for insufficient time.
Allow approximately 8 GiB for five retained checkpoints (roughly 1.58 GiB each
for 141255168 masters plus two moments), additional transactional-save space,
and logs/evidence. No permission to modify old data or
checkpoints is implied.

After completion, review the report **and** raw comparisons, performance/memory,
all receipt SHA256s, stop provenance and actual generated actions. A PASS qualifies
this targeted engineering sequence on the recorded source/stack. Remaining work:
full held-out validation across 34 episodes, scientific LR/48284-step-horizon
approval and all outstanding production release evidence (including the original
token certificate). Test episodes stay untouched for final reporting. A small
smoke does not establish task success, convergence or production readiness.
