# XR1 batch-24 recovery and reliability handoff

Status: **LOCAL CPU VALIDATION PASSED; PRODUCTION RECOVERY NOT CERTIFIED.**
No training job was submitted or run on a GPU. No original run, telemetry,
checkpoint, release, or evidence archive was modified. No SSH, commit, push, or
browser-opening command was run. Work remains on the isolated laptop branch.

Read [the incident analysis](PRODUCTION_5363645_ROOT_CAUSE.md) first. The cause is
an unresolved DataLoader delivery stall, not a demonstrated Lustre fault.

## Exact additions and fingerprint implications

No existing tracked source/configuration file was changed. Added files:

| Path | Purpose |
|---|---|
| `reliability/common.py` | Pinned identities, safe file operations, original source check, separate tool-bundle identity |
| `reliability/recover.py` | Evidence audit, CPU preparation, exact seed copy, reviewed-input validation; no training execution |
| `reliability/stage_cache.py` | Cache integrity seal, actual-space checks, optional verified scratch copy |
| `reliability/observe.py` | Passive bounded Linux process/kernel-wait snapshots; no signals or training hooks |
| `reliability/probe.py` | Explicit CPU-only, bounded replay with worker/sample/ZIP/collation markers and pre-timeout stacks |
| `reliability/tests/test_control.py` | Evidence preservation, source identity, observer and staging regression tests |
| `reliability/tests/test_worker.py` | Actual spawned PyTorch workers, hang/timeout/stacks, ordered ZIP reads |
| `reliability/tests/test_recovery_torch.py` | Actual CPU Lightning/AdamW step-8750 recovery, negative checks, incident reconstruction |
| `docs/claix_b24/PRODUCTION_5363645_ROOT_CAUSE.md` | Incident conclusions and uncertainty |
| `docs/claix_b24/RECOVERY_RELIABILITY_HANDOFF.md` | This handoff |
| `docs/claix_b24/reliability_evidence/baseline_cpu_tests.txt` | Initial suite attempt, showing missing local reference fixture |
| `docs/claix_b24/reliability_evidence/existing_cpu_tests.txt` | Existing 138-test suite rerun with the real reference fixture |
| `docs/claix_b24/reliability_evidence/control_tests.txt` | New control/staging tests |
| `docs/claix_b24/reliability_evidence/worker_tests.txt` | New real-worker tests |
| `docs/claix_b24/reliability_evidence/recovery_torch_tests.txt` | New real-optimizer/reconstruction tests |
| `docs/claix_b24/reliability_evidence/incident_audit.json` | Reproducible pinned archive audit |
| `docs/claix_b24/reliability_evidence/local_checks.txt` | Final source identity, syntax and change-scope checks |

The unchanged launcher and trainer both still compute
`852945709dac80152459c26ab5dab5e70e158b631ea27612d28ee2d172ef3d0b`.
New code/tests are deliberately outside the launcher's fingerprinted file set.
They do not monkeypatch the production launcher, loader, trainer, checkpoints or
fingerprint functions. `recover.py identity` separately hashes all reliability
Python files and both documents; `prepare` saves that manifest plus the actual
dirty checkout status in recovery provenance. `check` requires the reviewed
tool hash and lineage hash. This is an additional audit boundary, not a claim
that new operational tools were covered by the old release.

A wrapper is sufficient for checkpoint compatibility. The unchanged launcher
allows an external verified checkpoint when the new training target is absent.
It deliberately refuses original telemetry ending at 8758 against checkpoint
8750. The output root and checkpoint path are excluded from the resume config
digest. Original dataset/model path strings and all scientific parameters must
stay identical. The original engineering commit check also remains strict;
the tools are delivered as a reviewed additive bundle on that base, not relabeled
as part of its original release.

## Recovery transaction and scientific accounting

`prepare` first validates the pinned archives, source, re-composed Hydra config,
actual dataset inventory, model/language hashes, cache seal and full trusted
checkpoint. It acquires a shared, nonblocking, read-only lock on the original
run's existing `.production.lock`, refusing a concurrent production writer. It
checks every archived original-run file against its incident bytes.

Only then does it exclusively create a new persistent HPCWORK run root. Its
`_recovery/seed/step-008750.ckpt` and matching receipt are independent read-only
copies, not links and not reserialized checkpoints. It copies both evidence
archives and cache seal, saves exact overrides and a new plan, rechecks the parent
evidence, and finally writes `_recovery/recovery.json`. Partial preparation has
no valid completion marker and is not launchable by `check`; preserve it and
select another new root. Do not manufacture the missing marker.

The real training target remains absent:
`project_xr1_doosan/xr1_doosan_b24_results_v1`. The first training start must report
step 8750 and replay offset 210000. Its first completed optimizer update is 8751.
Previously observed steps 8751–8758 are recomputed from the saved state, followed
by 8759. Those eight volatile updates cannot be recovered from telemetry. The
lineage explicitly records the committed prefix, recomputed tail and first
previously unobserved step. Thesis analysis should use original steps 1–8750 plus
the selected recovery branch after 8750, while retaining the failed tail as
failure evidence. Never concatenate both tails as additional optimizer steps.

The original runtime restores FP32 trainables, FP32 AdamW moments, scheduler,
Lightning loop progress, Python/NumPy/Torch/CUDA RNG state and rank sampler offset.
Nothing resets the scheduler, optimizer or sampler. CUDA RNG topology must be one
device. Preparation validates saved state on CPU; actual CUDA restoration is a
remaining hardware check. The production checkpoint is trusted only after its
exact known SHA256 is checked, before pickle deserialization.

Checkpoint interval remains 250. The first normal new interval is 9000. Seed
8750 is outside the rotating target and always retained, including a failure
before 9000; it is never rewritten as a synthetic `latest.json`. The unchanged
rotation protects the two most recent usable target checkpoints, historical
milestones, and the current safe-stop pointer. The original run is outside all
new retention operations. Recalculate persistent free space after copying the
approximately 1.58 GiB seed; keep the original launcher 30 GiB/10,000-inode gate.

Graceful stop still occurs only at a completed optimizer boundary. Unexpected
exceptions emit `fatal` and do not save arbitrary partial state. A blocked
DataLoader can delay graceful stop until an exception; no new guarantee of
instantaneous checkpointing during an I/O hang is asserted. Do not save from a
signal handler, ignore a timeout, skip a batch, or automatically retry forever.

## Diagnostics: implemented scope and limitations

`observe.py` reads `/proc` from an independent process every five seconds. It
records PID/start-time identity, descendants, CPU/I/O counters, task wait channels,
system calls and kernel stacks where permissions allow. At most 16 processes,
32 tasks each, bounded text fields, and 24 snapshots are retained. It writes at
most two JSON reports: the first after 60 seconds of unchanged process counters,
and the final ring on process exit/deadline. It does not poll Lustre telemetry,
send signals, attach a debugger, hold training locks, or create threads before
worker fork. Inactivity can be legitimate; this is evidence, not diagnosis.
Failure to read `/proc/*/stack` is explicitly recorded as unavailable.

`probe.py` is a separate **CPU-only** executable. It keeps the scientific dataset
and sampler code intact, wraps their calls only in the probe, and observes real
four-worker, prefetch-two, persistent, ordered delivery. It records worker IDs,
PIDs, sample indices, episode/frame, archive/camera/source and ZIP member, with
markers around archive access, ZIP reads, sample completion and collation. A
60-second faulthandler alarm captures parent and active worker Python stacks
before the unchanged default 180-second timeout. Each process writes one current
JSON slot and a trace file on local scratch. The window is capped at 16 batches;
there are no unbounded per-sample Lustre logs. Copy the small diagnostic bundle
to HPCWORK after completion or failure, using the shell trap below.

The probe uses `spawn` to avoid adding unverified pre-fork thread/handle hazards.
This differs from production's default Linux worker start and cannot reproduce
its long-lived state. The original forensic probe's fresh default workers already
passed. The new probe adds observability, not a demonstrated repair. Production
uses `pin_memory=false`; CPU tests do not certify tensor IPC on CLAIX or any CUDA
pinning/transfer path. Workers stuck after `collate_done_awaiting_queue` with a
parent stack in queue receive suggest delivery/IPC; workers stuck in ZIP phases
suggest decode/I/O. Neither alone proves a storage subsystem fault.

Detailed production sample markers and Python worker traces are **not installed**
by this bundle. The passive observer can capture kernel evidence before timeout,
but it cannot produce missing Python stacks. A site-approved external stack
profiler would need separate permission/performance validation. An integrated
production worker watchdog would change the engineering fingerprint and require
a new release and explicit checkpoint compatibility design. Do not hide such a
hook behind `sitecustomize`, `PYTHONPATH`, monkeypatching or a Python wrapper while
claiming the original certified source is executing unchanged.

If an integrated source revision becomes necessary, keep the original payload
and source identity unchanged. Review a narrowly specified compatibility bridge
that validates the complete old contract and exact parent checksum before any
new-schema migration; bind old/new source, config, tensor/optimizer/RNG equality
and lineage in separately reviewed evidence. New source requires requalification
of all affected gates, including final-source H100 gates. Merely changing stored
`config_sha256`, expected hashes, or the source label is not validation. No such
migration or bypass is implemented here.

## Optional storage choices

| Choice | Benefit | Cost/limit |
|---|---|---|
| Existing verified Lustre ZIP cache + external liveness evidence | No bulk copy or exclusive-node request; unchanged scientific/source path | Cannot remove a possible transient shared-storage stall; cause remains unknown |
| Allocation-local scratch if independently confirmed usable | Removes repeated remote ZIP reads after verified copy | Actual mount, space, ownership, lifetime and performance must be measured; do not assume `$TMPDIR` is SSD or private |
| Explicit BeeOND allocation | Documented temporary SSD-backed namespace | Exclusive allocation, copy/verification startup time, scheduling and potential accounting cost; not enabled automatically |

RWTH documents approximately 635 GB local SSD on CLAIX-2023 ML nodes, and BeeOND
through `#SBATCH --beeond` and `$BEEOND`. BeeOND makes the allocation exclusive and
deletes its contents at job end. These are storage options, not evidence that
staging would cure this failure. See [RWTH CLAIX-2023 hardware](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/fbd107191cf14c4b8307f44f545cf68a/)
and [RWTH BeeOND instructions](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/42f947e2c9cf483fb97c3cb52c2d23aa/).

The cache's reported 145,751,023,616 bytes are about 135.74 GiB. Do not assume
635 GB is free. The staging tool checks current `statvfs`/free bytes/inodes inside
an existing allocation, retaining at least 8 GiB or 5% of copied bytes as reserve.
It accepts recognized local/BeeOND mount types at the CLI; the operator must
still establish that scratch belongs to this allocation. It does not request
BeeOND. It copies exactly the manifest's 646 ZIPs, their 646 sidecars and original
manifest bytes, preserving relative paths. This preserves the existing loader's
646-row contract; it does not copy model weights, datasets, runs or checkpoints.

The original manifest contains paths/frame metadata, not ZIP SHA256 values.
`seal` reads every archive, compares it with its existing sidecar's archive hash,
and binds those files and sidecars into a persistent index. Review and retain
that index's checksum independently before staging. This establishes current
integrity; it cannot retroactively prove cache bytes at incident time. `stage`
creates a new destination, copies files, fsyncs them, rehashes every destination,
and writes `STAGING_READY.json` only after the complete set verifies. A missing
marker, bad manifest, changed job ID, corrupt/missing file, wrong seal or inadequate
space fails closed. There is no implicit fallback to an incomplete copy.

Full copy plus destination verification costs approximately
`size / measured_copy_rate + size / measured_local_read_rate`, plus metadata.
For illustration only, 512 MiB/s copy and 1 GiB/s reread take about 6.8 minutes.
Sealing and preparation also require full persistent reads; do these in CPU
validation before paying for an H100 lease. Record actual rates. Staging inside
a GPU allocation consumes that lease and its safe-stop budget. Exclusivity may
prevent sharing the other GPUs/CPUs and increase queueing/accounting costs; confirm
site policy rather than estimating a monetary tariff. Check that BeeOND allocation
TRES still satisfies the unchanged exactly-one-GPU launcher guard; do not weaken
that guard to accommodate exclusive-node behavior.

Only the cache-root environment variable may point at scratch. All checkpoints,
receipts, lineage, releases and persistent logs stay under HPCWORK. The staging
receipt has a persistent audit copy. Temporary storage is never the sole location
of a checkpoint or provenance. First continuation should use the existing verified
cache unless staging has been separately validated and selected during review.

## Tests executed locally

| Suite | Result and scope |
|---|---|
| Existing offline suite | 138 tests passed, PyTorch 2.6.0+cu124 / Lightning 2.5.1.post0, CUDA hidden |
| New control/staging | 8 tests passed with real files, copies, ZIPs, `/proc` observation; space exhaustion is an injected measurement |
| New worker diagnostics | 2 tests passed with real four-worker PyTorch 2.7.1+cu128, spawn, ordered reads and an actual sleeping/hung worker |
| New recovery/incident | 5 tests passed; real CPU optimizer updates through 8758, verified 8750 restoration, exact weights/moments/scheduler and replay through 8758; RNG and rejection checks; all 192 reconstructed sample positions |

The initial existing-suite attempt failed because this isolated checkout lacked
`.claix_b24_reference/configs/hydra_overrides.json`. The rerun used a temporary
symlink to the existing actual local reference bundle; it was removed afterward.
Both logs are retained. No source fixture was invented to mark a gate PASS.

The existing offline harness declares its unavailable mmengine/backbone import
stubs. It also uses synthetic hardware fixtures for release/controller tests;
those are not H100 tests. The new recovery test uses the real checkpoint/runtime
and Lightning with a tiny network, a real rank-offset sampler and CPU
SingleDeviceStrategy, no optimizer/Trainer mocks. It uses zero workers to isolate
state restoration. Separate worker tests exercise four processes with NumPy
transport because this sandbox restricts Torch descriptor-passing IPC. They do
not claim a successful CLAIX tensor-IPC or pin-memory test. The CPU CLI probe
defaults to actual tensor transport and must be run on CLAIX.

Unverified: real production checkpoint payload and CUDA RNG restoration; actual
episode and full cache bytes; full `prepare` transaction on HPCWORK; exact CLAIX
Python/Torch/Lightning versions; full-model resumed update equivalence; process
stack permissions; sustained runtime instrumentation overhead; local/BeeOND mount
behavior and full 136 GiB staging performance. The original CPU replay is external
evidence, not a new job run by this investigation.

## Transfer, release gates and next allocation blockers

Transfer only the additive `reliability/` directory and these documents/evidence
through the established file-transfer process, into a **separate** CLAIX checkout
at the authoritative commit. Do not overwrite the original run or its checkout.
Hash a local review tarball, verify that hash after transfer, then run `identity`
on both sides and compare. The tarball must not contain original checkpoints or
modified release records. No commit or push is needed to review this additive
bundle. If the source base is changed later, its strict pin must be deliberately
reviewed; do not suppress it.

Before another H100 allocation:

1. Run the CPU commands below on the actual persistent files. Inspect the real
   step-8750 payload, all state, dataset/config identities and current cache seal.
2. Run the bounded CPU replay from 8751 through 8766 with real tensor IPC on the
   approved environment, plus hang/failure/stop tests. Retain versions and output.
3. Choose existing Lustre or separately validated staging; resolve outstanding
   diagnostic limitations explicitly. No evidence currently proves a stall fix.
4. Prepare a new run root and new release. The old release binds its old run root
   and storage report and must fail against the new plan. Keep all nine gates.
   Re-measure storage for the new root and preserve the original envelopes and
   artifact/provenance hashes. A human must review applicability of existing
   frozen split, token, workers, model-memory, throughput, resume, safe-stop,
   held-out and scientific evidence. Unchanged-source H100 evidence may be
   referenced truthfully if applicable; never invent a new job ID, mark new
   reliability behavior as hardware-tested, or automatically copy PASS decisions.
5. Review a separate recovery record binding tool-bundle, lineage, cache seal,
   release and actual CPU evidence hashes. Immutable reviewed release validation
   and allocation storage checks still run in the original launcher. If reviewers
   require a new H100 gate for changed operational behavior, that is an explicit
   blocker under this task's prohibition on GPU tests; do not declare it passed.

The recovery review schema expected by `check` is `xr1-recovery-review-v1` with
`status: REVIEWED`, real `reviewed_by`/`reviewed_at`, `lineage_sha256`,
`tools_sha256`, `cache_index_sha256`, `release_sha256`, and a nonempty `artifacts`
list of `{path, sha256}` pointing to real validation results. Create it only after
review; store it read-only. These fields are audit records, not cryptographic
signatures or substitutes for human review. No preapproved review is supplied.

## Exact read-only / CPU-only CLAIX commands

These commands are **prepared, not executed here**. Use an already authorized CPU
allocation for dataset replay/full-file hashing. They contain no SSH, `sbatch`,
GPU training or launcher `--execute`. First transfer the two pinned evidence
archives to the stated evidence directory and the additive bundle to the separate
checkout. Choose fresh output names if any destination already exists.

```bash
cd /home/zk443626/robotics_thesis/worktrees/xr1_b24_reliability_review
export XR1_PYTHON=/home/zk443626/robotics_thesis/envs/xr1_fp32/bin/python
export XR1_WORKTREE="$PWD"
source scripts/claix_b24_environment.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false MAX_LENGTH=12288
export PYTHONPATH="$PWD/xr1:$PWD/reliability"
export HPC=/hpcwork/zk443626/robotics_thesis
export ORIGINAL="$HPC/runs/xr1_b24_production/production_20261101_203343"
export XR1_INCIDENT_EVIDENCE="$HPC/evidence/production_failure_5363645"
export REVIEW="$HPC/diagnostics/xr1_reliability_v1"
export NEW="$HPC/runs/xr1_b24_recovery/recovery_5363645_step8750_v1"
export XR1_DOOSAN_ZIP_CACHE_ROOT="$HPC/cache/frames_zip/v1"
mkdir "$REVIEW"
mkdir -p "$HPC/runs/xr1_b24_recovery"

git rev-parse HEAD
git status --short
sacct -j 5363645 --parsable2 --format=JobID,State,ExitCode,Start,End,Elapsed,MaxRSS,ReqMem,NodeList
"$XR1_PYTHON" -B reliability/recover.py identity
"$XR1_PYTHON" -B reliability/recover.py audit-incident \
  --incident-archive "$XR1_INCIDENT_EVIDENCE/xr1_production_5363645_root_cause_evidence_v1.tar.gz" \
  --cpu-archive "$XR1_INCIDENT_EVIDENCE/cpu_rootcause_results.tar.gz"
sha256sum "$ORIGINAL/project_xr1_doosan/xr1_doosan_b24_results_v1/step-008750.ckpt"
cat "$ORIGINAL/project_xr1_doosan/xr1_doosan_b24_results_v1/step-008750.verified.json"
tail -n 12 "$ORIGINAL/project_xr1_doosan/xr1_doosan_b24_results_v1/telemetry.jsonl"
findmnt --json --target "$HPC" --output SOURCE,TARGET,FSTYPE
df -B1 "$HPC"
df -i "$HPC"
"$XR1_PYTHON" -B -c 'import torch, lightning; print(torch.__version__, lightning.__version__); assert not torch.cuda.is_available()'
```

Run the new tests and seal the unchanged cache. Tests create only synthetic files
in temporary directories; none write the original run.

```bash
"$XR1_PYTHON" -B -m unittest discover -s reliability/tests -p test_control.py -v \
  > "$REVIEW/control.txt" 2>&1
"$XR1_PYTHON" -B -m unittest discover -s reliability/tests -p test_worker.py -v \
  > "$REVIEW/workers.txt" 2>&1
"$XR1_PYTHON" -B reliability/tests/test_recovery_torch.py \
  > "$REVIEW/recovery.txt" 2>&1
"$XR1_PYTHON" -B reliability/stage_cache.py seal \
  --cache "$XR1_DOOSAN_ZIP_CACHE_ROOT" \
  --manifest-sha256 aabf3cae952aaf36fca827bf9d85c3df8430bbfc2aa8348b2feada797ddcff25 \
  --output "$REVIEW/cache-index.json"
sha256sum "$REVIEW/cache-index.json"
```

To rerun the original 138-test suite in this separate checkout, its untracked
reference fixture must exist. Recover the exact four original path overrides
from the pinned incident archive, refusing to overwrite a different fixture:

```bash
"$XR1_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
from recover import incident
e = Path(os.environ['XR1_INCIDENT_EVIDENCE'])
record = incident(e/'xr1_production_5363645_root_cause_evidence_v1.tar.gz', e/'cpu_rootcause_results.tar.gz')
path = Path('.claix_b24_reference/configs/hydra_overrides.json')
overrides = record['provenance']['plan']['overrides']
if path.exists():
    assert json.loads(path.read_text()) == overrides
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(overrides, stream, indent=2)
        stream.write('\n')
PY
"$XR1_PYTHON" -B xr1/tests/run_claix_cpu_tests.py > "$REVIEW/existing-cpu.txt" 2>&1
```

Review the seal and set `CACHE_INDEX_SHA256` to its approved checksum. Do not
silently regenerate a changed seal to make a failure pass. This next CPU-only
command verifies actual production payload/state and exclusively prepares the
new persistent root; it cannot launch training:

```bash
: "${CACHE_INDEX_SHA256:?Set the independently reviewed cache seal checksum}"
"$XR1_PYTHON" -B reliability/recover.py prepare \
  --original-run "$ORIGINAL" --new-run "$NEW" \
  --incident-archive "$XR1_INCIDENT_EVIDENCE/xr1_production_5363645_root_cause_evidence_v1.tar.gz" \
  --cpu-archive "$XR1_INCIDENT_EVIDENCE/cpu_rootcause_results.tar.gz" \
  --cache "$XR1_DOOSAN_ZIP_CACHE_ROOT" \
  --cache-index "$REVIEW/cache-index.json" --cache-index-sha256 "$CACHE_INDEX_SHA256"
```

For the bounded real dataset replay, first establish a writable allocation-local
scratch parent (not a Lustre directory). Keep the parent shell alive to retain
diagnostics even if the probe times out. `timeout` bounds the CPU process group;
no training process is involved. The shell trap copies partial local evidence if
the probe fails or is interrupted; complete node loss can still lose scratch.

```bash
: "${LOCAL_SCRATCH_PARENT:?Set allocation-owned local scratch after checking its mount}"
findmnt --json --target "$LOCAL_SCRATCH_PARENT" --output SOURCE,TARGET,FSTYPE
df -B1 "$LOCAL_SCRATCH_PARENT"
export CPU_SLOTS="$LOCAL_SCRATCH_PARENT/xr1-cpu-replay-${SLURM_JOB_ID}"
(
  trap 'if [ -d "$CPU_SLOTS" ]; then cp -a -- "$CPU_SLOTS" "$REVIEW/cpu-replay-partial-${SLURM_JOB_ID}"; fi' EXIT
  timeout --signal=TERM --kill-after=15s 20m \
    "$XR1_PYTHON" -B reliability/probe.py \
    --config "$ORIGINAL/project_xr1_doosan/xr1_doosan_b24_results_v1/resolved_compact_config.json" \
    --local "$CPU_SLOTS" --output "$REVIEW/cpu-replay-complete-${SLURM_JOB_ID}" \
    --first-step 8751 --batches 16 \
    > "$REVIEW/cpu-replay-${SLURM_JOB_ID}.out" 2> "$REVIEW/cpu-replay-${SLURM_JOB_ID}.err"
)
```

Optional staging, only after a separate storage decision and inside an existing
allocation. `LOCAL_CACHE_PARENT` may be `$BEEOND` **only if BeeOND was explicitly
requested by the operator**. It must not be assumed present or empty.

```bash
: "${LOCAL_CACHE_PARENT:?Set verified allocation-local or BeeOND parent}"
findmnt --json --target "$LOCAL_CACHE_PARENT" --output SOURCE,TARGET,FSTYPE
df -B1 "$LOCAL_CACHE_PARENT"
df -i "$LOCAL_CACHE_PARENT"
"$XR1_PYTHON" -B reliability/stage_cache.py stage \
  --cache "$HPC/cache/frames_zip/v1" \
  --index "$REVIEW/cache-index.json" --index-sha256 "$CACHE_INDEX_SHA256" \
  --destination "$LOCAL_CACHE_PARENT/xr1-cache-${SLURM_JOB_ID}" \
  --report "$REVIEW/staging-${SLURM_JOB_ID}.json"
```

Measure new persistent storage and prepare an **UNVERIFIED** release draft. Set
`PERSISTENT_SOURCE` to the operator-confirmed `findmnt` SOURCE, not a guessed
filesystem label. Do not reuse the old root's storage report.

```bash
: "${PERSISTENT_SOURCE:?Set the confirmed persistent findmnt SOURCE}"
"$XR1_PYTHON" -B scripts/claix_b24_storage_preflight.py \
  --run-root "$NEW" --filesystem-source "$PERSISTENT_SOURCE" --quota-diagnostics \
  > "$REVIEW/storage-submission.json"
"$XR1_PYTHON" -B scripts/claix_b24_release.py \
  --plan "$NEW/_recovery/plan.json" --output "$REVIEW/release.DRAFT.json"
```

After explicit release/recovery review, validate without training. Supply the
reviewed lineage/tool hashes printed by `prepare`. If using optional staging,
add `--staged-cache "$LOCAL_CACHE_PARENT/xr1-cache-${SLURM_JOB_ID}"`; it will
revalidate the same-job completion marker and all staged bytes.

```bash
: "${RECOVERY_LINEAGE_SHA256:?Set reviewed lineage checksum}"
: "${RECOVERY_TOOLS_SHA256:?Set reviewed tools checksum}"
"$XR1_PYTHON" -B reliability/recover.py check \
  --new-run "$NEW" --lineage-sha256 "$RECOVERY_LINEAGE_SHA256" \
  --tools-sha256 "$RECOVERY_TOOLS_SHA256" \
  --recovery-review "$REVIEW/recovery.REVIEWED.json" \
  --release-record "$REVIEW/release.REVIEWED.json"
```

This prints a quoted dry-run command for the unchanged launcher; it omits
`--execute`. Future authorized production continuation is a separate operator
action: use that exact plan/seed/release in a one-H100 allocation with the original
23:45:00 limit and 1800-second checkpoint margin. Restore the allocation's actual
CUDA visibility in that production shell; the CPU-validation shell intentionally
hides it. Retain the existing launcher stop-file, lock, monitoring and storage
checks. Review startup provenance, restored step 8750/offset 210000 and the first
new interval receipt. Nothing here authorizes or performs that allocation.

For passive observer CPU qualification, run it against a disposable CPU replay
parent PID and use a new report filename. Identify the parent and workers with
the read-only process listing and the probe's `workers.json`:

```bash
ps -u "$USER" -o pid,ppid,comm,stat,wchan:32,etime,args
: "${CPU_PROBE_PID:?Set the CPU replay parent PID from the same allocation}"
"$XR1_PYTHON" -B reliability/observe.py --pid "$CPU_PROBE_PID" \
  --seconds 600 --output "$REVIEW/process-observer-${SLURM_JOB_ID}.json"
```

Do not send `SIGUSR2` to production workers: the original trainer does not
register a handler for it. If a stall recurs, preserve fatal telemetry, stderr,
controller/observer reports, Slurm accounting, latest receipt and original seed.
Do not append a rollback to a target whose telemetry extends beyond its last
checkpoint. Allow the existing failure path to terminate; do not synthesize a
checkpoint from partially processed state. Re-audit the newest verified payload
and create a new reviewed recovery branch/root. A clean graceful stop whose
telemetry exactly reaches its verified checkpoint can use the unchanged normal
same-root resume guard. This incident-specific preparer deliberately accepts
only the original 8750 parent; later incident lineage requires a new explicit
review, not relaxing its pinned constants.
