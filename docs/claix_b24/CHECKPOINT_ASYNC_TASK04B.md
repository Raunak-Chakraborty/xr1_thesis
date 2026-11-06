# Task 04B — optional asynchronous checkpoint-500

Source implementation and isolated CPU transaction tests are complete. Overall
acceptance is **BLOCKED**: the available environment lacks Lightning, Hydra and
mmengine. Real Trainer/DDP capture, CUDA staging, Lustre durability/performance,
and the retained historical job evidence still require CLAIX acceptance. No
Slurm submission, GPU training, dependency installation, dataset/weight download,
browser, commit or push was performed.

```text
TASK04B_IMPLEMENTATION=PASS
TASK04B_CPU_TESTS=BLOCKED
TASK04B_FAILURE_INJECTION=PASS
TASK04B_SYNC500_FALLBACK=PASS
TASK04B_TWO_WAY_BENCHMARK=PREPARED
TASK04B_CLAIX_ACCEPTANCE=PENDING
TASK04B_GPU_JOBS=0
TASK04B_RESULT=BLOCKED
```

`IMPLEMENTATION=PASS` describes the implemented source, not qualification for
training. `FAILURE_INJECTION=PASS` describes the executed local CPU faults below,
not kill/power-loss testing on Lustre. `CPU_TESTS=BLOCKED` deliberately includes
the unexecuted real-framework requirement even though the isolated suites pass.
The prepared batch job must not be interpreted as approval to submit it.

## Scope and source audit

Worktree: `/home/horus/robotics_thesis/xr1_claix/worktrees/xr1_checkpoint500_async_v1`.
Starting and final HEAD: `1e528af3ef370f2aa2c46d4e418f185dcb48be32`; initially clean.
All changes are here. The original training directory and Task 04A worktree were
not modified. Task 04A's documentation, tests, YAML and wrapper are unchanged.
Its pre-existing real-framework acceptance blocker is not reclassified as PASS.

Read Task 04A's complete contract and evidence, the production/experimental
launcher and wrappers, environment bootstrap, checkpoint transaction and
retention code, runtime and entrypoint, experiment/recovery identity checks,
compact model save/load hooks, replay contracts, scientific configs and related
CPU fixtures. Ran the original 108-test focused suite before editing.

The synchronous `ProductionRuntime`, `VerifiedCheckpointIO`, checkpoint
transaction, retention, FP32 runner, sampler and original 250/500 configurations
are byte-identical. Only three existing files changed: the entrypoint selects a
separate async runtime/plugin; experiment binding recognizes its isolated
identity; the launcher recognizes the dedicated B/C batch contract. Defaults
still select sync-250. Task 04A's selector still selects its original sync-500
recipe and exact original wrapper. There is no async-to-sync retry path.

## Architecture and ownership

The writer is a **single non-daemon CPU thread**, created lazily for each request,
with **one in-flight snapshot and no waiting queue**. A thread avoids forking an
initialized CUDA/DDP process, multi-gigabyte IPC/shared-memory ownership, child
supervision and inherited-lock lifetime problems. It can overlap blocking POSIX
I/O with training. It does not provide process-level fault isolation or guarantee
serialization will release the GIL enough to improve throughput.

A spawn-based CPU process could isolate serialization better but would require
another reviewed transport, shared-buffer ownership and death/lock protocol.
Fork after CUDA initialization is unsuitable. This implementation deliberately
supports only the existing **single-rank** contract; multi-rank execution fails
before capture. It does not claim to generalize collective error handling.

The training thread performs these operations at the inherited completed
optimizer boundary:

1. Poll earlier writer errors at batch start and end. Before another save, join
   and acknowledge the prior transaction **before Lightning captures new state**.
   No checkpoint is dropped or coalesced, including historical pins.
2. Capture this rank's Python/NumPy/torch/CUDA RNG. Invoke the real
   `trainer.save_checkpoint(weights_only=False)` on the training thread. All
   Lightning hooks, loop capture, strategy work and barriers stay on this thread.
3. In the async CheckpointIO, recursively detach and copy **every** tensor to
   private CPU storage, including tensors already on CPU, views, FP32 model
   weights, optimizer moments/step tensors, loop and scheduler tensors and RNG
   tensors. NumPy RNG arrays and nested containers are copied too. Dtype is
   preserved. OrderedDict state metadata is preserved. Opaque/custom objects,
   object arrays/scalars and unsupported tensor subclasses/layouts fail closed.
4. Copies are blocking, unpinned and `non_blocking=False`. Training cannot update
   the source before staging returns. There are no deferred CUDA events or live
   CUDA references in the handoff. Validate the owned state, expected config and
   experiment; require increasing steps, a matching filename and world size one.
5. Hand the private snapshot to the thread and emit `checkpoint_staged` with
   `verified=false`. `last_saved` is **not** advanced. The writer is the sole
   owner; it only reads the state. “Immutable” here means no alias to any mutable
   training state and no writer mutation, not hardware read-only tensor pages.

The compact representation remains complete for this model: every trainable
weight and optimizer state is included, with frozen weights reconstructed by
the unchanged loader from the SHA-bound original pretrained checkpoint. No new
partial-state format or frozen-weight reconstruction shortcut was introduced.

The background thread runs the unchanged transaction: validate, serialize to a
temporary file, flush/fsync, full CPU read-back and validation, hash exact bytes,
no-clobber payload publication and directory fsync, exclusive verified receipt,
atomic `latest.json`, optional safe-stop pointer, then unchanged verified
retention. It never touches Trainer, the model, a CUDA stream, distributed
collectives, signals or telemetry. A handoff box is emptied on entry so Python's
Thread argument tuple does not retain staging through retention's extra reads.

Only after the entire transaction and rotation succeed does the training thread
acknowledge it, emit `checkpoint` with `verified=true`, and advance `last_saved`.
That row uses the **saved step**, with a separate `trainer_step_at_ack`; it cannot
mistakenly label step 500 as the later step when the writer finishes. The inherited
runner hook message `XR1_COMPACT_REPLAY_SAVED=PASS` describes payload construction
before IO, as it already did in sync mode; it is not verification evidence. Only
the transaction receipt and successful completion acknowledgement establish that.

## Stop, errors, recovery and retention

Safe stop and natural completion inherit the same immediate boundary triggers.
An off-interval stop at 751 first drains an in-flight 500, then captures and
waits for the actual 751 state. It cannot substitute the old 500 snapshot.
Terminal 48284 likewise drains and commits its own completed boundary. The
1500-step benchmark uses the existing `production.stop_after_step=1500`, leaving
trainer/data/scheduler horizons at 48284. Fit-end joins and closes the writer,
requires the final saved step, and performs the original full verified load.

Exceptions are sticky and propagate with their cause on collection, the next
batch boundary, close and teardown. No failure emits a successful writer
acknowledgement. Callback exception cleanup and the entrypoint's outer `finally`
both join the writer; loader shutdown is still attempted if close raises.
No background writer can outlive a gracefully exiting training process.

Joining intentionally has **no abandon-and-continue timeout**. Python cannot
safely kill a thread stuck in a filesystem call. A permanently stuck Lustre call
can prevent graceful completion until the allocation terminates. Consequently
the existing 1800-second stop margin is preserved but **not yet demonstrated
sufficient** for a prior write plus final write, read-back and rotation. Accepting
that tail bound on CLAIX is mandatory. Threads vanish with a hard-killed process;
hard kill does not imply a verified checkpoint or successful graceful shutdown.

Failures after link/replace can leave visible artifacts. The previous usable
payloads are retained; the run fails and requires review:

| Fault point | Permitted artifacts; never treated as successful completion |
| --- | --- |
| Partial serialization, initial file fsync, read-back validation | No new payload/receipt/pointer; local caught failures clean temporary files |
| Payload-directory fsync | Payload orphan may exist without receipt |
| Receipt file/link/directory failure | Payload and possibly visible receipt; no successful writer acknowledgement |
| Latest replacement/fsync | Verified payload/receipt may exist; latest may be old or visibly replaced but durability unacknowledged |
| Safe-stop pointer | Latest payload may be verified; requested stop is not acknowledged |
| SHA corruption found by rotation | Newest pointer may reference corrupt bytes; old usable payload remains; no rollback/promotion/retry |
| Retention failure | New receipt may already exist; no successful callback acknowledgement; old recovery payloads remain protected |

Receipt presence is not used to conceal a failed transaction. No incomplete
serialization passes read-back or obtains a receipt. A real kill can leave hidden
temporary orphans; no cleanup claim is made for abrupt termination or power loss.

Historical pins 2000/5000/10000/20000/30000/40000, the safe-stop pin and at least
two usable recent recovery states use the unchanged verified retention policy.
The benchmark retains receipts for 500, 1000 and 1500; normal retention can
delete **500's payload** after 1500. This is intentional. The preparation verifies
all receipts and all still-retained payloads, requiring 1000 and 1500, and uses
the successful checkpoint rows as evidence of 500's original full verification.

Async identities include `checkpoint_writer=async_thread_v1`, interval 500 and
the distinct name `xr1_doosan_b24_checkpoint500_async_v1`. They are bound into the
resume fingerprint, payload and receipt. Same-run recovery still requires the
same UUID, root, source tree/commit, complete launcher plan, latest receipt,
config, dataset/cache identity and exact telemetry cursor. Sync and async cannot
resume each other's output. Source changes require review as before.

The existing no-silent-rollback rule remains: telemetry ahead of latest blocks
automatic same-run recovery. Async can increase crash exposure beyond 500 steps:
with a sufficiently slow writer it can approach two intervals before the next
request's backpressure completes. A safe stop waits for its exact boundary; an
abrupt crash does not get that guarantee. No source-identity or recovery bypass
was added to hide this distinction.

## Preserved scientific configuration

The new YAML inherits Task 04A and changes only the named variant/output.
Tests compare complete resolved model and data dictionaries and the complete
trainer dictionary except the experiment name. Launch planning preserves:

- Original Xiaomi pretrained SHA256
  `94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7`, language SHA256
  `21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d`.
- Frozen train inventory `1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548`:
  255 episodes, 289703 frames; unchanged validation/test splits and audit logic.
- Last one transformer block policy, FP32 trainable storage, BF16 mixed compute,
  batch 24, four workers, prefetch two, persistent workers, ordered deterministic
  sampler seed 42, action horizon 30, no force inputs, accumulation one.
- Original AdamW and cosine schedule: lr multiplier 1.0, betas 0.9/0.95, weight
  decay 0.1, epsilon 1e-8, warmup 500, start 5e-7, peak 2e-5, floor 5e-6 and
  48284-step data/trainer/scheduler reference horizons.

No real data, split manifest or pretrained asset was downloaded or replaced.
E1–E4 were not implemented.

## Memory and timing budget

Using Task 04A's **observational estimate**, not a new measurement, a compact
payload is about 1.695 GB decimal (1.58 GiB). One staging copy plus one publication
read-back is about 3.39 GB (3.16 GiB). Additional transient finite-value masks,
contiguous copies, Python containers, allocator retention and filesystem cache
need headroom. There is no second queued snapshot and no pinned host allocation.
Staging is released before retention; retention loads one payload at a time.
After both training processes exit, the coordinator's exact final-state
comparison loads two compact checkpoints on CPU, also approximately 3.16 GiB.

Planning envelope within unchanged **96 GiB** allocation: budget **8 GiB** for
checkpoint-related transient/allocator use and reserve another **8 GiB** of
headroom, requiring the measured rest-of-job peak to fit below **80 GiB**. This is
a conservative acceptance target, **not an enforced or measured RSS cap**. Check
cgroup/Slurm peak memory including rank, torchrun, controller and all loaders;
summed process RSS can double-count shared pages. Do not increase CPUs/memory or
change loader science to make the comparison pass. If headroom is insufficient,
reject async.

The sync-500 nine timing fields remain unchanged. Async adds bounded telemetry:

| Field | Meaning |
| --- | --- |
| `state_capture_seconds` | RNG/Lightning/strategy capture before the CPU-copy function |
| `snapshot_staging_seconds` | Blocking recursive ownership copies |
| `snapshot_validation_seconds` | Caller-side validation of staged state and identity |
| Existing transaction phase fields | Prewrite validation; serialization/fsync; full read-back/validation; SHA256; publication |
| `safe_stop_pointer_seconds`, `rotation_verification_seconds` | Same work as sync, in writer |
| `background_writer_seconds` | Entire writer transaction including pointer and rotation |
| `completion_wait_seconds` | Main-thread join/collection wait at acknowledgement |
| `main_thread_blocking_seconds` | Capture/staging/submission/barrier plus completion wait; excludes telemetry append and polling overhead |
| Staged row `backpressure_seconds` | Wait before this request's capture, attributable also to the prior checkpoint's completion wait |

Do not sum overlapping aggregates or count backpressure twice. Main-thread timing
is a bounded proxy, not every nanosecond of callback overhead. Pipeline wait
retains the inherited definition, which includes earlier callback work.

The coordinator records each launcher wall time (startup/cleanup included), fit
wall time/throughput, GPU utilization step-sample mean, CPU and loader RSS sample
maxima, GPU allocated/reserved peaks, phase totals, blocking/backpressure and
paths to the unchanged 15-second GPU monitor. Step samples can miss CPU memory
spikes and are not time-weighted GPU utilization. Use monitor/Slurm/cgroup evidence
alongside them. Three checkpoint samples per leg cannot establish tail latency.

## Prepared two-way CLAIX benchmark — not executed

One new wrapper, `scripts/claix_b24_checkpoint_two_way.sbatch`, requests thes2452,
c23g, one node/task, `gpu:hopper:1`, 12 CPUs, 96 GiB, 23:45:00 and B:USR1@1800.
The launcher verifies the actual saved batch script byte-for-byte. Its former
250/500 wrappers still require their former exact scripts. The new B/C selector
is experimental-only, interval-500-only and fixes stop_after_step to 1500; it
does not permit arbitrary science overrides or bypass any storage/release guard.

The coordinator runs **exactly B then C**, once each:

| Leg | Config | Root beneath new `XR1_TWO_WAY_ROOT` |
| --- | --- | --- |
| B: Sync-500 | `claix_b24_checkpoint500` | `B_sync500` |
| C: Async-500 | `claix_b24_checkpoint500_async` | `C_async500` |

Each launcher exclusively creates a new root and UUID, starts original pretrained
weights at step zero with fresh optimizer/scheduler, uses seed 42 and the same
data order/horizon, and writes independent manifests/checkpoints/logs. No B
checkpoint initializes C. B's verified evidence is checked before C starts.
Signals are forwarded to the active launcher, which retains its boundary-stop
protocol; a signal/failure prevents the next leg. There is no automatic retry.

The coordinator preserves source/plan and results JSON, repeats the existing
allocation, persistent Lustre, user/project quota and cache checks, and retains
each launcher's source/config/dataset attestations and root lock. Source drift
between legs fails. After both finish it verifies science equality and compares
exact final model, AdamW, scheduler, loop, replay and RNG state on CPU. This is a
two-run determinism comparison, not a claim that actual XR-1 resumed execution
has been qualified. No third GPU training run is hidden in verification.

If C fails, preserve its root and partial evidence and B's independent results;
the coordinator reports BLOCKED and recommends abandoning async. Do not rerun it
automatically. A failure of B also prevents C from starting.

For a read-only plan with operator-supplied real paths, **without `--execute`**:

```bash
"$XR1_PYTHON" scripts/claix_b24_checkpoint_two_way.py \
  --python "$XR1_PYTHON" --run-root "$XR1_TWO_WAY_ROOT" \
  --overrides "$XR1_PATH_OVERRIDES" \
  --filesystem-source "$XR1_FILESYSTEM_SOURCE" \
  --project-id "$XR1_LUSTRE_PROJECT_ID"
```

The unsubmitted wrapper additionally requires `XR1_WORKTREE`, explicit
`XR1_EXPERIMENTAL_UNQUALIFIED=1`, existing `XR1_DOOSAN_ZIP_CACHE_ROOT`, the approved
offline environment, persistent submission/work/log paths, and unset `XR1_RESUME`
and `XR1_STOP_AFTER_STEP`. It executes the coordinator only inside the attested
allocation. The retained `two_way_plan.json` is a **synthetic path-only dry run**;
it is not a real storage/quota attestation.

Historical Sync-250 is **job 5504969**, 48129 completed optimizer steps, one H100,
interval 250, approximately 23.26 allocated GPU-hours. Those are supplied
historical facts, not newly measured results. Searches found no retained raw
telemetry/checkpoint evidence for that job in this checkout; Task 04A likewise
records it as supplied context. Historical source paths and hashes were not
invented. Before comparison, locate its retained telemetry, latest/stop receipt,
payload/hash evidence, launcher provenance and Slurm allocation accounting,
attribute them to 5504969, and retain read-only hashes/references with the report.
This evidence attribution remains **BLOCKED/PENDING**. Never launch a new
Sync-250 to fill the gap. Allocated GPU-hours and a full historical run are not
controlled equivalents of two 1500-step fit windows; cache warmth/order, startup,
Lustre contention, run duration and checkpoint retention history differ.

## Actual CPU tests and failure-injection evidence

Interpreter: `/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python`, torch
2.7.1+cu128, CUDA explicitly hidden. No dependencies were installed. Tests use
real torch tensors/AdamW/scheduler/RNG, threads, torch.save/load, hashes and POSIX
file transactions. Allocation/GPU/entrypoint registry fixtures are explicitly
synthetic. The async harness requires explicit `--allow-framework-stubs` to run
without dependencies and prints `TASK04B_REAL_FRAMEWORK=BLOCKED` even then.

| Executed command/check | Result and retained evidence |
| --- | --- |
| Original focused suite before edits | 108 passed, 10.039 s; `task04b_evidence/baseline_cpu.log` |
| Async isolated CPU suite | 43 passed, 5.319 s, no failures/errors/skips; `task04b_evidence/async_cpu.log` |
| Existing sync/launcher/retention regression suite | 108 passed, 10.208 s; `task04b_evidence/sync_regression_cpu.log` |
| Existing dataset/cache/I/O suite | 38 passed, 0.719 s; `task04b_evidence/io_cpu.log` |
| Async harness without stub permission | Exit 2, missing lightning/hydra/mmengine, no framework tests claimed; `task04b_evidence/real_framework_attempt.log` |
| Full existing CLAIX harness | Exit 1 during collection, missing lightning; `task04b_evidence/full_claix_cpu_attempt.log` |
| Static protection/syntax | 60 tracked files byte-identical, 9 Python AST parses, 9 bash syntax checks, git diff check; `task04b_evidence/static_checks.log` |
| Two-way plan, no execute | Exit 0, exactly B/C with distinct roots; `task04b_evidence/two_way_plan.json` |

Commands actually executed (the first was also run before edits):

```bash
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_experimental_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_async_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_async_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_cpu_tests.py
python3 -B docs/claix_b24/task04b_evidence/static_checks.py
python3 -B scripts/claix_b24_checkpoint_two_way.py --run-root /hpcwork/SYNTHETIC/task04b-two-way --overrides /hpcwork/SYNTHETIC/paths.json --filesystem-source SYNTHETIC --project-id 44885 --python /EXISTING_CLAIX_ENV/bin/python
```

The log files contain full test names. Principal executed coverage:

| Requirement | Actual evidence |
| --- | --- |
| Immutable optimizer boundary | Advance a real AdamW/scheduler after enqueue while serialization is blocked; persisted masters, all moments/step, loops and nested RNG remain the old values; storage pointers differ |
| Complete replay/RNG recovery | Restore Python/NumPy/torch RNG and optimizer/scheduler, take another optimizer update, compare exact parameters/moments/schedule; assert replay cursor |
| Ownership/memory | Copy CPU tensor views and OrderedDict metadata; reject opaque/object state; weak-reference test proves staging released before retention |
| Lifecycle/backpressure | Block writer with Events; refuse a second slot; runtime waits before next capture; close/teardown join; no surviving writer thread |
| Failure propagation | Writer failure raises before the next optimizer batch; callback exception and outer entrypoint fit-failure cleanup close the writer |
| Interrupted serialization | Write partial temporary bytes then raise OSError; no new receipt/latest, temporary cleanup, prior usable payload |
| fsync | Fail initial payload file fsync and individually calls 2–6: payload directory, receipt file/directory, latest file/directory; no success acknowledgement, prior payload verifies |
| SHA/read-back | Same-size payload corruption is rejected by rotation/load; malformed full read-back fails before publication |
| Receipt/latest/safe-stop/retention | Fail actual receipt hardlink, pointer replace and retention unlink; sticky error, no fallback/retry, previous payload unchanged |
| Safe stop/final during writing | Hold step 500 writer, request 751 USR1 or terminal 48284; verify both ordered writes, current stop pointer, final full verification and join |
| Final verification | Missing boundary and corrupted post-ack terminal bytes fail fit-end |
| Pins/identity | All historical pins and recent recovery states retained; foreign UUID/root/writer/config/source and telemetry-ahead recovery rejected |
| Launcher/benchmark | Closed selector, exact saved wrapper/resources, quota rejection, unique B/C identities, original initialization, failure/signal prevents next run, no retries |
| Evidence consumer | Real synthetic receipts/retention and CPU final-state comparison; corrupt bytes, missing receipt, changed science or valid-but-different model state fail |
| Sync-500 and sync-250 | Existing suites pass unmodified; source/config/wrapper byte comparisons pass |

During development, the new evidence fixture initially supplied a rank RNG dict
instead of the required list, and a memory-lifetime test initially used a Mock
that retained its arguments. Both test-fixture errors were corrected; the final
43-test run passes. The discovered Thread argument retention issue in the
implementation was independently fixed and is covered by the direct lifetime
test. No existing assertion was removed or weakened.

## Exact remaining CLAIX acceptance requirements

1. In the existing offline environment, run `run_claix_async_cpu_tests.py`
   **without** `--allow-framework-stubs`, the full existing CLAIX CPU harness,
   and the I/O harness without stubs. The new real tests must execute, not skip:
   `AsyncFrameworkTest.test_real_lightning_500_stop751_resume_rng_and_final_equality`
   (four persistent workers), real Hydra composition, and
   `AsyncDDPFrameworkTest.test_real_single_rank_cpu_ddp_checkpoint_io_handoff`
   (real Gloo/loopback). Failure or missing dependencies is BLOCKED. Keep Task
   04A's real-framework tests as mandatory unchanged acceptance too.
2. Review resolved B/C configurations and source digests; verify original weight
   and language hashes, all frozen split identities (including held-out splits),
   ZIP manifest/immutable policy, seed, 48284 horizons and original science.
   Confirm installed DDP exposes the exact IO plugin, all dump/hooks run on the
   caller, actual loop counters pass and each CUDA tensor copy completes before
   optimizer mutation. No opaque framework checkpoint objects may be silently
   dropped; reject unsupported state.
3. Before any separately authorized submission, verify exact wrapper attestation,
   persistent Lustre logs/root/cache, actual user and project quotas, source
   identity and 12 CPU/96 GiB/H100 allocation. Supply genuinely new parent/B/C
   roots. Do not use Task 04A's or historical run directories.
4. In the **only two new GPU runs**, require 1500 completed optimizer steps each,
   verified 500/1000/1500 receipts, fully readable retained 1000/1500 payloads,
   correct latest/safe-stop pointers, exact final state comparison, independent
   identities and no retry. Review every staged/acknowledged cursor and error.
   CPU framework stop/resume covers recovery before submission; actual H100
   resumed-execution qualification is not demonstrated by this two-fresh-run
   benchmark. If mandatory for adoption, leave adoption BLOCKED rather than
   silently adding a third GPU run or changing the fresh-run comparison.
5. Establish peak memory/headroom, writer CPU/GIL contention, D2H staging stalls,
   per-phase and total wall times, GPU utilization/memory and end-to-end
   throughput. Compare matched B/C windows, report startup/cleanup separately,
   and attribute historical 5504969 evidence without claiming a controlled
   full-run versus short-run comparison. A three-checkpoint sample does not
   establish production tails or a 48284-step speedup.
6. Qualify Lustre durability/error handling and worst-case stop drainage within
   1800 seconds (prior writer plus current checkpoint and retention), along with
   cleanup/root-lock lifetime on failures. Local exceptions do not establish
   crash/power-loss or CUDA-transfer correctness. Keep async unqualified if any
   of these cannot be established within the authorized work.
7. If C fails integrity, deterministic comparison, memory, shutdown or throughput
   acceptance, preserve all evidence, recommend abandoning async, and keep the
   unchanged sync-500 implementation as the reliable separate-run fallback.

## Exact changed-file inventory

| File | Change |
| --- | --- |
| `scripts/claix_b24_launch.py` | Dedicated closed B/C profile, fixed budget, exact wrapper attestation and async recovery identity |
| `xr1/mibot/utils/claix_experiment.py` | Isolated async variant/writer binding and same-run recovery name |
| `xr1/tools/train_doosan_compact.py` | Select async only by bound identity; outer close/join before loader cleanup |
| `xr1/mibot/utils/claix_async_checkpoint.py` | Owned CPU staging, bounded writer, explicit completion/error protocol and async callback |
| `xr1/configs/claix_b24_checkpoint500_async.yaml` | Inherited unchanged-science recipe with isolated identity/output |
| `scripts/claix_b24_checkpoint_two_way.py` | Sequential B/C coordinator, evidence validation, metrics and exact CPU final-state comparison |
| `scripts/claix_b24_checkpoint_two_way.sbatch` | Unsubmitted consolidated allocation wrapper |
| `xr1/tests/run_claix_async_cpu_tests.py` | Explicit stub permission and fail-closed real-framework requirement |
| `xr1/tests/test_claix_async_checkpoint.py` | Snapshot, replay, lifecycle, backpressure, stop, retention and leaf-I/O fault tests |
| `xr1/tests/test_claix_async_framework.py` | Required real Trainer/Hydra/Gloo acceptance tests, currently BLOCKED |
| `xr1/tests/test_claix_two_way.py` | Entrypoint/identity/science/launcher/coordinator/evidence regressions |
| `docs/claix_b24/CHECKPOINT_ASYNC_TASK04B.md` | This implementation, evidence and acceptance record |
| `docs/claix_b24/task04b_evidence/baseline_cpu.log` | Pre-edit 108-test execution |
| `docs/claix_b24/task04b_evidence/async_cpu.log` | Final 43-test async/failure-injection execution |
| `docs/claix_b24/task04b_evidence/sync_regression_cpu.log` | Unchanged 108-test regression execution |
| `docs/claix_b24/task04b_evidence/io_cpu.log` | 38-test I/O regression execution |
| `docs/claix_b24/task04b_evidence/real_framework_attempt.log` | Explicit missing-dependency BLOCKED result |
| `docs/claix_b24/task04b_evidence/full_claix_cpu_attempt.log` | Existing full harness collection failure |
| `docs/claix_b24/task04b_evidence/static_checks.py` | Reproducible source protection/syntax checks |
| `docs/claix_b24/task04b_evidence/static_checks.log` | Actual static results |
| `docs/claix_b24/task04b_evidence/two_way_plan.json` | Read-only synthetic two-leg plan; no execution |
