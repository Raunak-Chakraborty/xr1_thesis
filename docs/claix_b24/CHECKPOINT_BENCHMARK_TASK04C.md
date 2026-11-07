# Task 04C — benchmark hardening and CPU acceptance

The independent code audit and local hardening are complete. **Submission readiness
is BLOCKED**: real CLAIX framework acceptance, actual reviewed asset/split/cache
identities, current accounting evidence, and Lustre shutdown qualification remain
pending. No Slurm command that allocates/submits work, GPU training, installation,
download, commit, push, or historical checkpoint modification was performed.

```text
TASK04C_AUDIT=PASS
TASK04C_HARDENING=PASS
TASK04C_CPU_TESTS=PASS
TASK04C_SYNC500_PROTECTED=PASS
TASK04C_ASYNC500_PROTECTED=PASS
TASK04C_CPU_CLAIX_ACCEPTANCE=PREPARED
TASK04C_GPU_PREFLIGHT=PREPARED
TASK04C_TWO_WAY_BENCHMARK=PREPARED
TASK04C_GPU_JOBS_SUBMITTED=0
TASK04C_RESULT=BLOCKED
```

Here PASS for CPU tests means the bounded laptop suites in the execution table;
PREPARED means a reviewable procedure exists, not that its acceptance gates passed.
Neither framework stubs nor earlier Codex statements qualify the installed CLAIX
framework. There is no real CLAIX PASS receipt in this change.

## Provenance and independent audit

Started clean at `c3b2c6895fe2d3bfd19c3d301ef5b49a02841686`, branch
`thesis/xr1-checkpoint-benchmark-hardening-v1`. Task 04A is
`1e528af3ef370f2aa2c46d4e418f185dcb48be32`; the successful historical Sync-250
source is `0106f1a0e0dfbcbf18e9732f4924b8ccd9e937f0`. Job 5504969 and its
48,129 completed optimizer steps are supplied historical context. Retained Task
04A/B evidence contains local tests and missing-framework failures, not raw
historical job telemetry or checkpoint payloads. Historical attribution remains
PENDING. No new Sync-250 benchmark is authorized.

Read the Task 04A/B reports and retained logs, transaction/runtime/experiment/
async code, training entrypoint, launcher/coordinator/wrappers/environment,
inherited Hydra recipes, compact runner/save hooks, replay and scientific
contracts, storage/retention guards and their tests. The retained evidence index
records original hashes and test summaries. Original evidence and reports remain
byte-identical. The local static check independently compares protected files
against the exact starting commit; it does not reuse old PASS statements.

| Path / invariant | Independent finding and treatment |
| --- | --- |
| Completed optimizer boundary | Runtime requires exactly one completed step per batch, synchronizes CUDA, then snapshots at batch-end. Validator requires optimizer, scheduler, replay and Lightning batch counters at that exact boundary. Fit-end cannot create a checkpoint after loop counters advance. Real installed loop semantics remain mandatory CPU acceptance. |
| Lightning/DDP ownership | `Trainer.save_checkpoint`, dump/hooks, strategy barrier and all public IO control stay on the training thread. Only owned CPU state enters the worker. IO now binds the exact plugin instance; replacement fails. Exception cleanup drains the originally bound owner even if the strategy replaced its plugin. Real Gloo case additionally checks capture thread, saved steps and actual strategy teardown. |
| Tensor/Parameter/optimizer aliasing | Ordinary state_dict and optimizer state can alias live storage. Staging detaches and copies every CPU/CUDA tensor, Parameter, NumPy array, nested container and OrderedDict metadata. Blocking D2H copies finish before return; unsupported state fails. No asynchronous copy or CUDA work is delegated. Local tests mutate live model/moments after handoff and verify the saved old values. CUDA itself remains unqualified here. |
| Single flight | Prior result is joined/collected before the next capture. There is one snapshot, one writer, no second queue, retry, coalescing or fallback. At a stop during step-500 writing, the prior write drains before capturing the actual current boundary (e.g. 751). |
| Exception memory defect — fixed | Clearing only the outer worker traceback left chained inner publication frames owning staged tensors. All cause/context tracebacks are now detached, with cycle protection. Failed staging/thread startup stores a separate sticky diagnostic, avoiding a traceback back into the live capture. Weak-reference and startup fault tests cover this. |
| Controller cleanup defect — fixed | An experimental launcher exception could exit and remove its stop request while its child was alive. Cleanup now persists a stop request/evidence and reaps before releasing ownership. It never cancels a Python writer thread. |
| Unexpected controller / agent death — fixed | Launcher watches coordinator PID plus Linux start ticks; rank checks launcher identity at completed boundaries. B uses a benchmark-only synchronous callback; C uses its async callback. PID reuse/zombies fail liveness. A separate rank-owned `.checkpoint-writer.lock` prevents recovery while a writer survives controller/torchrun death. Recovery probes under the original root lock. Lock files are never unlinked. |
| Lock lifetime | Async closes the rank lock only after joining its writer, including errors. B closes after final verification or exception. Forked loader children can inherit the descriptor and conservatively extend lock lifetime until loader/process cleanup; a surviving helper must be reaped before recovery. Slurm cgroup/process cleanup still needs site qualification. |
| Close/teardown | Repeated successful close/teardown is safe; failure stays sticky. On exception, the original writer drains before callback cleanup. Entry point still has an outer async close before loader shutdown. Writer threads remain non-daemon. Strategy/loader lifetime in the actual installation is a required test, not inferred from the synthetic adapter. |
| Publication | Unchanged sync transaction: structural validation; temporary serialization/flush/file fsync; full CPU read-back/validation; SHA256; no-clobber payload link and directory fsync; exclusive receipt; atomic latest and directory fsync; optional stop pointer; verified rotation. No receipt/pointer precedes verification. Final payload is fully loaded and hash checked again. |
| Failure ordering | Every injected failure is fatal. A failure after a link/replace may leave an orphan, verified receipt or newer visible pointer without acknowledgement. These artifacts do not authorize success, retry, overwrite or automatic promotion. Existing usable payloads are retained. |
| Retention/recovery | Latest must name the newest receipt; safe-stop must name a verified retained payload. Two usable recent recovery states and historical pins are protected before deletion. Verification budget remains eight. Corruption, inconsistent pointers and budget exhaustion fail closed. Same-run recovery binds root, UUID, commit/source, full plan/config, writer, cache manifest and exact telemetry cursor. Telemetry ahead of latest blocks rollback. |
| Full verification limits | Read-back validates all required tensors/state structure and finite values; the byte hash binds the committed payload to its receipt. It is not an in-memory-versus-read-back bitwise tensor comparison. The final B/C comparison is exact for scientific state. No pickle payload from an untrusted source is authorized. |

### Shutdown and memory limits

**There is no guaranteed 1800-second graceful shutdown bound.** A Python
non-daemon thread in uninterruptible Lustre I/O cannot be safely killed. `close`
continues joining; it never marks a timeout closed or silently abandons an
unverified write. The launcher/coordinator can consequently wait indefinitely
inside the allocation. This is deliberate ownership retention, not a bounded
join implementation. Slurm's final hard kill/cgroup cleanup is the external last
protection; it is not graceful checkpoint success and can leave partial artifacts.
Even synchronous fsync/read-back or a stalled CUDA boundary can exceed a margin.

The bounded part is the request protocol: at deadline minus 1800 seconds or
USR1, stop at the next completed boundary, drain at most one prior writer, then
publish/verify the exact stop boundary and clean up. Completion time of those
operations is unbounded by POSIX. Qualify representative worst-case training
batch, D2H snapshot, prior writer, current publication, repeated retention reads,
final full read/hash, loader/agent cleanup and signal delivery on actual Lustre.
Include slow and interrupted I/O on disposable qualification files only. If the
1800-second operational target/headroom cannot be established, keep adoption
BLOCKED; a local Event released after 50 ms proves join ordering, not that bound.

At the supplied 1.695 GB payload estimate, staging plus read-back is roughly
3.39 GB. Noncontiguous copies, validation masks, exception state, allocator and
filesystem cache add transient memory. Staging is released before rotation;
rotation reads one payload at a time. Final comparison loads two payloads only
after both training processes exit. Retain the Task 04B planning target of 8 GiB
checkpoint transients plus 8 GiB spare, rest-of-job below 80 GiB within 96 GiB.
This is not a measured/enforced RSS limit. Measure Slurm/cgroup job peak; summed
process RSS may double-count shared pages and per-step samples miss spikes.

## Preserved science and two-way contract

All Hydra scientific recipes, original synchronous runtime/transaction,
Task 04A wrapper and historical experimental/production wrappers are unchanged.
The existing last_n=1 policy, original Xiaomi weight/language hashes, 255 training
episodes / 289703 frames, held-out 34/34 split identities, batch 24, four ordered
persistent workers, prefetch two, seed 42, action horizon 30, no force input,
FP32 trainables/BF16 frozen storage and compute, AdamW, original cosine schedule
and **48284** data/trainer/scheduler horizons remain intact. Preflight compares
complete protected recipe bytes, composes real Hydra, inventories frozen splits
without evaluating held-out labels, and hashes original required assets.

| Resource/identity | Exact contract |
| --- | --- |
| Account / partition | thes2452 / c23g |
| Allocation | one node, one task, exactly 12 CPUs, 96 GiB, `gpu:hopper:1`, actual one visible NVIDIA H100 |
| Time / signal | **23:45:00**, `B:USR1@1800` |
| Saved script | byte-for-byte `scripts/claix_b24_checkpoint_two_way.sbatch`, plus live allocation/resource checks |
| B | `claix_b24_checkpoint500`, sync, `<NEW parent>/B_sync500`, reviewed distinct UUID |
| C | `claix_b24_checkpoint500_async`, async, `<NEW parent>/C_async500`, reviewed distinct UUID |
| Stop / schedule | both `production.stop_after_step=1500`; save 500/1000/1500; schedule remains 48284 |
| Initialization | each a new process/model/optimizer/scheduler from original pretrained step zero; no B-to-C checkpoint input |
| Execution | exactly B then C once, no retry, no additional GPU warmup/run, no Sync-250 run |

Six hours was investigated and **not adopted**. Scaling supplied historical
23.26 GPU-hours by 3000/48129 gives about 1.45 hours, but this is not retained raw
matched-window performance evidence. Supplied mean checkpoint cost 19.64 seconds
and payload size provide no startup/hash, DataLoader, Lustre tail, full comparison
or shutdown upper bound. No defensible bound for these costs and the 1800-second
reserve is available. Keeping the existing limit follows the task's explicit
fallback rule. Tests reject 06:00:00, 23:44:00 and other wrong allocations, alongside
wrong wrappers/resources; no arbitrary walltime check was relaxed. The historical
23:45:00 experiment and Task 04A allocation are also unchanged.

Coordinator verifies B before starting C. Failure/interruption in B prevents C;
C failure preserves B and all partial C evidence. Signals are forwarded to the
active launcher, which uses its normal boundary-stop request. The first pending
signal is retained in bounded state. Exceptional cleanup terminates the launcher
by its handled TERM request and reaps it; a blocked writer still requires the
external hard-kill protection above. Real tiny child-process tests exercise
forwarding/reaping. Nonzero child codes propagate to the batch exit status;
negative signal exits become 128+signal. Any final verification/equality failure
is nonzero and records BLOCKED, including two zero child exit codes followed by
an RNG mismatch. Signals/source drift during final comparison also prevent success.

Final equality retains zero tolerance for model, AdamW, scheduler, loops, replay
and all rank RNG state, and now also compares epoch/global step, replay contract
and compact provenance. Exact tensor dtype/shape and scalar types are required.
Roots, UUIDs, experiment/config fingerprints, checkpoint names, timings and
telemetry timestamps legitimately differ because outputs are independent; those
are verified against each leg's reviewed identity instead of compared equal.
No scientific state field was removed from equality. Fresh B/C loop progress is
expected equal; resumed loop `current` counters may differ from an uninterrupted
run, so resume tests compare consumed samples, restored optimizer/schedule,
model and RNG while validating each checkpoint's boundary counters independently.

Timing aggregates overlap: callback duration contains synchronous subphases;
async main-thread blocking contains capture/submission and completion wait;
backpressure is also the prior checkpoint's wait. Keep each field's phase total
but never sum them into another total. Background writer time overlaps training.
`pipeline_wait_seconds` includes prior callback work. Launcher wall time includes
startup/hashing/cleanup, fit time includes loader startup/checkpoints. Step GPU
utilization is sample-based, not time-weighted. Coordinator also parses independent
15-second NVIDIA monitor samples; absence explicitly blocks performance review.
Three checkpoint samples cannot establish tails or a full-horizon speedup. B-first
OS page/cache warming can benefit C; no hidden warmup is added to disguise it.

## Dedicated real CLAIX CPU acceptance

Prepared files:
`scripts/claix_b24_task04c_cpu.sbatch`, `scripts/claix_b24_task04c_cpu.py`,
`xr1/tests/claix_cpu_report.py`. The wrapper uses the existing qualification
partition **c23ms**, thes2452, zero GPUs, 12 CPUs, 96 GiB, four hours. This partition
comes from the retained CPU wrapper; current site/account eligibility has not
been queried here. Obtain an operator-authorized CPU allocation separately after
checking eligibility. Nothing in this task submits it.

The wrapper sources the unchanged `scripts/claix_b24_environment.sh`: existing
`module load Python/3.11.5` and `$HOME/robotics_thesis/configs/xr1_hf_offline.env`.
It requires that setup's approved existing `XR1_PYTHON`; no Python/conda location
is invented. Set `XR1_WORKTREE` to the exact reviewed checkout and
`XR1_CPU_EVIDENCE_ROOT` to a NEW absolute persistent Lustre directory incorporating
job ID and a unique suffix. Use a persistent submission/work directory so unique
`xr1_task04c_cpu_%j.out/.err` are retained. No heavy tests run on a login node.

Inside that CPU allocation, the wrapper executes:

```bash
"$XR1_PYTHON" -B "$XR1_WORKTREE/scripts/claix_b24_task04c_cpu.py" \
  --output "$XR1_CPU_EVIDENCE_ROOT"
```

It validates the live allocation and saved CPU wrapper, explicitly hides CUDA,
imports the real Lightning/Hydra/PyTorch/mmengine packages, requires real Gloo,
records interpreter/module paths and installed versions, and executes **all four**
existing harnesses in order, each in a separate process:

```text
xr1/tests/run_claix_async_cpu_tests.py
xr1/tests/run_claix_experimental_cpu_tests.py
xr1/tests/run_claix_cpu_tests.py
xr1/tests/run_doosan_io_cpu_tests.py
```

There are no `--allow-framework-stubs` flags in this procedure. Existing small
backbone/registry test fixtures in the full harness remain explicit fixtures;
Lightning, Hydra, PyTorch, mmengine and actual framework acceptance cases are
real. This is tiny-model framework qualification, not CPU training of the 5B XR1.
Preflight separately checks required offline Qwen config/processor assets.

Keep each full `.log` and per-test `.json` plus `cpu-acceptance.json`. Console output
is one count/failure/error/skip summary per harness. Every unittest ID/outcome is
recorded; any missing harness, nonzero exit, empty result, failure, error, skip,
expected-failure, changed log/result hash, missing mandatory real case or source
change makes acceptance BLOCKED. All test IDs remain in detailed evidence.
Required cases include Sync-500 real Lightning stop/resume, Async-500 stop751/
resume equality, four-worker replay, real single-rank Gloo/DDP ownership and
teardown, real Hydra composition, checkpoint read-back, optimizer/scheduler and
full RNG equality, new normal 1500-step sync/async termination/equality, and the
existing Sync-250 regressions. No case is silently skipped when unavailable.
A timed-out/killed job without a final report cannot authorize preflight.

Run against a clean final reviewed commit and keep evidence outside the source
checkout; preflight rejects dirty-source or different-commit acceptance reports.
This session intentionally leaves all edits uncommitted, so this checkout is not
currently eligible. An authorized reviewer must establish the final immutable
commit later; this session does not commit it.

## Fail-closed preflight and reviewed inputs

`claix_b24_task04c_preflight.py` has no submission functionality. Copy
`task04c_review.template.json` outside the checkout, fill every null from actual
reviewed CLAIX evidence, then independently review and pin that JSON's SHA256.
The template cannot pass. Do not substitute synthetic paths from laptop fixtures.

Use the active branch above and the **final reviewed Task04C commit**, descending
from the starting commit; the unmodified parent cannot authorize these changes.
Bind the exact source digest (existing `claix_b24_launch.tree_digest()`), actual
`XR1_PYTHON`, existing four-path overrides and hash, canonical NEW parent run root,
distinct fresh lowercase 32-hex B/C UUIDs, actual filesystem SOURCE/project ID,
existing ZIP root/manifest hash, actual split roots and reviewed hashes, CPU report
and hash, exact two-way wrapper hash, and existing persistent Lustre log directory.
Approved `/hpcwork` aliases in staged input paths remain usable; the run root is
canonical because checkpoint identity uses its resolved value. No historical root
is accepted as NEW. Model/language paths and expected hashes come directly from
the protected composed recipe and existing overrides, not invented review fields.

In an authorized compute allocation, with the approved environment sourced:

```bash
"$XR1_PYTHON" -B "$XR1_WORKTREE/scripts/claix_b24_task04c_preflight.py" \
  --review "$XR1_BENCHMARK_REVIEW" \
  --review-sha256 "$XR1_BENCHMARK_REVIEW_SHA256"
```

Retain stdout/stderr in a unique persistent preflight log. Unknown/missing inputs
or unavailable queries fail CLOSED. Checks include clean tracked/untracked Git
state, exact tree/branch/commit and protected science, real environment matching
CPU acceptance, all real test IDs/evidence hashes, original model/language hashes,
frozen 255/34/34 split inventory/leakage checks, actual 646-entry ZIP manifest,
sidecars and ZIP member directories using the real runtime reader, ordered
four-worker settings and original schedule, nonexistent parent/B/C roots, storage
and user/project quotas, exact batch contract, live account jobs and saved scripts.
Any active two-way benchmark is conservatively rejected even if its job name was
overridden; inaccessible accounting/job metadata also blocks. Current job is excluded.

Preflight checks the actual locally cached Qwen3-VL configuration and processor
with original special-token IDs. It does **not** require Qwen backbone safetensors,
raw source videos, decoded PNG trees or another cache format: the actual runtime
constructs its backbone from config and loads original XR1 weights. ZIP manifests
bind approved row identities; runtime-compatible sidecar/directory validation is
not a full hash of every archive. External immutability of ZIPs/sidecars/episode
JSONs for worker lifetimes remains the original operational contract; CRC checks
still happen during runtime reads. Held-out inventories are hashed, not evaluated.

Two-way storage requires 60 GiB and 20000 free filesystem/user/project inodes
(where quota limits exist), verified inherited project accounting and mandatory
numeric user/project quota parsing. This is additional conservative preflight
headroom for both roots; original per-leg allocation storage gates remain intact.

Remaining project CPU/GPU allowance is **not** the Lustre storage quota. No
verified site project-compute budget parser is retained here, so none was invented.
Use `task04c_project_budget.template.json` only after a named reviewer reads fresh
authoritative site accounting, retains its exact raw output and hash, and converts
it to CPU-hours/GPU-hours with documented units. The review pins this artifact and
raw evidence; it must be at most 15 minutes old when checked and show at least
285 CPU-hours and 23.75 GPU-hours for the unchanged full request. This is an
explicit human-reviewed accounting gate, not a claim of an independent automatic
site-budget measurement. If the site cannot supply trustworthy matching quantities,
leave the gate BLOCKED and qualify a site accounting adapter before submission.
No fabricated values or old screenshots qualify it.

The two-way wrapper now also requires `XR1_BENCHMARK_REVIEW` and
`XR1_BENCHMARK_REVIEW_SHA256`. Coordinator reruns preflight, checks CLI root/Python/
overrides/filesystem/project against that exact review, assigns the pinned B/C
UUIDs, retains `two-way-preflight-review.json` and its path/hash in the plan, then
repeats live strict allocation and per-leg storage checks. In the GPU allocation,
actual stdout/stderr/work directories must equal the reviewed log directory.
There is no GPU submission command or automatic retry in this flow.

## Actual local execution and failure injection

Local interpreter:
`/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python` (existing environment,
torch 2.7.1+cu128), always `CUDA_VISIBLE_DEVICES=''`, `-B`. This is the laptop
interpreter, **not an invented CLAIX XR1_PYTHON**. Evidence is under
`task04c_evidence/local_20261107_01/`. Exact commands, exit codes, log hashes and
summaries are retained in `local-results.json`; full test names/errors remain in
logs. The final table is updated from actual final execution below.

| Executed command (all with the interpreter/CUDA settings above) | Actual result |
| --- | --- |
| `xr1/tests/run_claix_async_cpu_tests.py --allow-framework-stubs` before edits | 43 passed, exit 0 (`baseline_async.log`) |
| Same async/Task04C suite after hardening | See `final_verification_async.log` / local-results.json for final count |
| `xr1/tests/run_claix_experimental_cpu_tests.py` | 108 passed, exit 0; final verification retained separately |
| `xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs` | 38 passed, exit 0, 0.708 s (`io_local.log`) |
| `xr1/tests/run_claix_async_cpu_tests.py` without stubs | exit 2, lightning/hydra/mmengine unavailable; zero framework tests executed |
| `xr1/tests/run_claix_cpu_tests.py` | exit 1 during collection; missing Lightning; no complete framework suite executed |
| `xr1/tests/run_doosan_io_cpu_tests.py` without stubs | exit 1, three failed import placeholders, missing Lightning/mmengine; no real I/O cases executed |
| `python3 -B docs/claix_b24/task04c_evidence/static_checks.py` | Actual protected-byte, Python AST, shell syntax and diff checks in `static.log` |

Development logs are retained too: first hardening run had three occurrences of
one regression (an armed-slot change broke an existing validation test); restored
consumption behavior fixed it. Next run had a fake child PID fixture serialization
error; fixture now provides a real integer PID. Later runs passed. These failed
iterations are not presented as final results or hidden by replacing their logs.

Added/strengthened coverage includes normal 1500 sync/async stop with final
read-back, real USR1 during active writing, nested-exception tensor lifetime,
thread startup and every publication stage (including payload link/hash read,
file/directory fsync, receipt/latest/stop pointers and retention), stuck-writer
join limitations, owner/plugin identity, controller/PID loss and rank lock,
root/source/foreign-resume rejection, exact/wrong/shortened wrapper resources,
incomplete/tampered/skipped CPU evidence, missing assets, storage/inode/quota and
project-budget failures, dirty source, changed optimizer/scheduler/data recipes,
real child forwarding/reaping, B failure preventing C, C failure preserving B,
reviewed UUID binding, strict final state/RNG comparison and return-code propagation.
Site-facing preflight unit tests explicitly inject site measurements; these are
not actual mount/quota/framework acceptance receipts.

## Submission readiness, fallback and evidence handoff

Before any separately authorized submission, all of the following must hold:

1. Final reviewed immutable clean checkout/commit and complete exact input review;
   this uncommitted session is not eligible. All mandatory CLAIX CPU cases must
   actually execute successfully in the approved environment with no skips/stubs.
2. Actual original asset hashes, split identities, immutable ZIP identity, offline
   processor/config compatibility, canonical new roots, persistent log paths,
   live storage/inodes, fresh project-compute allowance and no conflicting job.
3. Document actual graceful-stop timing/memory qualification and limitations;
   do not claim the 1800-second bound for uninterruptible I/O. Resolve any required
   site durability/controller/cgroup qualification before adoption.
4. Independently authorize the exact two-way GPU allocation later. This task
   neither submits it nor creates an executable submission action automatically.

After the authorized benchmark, retain plan/review, per-launch provenance/source
checksums, allocation/storage reports, resolved configurations/inventories,
initialization records, full stdout/stderr, complete telemetry, independent NVIDIA
monitor logs, three verified receipts per leg, retained 1000/1500 payloads and
matching latest/safe-stop pointers, SHA verification and exact final comparison.
Step-500 payload may be removed by unchanged normal retention after 1500; its
receipt remains. Capture actual Slurm/cgroup accounting, peak memory and exit
states, and attribute historical 5504969 evidence read-only if available. Report
startup versus fit throughput, blocking versus overlapped phases and cache/order
limitations. Do not claim adoption or speedup from transaction integrity alone.

On any C integrity/equality/shutdown/memory/performance failure, preserve B/C
roots and all artifacts, mark BLOCKED, abandon async and retain Sync-500 as the
separate authorized fallback. Do not overwrite/delete payloads, silently retry C,
reuse B as C's initialization, or add a replacement Sync-250 benchmark. Task 04A's
unchanged Sync-500 recipe/wrapper is the fallback; a future run needs its own new
root and review. Failed/orphaned publications require operator recovery review,
not automatic promotion or rollback.

## Exact changed files

Existing files changed:

- `scripts/claix_b24_launch.py` — reviewed two-way UUID option, supervision,
  live-writer recovery probe and child ownership on experimental failure.
- `scripts/claix_b24_checkpoint_two_way.py` — mandatory exact preflight, reviewed
  identities, stronger evidence/state equality, process failure handling and monitor summary.
- `scripts/claix_b24_checkpoint_two_way.sbatch` — review inputs; original resources/time retained.
- `xr1/mibot/utils/claix_async_checkpoint.py` — exception memory, rank lock,
  owner/plugin lifecycle and controller loss.
- `xr1/mibot/utils/claix_experiment.py` — isolated two-way leg/writer/1500 identity binding.
- `xr1/tools/train_doosan_compact.py` — choose benchmark-only B supervisor.
- `xr1/tests/run_claix_async_cpu_tests.py` — include Task04C regressions.
- `xr1/tests/test_claix_async_framework.py` — actual framework/RNG/DDP/1500 acceptance additions.
- `xr1/tests/test_claix_two_way.py` — stronger identities, preflight fixture and shortened-time rejection.

New source/procedure files:

- `scripts/claix_b24_task04c_cpu.py`
- `scripts/claix_b24_task04c_cpu.sbatch`
- `scripts/claix_b24_task04c_preflight.py`
- `xr1/mibot/utils/claix_process.py`
- `xr1/mibot/utils/claix_benchmark_runtime.py`
- `xr1/tests/claix_cpu_report.py`
- `xr1/tests/test_claix_task04c.py`
- `docs/claix_b24/CHECKPOINT_BENCHMARK_TASK04C.md`
- `docs/claix_b24/task04c_review.template.json`
- `docs/claix_b24/task04c_project_budget.template.json`
- `docs/claix_b24/task04c_evidence/static_checks.py`

New evidence files under `task04c_evidence/local_20261107_01/` are enumerated with
hashes in `local-results.json`; all are actual local logs or evidence indexes,
not CLAIX acceptance receipts. No Task 04A/04B worktree was modified.
