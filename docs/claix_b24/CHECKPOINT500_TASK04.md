# Task 04A — isolated checkpoint-500 experimental variant

Source implementation and the available CPU checks are complete. Full acceptance
is **BLOCKED** pending real Hydra/Lightning acceptance in the existing CLAIX
environment. No GPU training, Slurm command, asynchronous checkpoint writer,
download, commit or push was performed.

```text
TASK04_SOURCE_AUDIT=PASS
TASK04_IMPLEMENTATION=PASS
TASK04_CPU_TESTS=PASS
TASK04_BASELINE_PROTECTION=PASS
TASK04_ASYNC_CHECKPOINTING=NOT_IMPLEMENTED
TASK04_GPU_JOBS=0
TASK04_RESULT=BLOCKED
```

`CPU_TESTS=PASS` means the executed, explicitly bounded laptop suites below.
It does not mean the full framework suite passed. Missing framework execution
prevents demonstrating the new variant's actual Lightning snapshot boundary,
strategy/plugin timing handoff and restore behavior end to end. Those checks
must finish before changing the overall result to PASS.

## Starting point and source audit

Worktree: `/home/horus/robotics_thesis/xr1_claix/worktrees/xr1_checkpoint500_v1`.
Branch: `thesis/xr1-checkpoint500-v1`. Starting HEAD:
`0106f1a0e0dfbcbf18e9732f4924b8ccd9e937f0`; the worktree was initially clean.
Historical job 5504969, 48,129 completed steps, its run identity and payloads are
context supplied by the task, not newly measured evidence. No historical payload
or baseline worktree was opened or modified.

Before editing, inspected the complete `claix_b24_launch.py`, experimental and
production batch wrappers and environment bootstrap; the selected
`train_doosan_compact.py` entrypoint and the generic `train.py`; experimental,
production and inherited Hydra recipes; `claix_experiment.py`,
`claix_contract.py`, `claix_resume_bridge.py`, `claix_runtime.py`,
`claix_checkpoint.py`, runner save/load hooks and deterministic replay contracts.
Inspected related launcher, experimental, checkpoint, runtime, release, retention
and storage CPU tests. Ran the existing focused suite before editing: 84 passed.

The actual CLAIX path uses `ProductionRuntime`, not Lightning's generic
`ModelCheckpoint` scheduling. Its batch-end hook requires exactly one newly
completed optimizer step; it saves for a stop request/deadline, interval,
step 2000, or terminal step. Fit-end refuses to succeed without that final
boundary save and loads/verifies it again. Saving at fit-end itself is forbidden
because Lightning has advanced its batch counters.

### Hardcoded 250 assumptions and output identities

| Location | Audited assumption and treatment |
| --- | --- |
| `claix_b24_production.yaml`, `doosan_compact_fp32.yaml` | 250-step intervals; unchanged |
| `claix_experiment.py` | Enforced interval 250 and `xr1_doosan_b24_experimental_v1`; now selects only the two named variants and binds each to its manifest/output |
| `claix_b24_launch.py` | Experimental plan interval, Hydra name, output target, recovery target, exact saved wrapper; now selects only reviewed 250/500 profiles |
| `claix_runtime.py` | Default interval 250; unchanged. Existing explicit step-2000 and stop/terminal triggers are unchanged |
| `claix_b24_release.py` | Strict production science contract includes 250; unchanged |
| `claix_b24_combined_qualification.py`, `claix_b24_targeted_h100.py` | Step-250 verification/signal gates, 260-step probes and warm windows; unchanged, do not qualify the new mode |
| `claix_b24_retention_benchmark.py` | Synthetic 40250/40500/40750 cadence; unchanged |
| Legacy checkpoint/runtime/launch/experimental/metrics/combined/release tests | Numerous intentional 250/750/2250 etc. fixtures and assertions; retained. Entrypoint fixture now also exercises the new recipe |

Repository source/config searches covered every `250`, experimental config/name
reference and checkpoint output target in `scripts/` and `xr1/`. Numerical data
normalization constants containing the digits 250 are not scheduling assumptions.
Inherited trainer `save_interval: 2` and fallback 1000/10000 values belong to other
entrypoints/probes, not this selected experiment.

The original experimental output name appears in its unchanged YAML and the
launcher/experiment helpers plus legacy tests. The original wrapper has its own
job/log names. All remain supported. Production's
`xr1_doosan_b24_results_v1`, `/hpcwork/zk443626/robotics_thesis/runs/xr1_results_finetune`,
compact defaults and qualification output names remain unchanged. The new YAML
sets its default root to null so it cannot inherit any of those roots.

### Fingerprint and recovery contracts

Launcher and training source digests hash the same sorted Python, Hydra YAML and
`scripts/claix_b24*` files, including tests. The resolved resume fingerprint binds
data, model, optimizer, scheduler, compute precision, devices, accumulation,
token limit, frozen train inventory and source digest; the experiment identity
is also included. Operational checkpoint interval was not independently in that
fingerprint. For 500 it is now explicitly in the experiment identity, along with
the output name, and therefore in payload, receipt and fingerprint.

The original 250 identity dictionary and plan shape are preserved. Source edits
necessarily change both source digests; no cross-source recovery exemption was
added. An original run still requires its original source/config identity.
Strict production's reviewed release record must also match the exact source
tree; an old release cannot authorize this modified tree.

Fresh execution exclusively creates a new root and random UUID. Recovery requires
the same root, UUID, source digest, commit, complete launch plan, latest receipt,
payload hash/config identity and exact telemetry cursor. A telemetry suffix
beyond latest still blocks automatic same-run recovery. A hard crash between
periodic saves therefore requires operator review under the existing policy;
this change does not silently roll back logged optimizer steps. Historical
step-21500 bridge authorization is unchanged and forbidden in experimental mode.

## Exact variant and preserved science

| Setting | Task 04A value |
| --- | --- |
| Hydra config | `claix_b24_checkpoint500`, inheriting `claix_b24_experimental`, then production and compact defaults |
| Named variant | `production.checkpoint_variant: checkpoint500_v1` |
| Experiment/output name | `xr1_doosan_b24_checkpoint500_v1` |
| Checkpoint directory | `<NEW_XR1_RUN_ROOT>/project_xr1_doosan/xr1_doosan_b24_checkpoint500_v1` |
| Manifest | `<NEW_XR1_RUN_ROOT>/experiment.json`, fresh random 32-character UUID, plan interval 500 and experiment name |
| Ordinary interval | 500 completed optimizer steps |
| Initial checkpoint | null; original pretrained Xiaomi weights, optimizer/scheduler step zero |
| Recovery | This variant's same-run latest verified checkpoint only |
| Batch wrapper | `scripts/claix_b24_checkpoint500.sbatch`; distinct job/stdout/stderr names |
| Allocation | thes2452 / c23g, one node/task, `gpu:hopper:1`, actual single NVIDIA H100, exactly 12 CPUs and 96 GiB, 23:45:00 |
| Safe stop | Existing `B:USR1@1800`, deadline minus 1800 seconds; no interval alignment delay |
| Storage | Existing persistent Lustre run/cache/output gates and mandatory user/project quotas |

Only `--experimental-checkpoint-interval {250,500}` was added. It is rejected
outside the explicit experimental double opt-in. Omitting it keeps 250. Each
selection requires the exact corresponding saved Slurm batch script. Neither
the four permitted staged path overrides nor scientific CLI controls were
broadened. Direct interval/output/manifest mismatches fail before model building.
Fresh roots cannot be reused, and existing original manifests cannot select 500.

For later read-only plan review, with the existing staged path JSON and a new
persistent root supplied by the operator:

```bash
"$XR1_PYTHON" scripts/claix_b24_launch.py \
  --mode experimental --experimental-unqualified \
  --experimental-checkpoint-interval 500 \
  --run-root "$XR1_RUN_ROOT" --workers 4 --overrides "$XR1_PATH_OVERRIDES" \
  --filesystem-source "$XR1_FILESYSTEM_SOURCE" --project-id "$XR1_LUSTRE_PROJECT_ID"
```

This command has no `--execute`. No real allocation or submission was attempted.
The new wrapper remains unsubmitted. Do not use a historical root or UUID.

The candidate's complete resolved model and data dictionaries equal the
original experimental recipe. Its optimizer, scheduler and all other scientific
trainer settings also equal the original recipe. Tests compare this full
inheritance, not just a few selected values. In particular:

- The frozen training identity remains
  `1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548`,
  255 episodes / 289703 frames. Train/validation/test split files and audit logic
  remain untouched (255/34/34 episodes). Actual validation/test inventory hashes
  must be checked against the existing reviewed CLAIX split identities; no new
  hashes or split manifests were invented locally.
- Batch 24; four workers; prefetch two; persistent workers; ordered deterministic
  replay; timeout 600; immutable ZIP policy; existing CRC/sidecar checks, cache
  bounds and diagnostic behavior remain. Dataset/model/checkpoint staging is
  persistent Lustre. The existing small temporary watchdog diagnostic spool is
  unchanged and is not data or checkpoint staging.
- Original architecture, action length 30, force inputs disabled, last one block
  policy, loss, sampler seed 42, FP32 masters, BF16 mixed compute, accumulation
  one and one H100 remain. Original pretrained SHA256 is
  `94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7`;
  language SHA256 is
  `21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d`.
- AdamW settings remain lr multiplier 1.0, betas 0.9/0.95, weight decay 0.1,
  epsilon 1e-8; cosine schedule remains 48284 steps, warmup 500, start 5e-7,
  peak 2e-5, floor 5e-6. Trainer and data horizons remain 48284.
- Production's existing 20:00:00 wrapper stays unchanged. Both experimental
  modes require 23:45:00, as verified at the requested starting commit. These
  distinct allocation contracts are intentional.

## Checkpoint invariants and failure evidence

The publication algorithm remains synchronous: completed-boundary snapshot,
structural validation, temporary payload serialization/flush/fsync, full CPU
read-back and structural validation, SHA256 over exact bytes, no-clobber payload
link and directory fsync, exclusive receipt publication/fsync, atomic latest
publication/fsync, then verified retention. Timing never changes this ordering.

| Required invariant | Source and executed CPU evidence |
| --- | --- |
| 1. Completed optimizer boundary | Existing progress validator checks optimizer, AdamW, scheduler, batch counters and epoch; malformed/fractional counters rejected. Actual hooks invoked with synthetic boundary state |
| 2. Full resumable state | Real torch round trips model, FP32 moments, scheduler, loops, replay and Python/NumPy/torch RNG; restored RNG draws and sample offset match |
| 3–5. Valid latest, verified receipt, SHA binding | Actual hashing/read-back; equal-size corruption rejected. Read-back validation failure publishes neither receipt nor pointer |
| 6–7. No overwrite, atomic publication/fsync | Existing destination/symlink guards retained. Serialization, fsync, receipt-link and latest-replace failures injected below the transaction interface |
| 8. Last usable recovery retained | Rotation verifies newest and two recovery states before deletion; corrupt newest/previous/safe-stop, budget exhaustion and unlink failure regressions retained |
| 9. Pinned retention | All milestones 2000/5000/10000/20000/30000/40000 retained plus two recent usable states; safe-stop 751 survives subsequent rotation |
| 10. Immediate safe stop | Signal flag at completed step 751 invokes real verified publication and safe-stop pointer, despite 751 not being divisible by 500 |
| 11, 15. Terminal verified publication | Hook at 48284 saves with natural_finish; fit-end loads it again, and rejects a missing final boundary or corrupted terminal bytes |
| 12. Interrupted serialization | Partial temporary bytes followed by OSError leave no valid new pointer/receipt; orphan-without-receipt loading is rejected |
| 13–14. Recovery identity and historical rejection | Same-run verified restore succeeds; foreign UUID/root/config/variant, changed source/plan, wrong latest/telemetry, symlinks and historical step-21500 initialization fail |

Ordinary scheduling is tested directly at 250/750 (no checkpoint), 500/1000
(verified checkpoint), 2000 (verified milestone) and 48284 (verified terminal).
The old 250 callback schedule/default, unchanged strict release gates and old
experimental entrypoint still pass their regressions.

Receipt publication failure leaves a durable payload orphan with no receipt;
the previous latest remains usable and reserialization to that name is refused.
Pointer publication failure leaves a verified new payload/receipt but the old
latest pointer; rotation rejects the inconsistent newest pointer. This requires
operator review, not overwrite, implicit promotion or automatic rollback. A real
process kill may leave a hidden temporary orphan; no receipt or pointer exists
for those partial bytes. Kill/power-loss durability on Lustre was not tested.

The synthetic hook tests do not establish the installed Lightning version's loop
counter behavior. That is the remaining acceptance gap, not a weakened validator.

## Bounded timing instrumentation

Only the 500 identity enables `VerifiedCheckpointIO(instrument=True)` in the
entrypoint. Other modes keep their prior telemetry fields and I/O counts. Each
successful checkpoint adds a fixed nine-number `timings` object to its existing
checkpoint telemetry row; no separate timing files, payload fields, receipt
fields, extra loads, hashes, background workers or growing caches are introduced.

| Field (seconds, monotonic wall clock) | Measured boundary |
| --- | --- |
| `state_capture_seconds` | Entry to Runtime.save through entry to CheckpointIO.save_checkpoint; includes RNG capture/gather, Lightning dump and strategy preparation |
| `prewrite_validation_seconds` | Publication entry/path guards through initial structural validation |
| `serialization_fsync_seconds` | Temporary-file creation, torch.save, flush, file fsync and close |
| `readback_validation_seconds` | Size ceiling check, torch.load to CPU and structural validation |
| `sha256_seconds` | Publication's full payload SHA256 calculation |
| `publication_seconds` | Receipt construction, payload link/unlink, directory fsync, receipt publication and latest publication/fsync |
| `safe_stop_pointer_seconds` | Optional safe-stop pointer and associated metadata work after the strategy barrier |
| `rotation_verification_seconds` | Complete existing retention call, including bounded verification reads/hashes and deletion/fsync |
| `callback_duration_seconds` | Runtime.save entry through retention, immediately before emitting its telemetry |

`callback_duration_seconds` is the end-to-end checkpoint work through publication
and retention, including state capture and strategy barrier. It excludes the
checkpoint telemetry append/console print and the final last_saved assignment;
it is a lower bound on the entire Python method duration. `duration_seconds`
retains its old definition (after RNG capture through retention) for comparison.
Pure Lightning dictionary capture cannot be separated from strategy preparation
without deeper framework hooks, so `state_capture_seconds` is an explicitly
bounded proxy, not a claimed pure snapshot/CUDA-copy timer. The existing CUDA
synchronization in batch-end precedes this measurement. No extra synchronization
or asynchronous CUDA timing was added. Retention's additional SHA/read-back work
is inside the retention aggregate, not double-counted as publication hashing.

Errors propagate; incomplete timings are never emitted as successful checkpoint
rows, and the IO object clears stale success timings before publication.
Tests confirm fixed keys, finite nonnegative durations, receipt schema, and
unchanged actual load/hash counts. These synthetic timings are not Lustre or
XR-1 performance measurements.

The next step's `pipeline_wait_seconds` still includes previous checkpoint work
under the established telemetry definition. Compare the preceding checkpoint
row and its timing breakdown before interpreting that value as DataLoader wait.
Subtracting checkpoint time alone is only an approximation because callback and
loader overlap/other host work remain; the metric was deliberately not relabeled
as pure loader time.

## Performance hypothesis

The supplied mean 19.64 s and 1.695 GB payload are observational inputs, not new
benchmarks. For a full 48284-step horizon without extra stops:

| | 250 interval | 500 interval |
| --- | ---: | ---: |
| Ordinary checkpoints | floor(48284/250) = 193 | floor(48284/500) = 96 |
| Separate terminal checkpoint | 1 | 1 |
| Total publications | 194 | 97 |
| Cost if each retains mean 19.64 s | 3810.16 s | 1905.08 s |

Expected avoided cost: **97 × 19.64 = 1905.08 seconds, about 31.8 minutes**.
Avoided payload writes alone are about 164.4 GB decimal; read-back, hashing and
retention traffic are additional and depend on actual behavior. The supplied
completed run's 193 events include its nonterminal safe stop, so it must not be
equated to 193 ordinary checkpoints at the full horizon. At the same 48129-step
stop, the comparison would be 192+1 versus 96+1, saving 96 events (~31.4 minutes).

There is no measured new training throughput or completion forecast. Lustre
contention, cache warmth, retention history, CPU costs, safe-stop frequency and
fixed compute/DataLoader stalls may change the mean. Lost-work exposure rises
from up to 250 to up to 500 completed steps after an abrupt failure. Safe-stop
and terminal guarantees remain immediate. Match step ranges/science, compare
median/tail durations and phase costs on a later authorized CLAIX run.

## Asynchronous feasibility assessment — not implemented

The frequency change should be measured first. Roughly 32 minutes of potential
savings already comes from fewer synchronous transactions. No per-phase CLAIX
data yet demonstrate that the remaining ~32 minutes justify another recovery
protocol and extra memory/CPU contention.

- **Thread versus process:** A dedicated CPU thread can share staged tensors
  without IPC and overlap blocking POSIX I/O; Python serialization/GIL work and
  CPU/memory contention may still limit overlap. A separate process gives better
  fault isolation, but introduces large IPC/shared-memory ownership, supervision
  and lock-lifetime issues. Fork after CUDA initialization is unsuitable; a
  spawn-based CPU-only worker and explicit shared-buffer ownership would need
  independent qualification. Neither option is present in Task 04A.
- **Allocated CPU:** Twelve allocated CPUs with four DataLoader workers suggests
  potential headroom, but does not prove a spare core: rank, loader service,
  diagnostics and controller/monitor also consume time. Measure per-core load,
  affinity and memory bandwidth before assigning a writer. Do not request more
  CPUs or alter the scientific loader configuration to hide writer contention.
- **Immutable staging:** At the completed step, detach and copy every mutable
  trainable parameter, AdamW moment/step, scheduler, loop/RNG/replay record and
  nested metadata into worker-owned immutable CPU storage. A state_dict or
  shallow dictionary copy alone aliases live state. Subsequent optimizer updates
  must be impossible until staging is demonstrably complete. Reconstruct frozen
  weights only from the same immutable original pretrained identity.
- **RAM/pinning:** One payload is about 1.695 GB (1.58 GiB); a full staging buffer
  plus writer read-back can require ~3.39 GB before Python/allocator overhead.
  Additional queued copies multiply this, and training/model/loader memory must
  fit within the unchanged 96 GiB request. The measured synchronous peak and
  available headroom are unknown. Pinned host memory may accelerate D2H copies,
  but it is scarce/locked memory; use bounded buffers, measure limits and retain
  them until CUDA completion and writer acknowledgement. Do not assume the
  entire payload must or can be pinned.
- **CUDA synchronization:** CPU writer must receive completed host bytes, never
  live CUDA tensor references. Explicit stream events/waits are needed before
  ownership transfer; nonblocking copy completion and optimizer mutation must
  have a proven order. Staging synchronization can remain the dominant stall.
- **Lightning assumptions:** Trainer.save_checkpoint and CheckpointIO are
  currently synchronous. Runtime marks last_saved only after verified publication
  and rotation; fit-end immediately verifies that checkpoint. Callback return,
  barriers and finalization must be redesigned around explicit staged versus
  committed states. Calling the existing Trainer API from a background thread
  while training mutates its connectors/loops is not justified.
- **Transaction order:** One publisher must enforce monotonic completed steps,
  SHA256 over exact staged bytes, read-back validation, no-clobber payload,
  directory fsync, exclusive receipt then atomic latest, followed by retention.
  Never publish latest on enqueue, before validation, or for a failed worker.
  The root lock must remain owned for the full lifetime of any active writer.
- **Safe stop/shutdown:** Stop optimizer updates at the current completed
  boundary, stage that state even off interval, drain earlier writes in order,
  await verified receipt/pointer/rotation acknowledgement, and verify final state
  before a successful exit. Bound draining within the 1800-second margin using
  measured tails. A pending old interval snapshot cannot stand in for the
  requested safe-stop/final boundary.
- **Worker crash:** Detect worker exit/exception, fail training closed and keep
  previous committed latest usable. Preserve uncommitted orphans for review;
  do not retry into an existing payload name or manufacture a receipt. Supervise
  child lifetime and inherited locks so a controller failure cannot free the run
  root while a writer is still active.
- **Backpressure:** Start with at most one in-flight immutable snapshot. At the
  next request, block if no buffer is free. Do not drop final, safe-stop or pinned
  milestones. Any coalescing of ordinary checkpoints would be a new policy that
  needs separate review and cannot silently change cadence.
- **Required future tests:** Mutation during staging/serialization, tensor
  aliasing, CUDA event/copy races, RNG/loop atomicity, full live-versus-resumed
  equality, exact boundary signal during an in-flight save, worker death at every
  publication stage, partial fsync/ENOSPC, stalled worker/full queue, out-of-order
  completion, duplicate destination, shutdown deadline expiry, controller/worker
  lock inheritance, final failure propagation and retention racing publication.
  Repeat on Lustre and the installed Lightning/CUDA versions before adoption.

## Tests actually executed and remaining acceptance

Existing local interpreter:
`/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python`, torch 2.7.1+cu128,
CUDA hidden. No dependency installation was performed. System python3 lacks
torch; the existing environment supplies torch/NumPy/PyYAML but lacks Lightning,
Hydra and mmengine.

Commands executed from this worktree:

```bash
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_claix_experimental_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_claix_cpu_tests.py
```

| Executed check | Actual result |
| --- | --- |
| Before editing: original focused suite | 84 passed, no failures/errors/skips, 8.470 s |
| Final focused suite: 84 existing + 24 new | 108 passed, no failures/errors/skips, 10.408 s |
| Dataset/cache/I/O regression suite | 38 passed, no failures/errors/skips, 0.689 s |
| Full CLAIX CPU harness attempt | Exit 1 during collection: missing lightning. No suite tests executed or claimed passed |
| Syntax, baseline comparisons and whitespace | Python AST: 9 files; bash -n: 7 wrappers + environment script; 55 protected tracked files byte-identical to starting commit; git diff --check passed |

The focused harness stubs only absent Callback/CheckpointIO bases for runtime
imports. Synthetic Trainer/registry/GPU/allocation boundaries in entrypoint and
launcher fixtures are explicit; actual main body, checkpoint validators,
torch.save/load, hashes, filesystem transaction, rotation, RNG restoration and
optimizer/scheduler operations run. Failure tests inject actual leaf I/O errors,
not successful fake verification. The I/O harness has explicit absent framework
base/registry/backbone stubs and real dataset/cache/DataLoader operations. These
results do not establish real Trainer/DDP/H100 integration. An early development
collection error from a misspelled fixture class import was corrected before
the final passing runs.

The static check ran as `python3 -B /tmp/xr1-task04-static-checks.py`; its complete
source and output are retained together in `task04_evidence/static_checks.log`
for reproduction. It checks the exact branch/HEAD and eight allowed tracked
source modifications, parses the nine changed/new Python files, syntax-checks
all seven wrappers and the environment bootstrap, compares the 55 protected
files against the starting commit and runs `git diff --check`.

Required later CLAIX checks (not executed here):

1. Run all three harnesses in the existing offline CLAIX environment; run the I/O
   harness without framework stubs when dependencies are available. Real
   `Checkpoint500HydraTest` and
   `RuntimeTest.test_checkpoint500_real_lightning_stop_resume_and_final_verification`
   must execute. The latter uses a tiny CPU model and real Trainer, saves 500,
   stops at 751, resumes through 1000 and verifies final 1001. It has no GPU need.
2. Verify resolved real Hydra configuration, frozen split inventories/ZIP
   manifest, pretrained/language hashes, 48284 horizon and all scientific fields.
   No test assets should be substituted for actual scientific identities.
3. Verify actual installed strategy exposes the configured CheckpointIO and
   timing handoff; confirm read-back/loop counter acceptance on real Lightning.
4. On an independently authorized future H100/Lustre acceptance run, check
   interval 500, off-interval USR1, terminal verification, same-run restore,
   exact allocation/script/storage guards and failure behavior. Requalify the
   500-specific path: existing qualification probes intentionally exercise 250.
5. Measure phase costs, callback/throughput tails, host RSS and storage headroom.
   Local synthetic execution establishes neither Lustre crash durability nor
   H100 throughput. Do not infer final-source production qualification.

## Compatibility with later scientific studies

E1 (last two blocks), E2 (lower peak LR), E3 (balanced task sampling) and E4
(lower LR floor) are not implemented. Task 04A is the unchanged-science checkpoint
engineering control. Use distinct named configs, roots, UUIDs and source/config
fingerprints for each later study and hold checkpoint cadence/telemetry constant
across comparisons. Start each from the original pretrained weights at step zero.
Do not resume a Task 04A checkpoint to initialize a new scientific experiment.

E1 currently conflicts with the strict last-one checkpoint metadata validator;
E3 conflicts with fixed replay/sampler contracts. Those require separately
reviewed contracts, not overrides or relaxed checks here. E2/E4 also change the
fingerprinted scheduler identity. This task deliberately adds no unrestricted
scientific experiment registry or generic Hydra escape hatch.

## Exact changed-file inventory

| File | Change |
| --- | --- |
| `scripts/claix_b24_launch.py` | Closed selector; named profile, wrapper, plan and recovery isolation |
| `scripts/claix_b24_checkpoint500.sbatch` | New dedicated 23:45:00 wrapper, identical scientific/resource request |
| `xr1/configs/claix_b24_checkpoint500.yaml` | New inherited recipe and isolated output |
| `xr1/mibot/utils/claix_experiment.py` | Variant identity binding and recovery directory selection |
| `xr1/mibot/utils/claix_checkpoint.py` | Optional fixed-size transaction timing output |
| `xr1/mibot/utils/claix_runtime.py` | Opt-in IO timing and save/retention phase telemetry |
| `xr1/tools/train_doosan_compact.py` | Enable timing only for bound 500 identity |
| `xr1/tests/test_claix_checkpoint500.py` | New 24 offline cases plus real Hydra acceptance case |
| `xr1/tests/test_claix_experimental.py` | Parameterize existing entrypoint fixture without weakening old assertions |
| `xr1/tests/test_claix_runtime.py` | Parameterize tiny CPU fixture; add real Lightning 500 stop/resume case |
| `xr1/tests/run_claix_experimental_cpu_tests.py` | Include the 24 new framework-free tests explicitly |
| `docs/claix_b24/CHECKPOINT500_TASK04.md` | This audit, design, evidence and handoff |
| `docs/claix_b24/task04_evidence/baseline_cpu.log` | Original 84-test pre-edit execution |
| `docs/claix_b24/task04_evidence/focused_cpu.log` | Final 108-test execution |
| `docs/claix_b24/task04_evidence/io_cpu.log` | 38-test I/O execution |
| `docs/claix_b24/task04_evidence/full_claix_cpu_attempt.log` | Actual missing-Lightning collection failure |
| `docs/claix_b24/task04_evidence/static_checks.log` | Syntax, protected-file comparisons and diff check |

Technical handoff: review the isolated recipe/selector and synchronous timing
delta, then run real Hydra/Lightning CPU acceptance on CLAIX. Resolve that evidence
gap before marking Task 04A PASS or authorizing any experimental training.
