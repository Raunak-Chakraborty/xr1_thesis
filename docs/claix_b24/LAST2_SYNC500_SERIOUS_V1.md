# Last-two-block Sync-500 serious run v1

Implementation and local CPU evidence only. **No GPU submission is authorized.**
Starting branch: `thesis/xr1-last2-sync500-serious-v1`; starting/unchanged HEAD:
`eeda3614db79c45f5e38e898e5c68c7da10d8db7`. No commit or push was made.
The pre-existing checkpoint/runtime retention edits were preserved and extended;
`xr1/tests/test_claix_serious_retention.py` was left unchanged.

The supplied reference job 5538134 already qualified Sync-500 last-one at 1500
steps and E1 last-two at 30 steps, including compact checkpoint inspection.
Its reported last-two count is 271570944 parameters, peak GPU memory about
71.81 GiB, physical capacity about 93.58 GiB, and checkpoint size about 3.26 GB.
These are supplied reference results, not measurements repeated in this task.

## Scientific and experiment identity

Configuration: `xr1/configs/claix_b24_last2_sync500_serious.yaml`.
Experiment: `xr1_doosan_b24_last2_sync500_serious_v1`.
Variant: `last2_sync500_serious_v1`.
The explicit launcher selector is `--serious-last2`, together with experimental
opt-in and checkpoint interval 500. Execution additionally requires the dedicated
audit JSON. These selectors do not authorize submission.

The profile inherits the existing Sync-500 scientific recipe; only last_n and
the isolated experiment identity change. The runtime rejects scientific
overrides even on a fresh invocation. Preserved invariants:

- Original Xiaomi pretrained initialization, step zero; never the E1 checkpoint.
- Last two blocks in each 24/36/36 stack (indices 22–23, 34–35, 34–35), plus
  the six existing robot-interface trainables. Actual runner count: 271570944.
- Frozen BF16 and trainable FP32 masters; original SDPA/compute configuration.
- Unchanged AdamW and cosine implementation/parameters; warmup 500; training,
  data and scheduler horizons all 48284. Accumulation 1, batch 24, world size 1.
- Four ordered persistent workers, prefetch 2, seed 42, immutable ZIP cache,
  action horizon 30, no force input, unchanged language and normalization.
- Frozen 255/34/34 train/validation/test episodes; 289703 training frames.
  Total exposure is 1158816 occurrences: four complete frame sets plus four.
  The existing shuffled fixed-horizon stream is preserved; there is no epoch wrap.

Original pretrained SHA256:
`94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7`.
Language SHA256:
`21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d`.
Train inventory SHA256:
`1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548`.
The original model baseline commit remains
`52676ccfa3b52f05de51d3d7407f33855159ac95`; the new engineering commit/tree
identity is recorded separately and must remain identical across allocations.

Checkpoint identity includes the same run UUID/root, source commit/tree,
scientific configuration digest, audited input digest, ZIP root/manifest,
original model/language hashes and frozen training identity. Receipts carry
that identity, exact bytes, SHA256 and resolved resume fingerprint. Selected
key hashes/counts, full FP32 masters/moments, optimizer parameter-to-key mapping,
weight decay groups, exact current cosine LR, scheduler state, Lightning batch
and optimizer loops, per-rank RNG and step-derived sampler cursor are checked.
The constructed model also enforces exact selected metadata on restore.
E1, last-one Sync-250/500 and historical production cannot cross-resume.

## Retention and storage

`ProductionRuntime` explicitly enables the existing serious milestone policy.
Synchronous publication occurs at every 500 completed optimizer steps, any
safe-stop boundary, and final step 48284. Permanent milestones:
**5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000**.
Final 48284, the latest two verified recovery states and the current safe-stop
state are also protected. The historical retention policy is unchanged.

Each publication completes serialization, fsync, full CPU read-back validation,
SHA256, immutable payload/receipt publication and atomic latest-pointer update.
Rotation still fully verifies **at most eight distinct payloads**. Old pins use
bounded receipt/stat checks; they are not deserialized nine times per rotation.
Missing, malformed, foreign or size-inconsistent milestone evidence aborts
before deletion, and missing earlier pins abort before new serialization.
Same-size corruption of an old pin is detected by its next full `load_verified`
audit/evaluation, not by the cheap rotation scan. Receipts remain after ordinary
payload rotation; unverified/corrupt payloads are never deleted to make space.

Fresh storage reserve is **77466091520 bytes (72.146 GiB)** and 10000 free inodes.
It uses a conservative payload ceiling of 3325960192 bytes
(`12 * 271570944 + 64 MiB`) rather than rounding the reported E1 size down.
Reserve: nine pins + final + two recovery + safe-stop + temporary serialization
+ eight possible rotation-backlog payloads = 22 payload slots, plus 4 GiB for
logs/metadata/headroom. Overlapping protected states are deliberately counted
separately. Resume budgeting reads actual receipt byte sizes, increases the
per-payload allowance if needed, and subtracts already resident verified-pin
sizes. Filesystem free space, user/project quota headroom, inherited project
accounting and free inodes must all pass. Before each save, free serialization
headroom is checked again. Quotas are rechecked on each allocation; an external
quota/space change during an allocation can still fail a write, preserving the
previous committed payload and failing closed without rollback.

Milestone payloads, receipts, `experiment.json`, immutable inventory/resume
contract, timestamped resolved configurations/start records, telemetry and
launch provenance supply evaluation provenance. Review each selected milestone
with `load_verified(..., resumable=False)` before reconstructing last-two FP32
weights over the original BF16 base. The historical evaluation CLI is hardcoded
to last-one and is not a serious-run evaluation entrypoint. No evaluation was
performed here; a later inference adapter must preserve the recorded metadata
and permit terminal inference without attempting terminal training resume.

## Fresh start, stop and continuation

1. Review this implementation and CPU evidence. A subsequently reviewed clean
   committed checkout is mandatory; the intentionally dirty current worktree
   cannot pass the submission audit. Committing was not authorized here.
2. On CLAIX, perform the CPU acceptance below using the existing offline Python.
   Supply verified absolute paths for original weights/language, frozen train,
   validation/test, reviewed split-identity JSON, ZIP cache and its reviewed
   manifest SHA256. Do not replace missing values with guessed paths or hashes.
3. Use `scripts/claix_b24_last2_serious_audit.py` to produce an exclusive audit
   report. It has no submission or training operation. Its `--help` lists all
   required inputs: source commit, exact Python, four-path override JSON, new
   persistent run root, Lustre source/project, split/cache identities and CPU
   acceptance evidence. All inputs are verified; unknown quotas fail closed.
   Fresh start requires a never-used root and no resume checkpoint.
4. After separate authorization, the dedicated prepared wrapper consumes that
   report. It rechecks live inputs/storage and its saved Slurm script. The first
   invocation creates `experiment.json` once with a new UUID and original-step-zero
   provenance. An existing root cannot be restarted as a fresh experiment.
5. For continuation, preserve the same source, UUID, root, overrides, scientific
   recipe, assets and CPU acceptance evidence. Select exactly the path named by
   `latest.json`; generate a new audit report with that resume path and supply
   `XR1_RESUME` to the same wrapper. The manifest/UUID is reused, not recreated.
   Horizon remains 48284, optimizer/scheduler and Lightning loops are restored,
   and the sampler begins at `global_step * 24` consumed rank-local samples.

The controller and torchrun inherit an exclusive run-root lock. A read-only
queue check rejects another active dedicated serious job; allocation validation
also requires the dedicated job name. SIGUSR1/TERM/INT request a stop in ordinary
controller code. At a completed optimizer boundary the callback synchronously
publishes a verified safe-stop checkpoint. The deadline is allocation end minus
1800 seconds. A crash with telemetry, a receipt or payload ahead of latest is
**not automatically recoverable**: preserve all evidence for explicit review.
No repeated/skipped optimizer steps, optimizer resets, automatic forks or silent
rollback are allowed. Failed first invocations that created evidence require
review and a genuinely new root; the tooling does not clear them.

Wrapper: `scripts/claix_b24_last2_sync500_serious.sbatch` (unsubmitted).
Resources: thes2452/c23g, one node/task, one Hopper H100, 12 CPUs, 96 GiB host
memory, exactly 23:45:00, `B:USR1@1800`. Saved script bytes, including line
endings, must exactly equal the source wrapper. Wrapper SHA256:
`25578638467630a9dc85426a4524a398cb323519c8f1ae0e05ba466c92bfaf01`.
Both the allocation GPU query and PyTorch device properties require at least
**90 GiB physical VRAM (92160 MiB)**; 80 GB H100s fail before model construction.
No unverified site-specific large-memory scheduling feature was invented;
placement availability still needs operator verification. Existing offline
environment setup, Lustre validation and path override contracts are reused.

## CPU checks and pending acceptance

Local interpreter: `/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python`,
torch 2.7.1+cu128, CUDA hidden. Lightning, Hydra and mmengine are absent. Nothing
was installed. The focused harness explicitly stubs absent Callback/CheckpointIO
bases; entrypoint tests substitute framework/model construction and unavailable
asset hashes. Torch optimizers, cosine scheduling, RNG, serialization, fsync,
read-back, SHA256, receipts, rotation and sampler continuation execute for real
on small synthetic CPU tensors. High-step fixtures relocate real optimizer state
to synthetic counters; they do not perform thousands of training updates.

Commands actually executed (all local CPU only):

```bash
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_last2_serious_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_experimental_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B xr1/tests/run_claix_cpu_tests.py
```

| Check | Actual result |
| --- | --- |
| Serious suite plus unchanged retention/E1/combined contracts | 45 passed; 0 failures/errors/skips |
| Existing synchronous checkpoint regression suite | 108 passed; 0 failures/errors/skips |
| Dataset/cache/I/O suite | 38 passed; 0 failures/errors/skips |
| Existing `ReplayTest`, explicit DataModule/registry stubs | 6 passed; real 0/4/8 worker processes, NumPy IPC |
| Full framework harness attempt | Blocked at import: missing Lightning; no tests executed |
| Python AST, dedicated wrapper `bash -n`, audit `--help`, whitespace | Passed; 13 Python files checked |
| Historical production/Sync-500 YAML and completed batch wrappers | Four files byte-identical to starting HEAD |

The replay command used the existing `run_claix_cpu_tests.bootstrap()`, an
explicit `LightningDataModule` base stub, and unittest loading of
`test_claix_replay.ReplayTest`. This does not certify production tensor IPC.
An intermediate new-suite run exposed a test-only lazy torch import interaction
with `patch.dict(sys.modules)` (1 failure/13 errors); fixture setup was corrected
and all subsequent runs passed. Initial attempts with `python` did not execute
tests because that command is absent; the existing interpreter above was used.
Final logs are in `last2_serious_v1_evidence/` beside this document.

Required **CLAIX CPU acceptance remains PENDING**. Run the full existing CPU
harness, including the new `test_claix_last2_serious_framework.py`: real Hydra
composition and real Lightning last-two compact save/resume with four persistent
ordered workers and actual CPU tensor IPC. The tiny model substitutes only the
backbone; real runner save/load, optimizer, scheduler, loops and callback execute.
Its safe-stops at 3/6 bound CPU work while the scheduler/data/trainer horizons stay
48284. Compare uninterrupted and resumed tensors/moments/LR, sample sequence,
loop counters and telemetry. This new framework test has only been syntax-checked
locally; no PASS is claimed. Use persistent scratch on Lustre for transaction
acceptance, verify signal/stop behavior, and retain the exact environment versions.

The audit requires a reviewed JSON with schema
`claix-last2-serious-cpu-acceptance-v1`, status `PASS`, `framework_stubs: false`,
exact `source_commit`, `source_tree_sha256`, `batch_script_sha256`, and installed
`torch_version`, `lightning_version`, `hydra_version`. Its `gates` must contain
`hydra_composition`, `lightning_same_run_resume`, `ordered_four_worker_replay`,
`serious_checkpoint_roundtrip`, `legacy_sync_regressions`, and `dataset_io`;
each needs `status: PASS`, an existing `evidence` path and its `sha256`.
Do not create this acceptance record from the local stubbed results.

Remaining gates: review/clean committed source, real CLAIX CPU acceptance,
actual approved Python/assets/split/cache identities, persistent root, live
filesystem/project/user quotas and inodes, no active duplicate, and verified
large-memory H100 placement. No CLAIX connection, GPU test, Slurm submission,
package installation, dataset/weight download, commit or push occurred.

## Exact source changes

Modified:

- `scripts/claix_b24_launch.py`
- `scripts/claix_b24_storage_preflight.py`
- `xr1/mibot/models/runner/doosan_compact_runner.py`
- `xr1/mibot/utils/claix_checkpoint.py` (preserves prior retention work)
- `xr1/mibot/utils/claix_experiment.py`
- `xr1/mibot/utils/claix_runtime.py` (preserves prior retention work)
- `xr1/tools/train_doosan_compact.py`

Added:

- `scripts/claix_b24_last2_serious_audit.py`
- `scripts/claix_b24_last2_sync500_serious.sbatch`
- `xr1/configs/claix_b24_last2_sync500_serious.yaml`
- `xr1/mibot/utils/claix_last2_serious.py`
- `xr1/tests/run_claix_last2_serious_cpu_tests.py`
- `xr1/tests/test_claix_last2_serious.py`
- `xr1/tests/test_claix_last2_serious_framework.py`
- `docs/claix_b24/LAST2_SYNC500_SERIOUS_V1.md`
- Evidence logs: `serious.log`, `synchronous.log`, `dataset_io.log`, `replay.log`,
  `framework_attempt.log`, `checks.txt` under `docs/claix_b24/last2_serious_v1_evidence/`.

The already-untracked `xr1/tests/test_claix_serious_retention.py` remains intact.

```text
SERIOUS_LAST2_SOURCE=PASS
SERIOUS_LAST2_IDENTITY=PASS
SERIOUS_LAST2_RESUME=BLOCKED
SERIOUS_LAST2_MILESTONES=PASS
SERIOUS_LAST2_CPU_TESTS=PASS
SERIOUS_LAST2_CLAIX_CPU_ACCEPTANCE=PENDING
SERIOUS_LAST2_GPU_SUBMISSION=NOT_AUTHORIZED
GPU_JOBS_SUBMITTED=0
```

PASS denotes implemented contracts and available local CPU checks. Resume remains
blocked for operational acceptance until real CLAIX Lightning restoration passes;
local same-run guard, optimizer-state and replay tests pass. This is not a release
or launch approval, and no GPU submission command is provided.
