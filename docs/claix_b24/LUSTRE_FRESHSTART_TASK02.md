# Task 02 — Explicit experimental fresh-start XR-1 training

Implemented on `thesis/xr1-lustre-freshstart-v1`, starting at committed Task 01B
`6c0ae1e` (Task 01: `5732f7b`). This is an **unqualified experimental mode**,
not production qualification. No release approval has been issued or inferred.

## Behavior and release boundary

Use the new `scripts/claix_b24_experimental.sbatch`. The Python launcher requires
both `--mode experimental` and `--experimental-unqualified`; the wrapper requires
`XR1_EXPERIMENTAL_UNQUALIFIED=1`. There is no automatic fallback. Experimental
mode rejects a release-record argument and the historical
`XR1_OPTION_B_RESUME_BRIDGE` environment setting. It never invokes that bridge.
The production wrapper and release validator are byte-identical to Task 01B;
production still requires all nine reviewed, checksum-bound release approvals.

`claix_b24_experimental.yaml` inherits the entire production training recipe.
Initial `trainer.ckpt_path` is explicitly null. The existing BaseRunner loads only
the `module` weights from the original Xiaomi pretrained checkpoint, rejects
missing/unexpected keys, and subsequently constructs a new AdamW and scheduler.
No old optimizer, scheduler, Lightning counters or replay cursor are restored.
Fine-tuning and the rank-offset sampler start at **optimizer step 0**. This is
pretrained initialization, not random initialization. The original pretrained
SHA256 remains
`94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7`;
Language V2 remains
`21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d`.
Both files are hashed before model construction, as before.

The experiment additionally requires the existing frozen train inventory SHA256
`1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548`,
255 episodes and 289703 frames. This is the previously recorded frozen identity,
not a new dataset or a locally fabricated measurement. Sampling, normalization,
augmentation, loss functions and architecture are unchanged. No dataset scan or
model initialization was performed locally.

| Setting | Effective value |
| --- | --- |
| Allocation | thes2452, c23g, one node/task, one H100 |
| CPU / host RAM / wall time | 12 / 96 GiB / 20:00:00 |
| Slurm notification / safe margin | B:USR1@1800 / 1800 seconds |
| Batch / workers / prefetch | 24 / 4 / 2 |
| Timeout / persistent / ordered | 600 seconds / true / true |
| Data / trainer / scheduler horizon | 48284 optimizer steps |
| Warmup / checkpoint interval | 500 / 250 steps |
| AdamW | lr multiplier 1, betas (0.9, 0.95), decay 0.1, eps 1e-8 |
| Schedule | start 5e-7, maximum 2e-5, minimum 5e-6 |
| Compute / trainables / last_n | BF16 mixed / FP32 / 1 |
| ZIP integrity policy | Task 01B immutable |

## Allocation and independent persistent-storage protection

Experimental allocation checks use live `scontrol show job` fields: matching job
ID, RUNNING, thes2452, c23g, exactly one node/task, twelve allocated CPUs and
CPUs/Task=12, exactly 96 GiB, one GPU and exactly 20 hours. The Slurm-visible
device must identify as H100 through a read-only nvidia-smi query; the entrypoint
also retains its actual CUDA device-count/name check. Authoritative EndTime
determines the unchanged 1800-second safe deadline. The saved batch script must
exactly match the dedicated wrapper, including `--signal=B:USR1@1800`. Live
StdOut/StdErr parents and WorkDir must pass the existing persistent-directory
checks. The workflow below explicitly supplies the same signal option to sbatch;
do not override it. The saved-script check verifies the request, not actual site
signal delivery, which remains a CLAIX acceptance check. The existing independent
controller deadline still requests a safe stop if a signal is delayed or absent.

The `CPUs/Task` field name and batch-script retrieval were checked against
[SchedMD's Slurm job-info implementation](https://github.com/SchedMD/slurm/blob/slurm-24-05-5-1/src/api/job_info.c).
Local Slurm output is mocked; no scheduler command was executed against a cluster.

Experimental storage has **no release-record dependency**. It uses the existing
filesystem inspector and thresholds: approved `/hpcwork`, `/work` or canonical
`/rwthfs/rz/cluster/hpcwork` roots, resolved-path containment, one actual persistent
mount, operator-specified matching SOURCE, at least 30 GiB available and 10000
free inodes. Experimental mode further requires Lustre for both run storage and
the existing ZIP cache. Local SSD, tmpfs, overlay, escaping symlinks and offline
frame staging are rejected.

Because this task explicitly requires quota checks, experimental preflight also
requires successful live `lfs quota -u UID` and `lfs quota -p PROJECT_ID` queries.
It parses default numeric KiB output, including CLAIX's bquota/blimit and
iquota/ilimit columns, identity labels and wrapped filesystem names. Each nonzero
soft/hard ceiling must leave the same 30 GiB and 10000 inode reserve; zero means
no corresponding quota limit. It conservatively treats soft limits as ceilings
even during grace. Unknown, ambiguous, failed or timed-out queries fail closed.
`lfs project -d` must confirm the supplied project ID and `P` inheritance flag on
the existing parent. Project ID is not inferred from the Unix GID or account name.
Actual quota usage may change between submission and execution: allocation always
measures it again. The unchanged production policy still treats optional raw quota
queries as diagnostics; experimental checks do not alter that policy.

Preflight runs **before** creating the run root. Failures appear in persistent
Slurm output. Successful reports are also retained as `launch-*.storage.json`.
Initial launch exclusively creates a new root, then acquires the existing
`.production.lock`; controller and torchrun inherit/hold that lock. Existing roots
cannot be reused as a new experiment, even after a failed initialization. Choose
a new root in that case; retain the old root as evidence.

## Provenance and recovery

`experiment.json` exclusively records the run UUID, explicit
`experimental_unqualified=true`, source commit/status, exact source fingerprint,
path overrides, operational configuration and storage binding. Launcher provenance
also records actual allocation fields, TRES, GPU identity, submitted-wrapper
checksum and initial checkpoint mode. The existing resolved-config/inventory/
resume-contract artifacts remain. `experimental-start-*.json` adds effective
configuration, original model/language identities, frozen dataset identity, ZIP
manifest identity, cache policy, initial step, checkpoint mode and whether the
optimizer/scheduler are new. Loader settings include timeout 600, four workers and
prefetch two. Existing worker `XR1_ZIP_CACHE_POLICY` and `XR1_LOADER_EFFECTIVE`
records continue to describe the actual loader/cache.

Future recovery keeps the **same root, UUID, source commit/fingerprint, path
overrides, effective scientific configuration, dataset and ZIP-manifest identity**.
The checkpoint fingerprint includes the experimental UUID/root/cache identity in
addition to the existing scientific/data/source fingerprint. Checkpoints also
carry explicit experimental metadata through the existing compact runner. Both
entrypoint preflight and VerifiedCheckpointIO require matching metadata and exact
fingerprints; the runner rechecks them during restore. A renamed/copied historical
checkpoint cannot gain authorization from its filename or directory.
Experimental receipts additionally bind the run metadata, so foreign/historical
receipts are rejected before payload deserialization; receipt and payload
experimental identities must match after full verification.

Only the same run's latest verified payload is accepted, with the existing bounded
receipt checks and complete contiguous telemetry ending at that step. The check
runs again under the lock. SHA256, full compact state, optimizer/scheduler/loop
counters, replay cursor and RNG validation remain mandatory. Recovery restores
those states and continues at the saved step; it is distinct from the initial
step-zero launch. Foreign runs, the old qualification checkpoint and historical
step 21500 are rejected. A future checkpoint at numerical step 21500 created by
this experiment is valid if every same-run check passes; the number itself is
not a historical identity.

Checkpoint transaction/publication/rotation algorithms are unchanged: persistent
same-directory temporary payload, fsync, read-back validation, checksum receipt,
atomic no-clobber publication and latest pointer, then bounded verified rotation.
Two ordinary recovery checkpoints, milestone pins and the verified safe-stop
pointer retain their existing policy. The 250-step interval and safe margin are
unchanged. Signals remain flag-only handlers; completed-step runtime callbacks
publish before stopping. CSV metrics, telemetry, launch logs, controller stop
evidence and GPU-monitor logs persist under the run root. Experimental launches
enable the existing watchdog through a persisted `launch-*.loader-diagnostics.json`:
30/90/150-second thresholds, one-second polling and reserved SIGUSR2. If an existing
diagnostic configuration is supplied, its timing and local-spool options are
preserved, while its persistent exporter destination is bound to this run's
`diagnostics/` directory. The existing watchdog implementation and worker cleanup
are unchanged. Its small local diagnostic spool is not dataset/model/checkpoint
staging; retained evidence is exported persistently by the existing exporter.

Task 01B's external immutability prerequisite still applies to manifests, JSONs,
ZIPs and sidecars for the entire worker lifetime. This implementation does not
prove write protection or replace the existing dataset integrity/preflight process.
Keep those inputs frozen; stop/revalidate/restart workers before any update.
There is no SSD staging, periodic rescan or altered sampling policy.

## Intended operational workflow — NOT executed

Set these variables to **existing, verified CLAIX paths/identities**. No local
path existence or CLAIX mount identity is inferred by this report:

```bash
export XR1_EXPERIMENTAL_UNQUALIFIED=1
: "${XR1_WORKTREE:?Set this experimental checkout on CLAIX}"
: "${XR1_PYTHON:?Set existing approved Python executable}"
: "${XR1_PATH_OVERRIDES:?Set existing JSON with exactly the four staged path overrides}"
: "${XR1_DOOSAN_ZIP_CACHE_ROOT:?Set the existing frozen Lustre ZIP cache}"
: "${XR1_FILESYSTEM_SOURCE:?Set verified persistent findmnt SOURCE}"
: "${XR1_LUSTRE_PROJECT_ID:?Set verified actual Lustre project ID}"
: "${XR1_EXPERIMENT_PARENT:?Set existing persistent experiment parent with project inheritance}"
: "${XR1_SUBMIT_DIR:?Set existing persistent directory for Slurm stdout/stderr}"
test -x "$XR1_PYTHON"
test -f "$XR1_PATH_OVERRIDES"
test -f "$XR1_WORKTREE/scripts/claix_b24_experimental.sbatch"
unset XR1_RESUME XR1_STOP_AFTER_STEP XR1_OPTION_B_RESUME_BRIDGE
unset XR1_DOOSAN_OFFLINE_FRAME_ROOT
export XR1_RUN_ROOT="$XR1_EXPERIMENT_PARENT/xr1-fresh-$("$XR1_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)')"
# Do NOT mkdir XR1_RUN_ROOT: the allocated launcher must exclusively create it.
cd "$XR1_SUBMIT_DIR"

# Read-only submission measurements; persist the actual result, never hand-edit PASS.
"$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_storage_preflight.py" \
  --experimental --stage submission --run-root "$XR1_RUN_ROOT" \
  --filesystem-source "$XR1_FILESYSTEM_SOURCE" --project-id "$XR1_LUSTRE_PROJECT_ID" \
  > "submission-$(basename "$XR1_RUN_ROOT").storage.json"
# Stop here if preflight fails. The allocation will independently recheck it.

"$XR1_PYTHON" "$XR1_WORKTREE/scripts/claix_b24_launch.py" \
  --mode experimental --experimental-unqualified --run-root "$XR1_RUN_ROOT" \
  --overrides "$XR1_PATH_OVERRIDES" --filesystem-source "$XR1_FILESYSTEM_SOURCE" \
  --project-id "$XR1_LUSTRE_PROJECT_ID" --python "$XR1_PYTHON"

# Intended initial submission; NOT executed by Task 02:
sbatch --signal=B:USR1@1800 --export=ALL "$XR1_WORKTREE/scripts/claix_b24_experimental.sbatch"
```

Use `set -euo pipefail` in an operational script so failed preflight stops the
workflow. The four path overrides are train split paths, original pretrained
weights, frozen language policy and default root. The launcher replaces the root
with XR1_RUN_ROOT; scientific overrides are rejected. The ignored
`.claix_b24_reference/configs/hydra_overrides.json` is absent in this worktree;
the workflow therefore explicitly requires an existing reviewed overrides file.
Do not point model weights at a fine-tuned or qualification checkpoint.

For recovery, retain the original environment/path overrides and unchanged source
checkout. Set XR1_RUN_ROOT to the original root, and select its pointer:

```bash
export XR1_RESUME="$("$XR1_PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["XR1_RUN_ROOT"])/"project_xr1_doosan"/"xr1_doosan_b24_experimental_v1"; print(p/json.loads((p/"latest.json").read_text())["path"])')"
# The launcher/entrypoint verify this pointer, receipt, payload and telemetry;
# reading latest.json alone never authorizes a restore.
cd "$XR1_SUBMIT_DIR"
# Intended recovery submission; NOT executed by Task 02:
sbatch --signal=B:USR1@1800 --export=ALL "$XR1_WORKTREE/scripts/claix_b24_experimental.sbatch"
```

Do not recover into a different root or select an older checkpoint to bypass
telemetry mismatch. If telemetry extends beyond the last verified checkpoint or
ends in a partial row, preserve the directory and investigate; automatic rollback
is deliberately refused. A run at the terminal horizon cannot resume further.

## CPU validation and remaining acceptance

Validation commands, from this worktree:

```bash
/home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_claix_experimental_cpu_tests.py
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
bash -n scripts/claix_b24_experimental.sbatch
git diff --check
```

The focused suite uses real CPU torch serialization, checksum/receipt validation,
publication, rotation, optimizer/schedule creation and runtime safe-stop callbacks.
The actual training-main function body executes with synthetic file identities
and mocked framework/registry/CUDA metadata boundaries. It exercises fresh startup,
verified same-run recovery and rejection before model construction. No real model
is constructed or initialized. Its YAML fixture composes the repository's global
defaults without Hydra. Missing Lightning Callback/CheckpointIO bases are explicitly
stubbed; legacy tests get explicit synthetic path overrides, not real staging.
The I/O suite uses its established mmengine/backbone/Lightning-base stubs and real
CPU dataset/DataLoader operations on synthetic files. No package was installed.

Final results: **112 CPU tests passed, 0 failed, 0 skipped** in the selected suites:
74 focused launcher/release/storage/checkpoint/runtime tests (4.947 seconds) and
38 Task 01/01B I/O regressions (0.666 seconds). Full outputs are retained in
[focused_cpu.txt](task02_evidence/focused_cpu.txt) and
[io_cpu.txt](task02_evidence/io_cpu.txt). Earlier harness-fixture failures were
corrected (command-line extraction, expected foreign-directory exception and
missing ignored path-overrides fixture); final reruns have no failures.

Python AST syntax parsing, Bash syntax and `git diff --check` passed. Tests verify
that the production wrapper/release validator/production recipe remain identical
to committed Task 01B. One pre-existing launcher walltime assertion still expected
23:45:00 despite Task 01B source requiring 20:00:00; that assertion now matches
the committed behavior. The legacy duplicate-root lock test covers its original
three profiles, while experimental recovery locking has a dedicated test.

Two legacy Hydra-dependent launch tests were not selected by the dependency-light
harness; this is **not** a claim that the complete framework suite ran. The
existing full Lightning/Hydra harness remains for the existing CLAIX environment.
No tests selected for execution were skipped.

## Exact modified and added files

Modified:

- `scripts/claix_b24_launch.py`: explicit mode, allocation/storage/run identity,
  recovery guards, persistent provenance and existing-watchdog configuration.
- `scripts/claix_b24_storage_preflight.py`: independent experimental storage,
  live quota headroom and project inheritance checks; production policy retained.
- `xr1/tools/train_doosan_compact.py`: experimental identity/provenance and
  separate same-run restore path without the historical bridge.
- `xr1/mibot/models/runner/doosan_compact_runner.py`: carry and verify experiment
  metadata through the existing compact checkpoint hooks.
- `xr1/mibot/utils/claix_checkpoint.py`: optional expected experimental identity,
  receipt binding and early foreign-checkpoint rejection.
- `xr1/mibot/utils/claix_contract.py`: existing frozen inventory identity constant
  and experiment-specific extension of the existing resume fingerprint.
- `xr1/mibot/utils/claix_runtime.py`: bind experimental checkpoint plugin loads to
  the expected fingerprint/run; default strict behavior preserved.
- `xr1/tests/test_claix_launch.py`: align stale walltime assertion and scope the
  existing lock fixture to its original profiles.

Added:

- `scripts/claix_b24_experimental.sbatch`: dedicated opt-in 20-hour H100 wrapper.
- `xr1/configs/claix_b24_experimental.yaml`: inherit unchanged scientific recipe.
- `xr1/mibot/utils/claix_experiment.py`: frozen identity and same-run resume checks.
- `xr1/tests/test_claix_experimental.py`: focused synthetic CPU tests.
- `xr1/tests/run_claix_experimental_cpu_tests.py`: explicit offline harness.
- `docs/claix_b24/LUSTRE_FRESHSTART_TASK02.md`: this report and operational workflow.
- `docs/claix_b24/task02_evidence/focused_cpu.txt`: actual final CPU test output.
- `docs/claix_b24/task02_evidence/io_cpu.txt`: actual Task 01/01B regression output.

Remaining CLAIX-only acceptance checks (not run): actual sbatch/GRES availability,
Slurm field and saved-script output compatibility, physical H100 visibility,
20-hour resource request and USR1 delivery; live user/project quota output,
project inheritance and HPCWORK mount identity; frozen data/ZIP provenance and
external immutability; original weight/language hashes; real Hydra composition and
Lightning/CUDA integration in the existing environment; actual step-zero counters,
optimizer/scheduler initialization, worker timeout/cache records, RSS/throughput,
watchdog diagnostics, persistent checkpoint publication/rotation/safe stop and
exact interrupted same-run restoration. These are experimental acceptance checks,
not fabricated release evidence or completed production qualification.

No Slurm job was submitted, no GPU training or standalone H100 qualification ran,
and no dataset/model was downloaded. Historical checkpoints, historical release
records, the old qualified worktree, architecture/losses/scientific recipes and
Task 01/01B cache/loader implementations were not modified. No commit or push was
performed. All repository edits are confined to this experimental worktree.
