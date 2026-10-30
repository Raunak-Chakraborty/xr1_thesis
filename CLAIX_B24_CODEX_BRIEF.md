# XR1-5B / Doosan — CLAIX production-readiness engineering

## Your role

Act as the senior PyTorch/Lightning distributed-training engineer
responsible for making the first results-producing XR1 fine-tuning
experiment safe, reproducible, measurable and resumable.

Do not merely propose changes. Implement code, tests, documentation
and launch artifacts within this isolated development branch.

You have permission to edit this Git worktree and execute local
lightweight CPU tests.

You do NOT have permission to submit Slurm jobs, run GPU jobs,
push to GitHub, rewrite history, delete user files, or change the
approved training dataset.

## Source authority

Pinned source commit:
52676ccfa3b52f05de51d3d7407f33855159ac95

Original branch:
thesis/doosan-xr1-fp32-compact-v1

Development branch:
thesis/claix-b24-production-v1

Read the repository's AGENTS.md if one exists.

Read ALL relevant files under:

  .claix_b24_reference/

Start with:

  .claix_b24_reference/SOURCE_STATE.txt
  .claix_b24_reference/configs/hydra_overrides.json
  .claix_b24_reference/configs/production_b24_v1/
  .claix_b24_reference/benchmark/benchmark_summary.json
  .claix_b24_reference/worker_tests/
  .claix_b24_reference/long_run_candidates/

These are actual CLAIX handoff artifacts, not generic examples.
They are reference material, not permission to overwrite them.

Verify the files and existing source implementation before editing.

## Experiment contract

Robot: Doosan M1013 with SCHUNK gripper.
Policy: Xiaomi XR1-5B, force-free imitation learning.
Data: real teleoperation demonstrations, two RGB cameras.

Training episodes: 255.
Training frames: 289703.
Validation episodes: 34.
Test episodes: 34.

Preserve existing Doosan state, action, normalization,
language policy, frame-selection and action-horizon semantics.

Training:
- 1 NVIDIA H100 on CLAIX c23g.
- Batch size 24 per GPU.
- MAX_LENGTH=12288.
- Certified maximum 413 tokens/sample.
- Worst-case 24-sample pack: 9912 tokens.
- Preserve all 24 samples per batch.
- BF16 model forward/backward.
- FP32 trainable master parameters and optimizer state.
- Train only the approved last transformer block.
- Keep frozen parameters frozen.
- DDP-compatible compact checkpoint/resume.
- Zero gradient accumulation unless explicitly changed.

Slurm:
- Requested allocation: 23:45:00.
- Aim to checkpoint and exit approximately 30 minutes
  before allocation expiry, leaving sufficient write margin.
- Training may finish naturally before the allocation limit.
- Do not alter batch size or sample ordering after training begins.

The 48284-step/four-frame-pass horizon in the CLAIX plan
is PROVISIONAL, not evidence that four passes are optimal.
Preserve it as a proposed configuration. Do not silently
choose different hyperparameters or claim convergence.

The earlier H100 benchmark tested batch sizes 1,2,4,8,
not 24. Its timings excluded some DataLoader waiting.
Do not claim validated batch-24 memory or throughput.

## Phase A — Source audit and tests

Inspect these implementations and their integration:

- tools/train_doosan_compact.py
- mibot/data/datamodule/doosan_datamodule.py
- mibot/data/datasets/doosan_json_dataset.py
- mibot/data/datasets/json_dataset.py
- mibot/data/collate/custom_collate.py
- mibot/utils/doosan_replay.py
- compact FP32 runner and checkpoint utilities
- scheduler implementation
- applicable tests and Hydra configurations

Identify actual assumptions, existing invariants, risks
and current test coverage.

Inspect the transferred CPU worker probe critically:
it has NEVER completed on CLAIX due to a Slurm outage.
Do not treat its PASS markers as verified outcomes.

Create docs/claix_b24/PROGRESS.md recording each phase,
test evidence and unresolved hardware-dependent checks.
Update that document throughout implementation.

## Phase B — Multiprocessing with deterministic resume

Enable configurable 4 and 8 DataLoader workers while
preserving the exact sample sequence and replay semantics.

Study DistributedSampler, RankOffsetSampler, worker seeds,
persistent_workers, prefetch_factor and Lightning state.

Use in-order delivery and fail-closed sampler-contract checks.
Distinguish prefetched indices from COMPLETED optimizer steps.

At a resume boundary, continue exactly at the correct
completed global step. Never advance the saved cursor merely
because workers prefetched samples.

Preserve deterministic per-sample augmentation.
Verify repeatability across workers and on resume.
Do not alter image augmentation policy as a shortcut.

Include tests for workers=0,4,8 and partial-batch resumes.
Explicitly test failure cases.

Avoid excessive CPU RAM consumption. Worker count is not
proof of GPU utilization: measure end-to-end throughput.

## Phase C — Data integrity and token packing

Make MAX_LENGTH=12288 an explicit production setting,
applied before collator initialization.

Audit CustomCollate for silent sample dropping. Production
must fail closed if a batch unexpectedly loses any sample.
Do not hide, silently discard, or silently requeue samples
in a way that changes replay order.

Verify 24 retained examples and two views per example
on REAL batch-24 data when CLAIX becomes available.

Handle potential packed-attention memory scaling honestly.
The 9912-token bound is certified, but batch-24 H100
GPU memory has NOT yet been measured.

## Phase D — Safe long-run execution and resume

Build a production launcher and a graceful-stop mechanism.

Requirements:
- 23:45:00 Slurm allocation maximum.
- Safe early stop with enough time to finish checkpoint I/O.
- Signal handlers may set flags but must not save state
  asynchronously or reentrantly inside a signal handler.
- Stop on a completed optimizer-step boundary.
- Save optimizer, scheduler, model trainable state,
  Lightning loop progress, sampler cursor and RNG state.
- Validate checkpoint completeness and readability.
- Atomic/transactional publication where feasible.
- Resume without repeating or skipping optimizer steps.
- Maintain correct cosine-scheduler state.
- Preserve a durable checkpoint at step 2000.
- Define rotation and checkpoint-retention policy.
- Validate compact checkpoint size and required metadata.
- Detect and reject incompatible resume configurations.
- Avoid launching duplicate production jobs.
- Do not depend solely on Slurm TERM delivery.
- Never claim a full checkpoint is safe just because a
  filename exists.

Add deterministic unit/integration tests for checkpoint
contract, step cursor and graceful-stop behavior.

## Phase E — Telemetry and validation

Measure actual end-to-end examples/second, including
DataLoader waiting, not only GPU-stage step timing.

Provide telemetry for:
- Global optimizer step and samples consumed.
- Packed token length and retained examples.
- CPU data wait/pipeline overhead.
- End-to-end throughput.
- GPU utilization and GPU memory.
- CPU memory and worker failures.
- Loss and learning rate.
- Checkpoint duration, path, size and verification result.
- Estimated remaining time and safe-stop deadline.
- Fatal errors, NaNs, nonfinite gradients and skipped steps.

Keep logs persistent, timestamped, concise and searchable.

Design a valid held-out evaluation protocol using the
34 validation episodes while keeping the 34 test episodes
untouched for final reporting.

Inspect whether the current architecture supports validation
directly. Do not falsely claim that Lightning validation
or a specific metric works without implementation/tests.

Separate validation data from training and avoid leakage.

## Phase F — Test and documentation gates

Run every practical local CPU test without loading XR1-5B
or requiring the complete CLAIX dataset.

Use synthetic fixtures or mocked dependencies where needed,
and clearly distinguish synthetic tests from real-data tests.

Do not download giant pretrained weights onto the laptop.
Do not invent successful benchmarks.
Do not submit anything to Slurm.
Do not change the production source on CLAIX.

Produce:
1. Working code modifications in the isolated branch.
2. Focused unit/integration tests.
3. Exact local test commands and outputs.
4. PROGRESS.md recording completed/unverified gates.
5. CLAIX batch-24 qualification launcher and test plan.
6. Production launcher draft and safety checks.
7. Resume/recovery documentation.
8. Exact file-level changes and their rationale.
9. Remaining compute-node-only verification steps.
10. Clearly identified blockers.

Make incremental, reviewable commits for successfully tested
phases if practical. Do not push to GitHub.

Do not ask for approval for routine local code investigation
or lightweight tests. Continue through the phases.

If a dependency is missing, try an isolated lightweight
test strategy rather than stopping immediately.

If correctness requires CLAIX hardware or data, mark that
gate UNVERIFIED and continue with independent tasks.

## Completion criteria

Do NOT describe the system as production ready merely
because local tests pass.

Finish by stating:
- Changes implemented.
- Tests actually executed and their results.
- Remaining GPU/Slurm/data checks.
- Risks and unresolved design decisions.
- Commits created, if any.
- Recommended next commands, without executing any
  GitHub push or Slurm submission.

Do not change the base branch or other thesis repositories.
