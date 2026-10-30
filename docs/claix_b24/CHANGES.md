# File-level engineering changes

All source edits are in the isolated production worktree. Baseline source commit
and transferred reference files are preserved. New code is opt-in through the new
production Hydra configuration and launchers, except stricter generic replay
validation and the Doosan sample-table/cache representation (tested for identical
selection/content semantics).

| File | Change and reason |
| --- | --- |
| `xr1/mibot/utils/doosan_replay.py` | Replay v2, workers 0/4/8, whole-batch/rank sampler checks, worker seeds, terminal-checkpoint validation and finite master/moment checks. |
| `xr1/mibot/data/datamodule/doosan_datamodule.py` | Ordered configurable workers, isolated generator, bounded prefetch/timeout, strict collator and original-epoch length for Lightning late resume. |
| `xr1/mibot/data/datasets/doosan_json_dataset.py` | Compact numeric sample table preserving original shuffle cycles; two-entry process-local JSON cache; separate canonical validation adapter without jitter. Training jitter and sample seeds unchanged. |
| `xr1/mibot/data/collate/custom_collate.py` | Optional production strict packing raises on overflow, >413 tokens/sample or wrong view count. Generic nonproduction default stays opt-out. |
| `xr1/mibot/models/runner/doosan_compact_runner.py` | Nonfinite loss/gradient and missing-gradient rejection; coordinated per-rank RNG/runtime metadata and verified config resume. Approved selection implementation remains unchanged. |
| `xr1/mibot/utils/claix_checkpoint.py` | Trusted compact-state validation, strict Lightning optimizer boundary, atomic no-clobber payload/receipt/pointer transactions, checksums, read-back, RNG helpers and retention. |
| `xr1/mibot/utils/claix_contract.py` | Content-hashed split inventory/leakage audit and immutable data/config/source resume fingerprint. |
| `xr1/mibot/utils/claix_runtime.py` | Step-boundary deadline/signal handling, checkpoint I/O plugin, sampler checks, persistent JSONL telemetry, finite/skipped-step failure reporting. |
| `xr1/mibot/utils/claix_evaluation.py` | Instruction-only inference payload without teacher-forcing targets, physical-unit masked MAE, micro/episode-macro aggregation and RNG isolation. |
| `xr1/tools/train_doosan_compact.py` | Production preflight, explicit MAX_LENGTH, source/data/config identity checks, verified resume, runtime callback/plugin, local CSV logger and per-invocation configs. Global warning suppression removed. |
| `xr1/configs/claix_b24_production.yaml` | Proposed one-GPU/batch-24/12288-token/48284-step/500-warmup recipe; bounded worker prefetch; automated Lightning checkpointing replaced only in this profile. |
| `scripts/claix_b24_launch.py` | Dry-run default, allocation-derived deadline, source/plan/evidence release gates, path-only staged overrides, run-root duplicate lock, persistent raw logs and controller signal handoff. Never submits jobs. |
| `scripts/claix_b24_production.sbatch` | Unsubmitted production-only 23:45:00 c23g H100/12-CPU/96-GiB draft; full release record and 30-minute early signal required. |
| `xr1/tools/claix_b24_worker_qualification.py` | Real-data compute-node worker/content/resume probe using the production DataModule and full frozen horizon. Not executed locally. |
| `xr1/tools/claix_b24_evaluate.py` | Standalone held-out smoke/full validation command with trusted compact reload and frozen split audit. Not executed against XR1 locally. |
| `xr1/tests/run_claix_cpu_tests.py` | Explicit isolated offline harness, mocks recorded, CUDA hidden, no weights/processor downloads. |
| `xr1/tests/test_claix_replay.py` | Real workers 0/4/8, actual image jitter, repeatability/resume/prefetch/partial-batch/fault and virtual multi-rank invariants. NumPy IPC workaround documented. |
| `xr1/tests/test_claix_collate.py` | Synthetic processor certification-bound 24-example/48-view pack and fail-closed overflow/view/token tests. |
| `xr1/tests/test_claix_checkpoint.py` | Verified roundtrip/corruption/orphan/disk-fault/retention/terminal-state and approved last-one freeze-policy tests. |
| `xr1/tests/test_claix_runtime.py` | Real CPU Lightning optimizer/scheduler/checkpoint/resume comparisons, deadline/signal, sampler/skipped-step/gradient failures, cosine endpoints. Rank lookup mocked due socket restrictions. |
| `xr1/tests/test_claix_evaluation.py` | Target-leakage exclusion, semantic masks/physical-unit aggregation, RNG/model-mode isolation, split audit and composed production Hydra contract. |
| `xr1/tests/test_claix_launch.py` | Dry-run/non-Slurm/release/override/walltime/duplicate-run safety checks; subprocess GPU launch mocked and never executed. |
| `xr1/tests/test_training_logic.py` | Correct case-sensitive `XR1` module import; existing flow-loss tests can now import on Linux with full dependencies. |
| `docs/claix_b24/PROGRESS.md` | Phase status, actual evidence, limitations and unresolved CLAIX gates, updated throughout. |
| `docs/claix_b24/RUNBOOK.md` | Qualification, release, validation, checkpoint retention and recovery commands/limitations. |
| `docs/claix_b24/release_record.template.json` | Every required real-qualification gate starts UNVERIFIED; no fabricated release. |
| `docs/claix_b24/evidence/` | Exact test output/commands, dry-run command, static-check results and transferred-reference checksum manifest. |

## Focused qualification/correctness review (2026-10-30)

| File | Additional change |
| --- | --- |
| `scripts/claix_b24_launch.py` | CPU/H100 qualification modes with independent authoritative allocation deadlines, GPU/CPU/RAM checks and 120/600-second margins; production alone requires exact 23:45:00 and all existing release gates. Explicit Hydra path quoting/output location, offline datasets/W&B environment, deadline request and restored signal handlers. |
| `scripts/claix_b24_cpu_qualification.sbatch` | New zero-GPU, 12-CPU, 48-GiB, 20-minute draft; all workers measured. |
| `scripts/claix_b24_h100_qualification.sbatch` | New one-H100, 12-CPU, 96-GiB, 45-minute draft; six-step smoke/resume. |
| `scripts/claix_b24_production.sbatch` | Fix mode to production and require release record; one H100, 12 CPUs, 96 GiB, 23:45:00. |
| `xr1/configs/claix_b24_production.yaml` | Remove unused hard-coded stop margin: walltime belongs to launcher profiles, shared GPU data/scheduler recipe is unchanged. |
| `xr1/mibot/data/datamodule/doosan_datamodule.py` | Prefetch capped at two for deterministic workers; shared-memory budget and explicit worker shutdown helpers; original-epoch ReplayDataLoader length preserved. |
| `xr1/tools/claix_b24_worker_qualification.py` | Testable complete-batch fetch, sample counts, timings, tensor-IPC/RSS estimates, deadline/signal failure without PASS, 60-second timeout, pinning off and explicit cleanup between every case. |
| `xr1/mibot/utils/doosan_replay.py` | Require exact AdamW/scheduler counters instead of truncating fractional values. |
| `xr1/mibot/utils/claix_checkpoint.py` | Require epoch-zero current/total batch-end progress and matching receipt/payload configuration fingerprint. |
| `xr1/mibot/utils/claix_runtime.py` | Version/IPC telemetry and final receipt validation; fit-end rate includes final checkpoint I/O/verification. |
| `xr1/tools/train_doosan_compact.py` | Verify one visible H100; explicitly clean persistent workers in fit finally. |
| `xr1/tests/test_claix_launch.py` | Short/incorrect/missing/expired allocation cases, every release gate/checksum, real Hydra composition with quoted paths, CPU offline launch fixture, all-mode locks and existing results. No real child training. |
| `xr1/tests/test_claix_checkpoint.py` | Reject wrong current/total batch-end progress, wrong epoch, fractional counters and mismatched receipt configuration. |
| `xr1/tests/test_claix_runtime.py` | Exact consumed suffix, weights/moments/counters/scheduler comparisons at 1/3/6/7 for workers 0/4/8; delayed-loader/save timing scope and final receipt. |
| `xr1/tests/test_claix_replay.py` | Explicitly reap real worker processes and assert exit before resumed iteration. |
| `xr1/tests/test_claix_worker_qualification.py` | New probe fixtures for every worker case, complete sample counts, suffix equality, deadline/drop/failure cleanup and shared-memory rejection. |
| `docs/claix_b24/{PROGRESS,RUNBOOK,CHANGES}.md` / `evidence/review_*` | Correct walltime/commands, evidence scope and remaining CLAIX-only gates. Previous session evidence preserved. |

The recovery commit `3c41f20` is untouched. This review creates no commit and leaves
all changes in this worktree; shared Git metadata is outside writable roots.

## Remaining blockers and decisions

- No batch-24 H100 memory/throughput result. The transferred benchmark cannot
  qualify this batch, the 9912-token SDPA pack, or sustained utilization.
- No production tensor-IPC or real DDP/GPU/Slurm resume test on the exact CLAIX
  stack. The CPU mock/transport scope is explicit in every local evidence claim.
- Real frozen dataset, ZIP cache, original certificate and split isolation must
  be verified on CLAIX. Local synthetic fixtures are not dataset qualification.
- Real filesystem hard-link/rename/fsync/flock guarantees, checkpoint size/write
  duration and graceful Slurm stop margin are unverified. SIGKILL cannot save.
- Real validation inference/prompt compatibility and all 34-episode metrics are
  unverified. No final-test inference has been performed.
- Worker count, host-memory request, site Slurm GPU syntax and safe write margin
  must be qualified. RSS includes shared pages. GPU telemetry may require NVML;
  null fields must be resolved before a utilization claim.
- Learning rate and four-pass horizon remain proposed scientific choices.
- Replay v1 checkpoints require the original pinned code for recovery; v2 does
  not silently migrate them. No migration is claimed or implemented.

No commits were created: this worktree's Git metadata lives in the shared
repository outside the permitted writable roots, and `.git` is read-only under
the supplied permissions. The reviewable working-tree changes remain available.
No push, Slurm submission, GPU training, base-branch change, or reference mutation
was performed.
