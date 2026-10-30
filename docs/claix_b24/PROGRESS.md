# CLAIX batch-24 engineering progress

Baseline: `52676ccfa3b52f05de51d3d7407f33855159ac95`, branch
`thesis/claix-b24-production-v1`. No AGENTS.md exists in this worktree or its
ancestor directories. The complete brief and all 18 transferred reference files
were read before editing. Reference checksums are recorded separately below.

## Phase A — audit (completed locally)

- Existing fixed-horizon replay uses a seeded DistributedSampler at epoch zero,
  rank-local offsets derived from completed optimizer steps, and rejects workers.
  Dataset frame selection cycles shuffled populations without replacement.
- Per-sample prompt choice and image augmentation use Python random, already
  scoped to a stable path/frame seed. Image augmentation policy stays unchanged.
- CustomCollate defaults to 4096 tokens and silently skips overflowing samples.
- Compact runner promotes only selected last-N parameters to FP32 and leaves
  frozen weights BF16. Checkpoints include AdamW, cosine LambdaLR and loop state,
  but lack explicit RNG/config contracts and transactional verification.
- Existing launcher has no walltime stop, duplicate-run guard, durable step-2000
  retention, or persistent end-to-end telemetry. BaseRunner has no validation_step.
  XR1 eval mode returns generated actions; training losses depend on train mode.
- SDPA packed attention may allocate quadratic masks; 9912 certified tokens is
  not a measurement of batch-24 memory safety.
- Transferred benchmark job 5341437 tested batches 1/2/4/8 only, three steps each
  (plus batch-1 resume). Timings exclude some loader waiting. No extrapolated
  batch-24 claims are justified.
- Transferred worker scripts print PASS only when run successfully; no outcome
  report was transferred and they never completed on CLAIX. They bypass the
  DataModule's zero-worker restriction, allocate short eight-step sample streams,
  and do not exercise optimizer/Lightning resume, faults, or production RAM.
- Local system Python lacks torch/Lightning. A separate `/tmp` test environment
  is being attempted using installed torch; no model weights will be downloaded.

## Phase B — deterministic workers

Implemented: ordered workers 0/4/8, isolated generator/worker seeds, whole-batch
offset checks, fail-closed sampler checks, compact numeric frame table and bounded
per-worker JSON cache. Exact original frame-selection cycles and jitter policy
are preserved. Synthetic tests cover prefetch beyond a committed cursor, abandoned
partial batch, cross-worker content equality, faults, and two virtual sampler ranks.
Lightning epoch-budget reporting now preserves absolute loop progress on late
resume. Replay schema v2 deliberately rejects old v1 checkpoints.

Local multiprocessing uses real workers with NumPy pickle transport because the
sandbox denies AF_UNIX tensor descriptor passing. Gloo loopback creation also
failed; CPU Trainer tests mock only DataModule rank lookup. Production tensor IPC,
real distributed process groups and CLAIX Lightning 2.5.3 remain UNVERIFIED.

## Phase C — packing and integrity

Implemented: explicit 12288 production limit before collator construction; strict
overflow, 413-token and two-view checks. Synthetic mocked-processor worst-case
pack retains 24 examples / 48 views / 9912 tokens. Real-data checks remain
UNVERIFIED until run on the approved CLAIX dataset.

## Phase D — checkpoint and safe stop

Implemented: optimizer-boundary deadline/signal stops, full compact state/RNG/config
contract, strict loop/cursor validation, read-back verification, fsync/atomic rename,
SHA256 receipts and latest pointer, newest-two retention plus durable step 2000.
Launcher defaults to dry-run and rejects execution without Slurm; production also
requires source/plan-bound evidence for every release gate. Shared run-root flock
prevents duplicate runs. Slurm USR1 goes to batch controller; its handler sets a
flag, and ordinary code writes a stop request for the training callback. A separate
absolute deadline prevents dependence on signal delivery.

CPU integration: uninterrupted eight-step runs and resumes at steps 3 and 6 match
FP32 model weights, AdamW moments/counters and scheduler states for workers 0/4/8.
Deadline stop saves a verified step-1 checkpoint. Fault injection proves disk-write
failure preserves the previous checkpoint/pointer; corruption/orphan/config/NaN
checkpoints are rejected. Actual CLAIX filesystem durability, Slurm signals,
H100 restart, compact size and 30-minute write margin are UNVERIFIED.

## Phase E — telemetry and evaluation

Implemented persistent UTC JSONL telemetry for steps/samples, tokens/examples/views,
pipeline overhead, end-to-end throughput, loss/LR, RSS, GPU utilization/memory when
available, remaining-time estimates, checkpoint verification and fatal errors.
CPU tests exercise callback logs; these synthetic rates are NOT GPU benchmarks.

Implemented a standalone generated-action evaluator, canonical validation dataset
without training jitter, inference payload stripping all teacher-forcing targets,
train-stat physical-unit masked MAEs and episode-macro aggregation. Synthetic model
tests cover no leakage, masks, aggregation and RNG/mode restoration. Split inventory
checks enforce 255/34/34 episodes, 289703 training frames and cross-split media/hash
isolation. Real XR1 inference/metric qualification and full 34-episode validation
are UNVERIFIED. No Lightning validation_step is claimed; training has none. Test
data is used only for split-identity audit, never inference or model selection.

## Phase F — evidence and handoff

Final local run: **57 tests passed in 3.565 seconds**, exit code 0, using PyTorch
2.7.1+cu128 (CUDA hidden), Python 3.11.15, torchvision 0.22.1+cu128,
transformers 4.48.1 and Lightning 2.5.1.post0. Exact commands and raw output are
recorded in [evidence/commands.txt](evidence/commands.txt) and
[evidence/local_cpu_tests.txt](evidence/local_cpu_tests.txt). Missing mmengine registry
and Qwen3-VL import dependencies are explicitly mocked in the offline harness;
existing flow-loss methods are extracted verbatim via AST. No pretrained weights
or processor resources are loaded. An attempted isolated pip download failed DNS;
cached lightweight wheels provided Lightning/Hydra instead.

Additional executed gates: `python3 -m compileall -q xr1 scripts`, `bash -n
scripts/claix_b24_production.sbatch`, `git diff --check`, qualification launcher
dry-run, and SHA256 recheck of all 18 transferred files: **all exit 0**. HEAD
remains the pinned commit and the branch remains the isolated development branch.
The raw launcher plan is in `evidence/qualification_dry_run.txt`; reference hashes
are in `evidence/reference_sha256.txt`.

Earlier local runs are not hidden: the initial 40-test run had three errors from
denied tensor-IPC/Gloo sockets; a 44-test rerun had four errors from rank lookup in
the socket-free fixture (fixed by explicit sampler rank/world arguments); a
48-test run had two metric-assertion failures because source ACTION_EPS is 1e-6,
not zero (assertions now account for the unchanged source normalization). The
52-test, 56-test and final 57-test runs passed. These failures did not involve real
CLAIX data, weights, GPUs or Slurm.

File-by-file rationale is in [CHANGES.md](CHANGES.md). Qualification, release gates,
validation protocol and recovery commands are in [RUNBOOK.md](RUNBOOK.md).

| Phase | Laptop outcome | Remaining CLAIX gate |
| --- | --- | --- |
| A | Brief/reference/source audit complete | Original certificate/frozen dataset evidence |
| B | Worker order/augmentation/resume/fault fixtures pass | Real tensor IPC, DDP and production RAM |
| C | Strict 24/48/9912 synthetic pack passes | Real batch-24 camera/token integrity and H100 memory |
| D | Atomic verified checkpoint, retention, CPU resume and stop pass | CLAIX 2.5.3 DDP/GPU/Slurm/storage/write-margin qualification |
| E | JSONL telemetry and standalone evaluator implemented/tested | GPU telemetry, real inference and all 34 validation episodes |
| F | 57 CPU tests and static/dry-run/reference checks pass; handoff complete | All release gates intentionally remain UNVERIFIED |

No commits were created because the shared Git metadata is outside the allowed
writable roots and `.git` is read-only in this managed worktree. Source changes
remain reviewable in the working tree. No GitHub push or Slurm submission occurred.

The 48284-step horizon
and 500-step warmup remain provisional; no convergence or production-readiness
claim is made. No Slurm submission, GPU training, GitHub push, base-branch edits,
or reference-file edits are authorized or performed.
