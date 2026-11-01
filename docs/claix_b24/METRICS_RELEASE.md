# Production metrics and retention delta

Reviewed parent: **02bd6c54d883b21f8ea039995038dc2c0da68124**.
The operator reports actual CLAIX job **5357168** passed all nine qualification
phases, exact six-step restart comparison, sustained 251-step training, safe stop
with verified checkpoint, and real validation smoke: **14.44 examples/s**, peak
reserved GPU memory **67.03 GiB**. Those results qualify that parent source. This
local change did not rerun or independently retrieve the cluster artifacts.
Earlier runbook qualification warnings describe historical development stages.

This change adds observability, checkpoint retention, and source evidence only.
It does not change loss calculations, gradients, optimizer behavior, replay,
dataset/normalization, action semantics, two-camera inputs, or compact payloads.
The original combined-qualification worktree, branch, and H100 experiment
artifacts are untouched. No job, GPU training, commit, push, or upload is performed.

## Exact source delta

| File | Change |
| --- | --- |
| `xr1/configs/claix_b24_production.yaml` | Add only `trainer.log_every_n_steps: 1`. |
| `xr1/mibot/utils/claix_checkpoint.py` | Preserve historical steps and latest stop; validate pointer and deletion candidates, reject unsafe receipt paths/symlinks, retain all receipts. |
| `xr1/mibot/utils/claix_runtime.py` | After verified boundary publication, write a small `safe-stop.json` receipt pointer for non-interval saves. Existing per-step telemetry stays intact. |
| `scripts/claix_b24_launch.py` | Per-invocation source/commit/release manifest; optional controller-side `nvidia-smi` sampler and cleanup. Release gates and signals retain their existing behavior. |
| `xr1/tests/test_claix_launch.py` | Mock the new read-only git identity probe in existing child-process fixtures. |
| `xr1/tests/test_claix_runtime.py` | Allow CPU fixture callers to supply a logger and logging interval; existing tests keep their defaults. |
| `xr1/tests/test_claix_metrics_release.py` | Additional real CPU Lightning CSV/resume/optimizer equality, retention, pointer, source gate, monitor lifecycle/failure tests. |
| `docs/claix_b24/RUNBOOK.md` and this document | Link current instructions; document evidence, limitations, retention and capacity. |

The scientific recipe stays at maximum LR **2e-5**, minimum **5e-6**, starting
**5e-7**, warmup **500**, horizon **48284**, batch **24**, workers **4**, last_n **1**,
BF16 compute / FP32 trainables, AdamW, checkpoint interval **250**, allocation
**23:45:00**, production safe-stop margin **1800 seconds**. The inherited
`gradient_clip_val: 1.0` and norm clipping are unchanged. No schedule or dataset
field is edited. The production configuration diff contains just the logging line.

Global gradient norm logging is deliberately omitted. Even an in-place-free
scalar reduction reads all **141255168** selected gradient elements and adds
kernels/synchronization every 50 updates. CPU fixtures cannot demonstrate
negligible overhead on the actual H100. The requirement to establish that
condition is therefore not met. Existing missing/nonfinite-gradient guards and
nonfinite-loss rejection remain unchanged, with their CPU tests passing.

## CSV interpretation and complete resume logs

The existing runner already logs all required components. The new interval lets
Lightning send them to its existing local CSVLogger after **every optimizer
update**. The local Lightning logger connector calls `logger.save()` on every
logged update; no custom writer or change to losses is needed.

| Required quantity | CSV column |
| --- | --- |
| Global step | `step` (Lightning's zero-based training log index) |
| Total training loss | `train/loss` |
| Frequency loss | `train/loss_freq` |
| L1 loss | `train/loss_l1` |
| MSE loss | `train/loss_mse` |
| Score loss | `train/loss_score` |
| Learning rate | `lr` |
| Packed token count | `train/token` |

With accumulation fixed at one, CSV `step=0` describes the first optimizer
update; JSONL `step=1` describes its completed boundary. Join on
`CSV.step + 1 = JSONL.step`. CSV LR is the rate used for that update, logged
inside `training_step`; JSONL LR is read after the scheduler advances, for the
next update. These existing semantics are preserved.

Each invocation/resume gets a new `metrics/version_N/metrics.csv`. Keep **all
versions**, ordered by invocation, to obtain the complete CSV history. The
existing `telemetry.jsonl` opens in append mode across resumes. A crash followed
by recovery from an earlier checkpoint can leave repeated step indices in the
historical logs; preserve them and use invocation/train-start boundaries to
identify the replayed segment. Do not erase or silently deduplicate raw evidence.
CPU tests confirm eight consecutive CSV updates over two logger versions (stop
at three, resume to eight), all eight required columns, matching per-step JSONL
losses, and exact reference weights, AdamW moments/counters and scheduler state.

## Exact evidence locations

Let **R = `$XR1_RUN_ROOT`**, and
**T = `R/project_xr1_doosan/xr1_doosan_b24_results_v1`**. All R/T paths are persistent
and reused on an explicit resume; the launcher refuses accidental overwrites.

| Evidence | Location |
| --- | --- |
| Every-step training telemetry and events | `T/telemetry.jsonl` (append) |
| CSV losses, LR and tokens | `T/metrics/version_N/metrics.csv` (new version per invocation) |
| Full resolved configuration per invocation | `T/resolved-<time_ns>.json` |
| Current expanded runtime configuration | `T/resolved_compact_config.json` (latest invocation; use the timestamped configs for history) |
| Dataset inventory and episode checksums | `T/training_inventory.json` (frozen identical inventory across valid resumes) |
| Resume fingerprint and replay contract | `T/resume_contract.json`; also embedded in every compact checkpoint |
| Verified compact payloads | `T/step-NNNNNN.ckpt` |
| Immutable receipts, including pruned payloads | `T/step-NNNNNN.verified.json` |
| Latest verified payload pointer | `T/latest.json` |
| Most recent verified stop/finish pointer | `T/safe-stop.json` |
| Exact checkout commit, dirty status, source file hashes, source-tree hash, production plan, release record and Slurm identifiers | `R/launch-<time_ns>.provenance.json` (new per invocation) |
| Child stdout/stderr, including early startup failures | `R/launch-<time_ns>.out` and `.err` |
| Controller stop request evidence | `R/launch-<time_ns>.stop.json`, when requested |
| Independent GPU samples / monitoring errors | `R/launch-<time_ns>.gpu.jsonl`, when monitoring is enabled and log storage works |
| Hydra output and its configuration files | `R/hydra/production-<time_ns>/` |
| Slurm/controller stdout/stderr with current sbatch defaults | `$SLURM_SUBMIT_DIR/claix_b24_production_<jobid>.out` and `.err` |

The resolved configurations preserve expected immutable model and language
SHA256s, checked against their external originals before construction. Dataset
checksums come from the full per-episode training inventory. Source manifests
hash all Python, YAML configuration (including normalization) and `claix_b24*`
script files using the same file scope as the release gate. Compact metadata
preserves backbone/language identities, selected FP32 keys/count/dtypes, optimizer
and replay/RNG/scheduler metadata. Frozen backbone weights remain external; no
pretrained 5B copy is made. `model.params.source_commit` retains its existing
baseline-lineage meaning; the **actual checkout commit** is recorded separately
in each launch manifest. A dirty checkout is explicitly recorded, never presented
as the pristine qualified commit.

Use a persistent submission working directory for the Slurm files, or explicitly
set Slurm output/error paths under R at a later authorized submission. No earlier
H100 directory should be reused. Archive the reviewed release record, actual
qualification reports/raw evidence referenced by it, and the exact reviewed source
snapshot alongside these records. Nothing is sent to W&B or an external service.

## Measurement scope

Every completed step emits the existing timestamp, `step_seconds`, cumulative
elapsed time, retained examples, views, packed tokens, consumed sample count,
loss, LR, pipeline wait, cumulative effective examples/s, remaining-time estimate,
safe-stop countdown, FP32/BF16 storage checks and CUDA allocated/reserved/peak
allocated/peak reserved counters when CUDA is available. CPU/worker RSS is queried
each step when psutil is available. Missing counters remain null/error-tagged;
they are not measurements. RSS includes shared pages; it is not unique physical
memory. Precision flags check storage dtypes, not every intermediate operation.

Step and throughput timers include prior callbacks/checkpoint I/O and loader
startup; the final `fit_end` rate includes final checkpoint verification. The
`pipeline_wait_seconds` interval also includes host/callback/I/O overhead and is
not an isolated dataloader-only timer. These timing scopes are unchanged.
Allocator peaks are cumulative process peaks, not newly reset per-step peaks.
Checkpoint, fatal, graceful-stop, and fit-end events are event-driven.

`torch.cuda.utilization()` is still attempted each step by existing telemetry;
on CLAIX its missing dependency can produce null plus `ModuleNotFoundError`.
The independent **nvidia-smi device utilization and device memory values are
sampled approximately every 15 seconds**, with query duration added to that
interval. They are driver samples, not step averages or PyTorch allocator counts.
UTC timestamps allow alignment. Device index/UUID and `CUDA_VISIBLE_DEVICES` are
recorded: nvidia-smi can list devices beyond the CUDA-visible training device;
select the allocation's UUID during analysis. Unsupported numeric counters are
null. The sampler uses an existing executable only, performs no installation or
download, runs outside the training process, and uses a three-second query timeout.
A missing executable, query/driver/parsing failure, or output failure disables
monitoring only. A daemon thread is stopped/joined at controller exit (up to four
seconds); timed-out query subprocesses are killed/reaped by `subprocess.run`.
No signal handler is installed or changed by monitoring, and Slurm signals retain
the reviewed controller stop path. Set `XR1_NVIDIA_SMI_MONITOR=0` to disable this
optional production-only monitor. The real CLAIX driver query remains untested.

## Retention and capacity preflight

After successful publication and verification, preserve the newest **two usable
verified recovery checkpoints**, plus permanent steps **2000, 5000, 10000, 20000,
30000, 40000**, and the most recent verified safe-stop/final checkpoint.
`safe-stop.json` pins the last allocation's final checkpoint while its resumed
run progresses; a subsequent verified stop replaces that pin. An overlapping
milestone/stop/recovery payload is stored only once. This does not recreate
historical checkpoints already deleted before this change; existing step 2000
and prior experiment directories are left untouched.

All receipts remain as immutable audit evidence after payload pruning. A receipt
without its payload is not resumable. Rotation requires canonical matching names,
schema, configuration identity, payload size, no symlinks, SHA256 and complete
checkpoint validation before deletion. Orphan, malformed, corrupted, foreign,
unverified, symlinked and unrelated files are preserved. The current `latest.json`
must equal the newest candidate's receipt and checksum; an inconsistent latest
or stop pointer prevents deletion. Publication's payload/receipt/pointer ordering,
no-clobber transaction, compact schema and resume validation remain unchanged.

At **141255168** FP32 trainables, masters plus two AdamW moments are
`12 * 141255168 = 1695062016 bytes`, approximately **1.579 GiB/checkpoint**, plus
small metadata. The existing size ceiling is **1.641 GiB** (adds 64 MiB headroom).
Six historical + two recovery + a distinct prior safe-stop = at most **nine**
normal retained payloads, **14.77 GiB** at the ceiling. A new payload before
rotation adds one: **16.42 GiB** at the ceiling. The temporary publication file
and published hard link share the same payload blocks; they are not two copies.
Receipts are small; budget another **1–2 GiB** for JSONL, CSV, child/Slurm logs,
configs, manifests and operational headroom. Orphans, corrupted/unrelated files,
prior runs and filesystem quotas can increase actual usage beyond this estimate.

**Recommended operator preflight:** require at least **30 GiB** of additional
usable filesystem space and user/project quota at the existing parent of R
before a fresh run. Inspect both `df -B1 <persistent-parent>` and the site's quota
report, inode availability, and existing run/orphan sizes. Before each resume,
recalculate free quota after existing retained payloads/logs; reserve at least
one full new checkpoint plus log growth and substantial safety headroom. Check
that the compact ceiling still matches actual receipts. Do not delete orphans
or unrelated files to satisfy capacity automatically. This is a recommendation,
not a changed launcher resource/release rule.

## Validation, performance and release requirements

The existing **86 CPU tests passed** before modification. The final suite and
focused command results are recorded in `evidence/metrics_release_verification.json`;
raw output is in `evidence/metrics_release_cpu_tests.txt` and
`evidence/metrics_release_focused_tests.txt`. These use the existing local
**torch 2.6.0+cu124 / Lightning 2.5.1.post0**, CPU only. The harness mocks unavailable
mmengine/backbone imports and DataModule rank lookup; real CPU Lightning, AdamW,
CSVLogger, replay and compact checkpoint I/O run. The focused loss components
are synthetic fixtures, not claims of real XR1 model evaluation.

Per-step CSV adds local filesystem writes/metric scalar handling. The optional
monitor adds one small subprocess about every 15 seconds, outside training, and
bounded controller shutdown delay. Stronger rotation checks reread/hash and
validate recovery/deletion candidates at checkpoint time, adding host RAM and
disk I/O; they do not allocate GPU state. This I/O may reduce throughput and must
fit the **unchanged 30-minute** production safe-stop margin. No new H100 overhead,
throughput or peak-memory result is claimed. CPU equality is evidence of unchanged
training arithmetic, not proof of H100 timing equivalence.

**Any Python/config change, including the new tests, changes the engineering
source digest and therefore the resume fingerprint.** Existing release records
and job 5357168 checkpoints do not bind this modified tree. The launcher still
requires the exact source/plan and all nine PASS gates with matching evidence
hashes; checkpoint resume still requires the exact configuration/source/data
fingerprint. No exemptions, rebinding of old checkpoints, or gate relaxation
are added. Do not update the old release record's digest and relabel its hardware
evidence as a run of this tree.

Before production, review the exact delta and scientific recipe; freeze a new
source revision/snapshot; preserve the parent job's genuine evidence; obtain and
review hardware evidence for the changed source (per-step CSV, exact restart,
checkpoint/stop I/O under the new retention policy, and monitor behavior). Retain
all other required gate evidence and explicit LR/horizon approval. Build a new
release record for the exact final source digest and production plan with honest
provenance for any reviewed historical evidence. Complete the storage/quota
preflight and confirm persistent Slurm log paths and site GRES spelling. Use a
fresh production run root; resume only checkpoints produced by that identical
frozen source/configuration. These remaining release actions were not executed
in this CPU-only development session.
