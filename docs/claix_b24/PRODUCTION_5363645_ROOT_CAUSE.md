# Production 5363645: evidence review and recovery boundary

Reviewed locally on 2026-11-02. **Underlying cause remains undetermined.** The
demonstrated failure is a 180-second PyTorch DataLoader result-delivery timeout,
after optimizer step 8758. There is no evidence sufficient to attribute it to
Lustre, a particular sample, or a particular worker. No GPU test, SSH session,
Slurm submission, commit, or push was performed for this investigation.

## Evidence and identity

Authoritative repository: `Raunak-Chakraborty/xr1_thesis`; checkout and branch:
`16da563ce61e2b58952281b86c474a0490e99a52`,
`thesis/xr1-b24-recovery-reliability-v1`.

| Item | SHA256 |
|---|---|
| Engineering source | `852945709dac80152459c26ab5dab5e70e158b631ea27612d28ee2d172ef3d0b` |
| Original incident archive | `9ab26a84c9ca3e4cd123bbda89d73cd1de7bcadd82ed669ca3bd0c20708b3912` |
| CPU forensic archive | `8234090f6aaffeca718d38a9e28c7ec356850340f406e8d7a632df20142b71fa` |
| Step-8750 payload, per original verified receipt | `a7476e50eee739b2acf6823125dda0f41fc25596a02a98d056eee9fe131c2c11` |
| Resume configuration | `87ec83b0200ec9f8ab9e3cafa75079dd0d7881dcea0913c0d03ca0a1571687ea` |
| Frozen training inventory | `1cb1cfc2491f4ae98942c26a7ece63bff5076fdd28322e98d9b83dcba287a548` |
| Cache manifest bytes | `aabf3cae952aaf36fca827bf9d85c3df8430bbfc2aa8348b2feada797ddcff25` |

Original archives were read from
`/home/horus/robotics_thesis/xr1_claix/evidence/production_failure_5363645/`.
Their exact names are
`xr1_production_5363645_root_cause_evidence_v1.tar.gz` and
`cpu_rootcause_results.tar.gz`. Inspection copies were extracted under
`/tmp/xr1_reliability_evidence/`; originals were not modified. All source files
included in the incident archive match the authoritative checkout byte-for-byte.
The launcher fingerprint was recomputed and matches the declared source hash.

Reviewed: full traceback; original step and checkpoint telemetry; controller GPU
samples; Slurm accounting; original resolved and Hydra configurations; inventory
and cache manifest; source dataset/ZIP/sampler/collator/runtime/checkpoint paths;
CPU probe implementation, job script, reconstructed samples, nine probe outcomes,
decode output and loader output. The archives contain **no checkpoint payload,
episode JSON payloads, ZIP archives, original worker stacks, or kernel I/O trace**.
The 1,695,132,435-byte production checkpoint must still be hashed and deserialized
on CLAIX. Inventory review is not a local revalidation of the actual dataset.

## Timeline

Times below are CET (UTC+1); telemetry stores UTC. Slurm job `5363645` ran on
`n23g0026` for 04:20:55 and ended `FAILED`, exit `1:0`.

| CET | Observation |
|---|---|
| Nov 1 21:03:50 | Slurm start |
| Nov 1 21:05:12.626 | `train_start`, replay offset 0 |
| Nov 2 01:18:01.829 | Optimizer step 8750 completed |
| 01:18:17.148 | Step-8750 checkpoint publication/rotation finished; 15.316 seconds |
| 01:18:18.834–01:18:30.475 | Steps 8751–8758 completed |
| 01:18:40.297 | First following GPU sample: 0% utilization, 69,915 MiB still allocated to the process/driver |
| 01:21:40.857 | `fatal`, global step still 8758: `DataLoader timed out after 180 seconds` |
| 01:21:55.597 | GPU memory down to 915 MiB |
| 01:24:45 | Torch elastic failure report and Slurm end |

There are 8,758 contiguous `step` records, 35 `checkpoint` records, one
`train_start`, one IPC-budget record and one `fatal`. There is no graceful stop or
verified final checkpoint beyond 8750. The last normal step took 1.658 seconds;
its recorded pipeline wait was 0.072 seconds. The 15.4-second wait on step 8751
includes the preceding checkpoint write, by design; it is not evidence of a
DataLoader stall.

The last step-to-fatal gap is about 190.38 seconds, not exactly 180. A DataLoader
timeout bounds a result-queue wait, not total wall time since the previous
optimizer callback. Ordered buffering, callback overhead and scheduling are not
separately timestamped. Likewise, roughly 184 seconds between `fatal` and job end
cannot be assigned conclusively to a worker shutdown, elastic cleanup, or another
wait. Treat both intervals as observations, not reconstructed worker stacks.

## What the evidence establishes

The traceback runs through Lightning's fetcher and combined loader into
`torch.utils.data.dataloader._next_data()` / `_get_data()`. The exception is raised
while obtaining another batch, before an optimizer update for step 8759. GPU idle
samples corroborate lack of GPU work during the wait; they do not identify why
CPU delivery stopped. No CUDA OOM, nonfinite-gradient, bad-sample exception,
wall-time expiry, or checkpoint-write exception is the reported termination.

The resolved recipe has batch 24, four workers, prefetch 2, persistent workers,
explicit ordered delivery, seed 42, and **`pin_memory=false`**. The original CPU
probe also used `pin_memory=false`. A pinned-memory delivery thread is therefore
not a candidate active component in this incident. Tensor serialization, shared
memory, worker/result queues, ordered buffering and the parent consumer remain
relevant. Do not enable pinning as a purported repair.

The initial IPC estimate required 3,370,345,760 bytes against 269,747,085,312 bytes
available in `/dev/shm`. Slurm reports MaxRSS 32,410,200 KiB against a 96 GiB
request. Post-incident HPCWORK inspection reported about 800 GiB available and
1% inode use. These observations weaken simple initial-capacity explanations;
they do not measure shared memory, storage latency, resource pressure or quotas
at the exact instant of failure. The post-incident node state was `mixed-` with
reason `none`, not an incident-time health certificate.

## CPU forensic replay validity and limits

The reconstruction is correct for the recorded contract:

1. Sort the 255 training episode paths; sum their frame counts to 289,703.
2. Use `random.Random(42)` to shuffle successive complete frame populations,
   taking exactly `48284 * 24 = 1,158,816` selected frames.
3. Apply epoch-zero PyTorch `DistributedSampler` permutation with seed 42,
   one rank, no padding and no dropping. At world size one this is precisely
   `torch.randperm(len(selected), generator=Generator().manual_seed(42))`.
4. Optimizer step `s` uses offsets `(s-1)*24` through `s*24-1`; map selected frame
   IDs through cumulative episode boundaries with `bisect_right`.

The forensic loader additionally checked each reconstructed index against the
actual dataset's `samples[index]` episode/frame pair. This local investigation
independently compared all 192 records for steps 8759–8766 with the real current
`DistributedSampler`, `RankOffsetSampler`, and `FrameSampleTable`; they match.
The original replay returned eight batches with packed lengths
9864, 9865, 9833, 9833, 9865, 9832, 9857, 9819. All eight direct-decode subprocesses
reported 24 successes; all nine subprocess outcomes were PASS. The full loader
probe took 36.076 seconds, including startup, in CPU job `5376522`. Original production used PyTorch
2.8.0+cu128 and Lightning 2.5.3.

This rules out a consistently failing decode or collate of those same samples
under the replay conditions. It does not prove the entire cache was healthy at
failure time. Direct decoding selected the first prompt's image list; the real
loader replay provided stronger coverage through the dataset's deterministic
prompt/augmentation path and collator. The eight-batch window corresponds to
four workers times prefetch two, but no recorded worker task IDs establish their
actual last operation in the original incident.

The replay used fresh processes in a different CPU allocation, a bounded batch
sampler, 120-second loader timeout, and no model execution. It did not recreate
four hours of ZIP-handle churn, queue state, memory pressure, node contention,
storage conditions or original worker-to-task assignment. Per-sample seeding
protects content against changed worker assignment; it does not reproduce
worker-local cache history. A later successful replay cannot rule out a transient
storage stall, intermittent IPC problem or long-lived deadlock.

## Storage stall versus deadlock

A storage stall could leave a worker inside `stat`, JSON/sidecar open, ZIP central
directory loading, ZIP member read or image decode awaiting filesystem data.
Contemporaneous kernel wait channels, system-call stacks, worker Python stacks,
I/O counters and filesystem/server telemetry would help distinguish these. A
worker in uninterruptible `D` state is evidence of a kernel wait, not proof that
Lustre itself is faulty. A warm, later replay can succeed after such a stall.

A multiprocessing deadlock may instead leave worker/feeder/parent threads waiting
on locks, queues, pipes or descriptor transfer. If all workers report completed
collation while the parent waits for data, IPC/delivery becomes a stronger lead.
If only later tasks complete, ordered delivery may be correctly waiting for a
missing earlier task. No reordering, sample skipping or timeout increase is an
acceptable scientific workaround. CPU/I/O counter inactivity alone cannot
distinguish legitimate sleep, storage waiting and deadlock.

## Instrumentation gaps

The production logs omit worker PID-to-index assignments, stage heartbeats,
episode/camera/member identities, Python worker stacks, queue wait stages and
pre-timeout kernel snapshots. ZIP handles are process-local with a bounded
12-entry LRU and PID reset; the source does not show a deliberately shared ZIP
handle across workers. That design reduces one known hazard but is not proof
against every multiprocessing failure.

Per-step GPU utilization calls failed with `ModuleNotFoundError`; the independent
15-second controller monitor supplied useful GPU utilization evidence. Existing
`pipeline_wait_seconds` mixes loader/transfer/callback/checkpoint overhead and
only appears for delivered batches. Fatal telemetry records the parent exception,
not the stalled worker operation. These gaps prevented diagnosis; there is no
evidence they caused the stall. The unchanged runtime intentionally does not save
from `on_exception`, since partial optimizer/loop state need not be resumable.

## Checkpoint/replay review and recovery conclusion

`publish_checkpoint` validates weights, FP32 AdamW moments, scheduler, loop and
replay metadata, writes/fsyncs a temporary payload, reads it back, then publishes
the payload without overwrite, its receipt, and finally `latest.json`. Orphaned
or partial publications are not valid resume points. `load_verified` checks
receipt schema, name, size, SHA256, step and configuration, then validates the
payload. Selection metadata is checked again against the instantiated model.
The compact model `source_commit` is the scientific selection baseline
`52676ccfa3b52f05de51d3d7407f33855159ac95`; it must not be relabeled as the
engineering commit.

There is no independent `engineering_source_sha256` field that can safely be
edited in this checkpoint. `resume_fingerprint` embeds the engineering source
digest in the calculation of `config_sha256`. It also binds complete data/model
configuration, optimizer/scheduler, precision, devices, accumulation, token
limit and dataset inventory. Trainer output root and checkpoint path are excluded;
data/model path spellings are not. Cache root is an environment variable outside
this fingerprint, so the recovery tool adds a separately reviewed byte-integrity
seal before allowing cache relocation.

The source digest includes every `xr1/**/*.py`, every `xr1/configs/**/*.yaml`, and
every immediate `scripts/claix_b24*` file, including tests/untracked matching
files. A change there changes the resume fingerprint even if architecture is
unchanged. New reliability tools and tests live outside that set, have their own
audited hash, and are not injected into production Python. No hash check is
patched or bypassed.

Step 8750 represents 210,000 committed samples. The offset sampler resumes there,
not at the iterator's prefetched position. `ReplayDataLoader.__len__` preserves the
full epoch length for Lightning's absolute restored counters. The runner restores
masters, moments, scheduler and loops; the runtime restores rank RNG at train
start. Independent loader RNG and deterministic sample/step seeding prevent
worker startup or discarded prefetch from advancing the scientific stream.

`check_resume_directory` correctly rejects an in-place resume: telemetry reaches
8758 while `latest.json` reaches 8750. A new target directory accepts the original
verified payload as a fork. **Steps 8751–8758 must be recomputed from 8750**; their
previous effects were volatile and are not in the checkpoint. Resuming at 8759
would skip eight updates. No durable update through 8750 is repeated. Preserve
the original tail as observed, uncommitted execution and do not count both tails
as separate updates in thesis plots. Literal zero recomputation of all completed
steps is impossible with only the step-8750 saved state.

The added CPU regression executes actual updates through 8758 on a tiny network,
injects a fatal exception, restores a genuine step-8750 checkpoint and reproduces
the eight updates exactly. It does not validate the unavailable production
payload, CUDA RNG restoration, XR1 numerical equivalence on H100, or CLAIX I/O.
See [the handoff](RECOVERY_RELIABILITY_HANDOFF.md) for release requirements and
commands. Production recovery remains **NOT CERTIFIED**.
