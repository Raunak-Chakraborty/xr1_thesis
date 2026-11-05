# Task 01 — XR-1 Lustre I/O reliability

Implemented on `thesis/xr1-lustre-freshstart-v1`, based on
`293117a8c6554c30bb344a061a86236cebe81cfa`. The new experiment must initialize from
original pretrained Xiaomi XR-1 weights at optimizer step **0**. Step 21500 is
historical evidence only. This patch implements I/O improvements; fresh-start
launch mode is Task 02. No resume, checkpoint bridging or release-gate changes
were made. The old qualified worktree and historical checkpoints were untouched.

The task records RWTH IT Center confirmation of Lustre metadata-latency spikes
around jobs 5363645 and 5383568. Inspection found a two-episode JSON LRU, 12 open
ZIPs per process, repeated sidecar checks/parsing on ZIP misses, central-directory
parsing on reopens, and path resolution on every image decode. Random frame
selection limits locality. A longer timeout tolerates stalls; it does not fix
filesystem performance.

The production recipe and DataModule fallback now use **600 seconds**. Batch size
24, four workers, prefetch two, persistent workers and ordered delivery remain
unchanged. So do sample selection, seeds, augmentation, normalization, state/action
outputs, masks, architecture and losses. Explicit CPU-probe overrides such as their
existing 60-second timeout still work. No new sampler or physical-read reordering
was implemented.

Settings under `data.params.train_datasets`:

| Setting | Initial value | Bound per worker |
| --- | ---: | --- |
| `json_cache_entries` | 16 | Decoded episodes per dataset |
| `json_cache_bytes` | 268435456 | 256 MiB estimated decoded JSON |
| `zip_cache_entries` | 32 | Open ZIPs, additionally capped by RLIMIT |
| `zip_cache_bytes` | 134217728 | 128 MiB estimated ZIP directory objects |
| `zip_metadata_entries` | 1024 | Validation records; accommodates all 646 archives |
| `zip_metadata_bytes` | 8388608 | 8 MiB estimated validation-record retention |
| `cache_report_every` | 0 | Disabled; positive intervals must be at least 128 lookups |

JSON caches are instance-local, bounded by count **and** decoded bytes. Zero
entries/bytes disables retention; oversized episodes load but are not retained.
Sizing includes containers, keys, strings and scalars, conservatively counting
aliases repeatedly. Parent enumeration clears its decoded cache; fork checks PID
before reuse, and spawn serializes policy only. Existing prompt deep-copy behavior
prevents conversation mutation across repeated samples. Caches consume no RNG.

ZIP eviction closes handles before opening replacements. Reconfiguration, explicit
cleanup and interpreter exit also close them; OS process exit releases descriptors.
The production cache closes inherited handles immediately after fork and checks
PID before reuse. Serialization contains policy only. Oversized directories are
not retained and their handles close in the decoder's `finally`, including failures.
The effective handle cap is `min(requested, soft_RLIMIT // 2, soft_RLIMIT - 64)`;
unlimited RLIMIT uses the requested limit. Insufficient capacity fails explicitly.
The resource limit is never raised; other libraries can still exhaust descriptors.

Every ZIP lookup checks archive and sidecar identities using device, inode, size,
nanosecond mtime and ctime. Missing/changed files close stale handles and invalidate
validation records. Initial checks retain schema, source, frame-count, crop and
archive-size verification, cap sidecar reads at 64 KiB, check every expected frame
name, and recheck identities during validation. Validated reopens skip sidecar
open/parsing, repeated full name checks and directory-memory measurement. Python
still parses the ZIP directory on reopen; retained handles avoid that entirely.
Root conversion is lexical instead of repeated filesystem resolution.

Consumed PNG members retain CRC verification, with a final drain to ensure PIL
cannot leave the member's CRC unchecked. Invalid provenance, missing/truncated
archives, bad directories and consumed corrupt members raise; there is no fallback,
skip, retry or accepted partial batch. Existing creation-time CRC/SHA checks and
integrity/preflight mechanisms remain unchanged. This patch does not add a full
136 GiB runtime checksum sweep: unread-member corruption is detected when read or
by whole-archive checks. File identities are not cryptographic proof. Manifest and
episode JSON immutability remain assumptions; concurrent mutation during a read
is unsupported, and identity checks do not make filesystem writes transactional.

Expected benefits are fewer JSON opens/parses, sidecar reads, and ZIP opens/directory
parses where reuse occurs. The safety tradeoff is explicit: ZIP-handle hits now
perform two path stats instead of zero; a validated reopen uses two path stats
plus an opened-file fstat without a sidecar read. Initial validation costs more.
JSON object sizing adds CPU work on misses. Synthetic tests verify these operations;
**no CLAIX performance benchmark or measured stall reduction is claimed**.

Sixteen episodes alone is not a memory safety argument. A 64-bit CPython float
usually needs 24 bytes plus its container pointer, before container overhead.
All 16 episodes fit only if their combined estimate is within 256 MiB. Retention
budgets total 392 MiB per worker, **1568 MiB (1.53 GiB)** for four workers. These
are object estimates, not RSS limits: allocator fragmentation, bookkeeping, a new
parse/directory, text buffers, decoded images, tensors, prefetched batches, shared
memory and host model allocations are additional. Oversized entries still incur
transient allocation. Actual frozen JSONs were unavailable locally; their decoded
sizes and total process memory need CLAIX measurement. Default retained ZIP FDs
increase from 48 to at most 128 across four workers, plus non-ZIP descriptors.

Counters cover JSON hits/misses/evictions, ZIP hits/misses/opens/reopens, sidecar
reads, metadata hits/evictions, invalidations and oversized entries. They saturate
at signed 64-bit maximum and retain no unbounded path history. Reopens are counted
within the bounded metadata history. Worker-local snapshots are available through
`dataset._json_cache.snapshot()` and `doosan_zip_images.cache_info()`; parent values
do not aggregate workers. With diagnostics bound, `cache_report_every=1024` emits
optional summaries to the existing fixed-size trace ring. Records may be overwritten.
There is no per-frame cache logging, synchronous telemetry I/O, new watchdog,
signal handler or deadline change.

Validation on 2026-11-05: **29 focused CPU tests passed** (0.559 s test execution),
including 19 new tests and 10 existing dataset/DataModule tests. Coverage includes
count/byte eviction, reopens, validation reuse, same-size replacement and in-place
changes, corruption, real fork isolation, serialization, RLIMIT, FD cleanup, RGB
parity, all training fields including augmented images/prompts, unchanged identities
and order, and real four-worker delivery with batch 24/prefetch two. Actual production
DataModule construction verifies timeout 600 and sampler offset zero; tests do not
wait 600 seconds to measure timeout expiration. Syntax checks and `git diff --check`
also pass.

The existing local Python 3.11 environment supplies torch 2.7.1+cu128 and torchvision
0.22.1+cu128; CUDA was hidden. No packages were installed. The focused harness
explicitly stubs missing mmengine registry/config and LightningDataModule base and
reuses the existing offline backbone bootstrap. Dataset, augmentation, sampler and
DataLoader execute for real; worker output uses canonical CPU bytes rather than
tensor FD IPC. No processor, weights, forward pass or optimizer is instantiated.
The full existing CPU suite was attempted but cannot start without Lightning.
Separate existing watchdog checks: handler-conflict rejection passed; transport
(fork/spawn) and USR1-preservation methods were blocked, with three failing subcases.
The sandbox rejects diagnostic startup's `setsockopt(SO_SNDBUF, 65536)` with
`PermissionError: Operation not permitted`. Watchdog code was not changed to bypass it.

Local commands, from this worktree root:

```bash
CUDA_VISIBLE_DEVICES='' \
  /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs

mkdir -p reliability/evidence/tmp
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=xr1:reliability/tests \
  /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B -m unittest -v \
  test_diagnostics.DiagnosticTests.test_order_and_tensor_transport_fork_and_spawn \
  test_diagnostics.DiagnosticTests.test_usr1_handler_unchanged \
  test_diagnostics.DiagnosticTests.test_handler_conflict_is_rejected

git diff --check
```

Exact CLAIX CPU-only rerun, from the experimental worktree with its existing offline
environment; these commands neither submit Slurm jobs nor invoke training:

```bash
export XR1_PYTHON=/home/zk443626/robotics_thesis/envs/xr1_fp32/bin/python
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export USE_TF=0 USE_FLAX=0
export PYTHONPATH="$PWD/xr1:$PWD/xr1/tests:$PWD/reliability/tests"
"$XR1_PYTHON" -B xr1/tests/run_doosan_io_cpu_tests.py
mkdir -p reliability/evidence/tmp
"$XR1_PYTHON" -B -m unittest -v \
  test_diagnostics.DiagnosticTests.test_order_and_tensor_transport_fork_and_spawn \
  test_diagnostics.DiagnosticTests.test_usr1_handler_unchanged \
  test_diagnostics.DiagnosticTests.test_handler_conflict_is_rejected
"$XR1_PYTHON" -B xr1/tests/run_claix_cpu_tests.py
```

Verify the actual `XR1_LOADER_EFFECTIVE` startup record reports `timeout: 600`,
`num_workers: 4`, `prefetch_factor: 2`, `persistent_workers: True`, `in_order: True`.
The same record appears during real loader construction after runtime overrides;
check it rather than trusting YAML alone. The CLAIX focused command deliberately
omits framework stubs.

For later read-only memory/locality measurement on the frozen HPCWORK train split,
use the existing JSON-only profiler explicitly from the new sequence's beginning:

```bash
: "${FROZEN_TRAIN_ROOT:?Set the existing frozen HPCWORK train-split path}"
export FROZEN_TRAIN_ROOT
PYTHONPATH="$PWD/reliability:$PWD/xr1" "$XR1_PYTHON" -B - <<'PY'
import json, os
from profile_json_cache import profile
print(json.dumps(profile(os.environ['FROZEN_TRAIN_ROOT'], first_step=1,
                         batches=128, workers=4, batch_size=24), indent=2))
PY
```

`first_step=1` is the profiler's one-based first batch, corresponding to optimizer
step zero before training. No checkpoint is loaded. Its count-only LRU simulation
and deduplicated object sizes are not measurements of this byte-limited cache or
RSS; compare with worker snapshots and process memory in a later CPU data probe.

Future locality recommendation: grouping episodes through a new sampler changes
the scientific sample sequence and potentially optimizer updates. Physical reads
can instead be grouped within an already assigned 24-sample batch, restoring
original positions before prompt/augmentation/collation. Explore a bounded
worker-local `__getitems__` implementation first, preserving duplicates, view order,
per-sample seeds, failure behavior and checkpoint cursor semantics. Require exact
pixel/tensor/prompt/order parity and real Lustre measurements. Do not widen prefetch
or change logical sampler order for that experiment. No such system is implemented
in Task 01.

Recommended **Task 02**: implement fresh-start experimental training mode without
the nine formal release approvals, loading only original pretrained weights at
global optimizer step zero while preserving storage validation, checkpointing,
locking, provenance and safe-stop protections. Release gates are unchanged here.

Exact modified/added files:

- `xr1/configs/claix_b24_production.yaml`
- `xr1/mibot/data/datamodule/doosan_datamodule.py`
- `xr1/mibot/data/datasets/doosan_json_dataset.py`
- `xr1/mibot/data/datasets/doosan_zip_images.py`
- `xr1/mibot/utils/doosan_io_cache.py` (new)
- `xr1/tests/run_doosan_io_cpu_tests.py` (new)
- `xr1/tests/test_doosan_io_cache.py` (new)
- `docs/claix_b24/LUSTRE_FRESHSTART_TASK01.md` (new)

No GPU training or Slurm submissions, dataset/checkpoint downloads or transfers,
node-local SSD staging, historical checkpoint writes, commits or pushes occurred.
