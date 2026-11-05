# Task 01B — Lustre metadata overhead

Implemented against committed baseline
`5732f7b319b424aea160fec4af06c78940b9d71d` on
`thesis/xr1-lustre-freshstart-v1`. This report supersedes Task 01's statements that
every ZIP handle hit must perform two path stats and that JSON sizing always
traverses the entire object.

The root cause was `ZipArchiveCache.get()` computing archive and sidecar stat
signatures **before** looking for a reusable open handle. With 48 image lookups,
even 48 hits caused 96 explicit path stats. These operations can incur filesystem
metadata latency; avoiding ZIP opens alone does not remove them.

## Integrity policy and implementation

`ZipArchiveCache(..., integrity_policy=...)` now accepts exactly two policies:

- `strict` remains the API/dataset default. Each lookup compares archive and
  sidecar device, inode, size, mtime_ns and ctime_ns. Changed identities invalidate
  held handles and validation records; missing files raise after cleanup.
- `immutable` is explicitly selected by `zip_integrity_policy: immutable` in the
  existing frozen-dataset recipe. A held-handle hit first compares manifest-row
  identity (source, camera, frame count, crop) in memory, then reuses the handle
  without stat/fstat/open/sidecar parsing. Any miss, including reopening an evicted
  archive, follows the same identity-validation path as strict mode. Changed
  identities invalidate metadata and force sidecar and directory revalidation.

The immutable policy requires the original frozen manifest, episode JSONs, ZIPs
and sidecars to be validated by the existing dataset integrity/preflight process
and externally protected from modification/replacement/deletion for each worker's
entire lifetime. Do not run cache generation or dataset updates against active
workers. Stop workers before updates and start new workers after revalidation.
If that operational prerequisite cannot be met, select `strict`.
**This task could not verify the actual frozen files or their write protection
locally.** Selecting immutable records an assumption, not proof of immutability.

Instantaneous external replacement detection and zero path verification I/O
cannot both be provided by an uncoordinated reader. An open descriptor can still
read an old inode after atomic replacement. In immutable mode the next hit may
serve the old, CRC-valid pixels and need never notice a replacement while that
handle stays cached. Sidecar deletion/mutation and same-size archive rewrites can
likewise go unnoticed until a miss. Tests explicitly demonstrate this behavior,
including serving different old/new valid PNGs before/after eviction. In-place
mutation can also interact with buffered data; reads are not a snapshot.

Strict mode checks the identities visible at each lookup; it is not a
transactional lock or a promise of truly instantaneous detection. Filesystem
attribute caching, races after stat, and concurrent writers remain limitations.
Stat identity is not a cryptographic content checksum. CRC detects consumed-member
corruption, not all valid external replacement or adversarial modification.
Neither policy establishes that a previously verified file stays unchanged
forever. No background watcher, periodic timer or full-dataset checksum sweep was
added.

Both policies preserve initial schema/source/frame/crop/archive-size checks,
64 KiB sidecar limit, every expected member name, opened-file fstats and final
initial-validation path rechecks. Member CRC checking and the decoder's final
member drain are unchanged. ZIP count/byte limits, validated-metadata count/byte
limits, RLIMIT cap, pre-open eviction, oversized-entry cleanup, fork isolation and
policy-only serialization remain in place. Changing the policy closes old handles.

Each cache/PID initialization emits `XR1_ZIP_CACHE_POLICY` to the existing runtime
stdout log, even with optional diagnostics disabled. It records the actual policy,
PID, effective handle limit, byte/metadata limits, expected hit path-stat count and
`immutable_dataset_required`. This is a startup/reconfiguration record, not per-frame
logging. `cache_info()` also includes `integrity_policy`; pickle/fork preserve the
policy and discard cached resources. Check each worker's record in a future run.

## Measured filesystem operations and remaining overhead

The tests wrap `os.stat`, `os.fstat`, `ZipFile` construction and `Path.open`, while
delegating to their real implementations on temporary local files. For the direct
cache lookup (excluding manifest/JSON initialization), observed counts are:

| Lookup | Explicit path stats | Explicit fstats | ZIP opens | Sidecar opens/reads |
| --- | ---: | ---: | ---: | ---: |
| Initial validation, either policy | 4 | 2 | 1 | 1 |
| Strict held-handle hit | 2 | 0 | 0 | 0 |
| Immutable held-handle hit | 0 | 0 | 0 | 0 |
| Unchanged reopen with validated metadata, either policy | 2 | 1 | 1 | 0 |
| Reopen after metadata eviction/change, either policy | 4 | 2 | 1 | 1 |

For **48 repeated hits**, the test observes **96 vs 0 path stats**, with zero
new ZIP opens, fstats or sidecar reads. This measures removed calls, not Lustre
latency or optimizer throughput. For 48 lookups comprising H held hits, R validated
reopens and V full validations, expected successful-path explicit path stats are
`2H + 2R + 4V` under strict and `2R + 4V` under immutable. Both also perform
`R + 2V` explicit fstats, `R + V` ZIP opens and `V` sidecar opens/reads. Thus 48
validated reopens still cost 96 path stats; 48 cold validations cost 192 path
stats. Low archive locality/byte eviction can limit the benefit substantially.
No real dataset hit/reopen distribution was available to estimate an average.

Stats/fstats, archive/sidecar opens, directory reads and member data reads are real
filesystem operations. Python's I/O internals may issue additional syscalls, such
as fstat during open, outside the wrapped Python APIs; these counts are **not a
complete syscall trace or Lustre MDT RPC count**. Kernel/client caching also means
one call need not equal one network metadata RPC. Manifest startup and JSON misses
still open/read their files. Member header/data read/seek operations remain on held
handles. Immutable cache lookup itself uses only lexical path construction, row
comparison, OrderedDict lookup/LRU updates and bounded counters in memory; PNG
decoding, CRC computation and object sizing are CPU work. There is no filesystem
exists/is_file/resolve call hidden in the hit path.

## JSON sizing measurement and correction

Task 01 recursively called Python for every scalar and built generator sums on
every JSON miss. Sizing can be significant relative to parsing. The new walker
uses `getsizeof` directly for list/tuple scalar leaves and stops as soon as a
partial conservative estimate exceeds the retention budget. If an object fits,
its estimate is **exactly the previous estimator's result**, including repeated
alias counting and container capacity. If rejected, its partial size is used only
to reject retention. No remembered-size cache or extra path stats were added.

`reliability/benchmark_doosan_json_sizing.py` keeps the old estimator as a benchmark
oracle and accepts `--episode` for existing real files. Local measurements on
2026-11-05 used Python 3.11.15, seven repetitions, median `perf_counter` time, GC
enabled with collection outside timing, already-read JSON text and already-decoded
objects. Inputs have joint_pos/wrench/semantic numeric arrays; they are synthetic,
**not actual frozen episodes**. Sizing excludes the tiny cache-key overhead.

| Synthetic frames | Text characters | Object estimate | Parse | Old sizing | New sizing, 256 MiB budget | New rejection, 1 MiB budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1,686,724 | 8,416,450 B | 27.29 ms | 73.88 ms | 59.42 ms | 6.93 ms |
| 100,000 | 16,201,697 | 84,003,874 B | 348.93 ms | 698.82 ms | 587.01 ms | 1.89 ms |

The measured retained-object sizing reduction is approximately 20% and 16% on
these two inputs. Exact estimates matched. The rejection budget is deliberately
small to exercise early exit; the larger list's container capacity contributes
most of 1 MiB immediately, explaining its shorter rejection time. These results
do not establish real-episode CPU cost, RSS reduction or training throughput.
Real frozen JSONs were unavailable locally; run the command below on CLAIX to
quantify their cost. No further optimization based on assumed homogeneous values,
file size ratios or unbounded identity memoization was introduced.

Retention limits stay 256 MiB JSON + 128 MiB ZIP directory estimates + 8 MiB
validation metadata per worker, with entry-count limits too: 392 MiB/worker,
1568 MiB across four workers. These remain estimates, not RSS limits. Full JSON
parsing happens before sizing; oversized parses, allocator fragmentation, dict/LRU
bookkeeping, images, tensors and prefetched batches can increase transient/total
memory. The new cutoff reduces sizing work, not parse allocation. A real CLAIX
memory/latency probe remains outstanding.

## Validation and scope

**38 CPU tests passed, 0 failed, 0 skipped**, 0.659 seconds test execution.
The suite includes nine additional test methods versus Task 01's 29, plus expanded
existing cases. Coverage includes real stat/fstat/open/read counts, repeated same
archive hits, eviction/reopening, archive/sidecar replacement/in-place rewrite and
deletion under both policies, bad initial provenance/directory/oversized sidecars,
CRC rejection for retained and oversized handles, exact RGB bytes, identical
sample/prompt/state/action/mask/augmented-image content and order, real four-worker
ordered delivery, FD cleanup, count/byte bounds, exact JSON byte-boundary behavior,
size-oracle parity, early cutoff, provenance, pickle, fork and RLIMIT.
Syntax parsing and `git diff --check` also passed.

Local dependencies: torch 2.7.1+cu128 was used on CPU with CUDA hidden. The focused
harness explicitly stubs unavailable mmengine Config/Registry and Lightning's
DataModule base, and uses its existing offline backbone-import bootstrap. The
dataset/augmentation/sampler/DataLoader run for real. Unstubbed framework/full-suite
validation remains for the existing CLAIX environment; no packages were installed.

Local commands, from the worktree root:

```bash
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  xr1/tests/run_doosan_io_cpu_tests.py --allow-framework-stubs
CUDA_VISIBLE_DEVICES='' /home/horus/anaconda3/envs/forcevla_thesis_env/bin/python -B \
  reliability/benchmark_doosan_json_sizing.py
git diff --check
```

CLAIX CPU commands, from this experimental worktree (no Slurm submission):

```bash
export XR1_PYTHON=/home/zk443626/robotics_thesis/envs/xr1_fp32/bin/python
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
"$XR1_PYTHON" -B xr1/tests/run_doosan_io_cpu_tests.py
"$XR1_PYTHON" -B reliability/benchmark_doosan_json_sizing.py
# Select an existing frozen episode; this reads JSON only, without cache writes.
: "${FROZEN_EPISODE_JSON:?Set an existing frozen train episode JSON path}"
"$XR1_PYTHON" -B reliability/benchmark_doosan_json_sizing.py --episode "$FROZEN_EPISODE_JSON"
```

Exact modified/added files:

- `xr1/mibot/data/datasets/doosan_zip_images.py`: explicit strict/immutable policy,
  in-memory handle-hit branch, policy provenance and snapshot/serialization support.
- `xr1/mibot/data/datasets/doosan_json_dataset.py`: pass configured integrity policy.
- `xr1/configs/claix_b24_production.yaml`: explicit frozen-dataset immutable opt-in
  and integrity prerequisites; release-policy fields unchanged.
- `xr1/mibot/utils/doosan_io_cache.py`: equivalent scalar sizing and safe cutoff.
- `xr1/tests/test_doosan_io_cache.py`: focused operation/safety/parity regressions.
- `reliability/benchmark_doosan_json_sizing.py` (new): repeatable CPU sizing benchmark.
- `docs/claix_b24/LUSTRE_FRESHSTART_TASK01B.md` (new): this report.

Batch 24, four workers, prefetch two, persistent/ordered loading and timeout 600
remain unchanged. Production-loader tests confirm these settings and sampler offset
zero. Model/pretrained source, losses, optimizer, scheduler, horizon, action semantics,
sampler and Lustre ZIP storage were not changed. No training or initialization from
historical fine-tuning checkpoints occurred. No GPU work, Slurm commands, dataset or
weight downloads, SSD staging, historical checkpoint modifications, release-policy
changes, commits or pushes occurred.
