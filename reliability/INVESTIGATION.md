# Loader reliability candidate — local development only

Baseline: `527be264ce6a5b462468c0a406f1a876c8bbdfa4`. Before edits the actual
protected digest was `852945709dac80152459c26ab5dab5e70e158b631ea27612d28ee2d172ef3d0b`.
Original engineering ancestor: `16da563ce61e2b58952281b86c474a0490e99a52`.

## Diagnosis before implementation

Neither historical timeout identifies its blocking operation. A DataLoader timeout
is a failure to obtain a result, not a diagnosis of storage. Job 5410198 completed
26,784 batches after step 21,500, with maximum wait 5.836842 s and mean 0.834527 s.
The imported CPU result records 22,706.967 s inside the replay; the supplied job
wall time is 6:18:37. These are different measurement boundaries.

The supplied 1,875 `cl_sync_io_wait`, 33 `ptlrpc_set_wait`, 1,707 ZIP-open overlaps,
and approximately 4.45 TB of aggregate process read_bytes demonstrate I/O pressure
during successful operation. They do not identify the cause of either timeout.

Source inspection found:

* Each worker retains 12 ZIP handles across a shuffled 255-episode training split
  with two cameras. Misses reread provenance and central directories. This is a
  plausible source of repeated I/O; changing it is not yet justified by a failure.
* JSON trajectories use a two-entry process-local LRU. Random frame ordering can
  cause frequent reparsing. Cache expansion needs decoded-memory measurements.
* Workers start when Lightning iterates the loader, after model construction.
  Default multiprocessing is inherited from Python; Linux production used fork.
  CUDA/native-library/thread state inherited at fork remains an untested mechanism.
* The CPU probe cancels its timer after collation, before the Queue feeder
  serializes tensors, exports storage handles, obtains its pipe lock, and sends.
  A successful Queue.put is not successful IPC delivery. Parent tensor handle
  reconstruction and pinned-memory delivery also need independent observation.
* Ordered delivery can wait for one task while later results are buffered. The
  watchdog must time the enclosing parent next() call, not CPU/I/O inactivity.
* The full CPU replay script used SIGUSR1 for stacks. Production StopRequest uses
  SIGUSR1/SIGTERM/SIGINT for stop requests. That diagnostic cannot be reused safely.
* Existing recovery tooling is bound to the first failure and step 8,750. It must
  not be relabeled as a step-21,500 recovery procedure.

## Scientific compatibility boundary

Both `claix_contract.engineering_source_digest` and `claix_b24_launch.tree_digest`
hash all `xr1/**/*.py`, `xr1/configs/**/*.yaml`, and `scripts/claix_b24*`, including
new Python files and tests. `reliability/` is separately hashed by recovery tooling.
The resume fingerprint includes that digest and the entire data configuration
except resume_step. Operational data options therefore also change compatibility.

Production hooks intentionally change the protected digest. Existing release and
checkpoint checks must reject this candidate. No hash allowlist, rewritten receipt,
false original digest, or checkpoint migration is part of this work. Deployment
requires a legitimate reviewed compatibility procedure and qualification evidence.

The step-21,500 receipt claims payload SHA256
`48fae2e1881e85ef8d760b30780ea21aa146053b1702b1084c56fb6644d64813`.
Reading the receipt is not payload verification. Independent payload rehashing and
full checkpoint validation after the second failure remain outstanding.

## Diagnostic design

Opt-in only. Bounded single-writer shared-memory rings record nested stage begin,
end, elapsed time, identities, true PyTorch task IDs, and parent ordering state.
Separate lanes distinguish worker computation from the multiprocessing feeder.
No producer performs diagnostic filesystem I/O or waits for a diagnostic reader.
An independent spawned watchdog triggers before the existing loader timeout,
collects bounded process/thread evidence, and requests Python stacks using only
reserved SIGUSR2. Stack output uses nonblocking sockets; a full socket loses
diagnostic bytes rather than blocking training. A separate exporter handles
persistent storage. Buffer overwrites, trace truncation, and unavailable native
evidence must be explicit. No diagnostics alter data, RNG, order, or retry samples.

Private Torch and CPython Queue hooks need tests on the exact CLAIX versions
(Python 3.11.5, Torch 2.8.0+cu128, Lightning 2.5.3). Local CPU tests cannot qualify
CUDA initialization, GPU pinning, H100 lifecycle, or real Lustre/SSD performance.

Later phases stay in separate modules/tests: verified optional staging; explicit
fork/spawn selection with unchanged default; cache measurements before tuning.
