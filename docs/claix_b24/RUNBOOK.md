# CLAIX batch-24 qualification and recovery

This is an engineering handoff, **not a production-ready declaration**. Nothing
here has been submitted to Slurm or run on a GPU. Use the isolated engineered
worktree on CLAIX; do not overwrite the pinned original checkout. Preserve the
transferred reference directory. The proposed 48284-step horizon, warmup 500,
max LR 2e-5 and minimum LR 5e-6 require experiment approval after qualification;
four frame passes are not established as optimal or converged.

## Local evidence

From the repository root, use an environment with CPU-usable torch, torchvision,
numpy, Pillow, transformers, Lightning, Hydra/OmegaConf:

```bash
CUDA_VISIBLE_DEVICES='' USE_TF=0 PYTHONPATH=xr1 \
  /tmp/claix-b24-test-env/bin/python -B xr1/tests/run_claix_cpu_tests.py
python3 scripts/claix_b24_launch.py --mode qualification --run-root /tmp/claix-plan-only
bash -n scripts/claix_b24_production.sbatch
git diff --check
```

The `/tmp` environment was built from an existing Python 3.11 environment and
cached pure-Python Lightning/Hydra wheels. It is temporary, not a portable CLAIX
environment. Full stdout/stderr, versions and command results are in
`evidence/local_cpu_tests.txt` and `evidence/commands.txt`. The explicit offline
harness mocks missing mmengine Config/Registry and heavyweight Qwen3-VL imports;
it executes real torch/Lightning/data loader/runner/checkpoint/scheduler logic.
Existing XR1 loss functions are extracted via AST to avoid loading the backbone.
The direct original offline script needs a full XR1 dependency environment.

The laptop sandbox denies loopback sockets and AF_UNIX tensor descriptor passing.
Tests use real worker processes and real augmentation with NumPy pickle transport,
and a real single-device Lightning Trainer with DataModule rank lookup mocked.
Those workarounds do not validate production tensor IPC or DDP collectives.
CLAIX's torch 2.8.0 / Lightning 2.5.3 / transformers 4.57.1 stack remains a separate
qualification gate; local torch 2.7.1 / Lightning 2.5.1.post0 is not its substitute.

## Fixed experiment and resume contract

Training remains force-free, two cameras, original state/action/normalization,
language prompts, frame selection, image jitter and 30-action horizon. The exact
existing `apply_last_n_policy(..., 1)` selection is preserved: final block of the
vision/language/DiT stacks **and the approved robot interface modules**, with all
other parameters frozen. Model storage is BF16 for frozen weights, FP32 for
selected masters and AdamW moments; forward/backward uses BF16 mixed precision.

Production uses batch 24, one GPU, no accumulation (`accumulate_grad_batches=1`),
ordered workers 0/4/8, MAX_LENGTH=12288, at most 413 tokens/sample and 9912 packed
tokens. Strict packing raises on overflow or view loss; it never requeues or drops.
The bound comes from transferred plan metadata, not a locally verified copy of
the original certificate (report SHA256
`87cfb0d0f2f3a310deeee7360e29a2ecb3f58a171985201c5497811fa20a4986`).

The checkpoint offset is `completed_global_step * 24` on each rank. Prefetched
indices and a partially processed next batch never advance it. Resume discards
in-flight work and reconstructs the exact remaining sampler suffix. There is one
fixed epoch and no epoch wrapping. ReplayDataLoader reports the **original** epoch
budget to Lightning's restored absolute batch counters, while iteration yields
only the suffix. CPU tests explicitly cover a resume beyond half the horizon.

Replay v2 intentionally rejects v1 checkpoints. Batch, workers, world size, seed,
data horizon, normalization, language/base identities, model configuration,
optimizer/scheduler settings and the engineering source tree are bound by the
resume contract. Training JSON contents are hashed, not just file names/counts.
Changing any bound field requires a new experiment; do not work around rejection.
Trainer stop step and checkpoint path may change for qualification/recovery while
the data/scheduler horizon stays frozen. Do not edit source after qualification.

## Compute-node qualification (UNVERIFIED)

The following commands are recommendations only. This session did not allocate
nodes, submit jobs, contact Slurm, or run these real-data/GPU commands. Run them
later inside an authorized **compute allocation**, with the existing CLAIX Python
and offline processor/model caches. Confirm scheduler GPU/resource flags and the
96 GiB host-memory request with actual site policy; the sbatch file is a draft.

1. Inventory the staged frozen 255/34/34 episode split and validate archived data,
   language, base model, ZIP cache and original token certificate checksums. The
   transfer contains a certificate hash, not the certificate itself. Verify
   train-only normalization and no episode/media overlap. Never decode/evaluate
   the test split during selection. Stage no replacement or modified dataset.
2. Run the real CPU probe using the **production** DataModule/sample horizon:

   ```bash
   export PYTHONPATH="$XR1_WORKTREE/xr1"
   export XR1_DOOSAN_ZIP_CACHE_ROOT=/path/to/approved/frames_zip/v1
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
   "$XR1_PYTHON" "$XR1_WORKTREE/xr1/tools/claix_b24_worker_qualification.py" \
     --overrides "$XR1_WORKTREE/.claix_b24_reference/configs/hydra_overrides.json" \
     --output /persistent/qualification/workers.json
   ```

   It compares 24-example/48-view contents and resume digests for workers 0/4/8.
   PASS is published only after completion. Fetch timings are CPU-fetch evidence,
   not GPU throughput. Measure real worker RSS/PSS and ZIP/processor overhead.
3. Review a GPU qualification plan (this prints only; it needs no allocation):

   ```bash
   python3 scripts/claix_b24_launch.py --mode qualification \
     --workers 4 --run-root /persistent/qualification/w4
   ```

   Inside an already authorized 23:45:00 allocation, add `--execute` and
   `--python "$XR1_PYTHON" --stop-after-step 3`. This executes six-step-budget
   qualification with a completed-step stop at 3. It keeps the full 48284-step
   data/scheduler horizon. Repeat with `--resume` pointing to the verified
   `project_xr1_doosan/qualification_b24_w4/step-000003.ckpt`, omitting
   `--stop-after-step`, to finish at 6. A fresh unique root provides uninterrupted
   six-step comparison. Do the same for workers 0 and 8; choose and freeze one
   count **before** results training. Do not change it mid-run/resume.
4. Compare real compact master tensors, AdamW moments/steps, scheduler states,
   losses, batch-content/order evidence and frozen-weight checksums against the
   uninterrupted run. Test a near-horizon resume on a separate, explicitly
   configured tiny synthetic DDP fixture on the exact CLAIX stack. The real
   qualification stream must retain the proposed full horizon.
5. Run enough additional approved qualification steps to measure warm and cold
   end-to-end behavior. The supplied six-step smoke is not a sustained throughput
   benchmark. Log 24 retained samples, 48 views, packed tokens, GPU peaks/reserved
   memory, utilization, host/worker memory, data wait and checkpoint overhead.
   Include long-token/tail-action samples and worst-case packed attention. SDPA
   mask/workspace costs can scale quadratically: the token bound does not prove
   H100 memory safety. No batch-24 H100 throughput or memory result exists here.
6. Exercise normal finish, callback deadline and batch-controller USR1 on the
   actual filesystem. Confirm save + read-back + checksum + fsync + rotation all
   finish within the 1800-second margin. Inject interrupted I/O only into a
   separate qualification directory; verify recovery from the previous receipt.
   Test controlled TERM behavior before expiry, process/worker failure and storage
   exhaustion. SIGKILL/node death cannot guarantee a final checkpoint.
7. Run validation smoke, then all 34 validation episodes as described below.
   Inspect actual inference prompt/tensor compatibility and metric meaning before
   opening its release gate. Review final learning rate/horizon as a scientific
   decision. Preserve all raw evidence and attach its hashes to release gates.

## Launcher and release gate

`scripts/claix_b24_production.sbatch` defaults to **qualification** mode. It requests
one c23g GPU, 23:45:00 and batch-controller USR1 1800 seconds before expiry. Supply
`XR1_WORKTREE`, `XR1_PYTHON`, `XR1_RUN_ROOT`, offline cache environment and optional
`XR1_WORKERS`, `XR1_RESUME`, `XR1_STOP_AFTER_STEP`. Production additionally needs
`XR1_MODE=production` and `XR1_RELEASE_RECORD`. No submission command is executed
by any launcher; the Python launcher runs only within an existing allocation.

The launcher reads authoritative `scontrol show job` EndTime, subtracts 1800
seconds and passes an absolute safe deadline. Missing/expired/incorrect allocation
metadata fails closed. The batch controller receives Slurm USR1; its handler sets
a flag. Normal controller code writes a request file, which the callback checks
at completed-step boundaries. Direct worker signal handlers also only set flags.
No signal handler saves a checkpoint. The independent deadline works without
Slurm signal delivery. A stuck GPU/kernel may still prevent callback execution.

`flock` is held by controller and inherited by torchrun, protecting the same run
root against duplicate active invocations. A nonempty run target requires explicit
resume. Shared-filesystem flock behavior must be verified on CLAIX. Different
run roots are separate experiments and cannot be protected by that one lock.

To prepare a release record, start from `release_record.template.json`, copy the
exact `production_plan` object and source-tree digest printed by a **production**
dry-run, and replace every UNVERIFIED gate with PASS plus an actual evidence file
path and SHA256. The launcher requires exact plan/source matches and checks all
evidence hashes. This is an operator-reviewed gate, not an automated assertion
that a JSON PASS proves scientific adequacy. Keep qualification logs and approval
evidence immutable; never fabricate PASS records or carry evidence across code
changes. The release gate is part of the requested production safety engineering,
not a request for permission to continue local work.

## Checkpoint publication and recovery

Each completed interval (250 steps), safe stop, natural finish, and step 2000
creates an immutable `step-NNNNNN.ckpt`. A temporary file is flushed/fsynced,
reloaded, and validated: selected FP32 tensors/key and element counts, finite
weights/moments, AdamW parameter groups/counters, cosine scheduler epoch, actual
Lightning optimizer/batch progress, rank cursor, per-rank RNG, configuration and
source/data provenance. Size must be no larger than `12 * selected_elements +
64 MiB`, excluding accidental 5B backbone serialization. The payload is published
with an atomic no-clobber hard link on the same filesystem. A checksum receipt is
published atomically, then `latest.json`. Directory metadata is fsynced. Verify
real checkpoint size, I/O time and filesystem guarantees on CLAIX.

Keep newest two committed checkpoints plus step 2000 forever. Rotation happens
after new publication. A filename alone is insufficient; a matching
`.verified.json` receipt, file size, SHA256 and successful contract validation are
mandatory. A crash between payload and receipt creates an orphan. Do not use it
or guess its completeness. Preserve/quarantine it for inspection before retrying
the same step, since existing destinations are deliberately never overwritten.

Recovery procedure:

1. Ensure the old job/processes ended and the run-root lock can be acquired.
2. Read `latest.json`, locate its named payload and matching receipt. Prefer the
   previous verified checkpoint if the latest transaction is damaged. Load only
   trusted locally generated checkpoints (torch pickle is not an untrusted format).
3. Use identical engineered source, immutable base/language, frozen dataset and
   bound configuration. Pass `--resume /absolute/path/step-NNNNNN.ckpt`. The
   entrypoint rejects checksum/configuration/cursor drift before constructing XR1.
4. Verify first resumed telemetry step is saved step + 1, sampler offset is
   saved step * 24, and LR follows the restored scheduler. Already-completed steps
   are not replayed. A terminal horizon checkpoint is valid for evaluation but
   intentionally rejected for further training.
5. Fatal loss/gradient/worker/I/O errors stop with a timestamped fatal event and
   preserve the last committed checkpoint. They do not trigger an unsafe save of
   an in-flight optimizer update. Recover from the last verified boundary.

Telemetry lives in `telemetry.jsonl`, with independent local CSV metrics and
per-invocation resolved configs. Throughput includes loader startup/wait, transfer,
compute, callbacks and preceding checkpoint I/O; it is not GPU-stage-only timing.
`pipeline_wait_seconds` includes host/callback/I/O overhead and is not presented
as a pure loader-only measurement. RSS sums include shared pages, not unique
physical memory. Unavailable GPU counters are null (with an error type when
available); a missing metric must not be called measured. The launcher also
captures raw child stdout/stderr in unique `launch-*.out` / `launch-*.err` files
under the persistent run root, covering model import/config errors and OOM before
callback initialization. Keep Slurm/controller stdout/stderr in a persistent
submission working directory as well.

## Validation protocol

There is no Lightning validation_step. Use standalone
`xr1/tools/claix_b24_evaluate.py` after a verified checkpoint; never merge validation
frames into the training sampler or recompute normalization from held-out data.
It audits train/validation/test identity and media separation, then evaluates
canonical sorted validation episodes and every frame once, without image jitter.
The dataset keeps training action/state/horizon semantics and training statistics.

The inference payload matches the deployed Doosan instruction/empty-CoT prompt,
uses the normalized current state, zero action prefix and semantic mask, and
removes both final teacher-forcing state/action turns and all target tensors.
Future actions are used only after inference for scoring. Seeds are stable per
episode/frame and isolated from training RNG. Model mode/RNG are restored.

Report masked generated-action MAE for translation (metres), rotation-vector
components (radians), and gripper target, using std + source ACTION_EPS to recover
physical units. Exclude padded future steps and unused 53 dimensions. Report both
frame-element micro MAE and equal-episode macro MAE. These are open-loop imitation
metrics, not closed-loop task success or a geometric SO(3) distance. Use validation
only to compare predefined checkpoints/configurations; keep the 34 test episodes
reserved for one final reporting protocol after all model choices are frozen.

Inside an authorized compute allocation, after setting offline environment:

```bash
"$XR1_PYTHON" xr1/tools/claix_b24_evaluate.py \
  --overrides .claix_b24_reference/configs/hydra_overrides.json \
  --checkpoint /persistent/run/project_xr1_doosan/qualification_b24_w4/step-000006.ckpt \
  --validation-root /approved/dataset_v2/splits/val \
  --test-root /approved/dataset_v2/splits/test \
  --output /persistent/qualification/validation-smoke.json --limit 4
```

Use the actual staged validation directory spelling. Use an overrides copy with
the matching worker count if evaluating a workers-0/8 checkpoint. Omit `--limit`
only for full validation; smoke output is explicitly marked as a subset. Actual
Qwen/XR1 inference, prompt compatibility and real metrics are still UNVERIFIED.
