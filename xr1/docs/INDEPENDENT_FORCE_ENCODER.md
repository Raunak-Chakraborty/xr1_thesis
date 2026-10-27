# Experiment B: independent Doosan force encoder

Implemented in the authorized encoder worktree from the user-reported source
`afe739d1c9aac9218036a13fee9f9eb182747159`, expected branch
`thesis/xr1-force/encoder`. Git metadata is intentionally inaccessible; neither
branch nor HEAD could be independently verified. No Git mutation was performed.

This implementation tests a representation hypothesis. Architecture, synthetic
tests, or a successful checkpoint load do not demonstrate better contact
reasoning, insertion success, or robustness to controller/cable disturbances.

## Architecture and tensor flow

Source inspection established these existing XR-1 paths:

* State `(B,1,60)` goes to `state_projector` → `(B,1,1024)` for DiT.
* In training, `state_projector_choice` also projects state into the VLM's
  auxiliary action-choice sequence. The DiT receives the VLM prefix KV cache.
* `t_embedder` and `t_projector` generate `(B,6,1024)` conditioning. Every
  existing DiT layer adds this to its `adaln_table`, yielding attention/MLP
  shifts, scales, and residual gates.

The extension extracts state `[...,20:26]` in **Fx,Fy,Fz,Tx,Ty,Tz** order.
It uses a `6 → 64 → 64 → 64` MLP with SiLU between linear layers. A separate
bias-free `64 → 6144` projection produces a `(B,6,1024)` residual that is added
to the existing timestep conditioning, before the unchanged DiT layers. This
reuses the model's existing conditioning interface without a transformer,
expert network, VLM change, token insertion, or resized pretrained tensor.

The default extension has **401,984 parameters** by arithmetic: 8,768 in the
encoder and 393,216 in fusion. The workstation script reports actual counts.

The current observation has exactly one state timestep. No future wrench,
window pooling, or action-label force is used. `forward()` computes the force
residual once, repeats it with the same `repeat_interleave` ordering as state
and actions for flow-matching training, and passes it through `dit_forward()`.
Generation uses the same `forward()` preparation and identical residual at
every ODE step. The asynchronous prefix-generation path also shares it.

Both original state projectors see a cloned state whose wrench channels are
zero, after normalization; all other state channels are preserved. The explicit
`dual_path=true` ablation retains force in state as well, and is off by default.
The auxiliary VLM choice/score loss remains force-free in the default setup;
encoder gradients come from the DiT action loss. That is intentional.

Disabled mode does not construct new modules, zero state channels, require a
Doosan input, or alter original parameter names/shapes and freeze flags. Passing
force in state while disabled retains that pre-existing behavior; a force-free
control must also use force-free data/zero state wrench channels. During a
same-instance diagnostic, changing the config's `enabled` flag bypasses fusion
but leaves already-created parameters registered; construct a disabled model
for baseline checkpoint/layout comparisons.

## Normalization and observation semantics

The source dataset normalizes the native 60D state before model forward.
Experiment B therefore consumes **already-normalized wrench**, not raw N/Nm:

```
normalized = clip(2 * ((raw - q01) / ((q99 + 1e-6) - q01)) - 1, -1, 1)
```

The float32 operation order matches the original numpy `normalize_quantile`.
`auto_cast` subsequently converts float32 model inputs to bfloat16 as before.
There is no normalization operation inside the encoder. Clipping saturates
outliers; raw zero force does not generally map to zero normalized features.

`q01` and `q99` must have the **XR-1 `(1,60)` layout**, not the converter's 25D
layout. All entries must be finite and all six force channels must have strictly
positive quantile spans. Missing stats, zero-padded force stats, and nonfinite
stats are errors. No real statistics are supplied or invented here. Estimate
them from the approved training split using the existing data workflow and use
the same statistics for validation, test, and inference. The independent-encoder
data flag validates these statistics and emits an int64 `force_normalization=1`
marker, collated once per observation. The model requires this marker and a
finite `(B,1,60)` state's wrench in `[-1,1]`. The marker is an API contract;
arbitrary callers must not falsely label raw or twice-normalized data.

Dataset state comes from `sample.frame_index`. Actions remain the converter's
exactly-once measured 7D semantic labels, windowed/padded to `(30,60)`, normalized
with original action stats and masked to channels `0:7` at real timesteps.
Neither action differencing nor the loss implementation was changed. Padded
timesteps do not enter masked MSE/choice loss, and incomplete horizons are
excluded by the existing frequency-loss rule.

Raw measurements remain Doosan BASE-frame N/Nm from
`RobotStateRt.external_tcp_force`, controller-reset-compensated upstream, with
motion/cable disturbances. No additional tare, reset, gravity compensation,
frame transform, or force-magnitude-derived contact label is introduced.

## Training configuration and optimizer

Compose `xr1/configs/force_encoder.yaml`. The default generic `config.yaml` and
`model/posttrain.yaml` remain unchanged. Experiment B requires:

* `data.type=DoosanDataModule`, `use_wrench=true`, `force_encoder_input=true`,
  `action_length=30`, and real paths/action statistics/state quantiles.
* An explicit pretrained checkpoint and a fresh output directory.

Model settings live under `model.params.model.force_encoder`:

| Setting | Class default | Experiment default / meaning |
|---|---|---|
| `enabled` | false | true |
| `hidden_dims` | [64,64] | MLP hidden widths; [] also permits a single encoder linear |
| `output_dim` | 64 | feature width, independent of DiT width |
| `fusion_type` | adaln_residual | only supported type; unknown types error |
| `fusion_init` | small_normal | `small_normal` or `zero` |
| `fusion_init_std` | 0.001 | positive finite normal standard deviation |
| `init_seed` | 42 | isolated CPU initialization seed |
| `dual_path` | false | explicit force-in-state + encoder ablation |
| `input_normalization` | state_quantile_once | only supported contract |
| `pretrained_policy` | original | frozen |
| `train_encoder` | true | encoder `requires_grad` |
| `train_fusion` | true | fusion `requires_grad` |

`pretrained_policy=original` retains XR-1's original trainability (including its
frozen input embeddings). `frozen` freezes all original parameters. `dit`
unfreezes DiT, state/action projectors, action output, timestep modules, and
sink. `last_vlm_block` unfreezes only the last VLM language-transformer block
among original parameters. The new encoder/fusion switches apply separately.
Disabled mode ignores these extension freeze policies.

The existing BaseRunner builds its optimizer from `requires_grad` parameters;
new weights use the existing decay group, encoder biases use no decay. No
replacement trainer, optimizer, scheduler, or loss was added. Checkpoints
contain registered encoder/fusion weights and the optimizer's corresponding
state through Lightning/DeepSpeed. For optimizer resume use the existing
`trainer.ckpt_path`, the exact architecture and trainability config, and the
same optimizer parameter grouping. Do not use pretrained import to resume an
optimizer. Distributed/ZeRO restoration still requires workstation validation.

The dedicated config sets `freq_excluded_dims=[6]` for Doosan's binary gripper.
Hold this, frequency-loss coefficient, async prefix training, trainable
pretrained modules, seed, data split, normalization, augmentations, and training
budget constant across comparisons. The new-encoder-only default is not a
matched comparison to the previously completed last-VLM-block pilot until the
freeze policies and other settings are matched.

## Initialization and checkpoint validation

Original `_build_model()` and all original submodule definitions are unchanged.
The new module is constructed afterwards, under an isolated CPU RNG context,
with explicit float32 CPU allocation, then moved to the original sink's
device/dtype. Encoder weights use Xavier uniform, encoder biases zero; fusion
uses Normal(0,0.001) by default, without a bias or separate zero gate. This
allows first-step encoder gradients through nonzero fusion weights. It is a
small perturbation, not exact initial baseline equality.

Optional zero fusion gives an exact initial zero residual. The first step
updates fusion; encoder gradients start after fusion becomes nonzero. A frozen
zero fusion with a trainable encoder is rejected because it disconnects learning.
Zero force produces a zero residual at initialization because all biases are
zero; learned biases may change that.

New state-dict keys are exactly `force_conditioner.encoder.*` and
`force_conditioner.fusion.weight` (prefixed `model.` inside BaseRunner).
`allow_base_checkpoint` defaults false in BaseRunner and is true in the
experiment config only for initial import. The loader validates every original
key and shape, rejects unexpected keys and partially missing new modules, and
permits missing keys only if they are exactly the complete new parameter set.
It fills that set from the initialized module and calls `load_state_dict` with
`strict=True`. Complete encoder checkpoints also load strictly. Deployment and
Lightning resume do not enable the missing-new-module exception.

The deployment loader expects the existing run directory checkpoint layout.
The verification tool accepts plain weights, Lightning `state_dict`, or
consolidated DeepSpeed `module` with optional `model.` prefixes, without
discarding unexpected keys. ZeRO partitions need the existing framework's
consolidation/export process. Real XR1 checkpoint compatibility is unverified
in this sandbox.

## Inference

`mibot.server.deploy` reads the run's saved config, validates experiment B,
loads complete model weights strictly, and selects the Doosan seven-channel
action mask for Doosan runs. Generic runs retain the original generic mask.
`DoosanClient(..., require_wrench=True)` requires current six-axis measurements
and emits `wrench_present=true` with the raw packed state. The server refuses
missing-wrench encoder requests, applies the matching float32 quantile formula,
adds the normalization marker, and calls the same model generation path.
The raw-state server API rejects requests already carrying a normalization
marker to avoid double normalization. Direct model callers must provide
normalized state plus the int64 marker themselves.

The resulting action still uses only the seven semantic channels and existing
denormalization. No ROS/controller or physical robot execution was added or
performed. The separate upstream `deploy/server.py` Hugging Face deployment is
unchanged and does not implement this thesis extension; use `xr1/mibot/server`.

## Checks and workstation commands

From the worktree root, CPU tests require no weights, assets, or GPU:

```bash
PYTHONPATH=xr1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python3 -B -m unittest discover -s xr1/tests -p 'test_force_encoder_*.py' -v
```

The torch tests use real PyTorch and actual XR-1 source methods, with a VLM test
double and one real DiT layer. Dataset tests replace image/JSON I/O with
synthetic fixtures and execute actual packing/normalization/windowing methods.
They are not full-model or dataset validation. Missing dependencies are skips;
dependency-free fake-weight tests check loader control flow only, not numerics.
Run existing regression tests in the workstation's existing XR-1 environment:

```bash
PYTHONPATH=xr1 CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s xr1/tests -p 'test_doosan_*.py' -v
PYTHONPATH=xr1 CUDA_VISIBLE_DEVICES='' python3 -m pytest xr1/tests/test_training_logic.py -q
```

GPU verification must run only on the lab workstation, with existing weights
and cached Qwen config/processor; do not run it in the laptop sandbox:

```bash
PYTHONPATH=xr1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python3 xr1/tools/verify_force_encoder.py \
  --checkpoint /PATH/TO/EXISTING/CONSOLIDATED_XR1_CHECKPOINT.pt \
  --output-dir /PATH/INSIDE/PROJECT/NEW_ENCODER_VERIFICATION_RUN \
  --device cuda:0 --training-repeat 1
```

The script refuses an existing output directory, uses tiny synthetic two-camera
observations and explicitly synthetic stats, loads the real checkpoint, runs
full training forward and backward, inspects every new parameter's gradient,
steps AdamW, compares force-free/enabled generation with matched noise, perturbs
each force axis, reports parameter counts and GPU memory, and verifies full
model + optimizer save/reload. It writes `verification.json` and a full-size
`synthetic_roundtrip.pt`; budget roughly another checkpoint's disk space.
No server, browser, network download, robot, or training job is launched.
It checks the default encoder architecture; use matching code/config to verify
a differently shaped encoder. A single-device check does not verify the
Lightning/DeepSpeed distributed checkpoint-resume path.

For a training smoke test, reuse the approved workstation experiment's real
stats, paths, final Language V2 instructions, and cached frames. Create a new
local Hydra config inheriting `force_encoder` and provide all mandatory fields;
do not copy fake stats from tests. Then use the unchanged training entry point:

```bash
cd xr1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python3 tools/train.py --config-name YOUR_LOCAL_ENCODER_CONFIG
```

Inspect `--cfg job --resolve` first. Run one optimizer step, save, then resume
with `trainer.ckpt_path` and confirm optimizer step/moments/new weights restore.
Use a fresh output directory for the resume verification to preserve the first
run's artifacts. Match the existing launcher/world-size setup for the later
two-GPU check. No full training or real robot evaluation is requested here.

## Integrating newer workstation source

The user reports workstation source
`27eb389e44ec381e3554b753dfabf8879f62419f`, including final Language V2 and an
offline PNG frame-cache workaround, and a successful force-free 2,000-step
last-transformer-block pilot on two RTX A5000 GPUs. Those facts were not
independently audited, and that repository/workstation was not accessed.

Port only the new modules and localized hooks into the newer code. Preserve its
`JsonDataset` image decoding/cache, prompts/language selection, train launcher,
and any existing freeze policy. Reconcile Doosan dataset flags/normalization
markers and the runner's strict pretrained loader with the newer equivalents;
do not replace files wholesale with this older branch. Recheck its actual DiT
conditioning shapes, names, checkpoint prefixes, collation/repeat ordering, and
normalization formula before applying the hook. The base dataset, collate, VLM,
ROS/controller paths, and generic config were not edited here.

Next: reconcile into the newer source, execute the CPU suite and real-checkpoint
GPU script, then verify a Lightning/DeepSpeed one-step save/resume with real
training-only statistics. Only after those checks should matched pilot runs be
scheduled. Encoder gradients, bfloat16 numerical behavior, real checkpoint
loading, image/language integration, GPU memory, distributed resume, and task
performance remain unverified until those tests run.
