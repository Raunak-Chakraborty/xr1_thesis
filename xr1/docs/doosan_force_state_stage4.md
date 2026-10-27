# Stage 4: force through existing XR-1 state conditioning

## Implemented scope

`data=doosan_force_state` inherits `DoosanDataModule` and its JSON dataset.
It sets `use_wrench: true` and requires a real statistics JSON through
`data.params.train_datasets.normalization_path`. No encoder, expert, new model
parameter, or checkpoint key is added. `data=doosan` remains force-free.

The adapter packs joint positions at 0:6, zero at 6, measured gripper at 7,
TCP position at 8:11, absolute TCP rotation vector at 11:14, joint velocity
at 14:20, wrench `[Fx,Fy,Fz,Tx,Ty,Tz]` at 20:26, and zeros at 26:60.
These are 19 force-free or 25 force-aware semantic values in 60 storage slots.
Action labels remain the converter's precomputed seven values at 0:7, with
60D storage and horizon 30. Both temporal padding and action padding are masked.

Training trace:

1. `DoosanJsonDataset._doosan_state` reads `proprios.wrench[frame]` when enabled;
   missing, wrong-width, or non-finite wrench fails rather than becoming zeros.
2. The dataset normalizes the complete `(1,60)` state. The unchanged collator
   stacks states without slicing features.
3. `XR1.py` projects all 60 features via `state_projector_choice` into the VLM's
   `<state>` token. The VLM injects this embedding using `masked_scatter`.
4. Independently, `state_projector` projects all 60 features into the DiT state
   token, concatenated with the sink and action tokens.

Important distinction: the existing evaluation branch does **not** insert a
state token into the VLM. Its visual/text VLM cache and full-state DiT projector
are preserved. Force enters both state projectors during training, and the DiT
state projector during the existing inference pathway. No claim of a new
inference VLM-state pathway is made.

## Required converter statistics contract

The normalization JSON must contain numeric lists:

- `mean`, `std`: shape `(30,60)`, finite float32-compatible values, nonnegative
  standard deviations; every action statistic outside 0:7 must be zero.
- `q01`, `q99`: shape `(1,60)`, finite float32-compatible values. Outside zero
  padding, q99 must exceed q01. Wrench occupies **storage indices 20:26**, not
  a densely packed semantic 19:25 range.
- Each of the six wrench quantile spans must be greater than `1e-6` after
  float32 conversion. Missing, all-zero, constant, reversed, non-finite, or
  float32-collapsed wrench bounds are rejected, not replaced with made-up ones.
- For force training, `normalization_metadata` is mandatory:

```text
schema: "doosan_state_v1"
source: "dataset"
split: "train"
use_wrench: true
dataset_id: nonempty identifier of the exact training split/manifest
num_state_samples: integer >= 2, the number actually used for state quantiles
wrench_order: ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
wrench_frame: nonempty name of the actual frame used in the converted data
force_unit: "N"
torque_unit: "Nm"
```

Compute q01/q99 from real synchronized states in training episodes only; keep
the same sensor compensation, frame, ordering, and units at deployment. Compute
action mean/std from converter-produced semantic labels without differencing
them again. Record the window/sampling policy and converter revision with the
dataset manifest. Constant force channels require investigation/re-export or a
separately approved normalization policy; this branch does not invent a span.

This interface validates declared provenance and numerical usability, not the
truth of the provenance. Verify the dataset manifest and checksums separately.
The external converter was not edited. `tools/compute_normalize.py` is the
unchanged generic two-arm example and must **not** be used for Doosan statistics.

`resolve_doosan_data(data)` accepts either inline statistics or a
`normalization_path`, never both (null placeholders are allowed with the file).
It returns a copy with statistics embedded and the file path removed. Thus the
saved training config is also the deployment contract. It zeros quantile bounds
for padding, including state index 6 and all inactive tail slots. Force-free
profiles also zero 20:26 even if shared input statistics contain force bounds.
Generic XR-1 profiles do not use this resolver.

## Deployment interface and local Stage 3 overlap

`DoosanClient(host=..., port=..., use_wrench=True)` is explicit force inference;
the default is false. Enabled calls require `robot_state["wrench"]`. Calls send
a boolean `use_wrench` request field. The server checks this against the saved
Doosan profile before normalization. Missing mode is accepted only for the
force-free profile, for legacy force-free clients. Opposite modes fail closed.

`DoosanClient.pack_state(robot_state, use_wrench=None)` retains the old static
packing helper's optional-wrench semantics; production calls always pass the
constructor's boolean. Static helper calls alone do not select a training mode.

`load_stats` chooses the seven-channel Doosan mask only for Doosan configs;
generic XR-1 retains its multi-arm mask. `Server.preprocess_state` validates
Doosan shape `(B,1,60)`, finiteness, mode agreement, and restores padded zeros.
Its default mode is `None`, retaining generic normalization behavior.

These local deployment mask/padding corrections may overlap the earlier Stage 3
work. No baseline worktree was accessed and no merge is claimed. Training and
deployment use the same resolved quantiles. The generic client, IO utilities,
Doosan adapter, datamodule, model architecture, and model YAML remain unchanged.

## Workstation verification (not executed on the laptop)

Use the lab's isolated environment with the versions in
`assets/requirements.txt`. The local lab snapshot documents Python 3.12;
validate CUDA, PyTorch 2.8.0, torchvision 0.23.0, FlashAttention 2, bf16,
Liger 0.6.5, Lightning 2.5.3, DeepSpeed 0.18.9, Transformers 4.57.1,
NumPy, Pillow, decord, Hydra, OmegaConf, mmengine, and tqdm on that workstation.
The snapshot is a reference, not an installer. Use the already-staged XR-1
checkpoint and cached Qwen3-VL processor/config assets. No laptop installation,
GPU workload, checkpoint load, or robot motion is required by this change.

Set paths to real workstation locations before running these commands:

```bash
export PROJECT=/home/horus/robotics_thesis/xr1_force_exp
export REPO="$PROJECT/worktrees/xr1_force_state"
export TRAIN_JSON=/absolute/path/to/converted/training_episode_json
export STATS=/absolute/path/outside/TRAIN_JSON/doosan_train_normalization.json
export CHECKPOINT=/absolute/path/to/existing/model_states.pt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/xr1${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$PROJECT/reports/stage4_verification/$RUN_ID"
export CHECK_DIR="$PROJECT/reports/stage4_verification/$RUN_ID"
cd "$REPO"
git status --short
git branch --show-current
git rev-parse HEAD
sha256sum "$CHECKPOINT" "$STATS"
python -B xr1/tools/validate_doosan_normalization.py "$STATS" --use-wrench
bash xr1/scripts/test_force_state_offline.sh
bash xr1/scripts/test_doosan_offline.sh
```

Require **zero skips** in the focused suite on the workstation. The offline
scripts hide CUDA for CPU tests only; the parent shell remains unchanged.
Do not treat an all-skipped tensor suite as a validated runtime.

Check actual Hydra composition without building a model:

```bash
cd "$REPO/xr1"
python -B tools/train.py data=doosan_force_state \
  "data.params.train_datasets.paths=[$TRAIN_JSON]" \
  "data.params.train_datasets.normalization_path=$STATS" \
  "model.params.pretrained=$CHECKPOINT" --cfg job --resolve
```

Run the provided one-batch smoke tool on a GPU with sufficient memory:

```bash
python -B tools/smoke_doosan_force_state.py --checkpoint "$CHECKPOINT" \
  --statistics "$STATS" --train-path "$TRAIN_JSON" \
  --output "$CHECK_DIR/force_smoke.json" --check-output-sensitivity
python -B tools/smoke_doosan_force_state.py --checkpoint "$CHECKPOINT" \
  --statistics "$STATS" --train-path "$TRAIN_JSON" \
  --output "$CHECK_DIR/baseline_smoke.json" --force-free
```

The tool uses the existing runner checkpoint loader, which rejects missing or
unexpected keys, runs a full-model forward/backward, and records finite losses,
input wrench gradients, and each original projector's six force-column gradient
norms. It performs **zero optimizer steps**, refuses existing report files, and
never connects to a robot. It is unsharded and may need substantially more GPU
memory than distributed fine-tuning. If it does not fit, use the approved lab
DeepSpeed strategy for a one-step trainer smoke instead; do not reduce dimensions
or silently change architecture to make it fit. Real checkpoint compatibility
is unverified until the lab executes this check successfully.

The optional inference probe uses the original visual/text-only inference
prompt, resets RNG for each generated output, changes one normalized wrench
channel at a time, and records maximum change over the seven action channels.
Check all six force values survive preprocessing; inspect projector gradients
and paired output differences. A zero change needs investigation (saturation,
bf16 precision, checkpoint weights, or genuinely unused features), not a claimed
positive result. Force-free projector force-column weight gradients should be
zero because its force input features are zero. Raw input-state gradients need
not be zero, since the model itself still has 60 input columns. A direct manual
injection into baseline model inputs bypasses deployment masking and is not a
valid force-free deployment test.

Short matched training comparison, only after successful smoke checks:

```bash
for MODE in false true; do
  torchrun --standalone --nproc_per_node=1 tools/train.py \
    data=doosan_force_state \
    "data.params.train_datasets.use_wrench=$MODE" \
    "data.params.train_datasets.paths=[$TRAIN_JSON]" \
    "data.params.train_datasets.normalization_path=$STATS" \
    data.params.train_datasets.num_workers=0 \
    data.params.train_datasets.persistent_workers=false \
    "model.params.pretrained=$CHECKPOINT" \
    trainer.num_nodes=1 trainer.seed=42 trainer.max_steps=100 \
    trainer.save_interval=100 trainer.scheduler.params.num_warmup_steps=10 \
    trainer.project=doosan_stage4 \
    "trainer.exp_name=wrench_${MODE}_${RUN_ID}" \
    "trainer.default_root_dir=$CHECK_DIR/training"
done
```

Use the approved distributed launch instead if one GPU does not fit. The two
runs use the same checkpoint, split, seed, optimizer, and state/action contracts;
the resolver masks force bounds only in the false run. Compare finite losses,
gradient norms, action masks, throughput, and held-out recorded episodes. A
100-step comparison checks functionality, not assembly performance or safety.
The existing generic config exporter also writes `xr1/assets/config.py`; preserve
any existing copy before training and review this known overwrite behavior.
Always choose fresh run names; the existing trainer does not refuse reused run
directories. No real-robot evaluation is authorized by these commands.

## Outstanding evidence

This sandbox has only Python 3.14 standard-library dependencies available.
Sixteen dependency-free tests pass; ten real-tensor tests are explicitly skipped.
All 44 Python files pass syntax compilation; both offline shell scripts pass
`bash -n`. The broad pre-existing Doosan suite has four dependency import errors.
No Hydra runtime composition, real checkpoint, image/MP4 batch, GPU execution,
training, or robot evaluation has been validated here.

Git metadata is unavailable: `.git` points at a main-repository gitdir not mounted
in this sandbox. The requested branch is `thesis/xr1-force/state` but cannot be
independently verified; HEAD is unknown. Do not repair or replace `.git` blindly.
The shared context files are stale and were intentionally not modified.
An unrelated existing `test_training_logic.py` imports the lowercase module
`mibot.models.VLA.xr1`, although the file is `XR1.py`; that test is outside the
Doosan unittest runner and was not changed or reported as passing.

Next implementation task: extend the external converter in its own authorized
workspace to emit this measured statistics/provenance contract and add an
episode-level real-data integrity/preflight test, then run the workstation gates.
