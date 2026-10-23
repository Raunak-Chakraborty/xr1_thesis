# XR1 thesis: CLAIX reproduction

## Repository

Personal thesis repository:

https://github.com/Raunak-Chakraborty/xr1_thesis

Official Xiaomi upstream:

https://github.com/XiaomiRobotics/Xiaomi-Robotics-1

Use the exact thesis Git commit when reproducing an experiment.

## Scope

This repository contains the XR1 model code and the
Doosan M1013 thesis integration.

The initial thesis baseline is force-free.

It uses:
- TCP RGB camera
- External RGB camera
- 19 informative proprioceptive state values
- 60D model-facing state
- 7D semantic Cartesian actions
- 60D model-facing actions
- 30-step action horizon

The wrench is excluded from this baseline.

## Other required repositories

The data-conversion and XR1 exchange exporter is maintained at:

https://github.com/Raunak-Chakraborty/doosan-forcevla-data-tools

Validated conversion checkpoint:

29776c5ddd4cf1e7d575612443e416d5e56d5b9d

Keep the conversion repository separate and version-pinned.

## Training data

The authoritative processed dataset was validated and frozen
after conversion.

It contains 323 episodes from the Cranfield and Siemens
assembly demonstrations.

Do not recommit the dataset into the source repository.

Copy or stage the dataset separately on CLAIX and verify
the archived checksums before training.

Preserve the episode hierarchy and both MP4 streams.

Create train/validation/test splits at episode level.

Compute XR1 normalization statistics from training episodes
only, never from validation or test episodes.

## Checkpoint

The lab checkpoint is stored outside Git:

xr1_lab/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt

The pretrained checkpoint is provided by Xiaomi Robotics
through the official model release.

Record and verify the checkpoint checksum when staging
the model on CLAIX.

Qwen3-VL-4B-Instruct processor/backbone resources may
also be required by the XR1 runtime.

## Lab environment reference

The validated lab uses:

- Python 3.12
- PyTorch 2.8.0+cu128
- Transformers 4.57.1
- Decord 0.6.0
- FlashAttention 2.8.3

See:

xr1/assets/doosan_lab_environment_snapshot.txt

This records exact lab package versions, including the
CUDA-dependent packages.

The snapshot is not guaranteed to install unchanged on
CLAIX. First inspect the actual CLAIX GPU nodes, drivers,
CUDA modules, Python modules, and cluster package policy.

Choose compatible CUDA-enabled PyTorch and FlashAttention
builds before recreating the remaining dependencies.

The original upstream installation instructions are in:

xr1/README.md

The upstream Python package is installed from the
xr1/ subdirectory, not from the repository root.

Do not copy the lab virtual environment to CLAIX.

## Offline unit tests

From the repository root, with a working XR1 Python environment:

    PYTHON_BIN=/path/to/python bash xr1/scripts/test_doosan_offline.sh

Expected baseline: 24 Doosan tests pass.

The tests must not download model files.

## Configuration

The force-free Doosan data profile is:

xr1/configs/data/doosan.yaml

Before training, provide actual exported XR1 dataset paths
and train-only normalization statistics.

Do not substitute dummy data, fake normalization values,
or artificial camera streams.

## Training reproducibility record

For every experiment, preserve:

- XR1 thesis commit
- Data-tools converter commit
- Dataset freeze/checksum manifest
- Train/validation/test episode split manifest
- Model checkpoint identity and checksum
- Exact resolved Hydra configuration
- Python/CUDA/PyTorch environment
- Random seed and distributed training settings
- Final checkpoint and evaluation results

Do not put datasets, model binaries, environment directories,
or generated training logs into normal Git history.
