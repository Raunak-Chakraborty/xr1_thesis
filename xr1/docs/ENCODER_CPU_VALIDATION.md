# Stage 5.1: persistent CPU validation environment

CPU validation passed on 2026-10-27: **61 tests passed, no failures, no skips,
no errors**. This includes 34 focused encoder tests, 24 existing Doosan tests,
and 3 existing training-logic tests. Among the focused tests, 21 execute actual
PyTorch/NumPy encoder, gradient, serialization, or data-path checks. The syntax
test is not numerical evidence.

Use the persistent sandbox environment, not the empty system Python:

```bash
bash xr1/scripts/test_encoder_cpu.sh
```

This launcher uses `/home/agent/.venvs/xr1-encoder/bin/python`, runs an import and
CPU FP32/BF16 autograd preflight, and then runs all three suites. It disables
CUDA visibility, model downloads, user-site imports, and unrelated pytest plugin
autoload. It keeps bytecode/pytest caches out of the source worktree and forwards
additional pytest arguments, for example `--junitxml=/NEW/REPORT/PATH/results.xml`.
`XR1_ENCODER_PYTHON` can explicitly select another already-prepared CPU venv.

Environment: Ubuntu 26.04 LTS x86_64, CPython 3.14.4, torch **2.9.1+cpu**,
torchvision **0.24.1+cpu**, NumPy 2.3.5, Transformers 4.57.1, Lightning and
PyTorch Lightning 2.5.3, Hydra 1.3.2, OmegaConf 2.3.0, mmengine 0.10.7,
Pillow 12.0.0, pytest 8.4.2, Decord 0.6.0, Liger 0.6.5, Triton 3.5.1.
All 70 installed versions, including transitive dependencies, are recorded in
`xr1/assets/requirements-encoder-cpu.lock.txt`. This is a CPU testing lock;
`xr1/assets/requirements.txt` remains the original workstation/GPU specification.

`torch.version.cuda` is `None`, `torch.cuda.is_available()` is false, and no
`nvidia-*` distributions are installed. Liger/Triton are installed solely because
the existing model definitions import them eagerly. No Triton kernel, GPU
driver/toolkit installation, or GPU workload ran. No pretrained assets were
downloaded. The system Python and host configuration were not changed.

The Python 3.12 source URL
`https://www.python.org/ftp/python/3.12.12/Python-3.12.12.tgz` returned HTTP 403.
No restriction was bypassed. Official CPU wheels were available for the installed
Python 3.14 interpreter, so it was reused. Torch/torchvision were moved from the
original 2.8.0/0.23.0 requirements to 2.9.1/0.24.1 for those wheels. NumPy/Pillow
also use versions with Python 3.14 wheels. Core source API dependencies retain
their original pins. Full compatibility with a different workstation environment
still needs testing there.

## Defect repaired

The first focused run had 32 passes and one failure: the mandatory experiment
output directory was not actually mandatory. OmegaConf merges `???` by retaining
an inherited concrete value, so `trainer.default_root_dir` silently stayed
`./test/`. The experiment now declares a new mandatory `encoder_run_dir` and
interpolates it into `trainer.default_root_dir`.

Supply a fresh path when composing the experiment:

```text
encoder_run_dir=/PATH/INSIDE/PROJECT/NEW_RUN
```

The regression checks missing-path rejection, explicit-path resolution, and the
unchanged generic default. No encoder, fusion, initialization, gradient, freeze,
checkpoint-loader, state/action, backbone, or loss redesign was needed.

## Remaining package metadata issue

Both `pip check` and `uv pip check` return nonzero for Decord 0.6.0. Its published
Python 3 wheel contains `Tag: cp36-cp36m-manylinux2010_x86_64` internally. This is
incompatible metadata on Python 3.14 even though the package imports, its native
ndarray/NumPy round trip passes, and all targeted tests pass. The package metadata
was not edited to suppress the warning. Video decoding on real episodes was not
tested, and this CPU result does not validate Decord's full media stack.

## Logs and limits

The complete report is:

```
/home/horus/robotics_thesis/xr1_force_exp/reports/codex_encoder_exports/CODEX_ENCODER_CPU_VALIDATION_LATEST.txt
```

Detailed installation, preflight, pytest, and JUnit logs are preserved in:

```
/home/horus/robotics_thesis/xr1_force_exp/reports/codex_encoder_exports/cpu_validation_20261027_171828/
```

The original Stage 5 handoff was preserved byte-for-byte. Initial failures and
collection errors remain in the logs; they were not replaced by passing runs.

The CPU harness runs actual XR-1 source methods and one DiT layer with a small
VLM cache test double. The original loss tests import the actual XR-1 model
definitions without constructing the 5B network. Checkpoint tests use synthetic
weights and actual PyTorch model/optimizer serialization. These results verify
CPU tensor behavior; they do not establish full pretrained XR-1 compatibility,
GPU/FlashAttention behavior, distributed Lightning/DeepSpeed resume, real dataset
integration, or assembly/contact performance.

Reconcile the localized extension into workstation source
`27eb389e44ec381e3554b753dfabf8879f62419f` manually. This worktree's user-reported
base remains `afe739d1c9aac9218036a13fee9f9eb182747159`. Preserve the workstation's
final Language V2, offline PNG cache, launcher, and freeze settings. Include the
new `encoder_run_dir` requirement in its configuration reconciliation. No Git
metadata workaround, merge, rebase, commit, push, or workstation access occurred.

Remaining workstation checks: existing `tools/verify_force_encoder.py` with real
pretrained weights and GPU; one-step Lightning/DeepSpeed save/resume; then real
converted Doosan data using genuine training-only statistics. No test in those
categories was executed here.

## Reusing or recreating the environment

Reuse the absolute venv executable or the launcher above in future sessions.
The venv and package caches are in the private microVM filesystem, outside Git.
They persist while this sandbox filesystem persists. If the microVM is recreated,
restore the environment from the saved lock instead of assuming the path exists.

Only if the environment is absent, recreate it without changing system Python:

```bash
uv venv --python /usr/bin/python3.14 --no-python-downloads /home/agent/.venvs/xr1-encoder
uv pip install --python /home/agent/.venvs/xr1-encoder/bin/python \
  --index-url https://download.pytorch.org/whl/cpu --only-binary :all: --no-deps \
  'torch==2.9.1+cpu' 'torchvision==0.24.1+cpu'
uv pip install --python /home/agent/.venvs/xr1-encoder/bin/python \
  --index-url https://pypi.org/simple \
  --only-binary torch,torchvision,numpy,pillow,scipy,matplotlib,tokenizers,safetensors,triton \
  -r xr1/assets/requirements-encoder-cpu.lock.txt
bash xr1/scripts/test_encoder_cpu.sh
```

The existing CPU wheels satisfy the exact `+cpu` pins in the lock; do not replace
them with PyPI's CUDA-enabled torch distribution. A small pure-Python ANTLR wheel
build is expected for Hydra. No native framework compilation is needed. The
Decord metadata warning described above remains a known limitation of this lock.
