#!/usr/bin/env bash
# Bounded laptop tests only. Never calls a scheduler, SSH, trainer or GPU model.
set -eu
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0
export MAX_LENGTH=12288
export PYTHONPATH="$PWD/xr1:$PWD/xr1/tests:$PWD/reliability"
# IPC socket path names must fit sockaddr_un; these are transient runtime sockets,
# not evidence. Every retained test artifact uses reliability/evidence explicitly.
export TMPDIR=/tmp
export MPLCONFIGDIR="$PWD/reliability/evidence/matplotlib"
export HF_HOME="$PWD/.loader_evidence/hf_home"
exec .venv/bin/python -m unittest "$@" -v
