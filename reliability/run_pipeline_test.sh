#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0
export TMPDIR=/tmp
export MPLCONFIGDIR="$PWD/reliability/evidence/matplotlib"
export HF_HOME="$PWD/.loader_evidence/hf_home"
case "$1" in
  baseline)
    options=(--source .loader_evidence/baseline/xr1)
    if ! test -f .loader_evidence/synthetic-camera-fixture/SYNTHETIC_CAMERA_FIXTURE.txt; then
      options+=(--build-fixture)
    fi
    ;;
  fork) options=(--source xr1 --method fork) ;;
  spawn) options=(--source xr1 --method spawn) ;;
  diagnostic-fork) options=(--source xr1 --method fork --diagnostics) ;;
  diagnostic-spawn) options=(--source xr1 --method spawn --diagnostics) ;;
  resume-spawn) options=(--source xr1 --method spawn --resume-step 2 --diagnostics) ;;
  *) exit 2 ;;
esac
exec .venv/bin/python reliability/qualify_pipeline.py "${options[@]}" \
  --config /home/horus/robotics_thesis/xr1_claix/evidence/dataloader_rootcause/xr1_cpu_job_5410198_20261103T183740Z_16793/unpacked/metadata/resolved_compact_config.json \
  --split-root /home/horus/robotics_thesis/dataset/xr1_claix_transfer_v1/dataset_v2/splits \
  --fixture "$PWD/.loader_evidence/synthetic-camera-fixture" \
  --output "$PWD/reliability/evidence/20261103/pipeline-$1.json"
