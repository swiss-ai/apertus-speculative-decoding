#!/usr/bin/env bash
set -euo pipefail
repo=${1:?} output=${2:?}
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cell=$(basename "$output")
export TRITON_CACHE_DIR=/tmp/bev-flex-${SLURM_JOB_ID}/${cell}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/bev-flex-${SLURM_JOB_ID}/${cell}/inductor
env | sort | grep -E '^(ALPS_|CUDA_|HIP_|ROCR_|NCCL_|RCCL_|FI_|OFI_|SLURM_JOB_|TORCH|TRITON|OMP_)' > "$output/environment.txt"
python3 "$repo/results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/flex-correctness/probe.py" \
  --mask-source "$repo/results/70b/dspark/beverin-nonfinite-20261030/pinned-source/models/dflash/attention.py" \
  --output "$output/result.json"
