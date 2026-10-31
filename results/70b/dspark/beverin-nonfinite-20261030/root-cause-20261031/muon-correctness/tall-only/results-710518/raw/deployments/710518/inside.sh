#!/usr/bin/env bash
set -euo pipefail
repo=${1:?} output=${2:?}
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SOURCE_REVISION=$(cat "$repo/REVISION") SOURCE_BRANCH=$(cat "$repo/SOURCE_BRANCH")
export NCCL_DEBUG=INFO
base="$repo/results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/muon-correctness"
env | sort | grep -E '^(ALPS_|CUDA_|HIP_|ROCR_|NCCL_|RCCL_|FI_|OFI_|SLURM_JOB_|TORCH|OMP_|SOURCE_|MUON_)' > "$output/environment.txt"
read -r -a oracle_shapes <<< "${MUON_ORACLE_SHAPES:-16x80 8192x40960}"
command=(python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2
  --max-restarts=0 --monitor-interval=1 "$base/probe.py" --output "$output"
  --device cuda --shapes "${oracle_shapes[@]}")
printf '%q ' "${command[@]}" > "$output/torchrun-command.txt"
printf '\n' >> "$output/torchrun-command.txt"
"${command[@]}"
