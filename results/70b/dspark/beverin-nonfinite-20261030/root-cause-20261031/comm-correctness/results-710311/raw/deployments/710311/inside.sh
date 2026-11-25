#!/usr/bin/env bash
set -euo pipefail
tools=${1:?tools path}
cell_dir=${2:?cell directory}
cell=${3:?cell name}
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET
export TRITON_CACHE_DIR=/tmp/bev-comm-${SLURM_JOB_ID}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/bev-comm-${SLURM_JOB_ID}/inductor
case "$cell" in
  upstream-default) unset NCCL_P2P_DISABLE ;;
  upstream-p2p-disabled) export NCCL_P2P_DISABLE=1 ;;
  alps-default) [[ "${NCCL_P2P_DISABLE:-}" == 1 ]] ;;
  *) exit 2 ;;
esac
env | sort | grep -E '^(NCCL_|RCCL_|FI_|OFI_|CUDA_|HIP_|ROCR_|GPU_|SLURM_JOB_|ALPS_)' \
  > "$cell_dir/environment-node${SLURM_NODEID}.txt"
hostname > "$cell_dir/hostname-node${SLURM_NODEID}.txt"
command=(python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=2
  --node-rank="$SLURM_NODEID" --rdzv-backend=c10d \
  --rdzv-endpoint="${HEAD_NODE}:29612" --rdzv-id="${SLURM_JOB_ID}-${cell}" \
  "$tools/probe.py" --output "$cell_dir" --iterations 32 \
  --shapes 128x1024 128x40960)
printf '%q ' "${command[@]}" > "$cell_dir/torchrun-command-node${SLURM_NODEID}.txt"
printf '\n' >> "$cell_dir/torchrun-command-node${SLURM_NODEID}.txt"
"${command[@]}"
