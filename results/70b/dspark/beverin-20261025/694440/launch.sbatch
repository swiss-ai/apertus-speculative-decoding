#!/bin/bash
#SBATCH -A infra01 -p mi300 -N 2 --gpus-per-node=4 --ntasks-per-node=1 -t 00:45:00 --nice=10000 --job-name=bev-dtensor-compare
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
b=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
repo="$b/repo-786ebde3"
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
p="$b/runs/dtensor-compare-$SLURM_JOB_ID"
mkdir -p "$p"
cp "$0" "$p/launch.sbatch"
cp "$b/dtensor-probe-isolated-20261025.py" "$p/probe.py"
scontrol show job "$SLURM_JOB_ID" > "$p/slurm-job.txt"
for image in 031 023; do
  cp "$repo/methods/dspark/beverin/env-vllm${image}.toml" "$p/environment-${image}.toml"
  sha256sum "$(sed -n 's|^image = "\(.*\)"|\1|p' "$p/environment-${image}.toml")" > "$p/container-${image}.sha256" &
  for nodes in 1 2; do
    export PROBE_NODES="$nodes" PROBE_OUTPUT="$p/torch-vllm${image}-${nodes}n.log" PROBE_SCRIPT="$p/probe.py"
    srun --input=none -N "$nodes" -n "$nodes" --ntasks-per-node=1 --environment="$p/environment-${image}.toml" --export=ALL bash -c '
      unset ROCR_VISIBLE_DEVICES
      export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
      torchrun --nnodes "$PROBE_NODES" --nproc_per_node=2 --node_rank "$SLURM_NODEID" --rdzv_backend=c10d --rdzv_endpoint "$HEAD_NODE:29503" --rdzv_id "$SLURM_JOB_ID-$PROBE_NODES" "$PROBE_SCRIPT"' > "$PROBE_OUTPUT" 2>&1
  done
done
wait
