#!/bin/bash
# RCCL over Slingshot between Beverin MI300A nodes in the ROCm vLLM image
# (aws-ofi-nccl hook). sbatch -N 2 methods/dspark/beverin/rccl-test.sbatch
#SBATCH -A infra01 -p mi300 -N 2 --gpus-per-node=4 --ntasks-per-node=1 -t 00:20:00 --job-name=bev-rccl
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
export REPO_ROOT="${SLURM_SUBMIT_DIR}"
EDF="${EDF:-${REPO_ROOT}/methods/dspark/beverin/env-vllm023.toml}"
export HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
LOG="${SLURM_SUBMIT_DIR%/repo}/logs/rccl-${SLURM_JOB_ID}.log"
echo "job ${SLURM_JOB_ID} nodes ${SLURM_JOB_NODELIST}"
srun --environment="${EDF}" --export=ALL bash -c '
  env | grep -E "^(NCCL|FI_)" | sort | sed "s/^/[env] /" | head -40
  torchrun --nnodes "${SLURM_JOB_NUM_NODES}" --nproc_per_node 4 --node_rank "${SLURM_NODEID}" \
    --rdzv_backend c10d --rdzv_endpoint "${HEAD_NODE}:29500" --rdzv_id "${SLURM_JOB_ID}" \
    "${REPO_ROOT}/methods/dspark/beverin/rccl_bench.py"' > "${LOG}" 2>&1 || echo "srun rc=$?"
# Full NCCL log in ${LOG}; the plugin, transport and results here.
grep -E "busbw|correct|Selected provider|Using network|NCCL_P2P|Error|Traceback" "${LOG}" | sort -u | head -60 || true
