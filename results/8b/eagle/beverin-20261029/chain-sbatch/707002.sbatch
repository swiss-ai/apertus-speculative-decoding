#!/bin/bash
#SBATCH --job-name=eagle-8b-e31-opb-thinkon
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --exclusive

#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/results/8b/eagle/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/results/8b/eagle/logs/%j.err
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff

#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail
export I_AM_ON_AN_ALLOCATED_GPU_JOB=1
export EAGLE_TRAIN_CONFIG=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/methods/eagle/configs/8b/train-e31-opb-thinkon-2n.yaml
export EAGLE_CUDA_DEVICES=0,1,2,3
export EAGLE_STEPS=train
export EAGLE_RUN_ID=stage1
# Wall-clock deadline for train_rollout (it stops, saves and exits 4 before this).
limit_s=$(echo "24:00:00" | awk -F: '{ if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2 }')
export EAGLE_STOP_AT=$(( $(date +%s) + limit_s ))
export TORCHSPEC_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/scratch/TorchSpec
export EAGLE_PYDEPS=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/scratch/pydeps
export HF_HOME=/iopsstor/scratch/cscs/${USER}/hf_home
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export EAGLE_PLATFORM=rocm
export EAGLE_BLAS_BACKEND=cublas
export EAGLE_START_GUARD_RUN_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/runs/e31-opb-thinkon-stage1
export EAGLE_MEM_LOG_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/logs
# No percent-strftime in this body: pyxis --environment sbatch mangled one (job 3491998).
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID} nodes=2"
python3 /iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/methods/eagle/beverin/start_guard.py "${EAGLE_START_GUARD_RUN_DIR}" || exit 3
train_once() {
export EAGLE_NNODES=2
export EAGLE_MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
srun --ntasks-per-node=1 --environment=/users/faruk_zahiragic/.sml/env_resolved_arm64_V8GmEu.toml \
  bash /iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff/methods/eagle/launch/train-eagle-on-node.sh
}
# A crash (not 0 = done, not 4 = saved at the deadline) relaunches on the same nodes
# from the last resume checkpoint instead of giving the allocation back to the queue.
attempt=1
while :; do
  set +e; train_once; rc=$?; set -e
  if [ "${rc}" = 0 ] || [ "${rc}" = 4 ]; then break; fi
  left=$(( EAGLE_STOP_AT - $(date +%s) ))
  if [ "${attempt}" -ge 3 ] || [ "${left}" -lt 1800 ]; then
    echo "END $(date -u -Iseconds) rc=${rc} attempts=${attempt}"
    exit "${rc}"
  fi
  echo "RETRY $(date -u -Iseconds) rc=${rc} attempt=${attempt} left_s=${left}"
  attempt=$(( attempt + 1 ))
  sleep 30
done
echo "END $(date -u -Iseconds) rc=${rc}"
exit "${rc}"
