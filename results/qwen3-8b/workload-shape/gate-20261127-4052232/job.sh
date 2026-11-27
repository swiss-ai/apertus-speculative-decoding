#!/bin/bash
#SBATCH --job-name=shape-gate-qwen3-8b
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/job.out
set -uo pipefail
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date -u +%FT%TZ)" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/slurm.txt"
( sha256sum "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/image.sha256" ) &
IDENT=$!
srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/environment.toml" bash -c '
  python3 -c "import torch, vllm, transformers; print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/runtime-versions.txt" 2>&1
  pip freeze > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/pip-freeze.txt" 2> /dev/null || true
  status=0
  ( bash /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/experiments/workload-shape/gate-arm.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/eagle3-k3;  ) & pids+=($!)
  ( bash /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/experiments/workload-shape/gate-arm.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/dflash-k3;  ) & pids+=($!)
  ( bash /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/experiments/workload-shape/gate-arm.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/dflash-k15;  ) & pids+=($!)
  ( bash /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/experiments/workload-shape/gate-arm.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/eagle3-k3-t1; bash /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/experiments/workload-shape/gate-arm.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/dflash-k3-t1;  ) & pids+=($!)
  for p in "${pids[@]}"; do wait ${p} || status=1; done
  exit ${status}
'
status=$?
wait ${IDENT} || true
echo "finished=$(date -u +%FT%TZ) status=${status}" >> "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/gate-20261127/slurm.txt"
exit ${status}
