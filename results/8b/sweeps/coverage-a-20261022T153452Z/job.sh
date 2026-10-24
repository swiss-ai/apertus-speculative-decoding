#!/bin/bash
#SBATCH --job-name=sweep-coverage-a
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/results/8b/sweeps/coverage-a-20261022T153452Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/results/8b/sweeps/coverage-a-20261022T153452Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_vfSyEY.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-coverage-a-20261022T153452Z" bash -c   'export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/experiments/system-8b/coverage-a.yaml" --output "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/results/8b/sweeps/coverage-a-20261022T153452Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
