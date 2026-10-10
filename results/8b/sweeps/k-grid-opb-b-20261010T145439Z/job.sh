#!/bin/bash
#SBATCH --job-name=sweep-k-grid-opb-b
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/k-grid-opb-b-20261010T145439Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/k-grid-opb-b-20261010T145439Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_Oi3nbn.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-k-grid-opb-b-20261010T145439Z" bash -c   'export PYTHONPATH="/users/faruk_zahiragic/apertus-spec-decoding/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/users/faruk_zahiragic/apertus-spec-decoding/experiments/system-8b/k-grid-opb-b.yaml" --output "/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/k-grid-opb-b-20261010T145439Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
