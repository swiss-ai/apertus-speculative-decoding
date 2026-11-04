#!/bin/bash
#SBATCH --job-name=sweep-arena-hard-a
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/arena-hard-a-20261021T125236Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/arena-hard-a-20261021T125236Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_c6jxoD.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-arena-hard-a-20261021T125236Z" bash -c   'export PYTHONPATH="/users/faruk_zahiragic/apertus-spec-decoding/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/users/faruk_zahiragic/apertus-spec-decoding/experiments/system-8b/arena-hard-a.yaml" --output "/users/faruk_zahiragic/apertus-spec-decoding/results/8b/sweeps/arena-hard-a-20261021T125236Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
