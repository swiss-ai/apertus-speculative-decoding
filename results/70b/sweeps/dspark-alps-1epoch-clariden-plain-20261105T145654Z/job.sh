#!/bin/bash
#SBATCH --job-name=sweep-dspark-alps-1epoch-clariden-plain
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-alps1epoch/results/70b/sweeps/dspark-alps-1epoch-clariden-plain-20261105T145654Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-alps1epoch/results/70b/sweeps/dspark-alps-1epoch-clariden-plain-20261105T145654Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/sml-state/env_resolved_arm64_1zZ6wc.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-dspark-alps-1epoch-clariden-plain-20261105T145654Z" bash -c   'export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-alps1epoch/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-alps1epoch/experiments/system-70b/dspark-alps-1epoch-clariden-plain.yaml" --output "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-alps1epoch/results/70b/sweeps/dspark-alps-1epoch-clariden-plain-20261105T145654Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
