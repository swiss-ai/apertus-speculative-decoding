#!/bin/bash
#SBATCH --job-name=sweep-think-1gpu
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/sweeps/think-1gpu-20261023T083948Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/sweeps/think-1gpu-20261023T083948Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/sweeps/think-1gpu-20261023T083948Z/slurm.txt"
# The image's digest, read on the compute node while the arms start.
sha256sum "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh" > "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/sweeps/think-1gpu-20261023T083948Z/image.sha256" &
DIGEST=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_jfsqg7.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-think-1gpu-20261023T083948Z" bash -c   'export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/experiments/high-concurrency-8b/think-1gpu.yaml" --output "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/sweeps/think-1gpu-20261023T083948Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
wait ${DIGEST} || true
exit ${status:-0}
