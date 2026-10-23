#!/bin/bash
#SBATCH --job-name=sweep-spec
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/results/8b/drafter-kv/kv-sparsity-20261022-a/allocations/preflight-runtime-graphs-03/results/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/results/8b/drafter-kv/kv-sparsity-20261022-a/allocations/preflight-runtime-graphs-03/results/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
if [ "1" = "1" ]; then
  srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_dqptD4.toml"     env PYTHONNOUSERSITE=1 python3 "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/serving/draft_kv/check_gpu.py"       --smoke --state-check --graph-helper-check > "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/results/8b/drafter-kv/kv-sparsity-20261022-a/allocations/preflight-runtime-graphs-03/results/state-kernels.log" 2>&1 || {
        status=$?
        kill ${SAMPLER} 2> /dev/null || true
        exit ${status}
      }
fi
srun --overlap --nodes=1 --ntasks=1 --environment="/users/faruk_zahiragic/.sml/env_resolved_arm64_dqptD4.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-spec-20261022T180043Z" bash -c   'export PYTHONPATH="/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/results/8b/drafter-kv/kv-sparsity-20261022-a/allocations/preflight-runtime-graphs-03/spec.yaml" --output "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity/results/8b/drafter-kv/kv-sparsity-20261022-a/allocations/preflight-runtime-graphs-03/results"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
