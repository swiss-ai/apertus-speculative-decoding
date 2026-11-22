#!/bin/bash
#SBATCH --job-name=sweep-sweep
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/slurm.txt"
# The image's digest, read on the compute node while the arms start.
sha256sum "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh" > "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/image.sha256" &
DIGEST=$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
if [ "0" = "1" ]; then
  srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/sml-state/env_resolved_arm64_DjkJNI.toml"     env PYTHONNOUSERSITE=1 python3 "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/serving/draft_kv/check_gpu.py"       --smoke --state-check --graph-helper-check > "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/state-kernels.log" 2>&1 || {
        status=$?
        kill ${SAMPLER} 2> /dev/null || true
        exit ${status}
      }
fi
if [ "1" = "1" ]; then
  # Node-wide GPU hardware counters for the whole sweep (serving/gpu-metrics-sampler.sh).
  srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/sml-state/env_resolved_arm64_DjkJNI.toml"     bash "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/serving/gpu-metrics-sampler.sh" "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/gpu-metrics"     "20" "120" &
  GPU_METRICS=$!
fi
srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/sml-state/env_resolved_arm64_DjkJNI.toml"   env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-loadtest/vllm-cache/sweep-sweep-20261122T173715Z" bash -c   'if [ "0" = "1" ]; then cd "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code"; fi; export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/src:${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/experiments/pd-decode-compute-bound/sweep.yaml" --output "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
if [ "1" = "1" ]; then
  # Let the chunk in progress finish and be written, then stop the sampler.
  touch "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/gpu-metrics/STOP"
  timeout 400 tail --pid=${GPU_METRICS} -f /dev/null || kill ${GPU_METRICS} 2> /dev/null || true
  # Per-second means and per-cell averages (never fatal: the reports stay).
  srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/sml-state/env_resolved_arm64_DjkJNI.toml"     env PYTHONNOUSERSITE=1 PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/src"     python3 -m apertus_bench.gpu_metrics export "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z" > "/iopsstor/scratch/cscs/faruk_zahiragic/pd-decode-compute-bound/code/results/8b/sweeps/sweep-20261122T173715Z/gpu-metrics/export.log" 2>&1 || true
fi
wait ${DIGEST} || true
exit ${status:-0}
