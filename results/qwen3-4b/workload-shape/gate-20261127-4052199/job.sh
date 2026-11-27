#!/bin/bash
#SBATCH --job-name=shape-gate-q4b
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/job.out
set -uo pipefail
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/slurm.txt"
( sha256sum "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/image.sha256" ) &
IDENT=$!
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 2000 > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/gpu.csv" 2> /dev/null &
SAMPLER=$!
srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/environment.toml" bash -c '
  export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-4b/code-75e9f01d7df0/src:${PYTHONPATH:-}"
  export HOME=/tmp/gate-home; mkdir -p ${HOME}
  export VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
  export XDG_CACHE_HOME=/tmp/xdg-cache XDG_CONFIG_HOME=/tmp/xdg-config TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
  export OUT="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127" TARGET="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/Qwen--Qwen3-4B" DRAFTER="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/RedHatAI--Qwen3-4B-speculator.dspark" CORPUS="/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/runs/full-20261127/inputs/corpus-measured.jsonl" SLICES="opb math-gsm8k probe-math probe-humaneval multi-turn code-multi json-schema tool-call lang-eng arena-hard-clean arena-hard-contaminated chat summarization long-context translate-to-en translate-from-en lang-deu lang-fra lang-ita lang-spa lang-roh lang-zho lang-jpn lang-hin lang-ara lang-swa lang-yor lang-amh"
  export REQUESTS="32" IMAGE="/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh" DRAFTER_SHA="7f13b62c900e88c7a8027055ae9882dcffaf23af66fbd494f792174be5c1245a"
  export TARGET_ID="Qwen/Qwen3-4B" TARGET_REV="1cfa9a7208912126459214e8b04321603b3df60c" DRAFTER_ID="RedHatAI/Qwen3-4B-speculator.dspark" DRAFTER_REV="bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23"
  export CORPUS_SHA="$(sha256sum "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/runs/full-20261127/inputs/corpus-measured.jsonl" | cut -d" " -f1)"
  export SOURCE_REVISION="$(cat "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-4b/code-75e9f01d7df0/SOURCE_REVISION" 2> /dev/null || echo unknown)"
  cd "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-4b/code-75e9f01d7df0"
  python3 -c "import torch, vllm, transformers; print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)" > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/runtime-versions.txt" 2>&1
  pip freeze > "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/pip-freeze.txt" 2> /dev/null || true
  gpu=0; pids=()
  for arm in k3-greedy k7-greedy k3-t1 k7-t1; do
    bash "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/arm.sh" "${arm}" "${gpu}" & pids+=($!); gpu=$((gpu + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  exit ${failed}
'
status=$?
kill ${SAMPLER} 2> /dev/null || true
wait ${IDENT} || true
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}" >> "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/gate/full-20261127/slurm.txt"
exit ${status}
