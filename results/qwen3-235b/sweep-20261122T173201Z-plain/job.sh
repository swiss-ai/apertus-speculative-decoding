#!/bin/bash
#SBATCH --job-name=qwen3-plain
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/job.out
set -uo pipefail
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/slurm.txt"
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/gpu-memory.csv" 2> /dev/null &
SAMPLER=$!
( sha256sum "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh" > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/image.sha256" ) &
IDENT=$!
srun --overlap --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/environment.toml" bash -c '
  set -uo pipefail
  export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/src:${PYTHONPATH:-}"
  # HOME is over quota: keep every cache, config and usage-stats write node-local.
  export HOME=/tmp/qwen3-home; mkdir -p ${HOME}
  export VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
  export XDG_CACHE_HOME=/tmp/xdg-cache XDG_CONFIG_HOME=/tmp/xdg-config TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
  python3 -c "import torch; print(\"torch\", torch.__version__)" > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/runtime-versions.txt" 2>&1
  python3 -c "import vllm; print(\"vllm\", vllm.__version__)" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/runtime-versions.txt" 2>&1
  pip freeze > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/pip-freeze.txt" 2> /dev/null || true
  vllm serve /iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/models/Qwen--Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8-plain --host 127.0.0.1 --port 8100 --tensor-parallel-size 4 --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --compilation-config.pass_config.fuse_allreduce_rms false  > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/server.log" 2>&1 &
  SERVER=$!
  start=$(date +%s); ready=0
  while [ $(( $(date +%s) - start )) -lt 3600 ]; do
    if ! kill -0 ${SERVER} 2> /dev/null; then echo "server exited before ready" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"; exit 3; fi
    if curl -sf http://127.0.0.1:8100/v1/models > /dev/null; then
      if curl -sf http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json"           -d "{\"model\": \"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8-plain\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}], \"max_tokens\": 8}"           > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/ready-probe.json"; then ready=1; break; fi
    fi
    sleep 15
  done
  [ ${ready} = 1 ] || { echo "not ready after 3600 s" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"; kill ${SERVER}; exit 4; }
  echo "ready_after_s=$(( $(date +%s) - start ))" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"
  failed=0
  for test in probe\|1\ 8\ 32\ 128\|--requests\ 32\ --requests-per-slot\ 4 arena-hard-clean\|1\ 8\ 32\ 128\|--requests\ 32\ --requests-per-slot\ 4 ; do
    IFS="|" read -r workload levels extra <<< "${test}"
    file="/users/faruk_zahiragic/apertus-spec-decoding/workloads/8b/arena-hard.jsonl"; [ "${workload}" = probe ] && file="/users/faruk_zahiragic/apertus-spec-decoding/workloads/8b/probe-speculator-benchmarks.jsonl"
    python3 -m apertus_bench loadtest --base-url http://127.0.0.1:8100 --model "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8-plain"       --workloads "${file}" --workload "${workload}" --variant "plain" --method none        --concurrencies ${levels} ${extra} --metrics-interval 1       --metadata deployment_id=qwen3-235b-fp8-plain-20261122T173201Z --metadata target_model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --metadata target_revision=e156cb4efae43fbee1a1ab073f946a1377e6b969 --metadata target_tensor_parallel_size=4 --metadata expert_parallel=0 --metadata image=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh --metadata speculative_arm=plain  --metadata slurm_job=${SLURM_JOB_ID} --metadata slurm_nodes=${SLURM_JOB_NODELIST}       --output "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/cells/${workload}" > "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/bench-${workload}.log" 2>&1       && echo "${workload}: ok" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"       || { echo "${workload}: failed" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"; failed=1; }
    kill -0 ${SERVER} 2> /dev/null || { echo "server died during ${workload}" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/status.txt"; failed=1; break; }
  done
  kill ${SERVER}; wait ${SERVER} 2> /dev/null
  exit ${failed}
'
status=$?
kill ${SAMPLER} 2> /dev/null || true
wait ${IDENT} || true
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}" >> "/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/repo/results/qwen3-235b/sweep-20261122T173201Z-plain/slurm.txt"
exit ${status}
