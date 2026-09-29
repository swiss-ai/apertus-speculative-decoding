#!/usr/bin/env bash
# Load test of one plain-target deployment, with server load and GPU memory sampled.
#   STAGE=8b ./serving/loadtest.sh
# Steps: launch the plain target (baseline.sh), wait for it, then inside the
# replica's own allocation (srun --overlap, so nothing heavy runs here):
#   - an nvidia-smi sampler (memory used/total, utilization, power, every 1 s)
#   - apertus-bench loadtest over LOADTEST_CONCURRENCIES on LOADTEST_WORKLOAD,
#     sampling /metrics every second (KV-cache usage, queue, preemptions)
# then copy the startup memory breakdown from the engine log and cancel.
# This script only waits; run it from a login node.
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SERVING_DIR}/.." && pwd)"
export STAGE="${STAGE:-8b}"
export MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-${HOME}/model-launch-apertus}"
export PATH="${HOME}/venvs/sml-apertus/bin:${PATH}"
export SML_PARTITION="${SML_PARTITION:-debug}"
export SML_TIME="${SML_TIME:-01:30:00}"
WORKLOADS="${LOADTEST_WORKLOADS:-${REPO_ROOT}/workloads/${STAGE}/eagle-8b-test.jsonl}"
WORKLOAD="${LOADTEST_WORKLOAD:-summarization}"
CONCURRENCIES="${LOADTEST_CONCURRENCIES:-1 8 32 64 128 256}"
PER_SLOT="${LOADTEST_REQUESTS_PER_SLOT:-4}"
MIN_REQUESTS="${LOADTEST_MIN_REQUESTS:-32}"
MAX_TOKENS="${LOADTEST_MAX_TOKENS:-}"
ENVIRONMENT_TOML="${LOADTEST_ENVIRONMENT:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEPLOYMENT_ID="apertus15-${STAGE}-baseline-loadtest-${STAMP}"
OUT="${REPO_ROOT}/results/${STAGE}/loadtest/${DEPLOYMENT_ID}"
mkdir -p "${OUT}"
# The serving environment does not mount /users; compile caches go to scratch.
CACHE_ROOT="/iopsstor/scratch/cscs/${USER}/apertus-loadtest/vllm-cache/${DEPLOYMENT_ID}"
mkdir -p "${CACHE_ROOT}"
export EXTRA_ENV="VLLM_CACHE_ROOT=${CACHE_ROOT}"

RUN_SUFFIX="loadtest-${STAMP}" "${SERVING_DIR}/baseline.sh" --no-tui > "${OUT}/launch.log" 2>&1
JOB="$(grep -aoE 'Job submitted: [0-9]+' "${OUT}/launch.log" | tail -1 | awk '{print $NF}')"
[ -n "${JOB}" ] || { echo "no job id; see ${OUT}/launch.log" >&2; exit 1; }
echo "job=${JOB}" > "${OUT}/job.txt"
SAMPLER=""
trap '[ -n "${SAMPLER}" ] && kill "${SAMPLER}" 2>/dev/null; scancel "${JOB}" 2>/dev/null || true' EXIT

started="$(date +%s)"
INFO="$("${SERVING_DIR}/wait-replica.sh" "${JOB}")" || { echo "status=not_ready" > "${OUT}/status.txt"; exit 1; }
echo "${INFO}" > "${OUT}/replica.txt"
eval "${INFO}"
echo "ready_seconds=$(( $(date +%s) - started ))" >> "${OUT}/replica.txt"
BASE="http://${IP}:8080"
LOGS="${HOME}/.sml/logs/${JOB}"

# GPU memory over the whole test, sampled on the replica node itself.
srun --jobid="${JOB}" --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" \
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000 > "${OUT}/gpu-memory.csv" 2> "${OUT}/gpu-sampler.err" &
SAMPLER=$!

# The load generator runs in the serving container on the replica's node.
SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENVIRONMENT_TOML}")"
srun --jobid="${JOB}" --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}" \
  --environment="${SML_ENVIRONMENT}" \
  env PYTHONPATH="${REPO_ROOT}/src" PYTHONNOUSERSITE=1 \
  python3 -m apertus_bench loadtest --base-url "${BASE}" --model "${MODEL}" \
    --workloads "${WORKLOADS}" --workload "${WORKLOAD}" \
    --concurrencies ${CONCURRENCIES} --requests "${MIN_REQUESTS}" \
    --requests-per-slot "${PER_SLOT}" --metrics-interval 1 \
    ${MAX_TOKENS:+--max-tokens "${MAX_TOKENS}"} \
    --variant "apertus15-${STAGE}-baseline" --method none \
    --metadata "deployment_id=${DEPLOYMENT_ID}" --metadata "slurm_job_id=${JOB}" \
    --metadata "slurm_node_id=${NODE}" --metadata "load_generator=replica node (srun --overlap)" \
    --output "${OUT}/cells" > "${OUT}/loadtest.log" 2>&1 \
  || echo "loadtest failed" | tee -a "${OUT}/status.txt"

kill "${SAMPLER}" 2>/dev/null || true
SAMPLER=""
cp "${LOGS}/log.out" "${OUT}/sml-log.out" 2>/dev/null || true
# Static memory: weights, KV pool, CUDA graphs, as the engine reports them at start.
grep -aE 'non-default args|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|CUDA graph|Graph capturing|gpu_memory_utilization|memory profiling|torch.compile took' \
  "${LOGS}/replica_0.out" > "${OUT}/engine-excerpt.txt" 2>/dev/null || true
cp "${LOGS}/replica_0.out" "${OUT}/replica_0.out" 2>/dev/null || true
grep -q "failed" "${OUT}/status.txt" 2>/dev/null || echo "status=ok" >> "${OUT}/status.txt"
echo "${OUT}"
