#!/bin/bash
# One deployment of the Qwen3-8B shape gate (written by submit-gate-qwen3.sh):
#   gate-arm.sh ARM_DIR
# ARM_DIR/arm.env defines GPU, PORT, SERVED, SERVE (array), BENCH (array), CORPUS, SLICES,
# REQUESTS, REPO_ROOT. Serves, waits for a chat completion, runs every slice at C=1, stops.
set -uo pipefail
dir="$1"
# shellcheck disable=SC1091
source "${dir}/arm.env"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HOME="/tmp/gate-home-${GPU}"; mkdir -p "${HOME}"
c="/tmp/gate-${GPU}"
export VLLM_CACHE_ROOT="${c}/vllm" TRITON_CACHE_DIR="${c}/triton" FLASHINFER_WORKSPACE_BASE="${c}/flashinfer"
export XDG_CACHE_HOME="${c}/xdg" XDG_CONFIG_HOME="${c}/xdg-config" TORCHINDUCTOR_CACHE_DIR="${c}/inductor"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
echo "gpu=${GPU} job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date -u +%FT%TZ)" > "${dir}/status.txt"
"${SERVE[@]}" > "${dir}/server.log" 2>&1 &
server=$!
start=$(date +%s); ready=0
while [ $(( $(date +%s) - start )) -lt 2400 ]; do
  kill -0 "${server}" 2> /dev/null || { echo "server exited before ready" >> "${dir}/status.txt"; exit 3; }
  if curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\": \"${SERVED}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}], \"max_tokens\": 8}" \
      > "${dir}/ready-probe.json"; then ready=1; break; fi
  sleep 10
done
[ "${ready}" = 1 ] || { echo "not ready" >> "${dir}/status.txt"; kill "${server}"; exit 4; }
echo "ready_after_s=$(( $(date +%s) - start ))" >> "${dir}/status.txt"
failed=0
for slice in ${SLICES}; do
  python3 -m apertus_bench loadtest --base-url "http://127.0.0.1:${PORT}" --model "${SERVED}" \
    --workloads "${CORPUS}" --workload "${slice}" --concurrencies 1 --requests "${REQUESTS}" \
    --requests-per-slot 1 --max-tokens 512 "${BENCH[@]}" \
    --metadata "slurm_job=${SLURM_JOB_ID:-local}" --metadata "slurm_node=$(hostname)" \
    --output "${dir}/cells/${slice}" > "${dir}/bench-${slice}.log" 2>&1 \
    && echo "${slice}: ok" >> "${dir}/status.txt" \
    || { echo "${slice}: failed" >> "${dir}/status.txt"; failed=1; }
  kill -0 "${server}" 2> /dev/null || { echo "server died during ${slice}" >> "${dir}/status.txt"; failed=1; break; }
done
kill "${server}"; wait "${server}" 2> /dev/null
echo "end=$(date -u +%FT%TZ) failed=${failed}" >> "${dir}/status.txt"
exit "${failed}"
