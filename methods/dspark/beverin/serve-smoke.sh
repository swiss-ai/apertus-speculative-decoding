#!/usr/bin/env bash
# Inside the container (serve-smoke.sbatch).
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
TP="${TP:-2}"; GPUS="${GPUS:-0,1}"; PORT=8001
LOG="${WORK}/logs/serve-${SLURM_JOB_ID}-tp${TP}.log"
python3 -c "import torch, vllm, transformers; print('torch', torch.__version__, 'hip', torch.version.hip, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES="${GPUS}" setsid vllm serve "${TARGET_MODEL}" --served-model-name apertus70b \
  --tensor-parallel-size "${TP}" --max-model-len 8192 --gpu-memory-utilization "${MEM:-0.85}" \
  ${EAGER:+--enforce-eager} --port "${PORT}" ${SERVE_ARGS:-} > "${LOG}" 2>&1 &
server=$!
trap 'kill -TERM -- -${server} 2> /dev/null || true; sleep 5; kill -KILL -- -${server} 2> /dev/null || true' EXIT
if ! wait_health "${PORT}" "${server}" 3000; then
  echo "SERVER_FAILED"; grep -nE "Error|error:|Traceback" "${LOG}" | tail -20; tail -40 "${LOG}"; exit 1
fi
echo "server up after $(( $(date +%s) - t0 )) s"
grep -E "Loading weights took|Model loading took|KV cache|GPU KV cache size|Maximum concurrency|init engine|graph" "${LOG}" | tail -8
python3 "${REPO_ROOT}/methods/dspark/beverin/generate_smoke.py" --port "${PORT}"
echo "serve smoke done"
