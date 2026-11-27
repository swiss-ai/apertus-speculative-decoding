#!/bin/bash
# One deployment on one GPU: serve, then one loadtest cell per slice.
#   arm.sh ARM GPU   (all other settings come from the environment of job.sh)
set -uo pipefail
ARM="$1"; GPU="$2"; PORT=$((8100 + GPU))
case "${ARM}" in
  k3-greedy) K=3; TEMP=0 ;;
  k7-greedy) K=7; TEMP=0 ;;
  k3-t1) K=3; TEMP=1 ;;
  k7-t1) K=7; TEMP=1 ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac
D="${OUT}/${ARM}"; mkdir -p "${D}"
SERVED="qwen3-4b-dspark-${ARM}"
SPEC="{\"method\": \"dspark\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${K}}"
SERVE=(vllm serve "${TARGET}" --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}"
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 40960 --max-num-seqs 8
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching --generation-config vllm --seed 0
  --speculative-config "${SPEC}")
printf '%q ' "${SERVE[@]}" > "${D}/launch-command.txt"; echo >> "${D}/launch-command.txt"
CUDA_VISIBLE_DEVICES=${GPU} "${SERVE[@]}" > "${D}/server.log" 2>&1 &
SERVER=$!
start=$(date +%s); ready=0
while [ $(( $(date +%s) - start )) -lt 1800 ]; do
  kill -0 ${SERVER} 2> /dev/null || { echo "server exited before ready" >> "${D}/status.txt"; exit 3; }
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then ready=1; break; fi
  sleep 10
done
[ ${ready} = 1 ] || { echo "not ready after 1800 s" >> "${D}/status.txt"; kill ${SERVER}; exit 4; }
echo "ready_after_s=$(( $(date +%s) - start ))" >> "${D}/status.txt"
failed=0
for slice in ${SLICES}; do
  python3 -m apertus_bench loadtest --base-url "http://127.0.0.1:${PORT}" --model "${SERVED}" \
    --workloads "${CORPUS}" --workload "${slice}" --variant "${ARM}" \
    --method dspark --num-speculative-tokens "${K}" \
    --concurrencies 1 --requests "${REQUESTS}" --requests-per-slot 1 --warmup-requests 1 \
    --max-tokens 512 --temperature "${TEMP}" --top-p 1 --seed 1 \
    --metrics-interval 5 \
    --metadata "deployment_id=${SERVED}-${SLURM_JOB_ID}" --metadata "checkpoint_sha256=${DRAFTER_SHA}" \
    --metadata "target_model=${TARGET_ID}" --metadata "target_revision=${TARGET_REV}" \
    --metadata target_tensor_parallel_size=1 --metadata "drafter_model=${DRAFTER_ID}" \
    --metadata "drafter_revision=${DRAFTER_REV}" --metadata "image=${IMAGE}" \
    --metadata "corpus_sha256=${CORPUS_SHA}" --metadata "source_revision=${SOURCE_REVISION}" \
    --metadata "slurm_job=${SLURM_JOB_ID}" --metadata "slurm_nodes=${SLURM_JOB_NODELIST}" \
    --metadata "gpu=${GPU}" --metadata "temperature=${TEMP}" \
    --output "${D}/cells/${slice}" > "${D}/bench-${slice}.log" 2>&1 \
    && echo "${slice}: ok" >> "${D}/status.txt" \
    || { echo "${slice}: failed" >> "${D}/status.txt"; failed=1; }
  kill -0 ${SERVER} 2> /dev/null || { echo "server died during ${slice}" >> "${D}/status.txt"; failed=1; break; }
done
kill ${SERVER}; wait ${SERVER} 2> /dev/null
exit ${failed}
