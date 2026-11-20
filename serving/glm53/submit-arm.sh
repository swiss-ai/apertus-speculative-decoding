#!/usr/bin/env bash
# Submit one GLM-5.3-Flash serving arm on one GH200 node (TP4, experts included)
# and load-test it with the repo's bench client (src/apertus_bench loadtest).
#   serving/glm53/submit-arm.sh ARM            # ARM: plain mtp-k1 mtp-k3 dflash-k3 dflash-k7
# Settings (environment):
#   GLM53_RUN       run directory (default results/glm53-flash/<stamp>-<arm>; one per deployment)
#   GLM53_SMOKE=1   load the model and serve a few short requests only
#   GLM53_PARTITION (debug for smoke, normal otherwise), GLM53_TIME, GLM53_DEPENDENCY
# Plan and decision rule: docs/glm53-flash-specdec.md. Only submits; check squeue.
set -euo pipefail

ARM="${1:?usage: submit-arm.sh ARM}"
SERVING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SERVING_DIR}/../.." && pwd)"
TARGET="/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.3-Flash"
DRAFTER="/capstor/store/cscs/swissai/infra01/hf_models/models/incoai/GLM-5.3-Flash-DFlash2"
ENV_SOURCE="${REPO_ROOT}/experiments/glm53-flash/env.toml"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${ENV_SOURCE}")"
WORKLOADS="${GLM53_WORKLOADS:-${HOME}/apertus-spec-decoding/workloads/8b}"
SMOKE="${GLM53_SMOKE:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
KIND="sweep"; [ "${SMOKE}" = "1" ] && KIND="smoke"
OUT="${GLM53_RUN:-${REPO_ROOT}/results/glm53-flash/${KIND}-${STAMP}-${ARM}}"
if [ "${SMOKE}" = "1" ]; then
  PARTITION="${GLM53_PARTITION:-debug}"; TIME_LIMIT="${GLM53_TIME:-01:30:00}"
else
  PARTITION="${GLM53_PARTITION:-normal}"; TIME_LIMIT="${GLM53_TIME:-04:00:00}"
fi
ACCOUNT="${GLM53_ACCOUNT:-infra01}"

case "${ARM}" in
  plain)     METHOD=none;   K="";  SPEC="" ;;
  mtp-k1)    METHOD=mtp;    K=1;   SPEC='{"method": "mtp", "num_speculative_tokens": 1}' ;;
  mtp-k3)    METHOD=mtp;    K=3;   SPEC='{"method": "mtp", "num_speculative_tokens": 3}' ;;
  dflash-k3) METHOD=dflash; K=3;   SPEC="{\"method\": \"dflash\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": 3}" ;;
  dflash-k7) METHOD=dflash; K=7;   SPEC="{\"method\": \"dflash\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": 7}" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

SERVED="zai-org/GLM-5.3-Flash-${ARM}"
PORT=8100
SERVE=(
  vllm serve "${TARGET}"
  --served-model-name "${SERVED}"
  --host 127.0.0.1 --port "${PORT}"
  # No --enable-expert-parallel: vLLM 0.30 copies it into the dense DFlash2
  # drafter's parallel config and rejects it (no experts), so every arm uses
  # plain TP4 for the experts as well, keeping the layouts identical.
  --tensor-parallel-size 4
  --language-model-only
  --max-model-len 16384
  --max-num-seqs 128 --max-num-batched-tokens 16384
  --gpu-memory-utilization "${GLM53_GPU_UTIL:-0.94}"
  --no-enable-prefix-caching
)
[ -n "${SPEC}" ] && SERVE+=(--speculative-config "${SPEC}")

if [ "${SMOKE}" = "1" ]; then
  # workload|concurrencies|extra client args
  TESTS=("probe|1 8|--requests 8 --requests-per-slot 1 --max-tokens 128")
else
  TESTS=(
    "probe|1 8 32 128|--requests 32 --requests-per-slot 4"
    "arena-hard-clean|1 8 32 128|--requests 32 --requests-per-slot 4"
  )
fi
BENCH_METHOD=(--method "${METHOD}")
[ -n "${K}" ] && BENCH_METHOD+=(--num-speculative-tokens "${K}")

mkdir -p "${OUT}"
cp "${ENV_SOURCE}" "${OUT}/environment.toml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/git-revision.txt"
git -C "${REPO_ROOT}" diff HEAD > "${OUT}/checkout.patch"
printf '%q ' "${SERVE[@]}" > "${OUT}/launch-command.txt"; echo >> "${OUT}/launch-command.txt"
{
  printf '%q ' "$0" "$@"; echo
  env | grep -E '^GLM53_' | sort || true
} > "${OUT}/submit-command.txt"
printf '%s\n' "${TESTS[@]}" > "${OUT}/loadtests.txt"
DEPLOYMENT_ID="glm53-flash-${ARM}-${STAMP}"

# Quote the parts that go into the job script.
SERVE_Q="$(printf '%q ' "${SERVE[@]}")"
BENCH_METHOD_Q="$(printf '%q ' "${BENCH_METHOD[@]}")"
TESTS_Q="$(printf '%q ' "${TESTS[@]}")"

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=glm53-${ARM}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -uo pipefail
echo "job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} started=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/slurm.txt"
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000 > "${OUT}/gpu-memory.csv" 2> /dev/null &
SAMPLER=\$!
# Identities: image digest, drafter weights, target config/index and shard sizes
# (hashing the 330 GB of target shards would take longer than the arm).
(
  sha256sum "${IMAGE}" > "${OUT}/image.sha256"
  sha256sum "${TARGET}/config.json" "${TARGET}/model.safetensors.index.json" "${TARGET}/chat_template.jinja" > "${OUT}/target-identity.sha256"
  ls -l "${TARGET}"/*.safetensors > "${OUT}/target-shards.txt"
  if [ "${METHOD}" = "dflash" ]; then
    sha256sum "${DRAFTER}/config.json" "${DRAFTER}/model.safetensors" > "${OUT}/drafter.sha256"
  fi
) &
IDENT=\$!
srun --overlap --nodes=1 --ntasks=1 --environment="${OUT}/environment.toml" bash -c '
  set -uo pipefail
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
  # HOME is over quota: keep every cache, config and usage-stats write node-local.
  export VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
  export XDG_CACHE_HOME=/tmp/xdg-cache XDG_CONFIG_HOME=/tmp/xdg-config TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)" > "${OUT}/runtime-versions.txt" 2>&1
  pip freeze > "${OUT}/pip-freeze.txt" 2> /dev/null || true
  ${SERVE_Q} > "${OUT}/server.log" 2>&1 &
  SERVER=\$!
  start=\$(date +%s); ready=0
  while [ \$(( \$(date +%s) - start )) -lt 3600 ]; do
    if ! kill -0 \${SERVER} 2> /dev/null; then echo "server exited before ready" >> "${OUT}/status.txt"; exit 3; fi
    if curl -sf http://127.0.0.1:${PORT}/v1/models > /dev/null; then
      if curl -sf http://127.0.0.1:${PORT}/v1/chat/completions -H "Content-Type: application/json" \
          -d "{\"model\": \"${SERVED}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}], \"max_tokens\": 8}" \
          > "${OUT}/ready-probe.json"; then ready=1; break; fi
    fi
    sleep 15
  done
  [ \${ready} = 1 ] || { echo "not ready after 3600 s" >> "${OUT}/status.txt"; kill \${SERVER}; exit 4; }
  echo "ready_after_s=\$(( \$(date +%s) - start ))" >> "${OUT}/status.txt"
  failed=0
  for test in ${TESTS_Q}; do
    IFS="|" read -r workload levels extra <<< "\${test}"
    file="${WORKLOADS}/arena-hard.jsonl"; [ "\${workload}" = probe ] && file="${WORKLOADS}/probe-speculator-benchmarks.jsonl"
    python3 -m apertus_bench loadtest --base-url http://127.0.0.1:${PORT} --model "${SERVED}" \
      --workloads "\${file}" --workload "\${workload}" --variant "${ARM}" ${BENCH_METHOD_Q} \
      --concurrencies \${levels} \${extra} --metrics-interval 1 \
      --metadata deployment_id=${DEPLOYMENT_ID} --metadata target_model=zai-org/GLM-5.3-Flash \
      --metadata target_tensor_parallel_size=4 --metadata expert_parallel=0 \
      --metadata image=${IMAGE} --metadata speculative_arm=${ARM} \
      --metadata slurm_job=\${SLURM_JOB_ID} \
      --output "${OUT}/cells/\${workload}" > "${OUT}/bench-\${workload}.log" 2>&1 \
      && echo "\${workload}: ok" >> "${OUT}/status.txt" \
      || { echo "\${workload}: failed" >> "${OUT}/status.txt"; failed=1; }
  done
  kill \${SERVER}; wait \${SERVER} 2> /dev/null
  exit \${failed}
'
status=\$?
kill \${SAMPLER} 2> /dev/null || true
wait \${IDENT} || true
echo "finished=\$(date -u +%Y-%m-%dT%H:%M:%SZ) status=\${status}" >> "${OUT}/slurm.txt"
exit \${status}
EOF

SUBMIT=(--parsable)
[ -n "${GLM53_DEPENDENCY:-}" ] && SUBMIT+=(--dependency "${GLM53_DEPENDENCY}")
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then echo "job script: ${OUT}/job.sh"; exit 0; fi
JOB="$(sbatch "${SUBMIT[@]}" "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} arm=${ARM} output=${OUT}"
