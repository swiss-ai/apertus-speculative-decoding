#!/usr/bin/env bash
# Submit one Qwen3-235B-A22B-Instruct-2507-FP8 serving arm on one GH200 node (TP4)
# and load-test it with the repo's bench client (src/apertus_bench loadtest).
# Ported from serving/glm53/submit-arm.sh (branch experiment/glm53-flash-specdec).
#   serving/qwen3/submit-arm.sh ARM            # ARM: plain eagle3-k3 eagle3-k5
# Settings (environment):
#   QWEN3_RUN        run directory (default results/qwen3-235b/<kind>-<stamp>-<arm>; one per deployment)
#   QWEN3_SMOKE=1    load the model and serve a few short requests only
#   QWEN3_PARTITION  (debug for smoke, normal otherwise), QWEN3_TIME, QWEN3_DEPENDENCY,
#   QWEN3_ACCOUNT    (default d_research_serving), QWEN3_GPU_UTIL (default 0.90)
#   QWEN3_MODELS     directory with the pinned downloads and their SHA256SUMS
# Plan and decision rule: docs/qwen3-235b-specdec.md. Only submits; check squeue.
set -euo pipefail

ARM="${1:?usage: submit-arm.sh ARM}"
SERVING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SERVING_DIR}/../.." && pwd)"
MODELS="${QWEN3_MODELS:-/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/models}"
TARGET="${MODELS}/Qwen--Qwen3-235B-A22B-Instruct-2507-FP8"
TARGET_ID="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
TARGET_REV="e156cb4efae43fbee1a1ab073f946a1377e6b969"
DRAFTER="${MODELS}/RedHatAI--Qwen3-235B-A22B-Instruct-2507-speculator.eagle3"
DRAFTER_ID="RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3"
DRAFTER_REV="87cd7aa934131942f5d7277617d7d6020f2df7f1"
ENV_SOURCE="${REPO_ROOT}/experiments/qwen3-235b/env.toml"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${ENV_SOURCE}")"
WORKLOADS="${QWEN3_WORKLOADS:-${HOME}/apertus-spec-decoding/workloads/8b}"
SMOKE="${QWEN3_SMOKE:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
KIND="sweep"; [ "${SMOKE}" = "1" ] && KIND="smoke"
OUT="${QWEN3_RUN:-${REPO_ROOT}/results/qwen3-235b/${KIND}-${STAMP}-${ARM}}"
if [ "${SMOKE}" = "1" ]; then
  PARTITION="${QWEN3_PARTITION:-debug}"; TIME_LIMIT="${QWEN3_TIME:-01:00:00}"
else
  PARTITION="${QWEN3_PARTITION:-normal}"; TIME_LIMIT="${QWEN3_TIME:-06:00:00}"
fi
ACCOUNT="${QWEN3_ACCOUNT:-d_research_serving}"

case "${ARM}" in
  plain)     METHOD=none;   K="" ;;
  eagle3-k3) METHOD=eagle3; K=3 ;;
  eagle3-k5) METHOD=eagle3; K=5 ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac
SPEC=""
[ "${METHOD}" = eagle3 ] && SPEC="{\"method\": \"eagle3\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${K}}"
if [ ! -f "${TARGET}/SHA256SUMS" ] || [ ! -f "${DRAFTER}/SHA256SUMS" ]; then
  echo "missing SHA256SUMS under ${MODELS}" >&2; exit 2
fi
DRAFTER_SHA="$(awk '$2 == "./model.safetensors" {print $1}' "${DRAFTER}/SHA256SUMS")"

SERVED="${TARGET_ID}-${ARM}"
PORT=8100
SERVE=(
  vllm serve "${TARGET}"
  --served-model-name "${SERVED}"
  --host 127.0.0.1 --port "${PORT}"
  # Same EP-free TP4 layout for every arm, as in the GLM-5.3-Flash sweep.
  --tensor-parallel-size 4
  --max-model-len 16384
  --max-num-seqs 128 --max-num-batched-tokens 16384
  --gpu-memory-utilization "${QWEN3_GPU_UTIL:-0.90}"
  --no-enable-prefix-caching
  # TP>1 on GH200 fails in the fused allreduce+RMSNorm pass (70B serving notes).
  --compilation-config.pass_config.fuse_allreduce_rms false
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
[ -n "${K}" ] && BENCH_METHOD+=(--num-speculative-tokens "${K}" --algorithm eagle3)
DEPLOYMENT_ID="qwen3-235b-fp8-${ARM}-${STAMP}"
META=(
  --metadata "deployment_id=${DEPLOYMENT_ID}" --metadata "target_model=${TARGET_ID}"
  --metadata "target_revision=${TARGET_REV}" --metadata target_tensor_parallel_size=4
  --metadata expert_parallel=0 --metadata "image=${IMAGE}" --metadata "speculative_arm=${ARM}"
)
if [ "${METHOD}" = eagle3 ]; then
  META+=(
    --metadata "checkpoint_sha256=${DRAFTER_SHA}" --metadata "drafter_model=${DRAFTER_ID}"
    --metadata "drafter_revision=${DRAFTER_REV}"
  )
fi

mkdir -p "${OUT}"
cp "${ENV_SOURCE}" "${OUT}/environment.toml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/git-revision.txt"
git -C "${REPO_ROOT}" diff HEAD > "${OUT}/checkout.patch"
printf '%q ' "${SERVE[@]}" > "${OUT}/launch-command.txt"; echo >> "${OUT}/launch-command.txt"
{
  printf '%q ' "$0" "$@"; echo
  env | grep -E '^QWEN3_' | sort || true
} > "${OUT}/submit-command.txt"
printf '%s\n' "${TESTS[@]}" > "${OUT}/loadtests.txt"
# Identities: pinned HF revisions, full-file SHA-256 manifests of both checkouts
# (computed once by the download job) and the workload corpus digests.
{
  echo "target ${TARGET_ID} ${TARGET_REV} ${TARGET}"
  echo "drafter ${DRAFTER_ID} ${DRAFTER_REV} ${DRAFTER}"
} > "${OUT}/model-identity.txt"
cp "${TARGET}/SHA256SUMS" "${OUT}/target.sha256"
cp "${DRAFTER}/SHA256SUMS" "${OUT}/drafter.sha256"
sha256sum "${WORKLOADS}/probe-speculator-benchmarks.jsonl" "${WORKLOADS}/arena-hard.jsonl" > "${OUT}/workloads.sha256"

SERVE_Q="$(printf '%q ' "${SERVE[@]}")"
BENCH_METHOD_Q="$(printf '%q ' "${BENCH_METHOD[@]}")"
META_Q="$(printf '%q ' "${META[@]}")"
TESTS_Q="$(printf '%q ' "${TESTS[@]}")"

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=qwen3-${ARM}
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
( sha256sum "${IMAGE}" > "${OUT}/image.sha256" ) &
IDENT=\$!
srun --overlap --nodes=1 --ntasks=1 --environment="${OUT}/environment.toml" bash -c '
  set -uo pipefail
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
  # HOME is over quota: keep every cache, config and usage-stats write node-local.
  export HOME=/tmp/qwen3-home; mkdir -p \${HOME}
  export VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
  export XDG_CACHE_HOME=/tmp/xdg-cache XDG_CONFIG_HOME=/tmp/xdg-config TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
  python3 -c "import torch; print(\"torch\", torch.__version__)" > "${OUT}/runtime-versions.txt" 2>&1
  python3 -c "import vllm; print(\"vllm\", vllm.__version__)" >> "${OUT}/runtime-versions.txt" 2>&1
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
      ${META_Q} --metadata slurm_job=\${SLURM_JOB_ID} --metadata slurm_nodes=\${SLURM_JOB_NODELIST} \
      --output "${OUT}/cells/\${workload}" > "${OUT}/bench-\${workload}.log" 2>&1 \
      && echo "\${workload}: ok" >> "${OUT}/status.txt" \
      || { echo "\${workload}: failed" >> "${OUT}/status.txt"; failed=1; }
    kill -0 \${SERVER} 2> /dev/null || { echo "server died during \${workload}" >> "${OUT}/status.txt"; failed=1; break; }
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
[ -n "${QWEN3_DEPENDENCY:-}" ] && SUBMIT+=(--dependency "${QWEN3_DEPENDENCY}")
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then echo "job script: ${OUT}/job.sh"; exit 0; fi
JOB="$(sbatch "${SUBMIT[@]}" "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} arm=${ARM} output=${OUT}"
