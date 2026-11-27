#!/usr/bin/env bash
# Validation gate of the Qwen3-4B workload-shape replay: serve Qwen3-4B with the
# Red Hat DSpark head in vLLM and measure acceptance per slice on the 896 prompts.
#   experiments/workload-shape/submit-gate-qwen3.sh OUT_DIR CORPUS
# CORPUS: the shape run's inputs/corpus-measured.jsonl (restricted; cluster only).
# One node, four independent deployments, one per GPU (TP=1, C=1, prefix caching
# off, 512-token cap, thinking off from each row's chat_template_kwargs):
#   GPU0 DSpark k=3 greedy   GPU1 DSpark k=7 greedy (the head's default)
#   GPU2 DSpark k=3 T=1      GPU3 DSpark k=7 T=1  (sampled-drafting check)
# Each arm runs the 28 slices one after another with apertus_bench loadtest
# (32 measured requests per slice, one warm-up), so the vLLM spec-decode counter
# deltas of a cell are that slice's acceptance. --generation-config vllm keeps the
# model's sampling defaults (T=0.6, top_k=20) out of the T=1 arms.
# GATE_ACCOUNT (d_research_serving), GATE_PARTITION (normal), GATE_TIME (03:00:00),
# GATE_MODELS (pinned downloads with SHA256SUMS), GATE_ARMS (default all four),
# GATE_SLICES (default all 28), GATE_REQUESTS (32). VALIDATE_ONLY=1 writes job.sh only.
set -euo pipefail

OUT="${1:?usage: submit-gate-qwen3.sh OUT_DIR CORPUS}"
CORPUS="${2:?usage: submit-gate-qwen3.sh OUT_DIR CORPUS}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS="${GATE_MODELS:-/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models}"
TARGET="${MODELS}/Qwen--Qwen3-4B"
TARGET_ID="Qwen/Qwen3-4B"
TARGET_REV="1cfa9a7208912126459214e8b04321603b3df60c"
DRAFTER="${MODELS}/RedHatAI--Qwen3-4B-speculator.dspark"
DRAFTER_ID="RedHatAI/Qwen3-4B-speculator.dspark"
DRAFTER_REV="bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23"
ENV_SOURCE="${REPO_ROOT}/experiments/workload-shape/env-qwen3.toml"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${ENV_SOURCE}")"
SUITE="${REPO_ROOT}/results/8b/workload-shape/inputs/suite.yaml"
ARMS="${GATE_ARMS:-k3-greedy k7-greedy k3-t1 k7-t1}"
SLICES="${GATE_SLICES:-$(python3 -c "import sys, yaml; print(' '.join(s['label'] for s in yaml.safe_load(open(sys.argv[1]))['slices']))" "${SUITE}")}"
REQUESTS="${GATE_REQUESTS:-32}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "${REPO_ROOT}" in
  /iopsstor/* | /capstor/*) ;;
  *) echo "run from a checkout on /iopsstor or /capstor; the container lacks ${REPO_ROOT}" >&2; exit 2 ;;
esac
for f in "${TARGET}/SHA256SUMS" "${DRAFTER}/SHA256SUMS" "${CORPUS}"; do
  test -f "${f}" || { echo "missing ${f}" >&2; exit 2; }
done
DRAFTER_SHA="$(awk '$2 == "./model.safetensors" {print $1}' "${DRAFTER}/SHA256SUMS")"
mkdir -p "${OUT}"
cp "${ENV_SOURCE}" "${OUT}/environment.toml"
{
  echo "source_revision=$(cat "${REPO_ROOT}/SOURCE_REVISION" 2> /dev/null || git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "repo_root=${REPO_ROOT}"
  echo "image=${IMAGE}"
  echo "arms=${ARMS}"
  echo "slices=${SLICES}"
  echo "requests=${REQUESTS}"
  echo "corpus=${CORPUS}"
  echo "corpus_sha256=$(sha256sum "${CORPUS}" | cut -d' ' -f1)"
  echo "target ${TARGET_ID} ${TARGET_REV} ${TARGET}"
  echo "drafter ${DRAFTER_ID} ${DRAFTER_REV} ${DRAFTER} model.safetensors=${DRAFTER_SHA}"
} > "${OUT}/provenance.txt"
cp "${TARGET}/SHA256SUMS" "${OUT}/target.sha256"
cp "${DRAFTER}/SHA256SUMS" "${OUT}/drafter.sha256"
{ printf '%q ' "$0" "$@"; echo; env | grep -E '^GATE_' | sort || true; } > "${OUT}/submit-command.txt"

ARM_SCRIPT="${OUT}/arm.sh"
cat > "${ARM_SCRIPT}" <<'ARM'
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
ARM
chmod +x "${ARM_SCRIPT}"

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=shape-gate-q4b
#SBATCH --account=${GATE_ACCOUNT:-d_research_serving}
#SBATCH --partition=${GATE_PARTITION:-normal}
#SBATCH --time=${GATE_TIME:-03:00:00}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -uo pipefail
echo "job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} started=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/slurm.txt"
( sha256sum "${IMAGE}" > "${OUT}/image.sha256" ) &
IDENT=\$!
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 2000 > "${OUT}/gpu.csv" 2> /dev/null &
SAMPLER=\$!
srun --overlap --nodes=1 --ntasks=1 --environment="${OUT}/environment.toml" bash -c '
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
  export HOME=/tmp/gate-home; mkdir -p \${HOME}
  export VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
  export XDG_CACHE_HOME=/tmp/xdg-cache XDG_CONFIG_HOME=/tmp/xdg-config TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
  export OUT="${OUT}" TARGET="${TARGET}" DRAFTER="${DRAFTER}" CORPUS="${CORPUS}" SLICES="${SLICES}"
  export REQUESTS="${REQUESTS}" IMAGE="${IMAGE}" DRAFTER_SHA="${DRAFTER_SHA}"
  export TARGET_ID="${TARGET_ID}" TARGET_REV="${TARGET_REV}" DRAFTER_ID="${DRAFTER_ID}" DRAFTER_REV="${DRAFTER_REV}"
  export CORPUS_SHA="\$(sha256sum "${CORPUS}" | cut -d" " -f1)"
  export SOURCE_REVISION="\$(cat "${REPO_ROOT}/SOURCE_REVISION" 2> /dev/null || echo unknown)"
  cd "${REPO_ROOT}"
  python3 -c "import torch, vllm, transformers; print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)" > "${OUT}/runtime-versions.txt" 2>&1
  pip freeze > "${OUT}/pip-freeze.txt" 2> /dev/null || true
  gpu=0; pids=()
  for arm in ${ARMS}; do
    bash "${ARM_SCRIPT}" "\${arm}" "\${gpu}" & pids+=(\$!); gpu=\$((gpu + 1))
  done
  failed=0
  for pid in "\${pids[@]}"; do wait "\${pid}" || failed=1; done
  exit \${failed}
'
status=\$?
kill \${SAMPLER} 2> /dev/null || true
wait \${IDENT} || true
echo "finished=\$(date -u +%Y-%m-%dT%H:%M:%SZ) status=\${status}" >> "${OUT}/slurm.txt"
exit \${status}
EOF
bash -n "${OUT}/job.sh"
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then echo "job script: ${OUT}/job.sh"; exit 0; fi
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
