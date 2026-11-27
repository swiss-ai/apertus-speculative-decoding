#!/usr/bin/env bash
# Validation gate of a workload-shape replay: serve the target with its drafters in vLLM
# (TP=1, one deployment per GPU lane, C=1) on the 28-slice corpus and record the served
# acceptance per slice with the repo's bench client (apertus_bench loadtest, 32 measured
# requests per slice, so a cell's spec-decode counter deltas are that slice's acceptance).
#   set -a; . experiments/workload-shape/<target>.env; set +a
#   experiments/workload-shape/submit-gate.sh OUT_DIR CORPUS
# CORPUS: the shape run's inputs/corpus-measured.jsonl (restricted; cluster only).
# Every request: 512-token cap, thinking off (the corpus rows' chat_template_kwargs),
# prefix caching off, --generation-config vllm (no model sampling defaults); greedy =
# temperature 0, T=1 = temperature 1, top_p 1, top_k -1; seed 1.
# Settings (the per-target .env files set them):
#   GATE_TARGET, GATE_TARGET_ID, GATE_TARGET_REV  target snapshot (with SHA256SUMS) and identity
#   GATE_HEADS      "name:path:hf_id:revision ..." (name = vLLM speculative method)
#   GATE_ARMS       "arm:head:k:greedy|t1:lane ..."; arms sharing a GPU lane run one
#                   after the other; one directory per arm under OUT_DIR
#   GATE_SERVED_PREFIX  served-model-name prefix; GATE_JOB_NAME (shape-gate)
#   GATE_MAX_MODEL_LEN (32768); GATE_SERVE_ARGS / GATE_BENCH_ARGS extra vllm serve /
#                   loadtest arguments
#   GATE_ACCOUNT (d_research_serving), GATE_PARTITION (normal), GATE_TIME (04:00:00),
#   GATE_SLICES (default all 28), GATE_REQUESTS (32), GATE_ONLY (subset of arm names).
# VALIDATE_ONLY=1 writes job.sh and the arm files without submitting.
set -euo pipefail

OUT="${1:?usage: submit-gate.sh OUT_DIR CORPUS}"
CORPUS="${2:?usage: submit-gate.sh OUT_DIR CORPUS}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="${GATE_TARGET:?source the per-target .env first}"
TARGET_ID="${GATE_TARGET_ID:?}"
TARGET_REV="${GATE_TARGET_REV:?}"
ENV_SOURCE="${REPO_ROOT}/experiments/workload-shape/env-qwen3.toml"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${ENV_SOURCE}")"
REQUESTS="${GATE_REQUESTS:-32}"
SUITE="${REPO_ROOT}/results/8b/workload-shape/inputs/suite.yaml"
SLICES="${GATE_SLICES:-}"
if [ -z "${SLICES}" ]; then
  SLICES="$(sed -n 's/^  - {label: \([^,]*\),.*/\1/p' "${SUITE}" | tr '\n' ' ')"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "${REPO_ROOT}" in
  /iopsstor/* | /capstor/*) ;;
  *) [ "${VALIDATE_ONLY:-0}" = 1 ] || {
       echo "run from a checkout on /iopsstor or /capstor; the container lacks ${REPO_ROOT}" >&2
       exit 2
     } ;;
esac
declare -A HEAD_PATH HEAD_ID HEAD_REV
for spec in ${GATE_HEADS:?}; do
  IFS=: read -r name path id rev <<< "${spec}"
  HEAD_PATH[${name}]="${path}"; HEAD_ID[${name}]="${id}"; HEAD_REV[${name}]="${rev}"
done
for f in "${TARGET}/SHA256SUMS" "${CORPUS}" "${HEAD_PATH[@]/%//SHA256SUMS}"; do
  test -s "${f}" || { echo "missing ${f}" >&2; exit 2; }
done
sha_of() { awk '$2 == "./model.safetensors" {print $1}' "$1/SHA256SUMS"; }
CORPUS_SHA="$(sha256sum "${CORPUS}" | cut -d' ' -f1)"
SOURCE_REVISION="$(cat "${REPO_ROOT}/SOURCE_REVISION" 2> /dev/null || git -C "${REPO_ROOT}" rev-parse HEAD)"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
test ! -e "${OUT}/job.sh" || { echo "${OUT} already holds a gate run; use a new directory" >&2; exit 2; }
cp "${ENV_SOURCE}" "${OUT}/environment.toml"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${OUT}/environment.toml"
{
  echo "source_revision ${SOURCE_REVISION}"
  echo "target ${TARGET_ID} ${TARGET_REV} ${TARGET}"
  for name in "${!HEAD_PATH[@]}"; do
    echo "${name} ${HEAD_ID[${name}]} ${HEAD_REV[${name}]} ${HEAD_PATH[${name}]} $(sha_of "${HEAD_PATH[${name}]}")"
  done
  echo "corpus ${CORPUS} ${CORPUS_SHA}"
  echo "image ${IMAGE}"
} > "${OUT}/identity.txt"
cp "${TARGET}/SHA256SUMS" "${OUT}/target.sha256"
for name in "${!HEAD_PATH[@]}"; do cp "${HEAD_PATH[${name}]}/SHA256SUMS" "${OUT}/${name}.sha256"; done
{ printf '%q ' "$0" "$@"; echo; env | grep -E '^GATE_' | sort || true; } > "${OUT}/submit-command.txt"

# One environment file per deployment; experiments/workload-shape/gate-arm.sh runs it.
arm_env() {
  local arm="$1" head="$2" k="$3" mode="$4" gpu="$5"
  local port=$((8100 + gpu)) drafter="${HEAD_PATH[${head}]:?unknown head ${head}}"
  local gen=(--temperature 0 --top-p 1 --seed 1)
  case "${mode}" in
    greedy) ;;
    t1) gen=(--temperature 1 --top-p 1 --top-k -1 --seed 1) ;;
    *) echo "unknown sampling ${mode} in arm ${arm}" >&2; exit 2 ;;
  esac
  local spec="{\"method\": \"${head}\", \"model\": \"${drafter}\", \"num_speculative_tokens\": ${k}}"
  local dir="${OUT}/${arm}" served="${GATE_SERVED_PREFIX:?}-${arm}"
  mkdir -p "${dir}"
  # shellcheck disable=SC2206
  local serve=(vllm serve "${TARGET}" --served-model-name "${served}" --host 127.0.0.1
    --port "${port}" --tensor-parallel-size 1 --dtype bfloat16
    --max-model-len "${GATE_MAX_MODEL_LEN:-32768}" --max-num-seqs 8
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching --generation-config vllm
    ${GATE_SERVE_ARGS:-} --speculative-config "${spec}")
  local bench=(--variant "${arm}" --method "${head}" --num-speculative-tokens "${k}")
  if [ "${head}" = eagle3 ]; then bench+=(--algorithm eagle3); fi
  # shellcheck disable=SC2206
  bench+=("${gen[@]}" ${GATE_BENCH_ARGS:-}
    --metadata "deployment_id=${served}-${STAMP}" --metadata "target_model=${TARGET_ID}"
    --metadata "target_revision=${TARGET_REV}" --metadata target_tensor_parallel_size=1
    --metadata "checkpoint_sha256=$(sha_of "${drafter}")"
    --metadata "drafter_model=${HEAD_ID[${head}]}" --metadata "drafter_revision=${HEAD_REV[${head}]}"
    --metadata "image=${IMAGE}" --metadata "sampling=${mode}" --metadata "gate_arm=${arm}"
    --metadata "gpu=${gpu}" --metadata "corpus_sha256=${CORPUS_SHA}"
    --metadata "source_revision=${SOURCE_REVISION}")
  printf '%q ' "${serve[@]}" > "${dir}/launch-command.txt"; echo >> "${dir}/launch-command.txt"
  {
    printf 'GPU=%q\nPORT=%q\nSERVED=%q\nCORPUS=%q\nSLICES=%q\nREQUESTS=%q\nREPO_ROOT=%q\n' \
      "${gpu}" "${port}" "${served}" "${CORPUS}" "${SLICES}" "${REQUESTS}" "${REPO_ROOT}"
    printf 'SERVE=('; printf '%q ' "${serve[@]}"; printf ')\n'
    printf 'BENCH=('; printf '%q ' "${bench[@]}"; printf ')\n'
  } > "${dir}/arm.env"
}

LANES=("" "" "" "")
ARMS=()
for spec in ${GATE_ARMS:?}; do
  IFS=: read -r arm head k mode lane <<< "${spec}"
  if [ -n "${GATE_ONLY:-}" ] && [[ " ${GATE_ONLY} " != *" ${arm} "* ]]; then continue; fi
  arm_env "${arm}" "${head}" "${k}" "${mode}" "${lane}"
  LANES[${lane}]+="bash ${REPO_ROOT}/experiments/workload-shape/gate-arm.sh ${OUT}/${arm}; "
  ARMS+=("${arm}")
done

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=${GATE_JOB_NAME:-shape-gate}
#SBATCH --account=${GATE_ACCOUNT:-d_research_serving}
#SBATCH --partition=${GATE_PARTITION:-normal}
#SBATCH --time=${GATE_TIME:-04:00:00}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -uo pipefail
echo "job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} started=\$(date -u +%FT%TZ)" > "${OUT}/slurm.txt"
( sha256sum "${IMAGE}" > "${OUT}/image.sha256" ) &
IDENT=\$!
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \\
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \\
  --format=csv,noheader,nounits -lms 2000 > "${OUT}/gpu.csv" 2> /dev/null &
SAMPLER=\$!
srun --overlap --nodes=1 --ntasks=1 --environment="${OUT}/environment.toml" bash -c '
  python3 -c "import torch, vllm, transformers; print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)" > "${OUT}/runtime-versions.txt" 2>&1
  pip freeze > "${OUT}/pip-freeze.txt" 2> /dev/null || true
  status=0
$(for gpu in 0 1 2 3; do [ -n "${LANES[${gpu}]}" ] && echo "  ( ${LANES[${gpu}]} ) & pids+=(\$!)"; done)
  for p in "\${pids[@]}"; do wait \${p} || status=1; done
  exit \${status}
'
status=\$?
kill \${SAMPLER} 2> /dev/null || true
wait \${IDENT} || true
echo "finished=\$(date -u +%FT%TZ) status=\${status}" >> "${OUT}/slurm.txt"
exit \${status}
EOF
bash -n "${OUT}/job.sh"
for arm in "${ARMS[@]}"; do bash -n "${OUT}/${arm}/arm.env"; done
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then echo "job script: ${OUT}/job.sh"; exit 0; fi
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
