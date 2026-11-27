#!/usr/bin/env bash
# Validation gate of the Qwen3-8B workload-shape replay: serve Qwen/Qwen3-8B with each
# Red Hat drafter in vLLM (TP1, one deployment per GPU, C=1) on the 28-slice corpus and
# record the served acceptance per slice with the repo's bench client.
#   experiments/workload-shape/submit-gate-qwen3.sh OUT_DIR
# Arms (GPU: deployments, one after the other):
#   0: eagle3-k3 greedy      1: dflash-k3 greedy      2: dflash-k15 greedy (the card's k)
#   3: eagle3-k3 T=1, then dflash-k3 T=1 (validates the sampled replay)
# Every request: 512-token cap, thinking off (corpus chat_template_kwargs), prefix caching
# off, --generation-config vllm (no model sampling defaults), greedy = temperature 0;
# T=1 = temperature 1, top_p 1, top_k -1, seed 1. Each deployment gets its own
# directory under OUT_DIR/<arm>/ with the launch command, cells and logs.
# Env: GATE_MODELS (pinned downloads with SHA256SUMS), GATE_CORPUS (corpus-measured.jsonl),
#      GATE_ACCOUNT (d_research_serving), GATE_PARTITION (normal), GATE_TIME (04:00:00),
#      GATE_ARMS (subset, e.g. "eagle3-k3"), GATE_SLICES (subset of slice labels), GATE_REQUESTS (32).
set -euo pipefail

OUT="${1:?usage: submit-gate-qwen3.sh OUT_DIR}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b
MODELS="${GATE_MODELS:-${W}/models}"
TARGET="${MODELS}/Qwen--Qwen3-8B"
TARGET_ID="Qwen/Qwen3-8B"
TARGET_REV="b968826d9c46dd6066d109eabc6255188de91218"
EAGLE3="${MODELS}/RedHatAI--Qwen3-8B-speculator.eagle3"
EAGLE3_REV="08610ffa01dd9f16731fe8f627b85905b6aa51c4"
DFLASH="${MODELS}/RedHatAI--Qwen3-8B-speculator.dflash"
DFLASH_REV="1a11b170eb65c8a62c80ecd01dfe22a5907298e6"
CORPUS="${GATE_CORPUS:?set GATE_CORPUS to RUN_DIR/inputs/corpus-measured.jsonl}"
ENV_SOURCE="${REPO_ROOT}/experiments/workload-shape/env-qwen3.toml"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${ENV_SOURCE}")"
ACCOUNT="${GATE_ACCOUNT:-d_research_serving}"
PARTITION="${GATE_PARTITION:-normal}"
TIME_LIMIT="${GATE_TIME:-04:00:00}"
REQUESTS="${GATE_REQUESTS:-32}"
ARMS="${GATE_ARMS:-eagle3-k3 dflash-k3 dflash-k15 eagle3-k3-t1 dflash-k3-t1}"
SUITE="${REPO_ROOT}/results/8b/workload-shape/inputs/suite.yaml"
SLICES="${GATE_SLICES:-}"
if [ -z "${SLICES}" ]; then
  SLICES="$(sed -n 's/^  - {label: \([^,]*\),.*/\1/p' "${SUITE}" | tr '\n' ' ')"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

for f in "${TARGET}/SHA256SUMS" "${EAGLE3}/SHA256SUMS" "${DFLASH}/SHA256SUMS" "${CORPUS}"; do
  test -s "${f}" || { echo "missing ${f}" >&2; exit 2; }
done
sha_of() { awk '$2 == "./model.safetensors" {print $1}' "$1/SHA256SUMS"; }
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
test ! -e "${OUT}/job.sh" || { echo "${OUT} already holds a gate run; use a new directory" >&2; exit 2; }
cp "${ENV_SOURCE}" "${OUT}/environment.toml"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${OUT}/environment.toml"
{
  echo "source_revision $(cat "${REPO_ROOT}/SOURCE_REVISION" 2> /dev/null || git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "target ${TARGET_ID} ${TARGET_REV} ${TARGET}"
  echo "eagle3 RedHatAI/Qwen3-8B-speculator.eagle3 ${EAGLE3_REV} ${EAGLE3} $(sha_of "${EAGLE3}")"
  echo "dflash RedHatAI/Qwen3-8B-speculator.dflash ${DFLASH_REV} ${DFLASH} $(sha_of "${DFLASH}")"
  echo "corpus ${CORPUS} $(sha256sum "${CORPUS}" | cut -d' ' -f1)"
  echo "image ${IMAGE}"
} > "${OUT}/identity.txt"
cp "${TARGET}/SHA256SUMS" "${OUT}/target.sha256"
cp "${EAGLE3}/SHA256SUMS" "${OUT}/eagle3.sha256"
cp "${DFLASH}/SHA256SUMS" "${OUT}/dflash.sha256"
{ printf '%q ' "$0" "$@"; echo; env | grep -E '^GATE_' | sort || true; } > "${OUT}/submit-command.txt"

# One environment file per deployment; experiments/workload-shape/gate-arm.sh runs it.
arm_env() {
  local arm="$1" gpu="$2" port="$3" drafter method k mode rev
  case "${arm}" in
    eagle3-k3 | eagle3-k3-t1) drafter="${EAGLE3}"; method=eagle3; k=3; rev="${EAGLE3_REV}" ;;
    dflash-k3 | dflash-k3-t1) drafter="${DFLASH}"; method=dflash; k=3; rev="${DFLASH_REV}" ;;
    dflash-k15) drafter="${DFLASH}"; method=dflash; k=15; rev="${DFLASH_REV}" ;;
    *) echo "unknown arm ${arm}" >&2; exit 2 ;;
  esac
  local gen=(--temperature 0 --top-p 1 --seed 1)
  mode=greedy
  if [ "${arm%-t1}" != "${arm}" ]; then
    mode=t1
    gen=(--temperature 1 --top-p 1 --top-k -1 --seed 1)
  fi
  local spec="{\"method\": \"${method}\", \"model\": \"${drafter}\", \"num_speculative_tokens\": ${k}}"
  local dir="${OUT}/${arm}" served="qwen3-8b-${arm}" dep="qwen3-8b-shape-gate-${arm}-${STAMP}"
  local ck corpus_sha
  ck="$(sha_of "${drafter}")"
  corpus_sha="$(sha256sum "${CORPUS}" | cut -d' ' -f1)"
  mkdir -p "${dir}"
  local serve=(vllm serve "${TARGET}" --served-model-name "${served}" --host 127.0.0.1
    --port "${port}" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768
    --max-num-seqs 8 --gpu-memory-utilization 0.85 --no-enable-prefix-caching
    --generation-config vllm --speculative-config "${spec}")
  local bench=(--variant "${arm}" --method "${method}" --num-speculative-tokens "${k}")
  if [ "${method}" = eagle3 ]; then bench+=(--algorithm eagle3); fi
  bench+=("${gen[@]}"
    --metadata "deployment_id=${dep}" --metadata "target_model=${TARGET_ID}"
    --metadata "target_revision=${TARGET_REV}" --metadata target_tensor_parallel_size=1
    --metadata "checkpoint_sha256=${ck}" --metadata "drafter_model=RedHatAI/Qwen3-8B-speculator.${method}"
    --metadata "drafter_revision=${rev}" --metadata "image=${IMAGE}" --metadata "sampling=${mode}"
    --metadata "gate_arm=${arm}" --metadata "corpus_sha256=${corpus_sha}")
  printf '%q ' "${serve[@]}" > "${dir}/launch-command.txt"; echo >> "${dir}/launch-command.txt"
  {
    printf 'GPU=%q\nPORT=%q\nSERVED=%q\nCORPUS=%q\nSLICES=%q\nREQUESTS=%q\nREPO_ROOT=%q\n' \
      "${gpu}" "${port}" "${served}" "${CORPUS}" "${SLICES}" "${REQUESTS}" "${REPO_ROOT}"
    printf 'SERVE=('; printf '%q ' "${serve[@]}"; printf ')\n'
    printf 'BENCH=('; printf '%q ' "${bench[@]}"; printf ')\n'
  } > "${dir}/arm.env"
}

# GPU lanes: arms that share a lane run one after the other.
declare -A LANE=([eagle3-k3]=0 [dflash-k3]=1 [dflash-k15]=2 [eagle3-k3-t1]=3 [dflash-k3-t1]=3)
LANES=("" "" "" "")
for arm in ${ARMS}; do
  gpu="${LANE[${arm}]:?unknown arm ${arm}}"
  arm_env "${arm}" "${gpu}" $((8100 + gpu))
  LANES[${gpu}]+="bash ${REPO_ROOT}/experiments/workload-shape/gate-arm.sh ${OUT}/${arm}; "
done

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=shape-gate-qwen3-8b
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -uo pipefail
echo "job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} started=\$(date -u +%FT%TZ)" > "${OUT}/slurm.txt"
( sha256sum "${IMAGE}" > "${OUT}/image.sha256" ) &
IDENT=\$!
srun --overlap --nodes=1 --ntasks=1 --environment="${OUT}/environment.toml" bash -c '
  python3 -c "import torch, vllm, transformers; print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)" > "${OUT}/runtime-versions.txt" 2>&1
  pip freeze > "${OUT}/pip-freeze.txt" 2> /dev/null || true
  status=0
$(for gpu in 0 1 2 3; do [ -n "${LANES[${gpu}]}" ] && echo "  ( ${LANES[${gpu}]} ) & pids+=(\$!)"; done)
  for p in "\${pids[@]}"; do wait \${p} || status=1; done
  exit \${status}
'
status=\$?
wait \${IDENT} || true
echo "finished=\$(date -u +%FT%TZ) status=\${status}" >> "${OUT}/slurm.txt"
exit \${status}
EOF
bash -n "${OUT}/job.sh"
for arm in ${ARMS}; do bash -n "${OUT}/${arm}/arm.env"; done
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then echo "job script: ${OUT}/job.sh"; exit 0; fi
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
