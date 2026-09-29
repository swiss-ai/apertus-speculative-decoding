#!/usr/bin/env bash
# Regenerate open-perfectblend answers with the 8B target (thinking off, greedy).
#   ./methods/eagle/launch/submit-regenerate.sh
# Job 1 shards the checked corpus; job 2 is an array of exclusive nodes, one
# generate_targets process per GPU and shard. Finished shards are skipped, so
# resubmitting resumes. Nothing runs on the login node.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"

ENV_SOURCE="${EAGLE_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
ACCOUNT="${EAGLE_TRAIN_ACCOUNT:-infra01}"
PARTITION="${REGEN_PARTITION:-normal}"
NODES="${REGEN_NODES:-4}"
GPUS=4
SHARDS=$(( NODES * GPUS ))
TIME_LIMIT="${REGEN_TIME:-04:00:00}"
DATA_DIR="${REGEN_DATA_DIR:-/iopsstor/scratch/cscs/${USER}/apertus-eagle/data/open-perfectblend}"
GEN_DIR="${DATA_DIR}/generated-8b-think-off"
REPORT_DIR="${REPO_ROOT}/results/8b/eagle/data-perfectblend"
LOG_ROOT="${REPO_ROOT}/results/8b/eagle/logs"
CONTRACT="${REPO_ROOT}/targets/8b/contract.json"
# Longest checked conversation is 7,785 tokens; p99 2,289.
MAX_SEQ="${REGEN_MAX_SEQ:-8192}"
MAX_PROMPT="${REGEN_MAX_PROMPT:-6144}"
MAX_NEW="${REGEN_MAX_NEW:-4096}"

SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
mkdir -p "${LOG_ROOT}" "${GEN_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMON="#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --chdir=${REPO_ROOT}"
ENV_LINES="set -euo pipefail
export HF_HOME=/iopsstor/scratch/cscs/\${USER}/hf_home
export PYTHONPATH=${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1"

PREP="${LOG_ROOT}/submit-regen-prep-${STAMP}.sbatch"
cat > "${PREP}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eagle-8b-regen-prep
${COMMON}
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_ROOT}/%j.out
#SBATCH --error=${LOG_ROOT}/%j.err
${ENV_LINES}
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID}"
python3 -m apertus_eagle.perfectblend_shards --data-dir ${DATA_DIR} --shards ${SHARDS} \\
  --report ${REPORT_DIR}/shards.json
echo "END \$(date -u -Iseconds)"
SBATCH

GEN="${LOG_ROOT}/submit-regen-gen-${STAMP}.sbatch"
cat > "${GEN}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eagle-8b-regen
${COMMON}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-$(( NODES - 1 ))
#SBATCH --output=${LOG_ROOT}/%A_%a.out
#SBATCH --error=${LOG_ROOT}/%A_%a.err
${ENV_LINES}
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
pids=()
for gpu in \$(seq 0 $(( GPUS - 1 ))); do
  shard=\$(printf "%03d" \$(( SLURM_ARRAY_TASK_ID * ${GPUS} + gpu )))
  out=${GEN_DIR}/shard-\${shard}
  if [ -f "\${out}/generation-summary.json" ]; then echo "shard \${shard} done, skipping"; continue; fi
  mkdir -p "\${out}"
  CUDA_VISIBLE_DEVICES=\${gpu} python3 -m apertus_eagle.generate_targets --contract ${CONTRACT} \\
    --input ${DATA_DIR}/shards/shard-\${shard}.jsonl --output-dir "\${out}" \\
    --max-seq-length ${MAX_SEQ} --max-prompt-tokens ${MAX_PROMPT} --max-new-tokens ${MAX_NEW} \\
    > "\${out}/generate.log" 2>&1 &
  pids+=(\$!)
done
status=0
for pid in "\${pids[@]}"; do wait "\${pid}" || status=1; done
echo "END \$(date -u -Iseconds) status=\${status}"
exit \${status}
SBATCH

bash -n "${PREP}"; bash -n "${GEN}"
echo "prep=${PREP} gen=${GEN} shards=${SHARDS} out=${GEN_DIR}"
if [ "${REGEN_DRY_RUN:-0}" = "1" ]; then echo "dry run; not submitting"; exit 0; fi
if [ "${REGEN_SKIP_PREP:-0}" = "1" ]; then
  sbatch "${GEN}"
else
  PREP_JOB="$(sbatch --parsable "${PREP}")"
  echo "prep job ${PREP_JOB}"
  sbatch --dependency="afterok:${PREP_JOB}" --kill-on-invalid-dep=yes "${GEN}"
fi
