#!/usr/bin/env bash
# Submit one ALPS campaign stage from the Beverin login node. No job chaining.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# shellcheck disable=SC1091
. "${REPO_ROOT}/methods/dspark/beverin/alps_image.sh"
STAGE="${STAGE:?STAGE=a1|a2|a3|b|c|data|regen}"
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
CAMPAIGN="${WORK}/campaigns/alps-1epoch-20261031"
mkdir -p "${CAMPAIGN}"
squeue -u "${USER}" -o "%.18i %.9P %.40j %.2t %.10M %.6D %R" > "${CAMPAIGN}/squeue-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ).txt" || true
# Protected jobs are observed and never cancelled by this launcher.
RUNTIME="${RUNTIME:-${REPO_ROOT}}"
EDF="${RUNTIME}/methods/dspark/beverin/env-vllm-alps.toml"
LOG_ROOT="${WORK}/logs"
mkdir -p "${LOG_ROOT}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

submit_one() {  # name time script-body
  local name="$1" time_limit="$2" body="$3" nice_flag="${4:-}"
  local script="${LOG_ROOT}/${name}-${STAMP}.sbatch"
  cat > "${script}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --time=${time_limit}
#SBATCH --nodes=${STAGE_NODES:-1}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --chdir=${RUNTIME}
${nice_flag}
set -euo pipefail
export REPO_ROOT=${RUNTIME}
export WORK=${WORK}
export RUN_ID=${RUN_ID}
export SPECULATORS=${SPECULATORS:-${WORK}/speculators-7046ac6-provenance}
export SIDE_DEPS=${SIDE_DEPS:-${WORK}/py312/pydeps}
export TARGET_MODEL=${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}
export CAMPAIGN_DECISION=${CAMPAIGN}/blas-decision.json
export BEVERIN_DEPLOYMENT=${CAMPAIGN}/${name}-\${SLURM_JOB_ID}
mkdir -p "\${BEVERIN_DEPLOYMENT}"
cp "\$0" "\${BEVERIN_DEPLOYMENT}/launch.sbatch"
cp "${EDF}" "\${BEVERIN_DEPLOYMENT}/environment.toml"
printf '%s\n' "${RUNTIME_SHA:-unknown}" > "\${BEVERIN_DEPLOYMENT}/repo-revision.txt"
sha256sum "${ALPS_SQUASHFS}" > "\${BEVERIN_DEPLOYMENT}/container.sha256"
got=\$(awk '{print \$1}' "\${BEVERIN_DEPLOYMENT}/container.sha256")
if [ "\${got}" != "${ALPS_SQUASHFS_SHA256}" ]; then
  echo "ALPS image hash mismatch: \${got}" >&2
  exit 1
fi
scontrol show job "\${SLURM_JOB_ID}" > "\${BEVERIN_DEPLOYMENT}/slurm-job.txt"
${body}
SBATCH
  sbatch --parsable "${script}"
}

case "${STAGE}" in
  a1|a2|a3)
    RUN_ID="${RUN_ID:?}"
    STAGE_NODES=1
    id="$(submit_one "dspark-a-${STAGE}-${RUN_ID}" "02:00:00" \
      "srun --network=disable_rdzv_get --ntasks-per-node=1 --environment=${EDF} --export=ALL bash ${RUNTIME}/methods/dspark/beverin/stage_a_inside.sh ${STAGE}")"
    echo "${id}" | tee "${CAMPAIGN}/${STAGE}.jobid"
    ;;
  b|c)
    RUN_ID="${RUN_ID:?}"
    export BEVERIN_ALPS=1 WORK RUNTIME RUN_ID
    export EDF SPECULATORS="${SPECULATORS:-${WORK}/speculators-7046ac6-provenance}"
    export SIDE_DEPS="${SIDE_DEPS:-${WORK}/py312/pydeps}"
    export DATA="${DATA:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/data}"
    export DATA_PROVENANCE="${DATA_PROVENANCE:-${WORK}/provenance/data-20261025}"
    export NODES=32 CHAIN=1 MAX_ATTEMPTS=1 EXTRACT_START_ATTEMPTS=1
    export STOP_MARGIN=2400 SBATCH_EXTRA=--no-requeue
    export SEQ=8192 ANCHORS=512 EPOCHS=1 LR=8.48528137423857e-4
    export CKPT_FREQ=0.015625 DSPARK_DTENSOR_ALLGATHER=1
    export LAYERS='1 20 39 58 77'
    export EXTRACT_MEM=0.62 EXTRACT_FLAGS='--api-server-count 1 --renderer-num-workers 1'
    export WORKERS=1 PREFETCH=1 BEVERIN_EMPTY_CACHE_ON_SAVE=1
    export BEVERIN_CHECK_ACTIVATIONS=1 BEVERIN_CAPTURE_BAD_BATCH=1 NCCL_DEBUG=WARN
    export BEVERIN_POST_SCAN=1
    unset MAX_STEPS BEVERIN_PILOT_RESUME BEVERIN_PROFILE_STEPS DEPENDENCY
    unset BEVERIN_REDUCE_DTYPE BEVERIN_NCCL_PROTO BEVERIN_NCCL_MIN_NCHANNELS
    unset BEVERIN_NCCL_NCHANNELS_PER_NET_PEER BEVERIN_FI_MR_CACHE_MONITOR
    if [ "${STAGE}" = b ]; then
      test ! -e "${WORK}/runs/${RUN_ID}"
      export TIME=06:00:00 BEVERIN_SEGMENT_STEPS=640 BEVERIN_SINGLE_EPOCH=0
      export BEVERIN_STABLE_STEP=640
      unset BEVERIN_EXPECTED_EPOCH_STEPS
      printf '%s\n' "${RUN_ID}" > "${CAMPAIGN}/run-id.txt"
    else
      export TIME=24:00:00 BEVERIN_SINGLE_EPOCH=1 BEVERIN_EXPECTED_EPOCH_STEPS=4077
      unset BEVERIN_SEGMENT_STEPS BEVERIN_STABLE_STEP
    fi
    if [ -f "${CAMPAIGN}/blas-decision.json" ]; then
      backend="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["backend"])' "${CAMPAIGN}/blas-decision.json")"
      if [ "${backend}" = default ]; then
        unset BEVERIN_BLAS_BACKEND
      else
        export BEVERIN_BLAS_BACKEND=cublas
      fi
    else
      export BEVERIN_BLAS_BACKEND=cublas
    fi
    bash "${RUNTIME}/methods/dspark/beverin/submit-train.sh"
    ;;
  data)
    RUN_ID="${RUN_ID:-expansion-corpus-${STAMP}}"
    STAGE_NODES=1
    id="$(submit_one "dspark-expansion-${STAMP}" "06:00:00" \
      "srun --network=disable_rdzv_get --ntasks-per-node=1 --environment=${EDF} --export=ALL bash ${RUNTIME}/methods/dspark/beverin/data_expansion/run.sh" \
      "#SBATCH --nice=10000")"
    echo "${id}" | tee "${CAMPAIGN}/data.jobid"
    ;;
  regen)
    echo "regen submission is started by launch-alps only after the corpus and the pilot exist" >&2
    exit 2
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
