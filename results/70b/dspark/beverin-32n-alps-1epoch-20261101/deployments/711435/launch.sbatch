#!/bin/bash
#SBATCH --job-name=dspark-70b-opb-thinkoff-32n-alps-1epoch-20261101T100117Z
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --time=06:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
export REPO_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e
export DSPARK_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
export SPECULATORS=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/speculators-7046ac6-provenance
export SIDE_DEPS=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/py312/pydeps
export DATA=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/data
export RUN_ID=opb-thinkoff-32n-alps-1epoch-20261101T100117Z
export NODES=32
export SEQ=8192
export ANCHORS=512
export LR=8.48528137423857e-4
export EPOCHS=1
export CKPT_FREQ=0.015625
export DSPARK_DTENSOR_ALLGATHER=1
export LAYERS=1\ 20\ 39\ 58\ 77
export EXTRACT_MEM=0.62
export EXTRACT_FLAGS=--api-server-count\ 1\ --renderer-num-workers\ 1
export PY_PRELUDE=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e/methods/dspark/beverin/apertus15_run.py
export MAX_ATTEMPTS=1
export WORKERS=1
export PREFETCH=1
export BEVERIN_SEGMENT_STEPS=640
export BEVERIN_EMPTY_CACHE_ON_SAVE=1
export NCCL_DEBUG=WARN
export DATA_PROVENANCE=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/provenance/data-20261025
export EXTRACT_START_ATTEMPTS=1
export EXTRACT_HEALTH_SECONDS=7200
export BEVERIN_BLAS_BACKEND=cublas
export BEVERIN_SINGLE_EPOCH=0
export BEVERIN_CHECK_ACTIVATIONS=1
export BEVERIN_CAPTURE_BAD_BATCH=1
export BEVERIN_ALPS=1
export BEVERIN_CHECKPOINT_RING=1
export BEVERIN_ROLLBACK=1
export BEVERIN_MILESTONE_INTERVAL=512
export BEVERIN_RING_KEEP=2
export DSPARK_CXI_EAGER=1
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} edf=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e/methods/dspark/beverin/env-vllm-alps.toml"
cat "${REPO_ROOT}/REVISION" 2> /dev/null || true
export WORK=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
export BEVERIN_DEPLOYMENT="${WORK}/runs/${RUN_ID}/deployments/${SLURM_JOB_ID}"
mkdir -p "${BEVERIN_DEPLOYMENT}"
cp "$0" "${BEVERIN_DEPLOYMENT}/launch.sbatch"
cp "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e/methods/dspark/beverin/env-vllm-alps.toml" "${BEVERIN_DEPLOYMENT}/environment.toml"
cp "${REPO_ROOT}/REVISION" "${BEVERIN_DEPLOYMENT}/repo-revision.txt"
scontrol show job "${SLURM_JOB_ID}" > "${BEVERIN_DEPLOYMENT}/slurm-job.txt"
git -C "${SPECULATORS}" rev-parse HEAD > "${BEVERIN_DEPLOYMENT}/speculators-revision.txt"
git -C "${SPECULATORS}" diff HEAD > "${BEVERIN_DEPLOYMENT}/speculators.patch"
sha256sum "${DATA}/state.json" "${DATA}/dataset_info.json" > "${BEVERIN_DEPLOYMENT}/dataset-metadata.sha256"
if [ -n "${DATA_PROVENANCE:-}" ]; then
  cp -r "${DATA_PROVENANCE}" "${BEVERIN_DEPLOYMENT}/data-provenance"
fi
python3 "${REPO_ROOT}/methods/dspark/beverin/audit_train.py" start \
  --work "${WORK}" --run-id "${RUN_ID}" --job-id "${SLURM_JOB_ID}"
if [ "${BEVERIN_SINGLE_EPOCH:-0}" = 1 ]; then
  guard_rc=0
  python3 "${REPO_ROOT}/methods/dspark/beverin/single_epoch.py" \
    --checkpoints "${WORK}/runs/${RUN_ID}/checkpoints" \
    --expected-steps "${BEVERIN_EXPECTED_EPOCH_STEPS}" || guard_rc=$?
  if [ "${guard_rc}" = 0 ]; then
    python3 "${REPO_ROOT}/methods/dspark/beverin/audit_train.py" finish \
      --work "${WORK}" --run-id "${RUN_ID}" --job-id "${SLURM_JOB_ID}" --completion-skip
    echo "FINISH $(date -u -Iseconds) already trained and validated; containers skipped"
    exit 0
  fi
  [ "${guard_rc}" = 1 ] || exit "${guard_rc}"
fi
IMAGE_PATH="/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh"
if [ "1" = 1 ]; then
  sha256sum "${IMAGE_PATH}" > "${BEVERIN_DEPLOYMENT}/container.sha256"
  got=$(awk '{print $1}' "${BEVERIN_DEPLOYMENT}/container.sha256")
  if [ "${got}" != "416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928" ]; then
    echo "ALPS image hash mismatch: ${got}" >&2
    exit 1
  fi
  echo "ghcr.io/eth-cscs/alps-extended-images/vllm-rocm@sha256:7d95232cbc7511be5672d272f7a567ea52a3c5fbd5310ce0903956f0f37b5cc7" > "${BEVERIN_DEPLOYMENT}/image-oci.txt"
  image_hash_pid=""
else
  sha256sum "${IMAGE_PATH}" > "${BEVERIN_DEPLOYMENT}/container.sha256" &
  image_hash_pid=$!
fi
# The trainer saves and exits STOP_MARGIN s before the time limit
# (speculators-deadline-stop.patch).
export DSPARK_STOP_AT=$(( $(date +%s) + 21600 - 2400 ))
export HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
rc=0
# shellcheck disable=SC2086
srun --network=disable_rdzv_get --ntasks-per-node=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e/methods/dspark/beverin/env-vllm-alps.toml --export=ALL \
  bash "${REPO_ROOT}/methods/dspark/beverin/train-node.sh" || rc=$?
if [ -n "${image_hash_pid}" ]; then wait "${image_hash_pid}"; fi
if [ "1" = 1 ] && [ "${rc}" = 0 ]; then
  # shellcheck disable=SC2086
  srun --network=disable_rdzv_get --ntasks=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-83a9c53e/methods/dspark/beverin/env-vllm-alps.toml --export=ALL \
    python3 "${REPO_ROOT}/methods/dspark/beverin/scan_checkpoint.py" \
    --checkpoint "${WORK}/runs/${RUN_ID}/checkpoints/0" \
    --output "${BEVERIN_DEPLOYMENT}/finiteness-scan.json" --hash || rc=$?
fi
if [ -n "640" ] && [ "${rc}" = 0 ]; then
  stable="${WORK}/runs/${RUN_ID}/stable/step-640"
  if [ ! -e "${stable}" ]; then
    mkdir -p "${WORK}/runs/${RUN_ID}/stable"
    cp -a "${WORK}/runs/${RUN_ID}/checkpoints/0" "${stable}"
    echo "STABLE_CHECKPOINT ${stable}"
  fi
fi
audit_rc=0
python3 "${REPO_ROOT}/methods/dspark/beverin/audit_train.py" finish \
  --work "${WORK}" --run-id "${RUN_ID}" --job-id "${SLURM_JOB_ID}" --return-code "${rc}" || audit_rc=$?
[ "${rc}" != 0 ] || rc=${audit_rc}
echo "FINISH $(date -u -Iseconds) rc=${rc}"
exit "${rc}"
