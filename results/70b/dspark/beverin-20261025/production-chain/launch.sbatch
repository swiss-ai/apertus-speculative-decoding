#!/bin/bash
#SBATCH --job-name=dspark-70b-opb-thinkoff-1n-beverin
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
export REPO_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200
export DSPARK_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
export SPECULATORS=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/speculators-7046ac6-provenance
export SIDE_DEPS=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/py312/pydeps
export DATA=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/data
export RUN_ID=opb-thinkoff-1n-beverin
export SEQ=8192
export ANCHORS=512
export LR=1.5e-4
export EPOCHS=10
export CKPT_FREQ=0.015625
export DSPARK_DTENSOR_ALLGATHER=1
export LAYERS=1\ 20\ 39\ 58\ 77
export EXTRACT_MEM=0.62
export EXTRACT_FLAGS=--api-server-count\ 1\ --renderer-num-workers\ 1
export PY_PRELUDE=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200/methods/dspark/beverin/apertus15_run.py
export WORKERS=1
export PREFETCH=1
export BEVERIN_EMPTY_CACHE_ON_SAVE=1
export DATA_PROVENANCE=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/provenance/data-20261025
export BEVERIN_BLAS_BACKEND=cublas
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} edf=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200/methods/dspark/beverin/env-vllm031.toml"
cat "${REPO_ROOT}/REVISION" 2> /dev/null || true
export WORK=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
export BEVERIN_DEPLOYMENT="${WORK}/runs/${RUN_ID}/deployments/${SLURM_JOB_ID}"
mkdir -p "${BEVERIN_DEPLOYMENT}"
cp "$0" "${BEVERIN_DEPLOYMENT}/launch.sbatch"
cp "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200/methods/dspark/beverin/env-vllm031.toml" "${BEVERIN_DEPLOYMENT}/environment.toml"
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
sha256sum "/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh" > "${BEVERIN_DEPLOYMENT}/container.sha256" &
image_hash_pid=$!
# The trainer saves and exits STOP_MARGIN s before the time limit
# (speculators-deadline-stop.patch).
export DSPARK_STOP_AT=$(( $(date +%s) + 86400 - 2400 ))
export HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
rc=0
srun --ntasks-per-node=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-ada15200/methods/dspark/beverin/env-vllm031.toml --export=ALL \
  bash "${REPO_ROOT}/methods/dspark/beverin/train-node.sh" || rc=$?
wait "${image_hash_pid}"
audit_rc=0
python3 "${REPO_ROOT}/methods/dspark/beverin/audit_train.py" finish \
  --work "${WORK}" --run-id "${RUN_ID}" --job-id "${SLURM_JOB_ID}" --return-code "${rc}" || audit_rc=$?
[ "${rc}" != 0 ] || rc=${audit_rc}
echo "FINISH $(date -u -Iseconds) rc=${rc}"
exit "${rc}"
