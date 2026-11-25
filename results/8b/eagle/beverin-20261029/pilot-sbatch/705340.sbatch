#!/bin/bash
# P0 of the Beverin move of the EAGLE thinking-on run, 1 MI300A node, from a
# repo-<sha> snapshot (sbatch --chdir=<snapshot> methods/eagle/beverin/p0.sbatch):
# 1. copies the production checkpoints/resume (+ history.jsonl, evals.json) into
#    the pilot run ids stage1-bev-p0 / -pilot-lt / -pilot-rb (production read-only);
# 2. p0_probe.py: imports, teacher load, teacher features hipBLASLt vs rocBLAS;
# 3. a short 1-node training replay from the copy (train-e31-opb-thinkon.yaml:
#    4 ranks x 8 = the same 32 rows per step), rocBLAS, stopping after 40 steps.
#SBATCH -A infra01 -p mi300 -N 1 --ntasks-per-node=1 --gpus-per-node=4 --exclusive
#SBATCH -t 01:15:00 --job-name=eagle-bev-p0
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/logs/%x-%j.out
set -euo pipefail
REPO_ROOT="$(pwd)"
RUNS=/iopsstor/scratch/cscs/${USER}/apertus-eagle/8b/runs
PROD="${RUNS}/e31-opb-thinkon-stage1"
EDF="${REPO_ROOT}/methods/eagle/configs/train-env-beverin.toml"
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} node=$(hostname) repo=${REPO_ROOT} rev=$(cat REVISION)"
for id in stage1-bev-p0 stage1-bev-pilot-lt stage1-bev-pilot-rb; do
  dest="${RUNS}/e31-opb-thinkon-${id}"
  if [ ! -f "${dest}/checkpoints/resume/state.json" ]; then
    mkdir -p "${dest}/checkpoints"
    cp -a "${PROD}/checkpoints/resume" "${dest}/checkpoints/resume.part"
    cp -a "${PROD}/history.jsonl" "${PROD}/evals.json" "${dest}/"
    mv "${dest}/checkpoints/resume.part" "${dest}/checkpoints/resume"
  fi
  python3 -c "import json,sys; print(sys.argv[1], 'resume step', json.load(open(sys.argv[2]))['step'])" \
    "${id}" "${dest}/checkpoints/resume/state.json"
done
( cd "${PROD}/checkpoints/resume" && sha256sum state.json master.pt optimizer.pt draft.safetensors ) \
  > "${RUNS}/e31-opb-thinkon-stage1-bev-p0/source-resume.sha256" &
hash_pid=$!
export I_AM_ON_AN_ALLOCATED_GPU_JOB=1 EAGLE_PLATFORM=rocm EAGLE_CUDA_DEVICES=0,1,2,3
export TORCHSPEC_ROOT="${REPO_ROOT}/scratch/TorchSpec" EAGLE_PYDEPS="${REPO_ROOT}/scratch/pydeps"
export HF_HOME=/iopsstor/scratch/cscs/${USER}/hf_home TMPDIR=/tmp WANDB_MODE=offline WANDB_DISABLED=true
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
OUT=/iopsstor/scratch/cscs/${USER}/beverin-eagle/p0/${SLURM_JOB_ID}
mkdir -p "${OUT}"
srun --environment="${EDF}" bash -c '
  set -euo pipefail
  unset ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  C=/tmp/bev-cache-p0; export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg MIOPEN_USER_DB_PATH=$C/miopen MIOPEN_CUSTOM_CACHE_DIR=$C/miopen
  export PYTHONPATH="${EAGLE_PYDEPS}:'"${REPO_ROOT}"'/methods/eagle:'"${REPO_ROOT}"'/src:${TORCHSPEC_ROOT}"
  rocm-smi --showproductname || true
  python3 '"${REPO_ROOT}"'/methods/eagle/beverin/p0_probe.py --contract targets/8b/contract.json \
    --rows /iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/data/dspark-opb-thinking-on/heldout.jsonl \
    --output '"${OUT}"'/p0-probe.json
  rm -rf $C
' || echo "P0 probe failed rc=$?"
wait "${hash_pid}"
cmp <(cut -d" " -f1 "${RUNS}/e31-opb-thinkon-stage1-bev-p0/source-resume.sha256") \
    <(cd "${RUNS}/e31-opb-thinkon-stage1-bev-p0/checkpoints/resume" && sha256sum state.json master.pt optimizer.pt draft.safetensors | cut -d" " -f1) \
  && echo "pilot copy matches production resume"
export EAGLE_TRAIN_CONFIG="${REPO_ROOT}/methods/eagle/configs/8b/train-e31-opb-thinkon.yaml"
export EAGLE_STEPS=train EAGLE_RUN_ID=stage1-bev-p0 EAGLE_BLAS_BACKEND=cublas
export EAGLE_STOP_AFTER_STEPS=$(( 166654 + 40 ))
export EAGLE_STOP_AT=$(( $(date +%s) + 3600 ))
export EAGLE_MEM_LOG_DIR=/iopsstor/scratch/cscs/${USER}/beverin-eagle/logs
rc=0
srun --environment="${EDF}" bash "${REPO_ROOT}/methods/eagle/launch/train-eagle-on-node.sh" || rc=$?
echo "END $(date -u -Iseconds) rc=${rc}"
