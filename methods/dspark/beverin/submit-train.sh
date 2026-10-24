#!/usr/bin/env bash
# Submit DSpark 70B training on Beverin (MI300A), the counterpart of
# methods/dspark/train/submit-train.sh on Clariden. Run on the Beverin login
# node from the repository copy:
#   RUN_ID=opb-thinkoff-16n-beverin NODES=16 [CHAIN=3] [TIME=24:00:00] \
#     methods/dspark/beverin/submit-train.sh
# Same train-node.sh, recipe and data as Clariden's 16-node run; Beverin
# specifics: the vLLM 0.31 ROCm image + apertus15_text plugin (PY_PRELUDE),
# stock vLLM server flags, ROCR_VISIBLE_DEVICES unset (env.sh), runs under
# beverin-feasibility/runs, 24 h links (mi300 limit), and a 10 s memory log
# per node (GPUs and host share one ~501 GB pool on MI300A).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
: "${RUN_ID:?set RUN_ID}"
NODES="${NODES:-16}"
TIME="${TIME:-24:00:00}"
CHAIN="${CHAIN:-1}"
EDF="${EDF:-${REPO_ROOT}/methods/dspark/beverin/env-vllm031.toml}"
LOG_ROOT="${WORK}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOG_ROOT}"

# Clariden 16-node settings (opb-thinkoff-16n), overridable for pilots.
export DSPARK_DIR="${WORK}" SPECULATORS="${SPECULATORS:-${WORK}/speculators}"
export SIDE_DEPS="${SIDE_DEPS:-${WORK}/py312/pydeps}"
export DATA="${DATA:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/data}"
export RUN_ID SEQ="${SEQ:-8192}" ANCHORS="${ANCHORS:-512}" LR="${LR:-6e-4}" EPOCHS="${EPOCHS:-10}"
export CKPT_FREQ="${CKPT_FREQ:-0.0625}" DSPARK_DTENSOR_ALLGATHER="${DSPARK_DTENSOR_ALLGATHER:-1}"
export LAYERS="${LAYERS:-1 20 39 58 77}"
export EXTRACT_MEM="${EXTRACT_MEM:-0.62}" EXTRACT_FLAGS="${EXTRACT_FLAGS-}"
export PY_PRELUDE="${REPO_ROOT}/methods/dspark/beverin/apertus15_run.py"

pass_env() {
  local v
  for v in DSPARK_DIR SPECULATORS SIDE_DEPS DATA RUN_ID SEQ ANCHORS LR EPOCHS CKPT_FREQ DSPARK_DTENSOR_ALLGATHER \
           LAYERS EXTRACT_MEM EXTRACT_FLAGS PY_PRELUDE MAX_STEPS MAX_ATTEMPTS WORKERS PREFETCH STALL_MINUTES \
           PYTORCH_HIP_ALLOC_CONF; do
    if [ -n "${!v+x}" ]; then printf 'export %s=%q\n' "${v}" "${!v}"; fi
  done
}

IFS=: read -r th tm ts <<< "${TIME}"
LIMIT_S=$(( 10#${th} * 3600 + 10#${tm} * 60 + 10#${ts} ))
SCRIPT="${LOG_ROOT}/submit-train-${RUN_ID}-${STAMP}.sbatch"
cat > "${SCRIPT}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=dspark-70b-${RUN_ID}
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --time=${TIME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --output=${LOG_ROOT}/%x-%j.out
set -euo pipefail
export REPO_ROOT=${REPO_ROOT}
$(pass_env)
echo "START \$(date -u -Iseconds) job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} edf=${EDF}"
cat "\${REPO_ROOT}/REVISION" 2> /dev/null || true
sha256sum "$(sed -n 's|^image = "\(.*\)"|\1|p' "${EDF}")" 2> /dev/null | cut -c1-64 | sed 's/^/image sha256 /' &
# The trainer saves and exits STOP_MARGIN s before the time limit
# (speculators-deadline-stop.patch).
export DSPARK_STOP_AT=\$(( \$(date +%s) + ${LIMIT_S} - ${STOP_MARGIN:-2400} ))
export HEAD_NODE=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -1)
srun --ntasks-per-node=1 --environment=${EDF} --export=ALL bash -c '
  . "\${REPO_ROOT}/methods/dspark/beverin/env.sh"
  ( while sleep 10; do echo "\$(date -u +%FT%TZ) \$(free -g | awk "/Mem:/ {print \"used_gb=\" \\\$3 \" avail_gb=\" \\\$7}")"; done ) \\
    > "\${WORK}/logs/mem-\${SLURM_JOB_ID}-node\${SLURM_NODEID}.log" &
  exec bash "\${REPO_ROOT}/methods/dspark/train/train-node.sh"'
SBATCH

dep="${DEPENDENCY:-}"
for i in $(seq 1 "${CHAIN}"); do
  # shellcheck disable=SC2086
  id="$(sbatch --parsable ${dep:+--dependency=${dep}} ${SBATCH_EXTRA:-} "${SCRIPT}")"
  echo "submitted ${id} (${i}/${CHAIN}) ${SCRIPT}"
  dep="afterany:${id}"
done
