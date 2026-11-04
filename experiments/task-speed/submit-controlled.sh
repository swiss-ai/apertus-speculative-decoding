#!/usr/bin/env bash
# Submit the controlled per-task speed run (run-controlled.py) on one 4-GPU node.
# Run on a Clariden login node from the task-speed checkout on scratch (the
# serving container mounts /iopsstor and /capstor, not /users):
#   experiments/task-speed/submit-controlled.sh
# Extension pairs for inconclusive cells:
#   TASK_SPEED_CORPUS=<first run>/corpus.jsonl TASK_SPEED_SELECT=<select.json> \
#   TASK_SPEED_FIRST_PAIR=5 experiments/task-speed/submit-controlled.sh
# Pipeline smoke on debug (2 slices, 2 prompts, 2 GPUs, minutes):
#   TASK_SPEED_SUITE=experiments/task-speed/smoke-suite.yaml TASK_SPEED_PARTITION=debug \
#   TASK_SPEED_TIME=01:00:00 TASK_SPEED_GPUS=0,1 experiments/task-speed/submit-controlled.sh
# Only stages inputs and submits; VALIDATE_ONLY=1 writes job.sh without submitting.
set -euo pipefail

CODE="$(cd "$(dirname "$0")/../.." && pwd)"
SUITE="${TASK_SPEED_SUITE:-${CODE}/experiments/task-speed/suite.yaml}"
# The pilot whose image, overlays, chat template and checkpoints the live
# Kubernetes Deployment was frozen from (serving/kubernetes/prepare.py BENCH).
PILOT="${TASK_SPEED_PILOT:-/iopsstor/scratch/cscs/${USER}/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-12h-20261029-a}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${TASK_SPEED_OUTPUT:-${CODE}/results/8b/task-speed/controlled-${STAMP}}"
INPUTS="${TASK_SPEED_INPUTS:-${CODE}/workloads/8b/task-speed-input}"
PARTITION="${TASK_SPEED_PARTITION:-normal}"
TIME_LIMIT="${TASK_SPEED_TIME:-06:00:00}"
ACCOUNT="${TASK_SPEED_ACCOUNT:-infra01}"
GPUS="${TASK_SPEED_GPUS:-0,1,2,3}"
FIRST_PAIR="${TASK_SPEED_FIRST_PAIR:-1}"

case "${CODE}" in
  /iopsstor/* | /capstor/*) ;;
  *) echo "run from a checkout on /iopsstor or /capstor; the container lacks ${CODE}" >&2; exit 2 ;;
esac
test -f "${PILOT}/environment.toml" || { echo "missing pilot ${PILOT}" >&2; exit 2; }
if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then
  echo "tracked changes in ${CODE}; commit and sync first" >&2
  exit 2
fi
BRANCH="$(git -C "${CODE}" branch --show-current)"
REVISION="$(git -C "${CODE}" rev-parse HEAD)"

if [ -n "${TASK_SPEED_CORPUS:-}" ]; then
  SOURCE_ARGS="--corpus ${TASK_SPEED_CORPUS}"
else
  # Stage the frozen sources next to the code; build_corpus checks every hash.
  mkdir -p "${INPUTS}"
  python3 - "${SUITE}" "${INPUTS}" <<'PY'
import hashlib, shutil, sys
from pathlib import Path
import yaml
suite, inputs = yaml.safe_load(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
for name, source in suite["sources"].items():
    target = inputs / source["file"]
    if not target.exists():
        shutil.copy2(source["origin"], target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        sys.exit(f"{name}: {digest} != frozen {source['sha256']}")
    print(f"{name}: {target} ok")
PY
  SOURCE_ARGS="--inputs ${INPUTS}"
fi
SELECT_ARGS=""
if [ -n "${TASK_SPEED_SELECT:-}" ]; then SELECT_ARGS="--select ${TASK_SPEED_SELECT}"; fi

mkdir -p "$(dirname "${OUT}")"
LAUNCH="${OUT}.launch"
mkdir "${LAUNCH}"
cat > "${LAUNCH}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=task-speed
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${LAUNCH}/job.out
set -euo pipefail
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm \
  --format=csv,noheader,nounits -lms 1000 > "${LAUNCH}/gpu.csv" 2> /dev/null &
SAMPLER=\$!
# Extend the image's PYTHONPATH (it carries /workspace/vllm), never replace it.
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --environment="${PILOT}/environment.toml" \
  env PYTHONNOUSERSITE=1 TASK_SPEED_BRANCH="${BRANCH}" TASK_SPEED_REVISION="${REVISION}" bash -c \
  'export PYTHONPATH="${CODE}/src:\${PYTHONPATH:-}"; exec python3 "${CODE}/experiments/task-speed/run-controlled.py" --suite "${SUITE}" --pilot "${PILOT}" ${SOURCE_ARGS} ${SELECT_ARGS} --gpus ${GPUS} --first-pair ${FIRST_PAIR} --output "${OUT}"' \
  || status=\$?
kill \${SAMPLER} 2> /dev/null || true
exit \${status:-0}
EOF
printf '%s\n' "branch=${BRANCH}" "revision=${REVISION}" "pilot=${PILOT}" "output=${OUT}" \
  > "${LAUNCH}/launch.txt"
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  echo "validate_only=1; job script in ${LAUNCH}/job.sh"
  exit 0
fi
JOB="$(sbatch --parsable "${LAUNCH}/job.sh")"
echo "job=${JOB}" >> "${LAUNCH}/launch.txt"
echo "job=${JOB} output=${OUT}"
