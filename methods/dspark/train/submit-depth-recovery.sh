#!/usr/bin/env bash
# Queue the bounded, fail-closed T3 continuation after the replacement training.
# Usage: submit-depth-recovery.sh TRAIN_JOB FRESH_CAMPAIGN_DIRECTORY TRAIN_SUBMIT_SCRIPT
set -euo pipefail
TRAIN_JOB="${1:?training job ID required}"
CAMPAIGN="${2:?fresh absolute campaign directory required}"
TRAIN_SUBMIT="${3:?exact submitted training script required}"
[[ "${TRAIN_JOB}" =~ ^[0-9]+$ && "${CAMPAIGN}" = /* ]]
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
REVISION="$(git -C "${REPO}" rev-parse HEAD)"
export PYTHONDONTWRITEBYTECODE=1
mkdir "${CAMPAIGN}"
mkdir "${CAMPAIGN}/source"
git -C "${REPO}" archive "${REVISION}" src serving methods/dspark/train \
  methods/eagle/configs/train-env.toml targets/8b/contract.json \
  experiments/smoke-70b/experiment.yaml | tar -x -C "${CAMPAIGN}/source"
SOURCE="${CAMPAIGN}/source"
python3 "${SOURCE}/methods/dspark/train/depth_campaign.py" snapshot "${CAMPAIGN}" \
  --training-repo "${REPO}" --revision "${REVISION}" --train-job "${TRAIN_JOB}" \
  --training-submit "${TRAIN_SUBMIT}"

PREVIOUS="${TRAIN_JOB}"
if [ "${DEPTH_TRAINING_COMPLETE:-0}" = 1 ]; then
  sacct -X -j "${TRAIN_JOB}" -n -P -o JobIDRaw,State,ExitCode > "${CAMPAIGN}/training-accounting.txt"
  python3 "${SOURCE}/methods/dspark/train/depth_campaign.py" resume "${CAMPAIGN}" \
    --accounting "${CAMPAIGN}/training-accounting.txt"
  PREVIOUS=""
fi
for STAGE in validate a b; do
  LIMIT=04:00:00
  [ "${STAGE}" != validate ] || LIMIT=02:00:00
  STAGE_LOG="${CAMPAIGN}/${STAGE}-job.out"
  cat > "${CAMPAIGN}/${STAGE}.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=depth-recovery-${STAGE}
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=${LIMIT}
#SBATCH --chdir=${SOURCE}
#SBATCH --output=${STAGE_LOG}
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
trap 'python3 "${SOURCE}/methods/dspark/train/depth_campaign.py" fail "${CAMPAIGN}" --stage "${STAGE}"' ERR
# Pyxis has rejected long overlay mount lists before. The Slurm job ID makes
# this short path private to this allocation; patch-vllm may safely replace it.
export VLLM_PATCH_DIR="/iopsstor/scratch/cscs/${USER}/sd3-\${SLURM_JOB_ID}"
export SML_ENV_DIR="${CAMPAIGN}/environments"
export DEPTH_SOURCE_MANIFEST="${CAMPAIGN}/source-manifest.json"
python3 "${SOURCE}/methods/dspark/train/depth_campaign.py" verify "${CAMPAIGN}"
PATCHES=("${SOURCE}"/serving/patches/vllm-apertus-image-token.patch
  "${SOURCE}"/serving/patches/vllm-apertus-eagle3-aux-layers.patch
  "${SOURCE}"/serving/patches/vllm-apertus-eagle-mrv2-inner.patch
  "${SOURCE}"/serving/patches/vllm-apertus-dspark-mrv2-inner.patch
  "${SOURCE}"/serving/patches/vllm-apertus-dspark-anchor-layout.patch
  "${SOURCE}"/serving/patches/vllm-apertus-dspark-num-layers.patch)
EXTRA_MOUNTS="\$("${SOURCE}/serving/patch-vllm.sh" \
  /capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh \
  "\${PATCHES[@]}")"
export EXTRA_MOUNTS
DEPTH_RESOLVED_ENV="\$("${SOURCE}/serving/resolve-env.sh" \
  "${SOURCE}/methods/eagle/configs/train-env.toml")"
export DEPTH_RESOLVED_ENV
cp "\${DEPTH_RESOLVED_ENV}" "${CAMPAIGN}/${STAGE}-environment.toml"
printf '%s\\n' "\${PATCHES[@]}" > "${CAMPAIGN}/${STAGE}-patches.txt"
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000 > "${CAMPAIGN}/${STAGE}-gpu-memory.csv" 2>/dev/null &
SAMPLER=\$!
trap 'kill "\${SAMPLER}" 2>/dev/null || true' EXIT
srun --overlap --nodes=1 --ntasks=1 --environment="\${DEPTH_RESOLVED_ENV}" \
  env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  SWEEP_CACHE_ROOT="${CAMPAIGN}/cache-${STAGE}" \
  bash -c 'cd "${SOURCE}"; export PYTHONPATH="${SOURCE}/src:\${PYTHONPATH:-}"; exec python3 "${SOURCE}/methods/dspark/train/depth_campaign.py" run "${CAMPAIGN}" --stage "${STAGE}"'
EOF
  DEPENDENCY_ARGS=()
  [ -z "${PREVIOUS}" ] || DEPENDENCY_ARGS=(--dependency="afterok:${PREVIOUS}")
  JOB="$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" "${CAMPAIGN}/${STAGE}.sbatch")"
  JOB="${JOB%%;*}"
  python3 - "${CAMPAIGN}" "${STAGE}" "${JOB}" "${PREVIOUS}" <<'PY'
import json
import sys
from pathlib import Path
campaign, stage, job, dependency = sys.argv[1:]
path = Path(campaign) / "campaign.json"
state = json.loads(path.read_text())
state["job_ids"].append(job)
prerequisite = f"afterok:{dependency}" if dependency else f"completed-training:{state['training_job']}"
state.setdefault("stages", {})[stage] = {"job_id": job, "dependency": prerequisite}
state["state"] = "waiting"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  echo "stage=${STAGE} job=${JOB} prerequisite=${PREVIOUS:-completed-training-${TRAIN_JOB}} campaign=${CAMPAIGN}"
  PREVIOUS="${JOB}"
done
echo "state=${CAMPAIGN}/campaign.json source_revision=${REVISION}"
