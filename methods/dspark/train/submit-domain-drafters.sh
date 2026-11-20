#!/usr/bin/env bash
# Submit the per-domain DSpark drafters (canvas A3) from the cluster checkout.
#   WHAT=runtime ./methods/dspark/train/submit-domain-drafters.sh # no traces needed: side deps, rebind, e10 on OPB
#   WHAT=smoke ./methods/dspark/train/submit-domain-drafters.sh   # pilot slice, debug, baseline+mixed
#   WHAT=full  [AFTER=<jobid>] ./methods/dspark/train/submit-domain-drafters.sh
# full = one baseline job (rebind, evaluation sets, e10 evaluations), then one
# job per head (6 domains + mixed), each afterok on the baseline. Every submit
# gets its own vLLM overlay directory (patch-vllm.sh deletes and rebuilds it).
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${TRAIN_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
WHAT="${WHAT:?smoke or full}"
DATA_BASE=/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
case "${WHAT}" in
  runtime) PLAN=methods/dspark/configs/8b/domain-drafters-runtime.json
         ROOT="${DOMAIN_ROOT:-${DATA_BASE}/dspark-runtime-${STAMP}}" ;;
  smoke) PLAN=methods/dspark/configs/8b/domain-drafters-pilot.json
         ROOT="${DOMAIN_ROOT:-${DATA_BASE}/dspark-pilot-${STAMP}}" ;;
  full)  PLAN=methods/dspark/configs/8b/domain-drafters.json
         ROOT="${DOMAIN_ROOT:-${DATA_BASE}/dspark-r0-${STAMP}}" ;;
  *) echo "WHAT must be runtime, smoke or full" >&2; exit 2 ;;
esac
mkdir -p "${ROOT}/logs" "${ROOT}/env" "${ROOT}/submissions"
REV="$(cat .source-revision 2> /dev/null || git rev-parse HEAD)"
common=(STAGE=8b MODE=domain DOMAIN_ROOT="${ROOT}" DOMAIN_PLAN="${REPO_ROOT}/${PLAN}"
  SML_ENV_DIR="${ROOT}/env" DSPARK_LOG_ROOT="${ROOT}/logs"
  DOMAIN_SIDE_DEPS="${DATA_BASE}/dspark-runtime/pydeps")
# AFTER only gates while that job is still queued or running: a finished job
# (sbatch rejects afterok on it) is checked here instead, through sacct.
if [ -n "${AFTER:-}" ] && [ -z "$(squeue -h -j "${AFTER}" -o %i 2> /dev/null)" ]; then
  state="$(sacct -n -X -j "${AFTER}" -o State%20 | awk 'NF{print $1; exit}')"
  if [ "${state}" != COMPLETED ]; then echo "AFTER job ${AFTER} ended ${state:-unknown}" >&2; exit 1; fi
  echo "AFTER job ${AFTER} already COMPLETED; no dependency" >&2
  AFTER=""
fi
submit() {  # submit HEADS PARTITION TIME DEPENDENCY -> job id
  local out
  out="$(env "${common[@]}" HEADS="$1" DSPARK_PARTITION="$2" TIME="$3" DEPENDENCY="$4" \
    DSPARK_TRAIN_PATCH_DIR="${ROOT}/vllm-patch-${1// /-}-${STAMP}" \
    ./methods/dspark/train/submit-train.sh)"
  echo "${out}" >&2
  sed -n 's/^submitted \([0-9]*\).*/\1/p' <<< "${out}" | tail -1
}
record() { printf '{"what":"%s","heads":"%s","job":"%s","dependency":"%s","source_revision":"%s","plan":"%s","plan_sha256":"%s","submitted_at":"%s"}\n' \
  "${WHAT}" "$1" "$2" "$3" "${REV}" "${PLAN}" "$(sha256sum "${PLAN}" | cut -d' ' -f1)" "${STAMP}" >> "${ROOT}/submissions/submissions.jsonl"; }

if [ "${WHAT}" = runtime ]; then
  common+=(INSTALL_SIDE_DEPS=1)
  id="$(submit baseline debug 01:30:00 "${AFTER:+afterok:${AFTER}}")"
  record baseline "${id}" "${AFTER:-}"
  echo "runtime ${id} root ${ROOT}"
  exit 0
fi
if [ "${WHAT}" = smoke ]; then
  common+=(TRAIN_MINUTES=15)  # 6 steps; the 90-min full-run bound paused job 3988408
  id="$(submit "baseline mixed" debug 01:30:00 "${AFTER:+afterok:${AFTER}}")"
  record "baseline mixed" "${id}" "${AFTER:-}"
  echo "smoke ${id} root ${ROOT}"
  exit 0
fi
base="$(submit baseline normal 02:30:00 "${AFTER:+afterok:${AFTER}}")"
record baseline "${base}" "${AFTER:-}"
ids=()
for h in math code chat summarization multilingual tool mixed; do
  id="$(submit "${h}" normal 04:00:00 "afterok:${base}")"
  record "${h}" "${id}" "afterok:${base}"
  ids+=("${h}=${id}")
done
echo "full baseline=${base} ${ids[*]} root ${ROOT}"
