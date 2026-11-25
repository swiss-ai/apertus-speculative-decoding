#!/usr/bin/env bash
# Domain drafters (A3), EAGLE grid: submit the pilot smoke or the seven arms.
# Run on a Clariden login node from the cluster checkout (orchestration only).
#   methods/eagle/launch/domain/submit-grid.sh pilot   # 4-step smoke on the pilot slice
#   methods/eagle/launch/domain/submit-grid.sh full    # one 1-node job per arm
#   methods/eagle/launch/domain/submit-grid.sh status  # which READY markers exist
#   methods/eagle/launch/domain/submit-grid.sh export-smoke  # 4 steps on OPB rows, no traces
# Traces come from the domain-drafters data branch (contract: traces/<domain>/
# {train,dev}.jsonl with a READY-<domain> marker; traces/pilot/<domain>/ with
# READY-pilot). The mixed arm takes an equal share of every domain's first rows,
# interleaved round-robin, 8,000 rows in total (the per-domain arms' size).
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../../.." && pwd)"
MODE="${1:?pilot|full|status|export-smoke}"
DATA="${DOMAIN_DATA:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters}"
TRACES="${DATA}/traces"
DOMAINS=(math code chat summarization multilingual tool)
CONFIGS="${REPO_ROOT}/methods/eagle/configs/8b/domain"
export EAGLE_CUDA_DEVICES=0,1,2,3
export EAGLE_TRAIN_PARTITION="${EAGLE_TRAIN_PARTITION:-normal}"
# HOME is at its quota: keep the patched-vLLM overlay on scratch.
export EAGLE_VLLM_PATCH_DIR="${EAGLE_VLLM_PATCH_DIR:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters/eagle/vllm-patch-train}"
export SML_ENV_DIR="${SML_ENV_DIR:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters/eagle/sml-env}"

ready() {  # domain -> 0 when its marker and files exist
  local d="$1"
  { [ -e "${TRACES}/READY-${d}" ] || [ -e "${TRACES}/${d}/READY-${d}" ]; } &&
    [ -s "${TRACES}/${d}/train.jsonl" ] && [ -s "${TRACES}/${d}/dev.jsonl" ]
}
pilot_ready() {
  { [ -e "${TRACES}/pilot/READY-pilot" ] || [ -e "${TRACES}/READY-pilot" ]; } || return 1
  for d in "${DOMAINS[@]}"; do [ -s "${TRACES}/pilot/${d}/dev.jsonl" ] || return 1; done
  [ -s "${TRACES}/pilot/math/train.jsonl" ]
}
[ -s "${DATA}/opb/heldout.jsonl" ] || { echo "missing ${DATA}/opb/heldout.jsonl (run opb-heldout.sh)" >&2; exit 1; }
[ -s "${DATA}/init/eagle-e10-thinkoff-hf/model.safetensors" ] || { echo "missing e10 export" >&2; exit 1; }

case "${MODE}" in
status)
  pilot_ready && echo "pilot: ready" || echo "pilot: not ready"
  for d in "${DOMAINS[@]}"; do ready "${d}" && echo "${d}: ready" || echo "${d}: not ready"; done
  ;;
export-smoke)
  EAGLE_TRAIN_CONFIG="${CONFIGS}/domain-export-smoke.yaml" EAGLE_TRAIN_TIME="${EAGLE_TRAIN_TIME:-01:00:00}" \
    "${REPO_ROOT}/methods/eagle/launch/submit-eagle-train.sh"
  ;;
pilot)
  pilot_ready || { echo "pilot slice not ready" >&2; exit 3; }
  EAGLE_TRAIN_CONFIG="${CONFIGS}/domain-pilot-smoke.yaml" EAGLE_TRAIN_TIME="${EAGLE_TRAIN_TIME:-01:00:00}" \
    "${REPO_ROOT}/methods/eagle/launch/submit-eagle-train.sh"
  ;;
full)
  for d in "${DOMAINS[@]}"; do ready "${d}" || { echo "${d} traces not ready" >&2; exit 3; }; done
  mixed="${DATA}/eagle/mixed/train-n8000.jsonl"
  if [ ! -s "${mixed}" ]; then
    mkdir -p "$(dirname "${mixed}")"
    python3 - "${mixed}" "${DOMAINS[@]/#/${TRACES}/}" <<'PY'
import hashlib, json, sys
out, dirs = sys.argv[1], sys.argv[2:]
total, n = 8000, len(dirs)
shares = [total // n + (i < total % n) for i in range(n)]
pools = []
for d, k in zip(dirs, shares):
    with open(f"{d}/train.jsonl") as f:
        rows = [line if line.endswith("\n") else line + "\n" for _, line in zip(range(k), f)]
    if len(rows) != k:
        raise SystemExit(f"{d}: {len(rows)} rows, wanted {k}")
    pools.append(rows)
with open(out, "w") as f:
    for i in range(max(shares)):
        for rows in pools:
            if i < len(rows):
                f.write(rows[i])
manifest = {"rows": total, "shares": dict(zip(dirs, shares)), "order": "round-robin by row index",
            "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
open(out + ".manifest.json", "w").write(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
PY
  fi
  for arm in "${DOMAINS[@]}" mixed; do
    EAGLE_TRAIN_CONFIG="${CONFIGS}/domain-${arm}-n8k.yaml" EAGLE_TRAIN_TIME="${EAGLE_TRAIN_TIME:-04:00:00}" \
      "${REPO_ROOT}/methods/eagle/launch/submit-eagle-train.sh"
  done
  ;;
*) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
