#!/usr/bin/env bash
# Drafter updates (T5), step B inputs: nested trace subsets and replay files.
#   <data>/adapt-data/<domain>-n<N>.jsonl   first N rows of traces/<domain>/pool.jsonl
#   <data>/adapt-data/opb-replay-n<R>.jsonl first R rows of the e10 training corpus
# (already in the colleague's shuffled order), R = N/4, so replay is 20% of a
# mixed run's rows. Only head and sha256: light enough for the login node.
# Writes adapt-data/manifest.json (counts and digests, no text).
set -euo pipefail

DATA_ROOT="${DRAFTER_DATA_ROOT:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/drafter-updates}"
OPB_TRAIN="${OPB_TRAIN:-/iopsstor/scratch/cscs/${USER}/apertus-eagle/data/dspark-opb-thinking-off/train.jsonl}"
DOMAINS="${DRAFTER_DOMAINS:-chat summarization multilingual}"
SIZES="${DRAFTER_SIZES:-500 2000 8000}"
OUT="${DATA_ROOT}/adapt-data"
mkdir -p "${OUT}"

entries=()
add() {  # name path source rows
  local rows sha
  rows="$(wc -l < "$2")"
  [ "${rows}" -eq "$4" ] || { echo "$2: ${rows} rows, wanted $4" >&2; exit 1; }
  sha="$(sha256sum "$2" | cut -d' ' -f1)"
  entries+=("\"$1\": {\"path\": \"$2\", \"source\": \"$3\", \"rows\": ${rows}, \"sha256\": \"${sha}\"}")
}
for domain in ${DOMAINS}; do
  pool="${DATA_ROOT}/traces/${domain}/pool.jsonl"
  for n in ${SIZES}; do
    target="${OUT}/${domain}-n${n}.jsonl"
    [ -f "${target}" ] || head -n "${n}" "${pool}" > "${target}"
    add "${domain}-n${n}" "${target}" "${pool}" "${n}"
  done
  add "${domain}-dev" "${DATA_ROOT}/traces/${domain}/dev.jsonl" "traces" "$(wc -l < "${DATA_ROOT}/traces/${domain}/dev.jsonl")"
done
for n in ${SIZES}; do
  r=$((n / 4))
  target="${OUT}/opb-replay-n${r}.jsonl"
  [ -f "${target}" ] || head -n "${r}" "${OPB_TRAIN}" > "${target}"
  add "opb-replay-n${r}" "${target}" "${OPB_TRAIN}" "${r}"
done
{
  echo "{"
  printf '  %s' "${entries[0]}"
  for entry in "${entries[@]:1}"; do printf ',\n  %s' "${entry}"; done
  printf '\n}\n'
} > "${OUT}/manifest.json"
python3 -m json.tool "${OUT}/manifest.json" > /dev/null
cat "${OUT}/manifest.json"
