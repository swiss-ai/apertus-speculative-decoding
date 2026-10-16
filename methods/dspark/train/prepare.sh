#!/usr/bin/env bash
# Build the speculators training dataset inside the container (Yu's
# apertus_opb_prepare.sh): prepare-data packs at 8,192 tokens.
#   IN=answers.jsonl OUT=dir bash prepare.sh        # any regenerated jsonl (pilot)
#   FULL=1 bash prepare.sh                          # the whole corpus
# FULL=1 first merges the parts in the 8B corpus's row order, then checks with
# Yu's val_ids.py that the trainer holds out the 8B run's conversations, and
# copies the corpus to the shared store.
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/train/common.sh"
check_side_deps
STORE_DIR="/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-70b-thinking-off"
CORPUS="open-perfectblend_apertus15-70b_thinking-off_seed0.jsonl"

if [ "${FULL:-0}" = 1 ]; then
  IN="${DSPARK_DIR}/${CORPUS}"
  OUT="${DSPARK_DIR}/data"
  stamp "merge"
  python3 "${REPO_ROOT}/methods/dspark/train/merge_corpus.py" \
    --parts "${DSPARK_DIR}/regen/part-*.jsonl" --order "${CORPUS_8B_DIR}/open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl" \
    --out "${IN}" --missing "${DSPARK_DIR}/missing-ids.txt" ${ALLOW_MISSING:+--allow-missing}
fi
: "${IN:?}" "${OUT:?}"
stamp "prepare-data ${IN} -> ${OUT}"
python3 -m speculators prepare-data --model "${TARGET_MODEL}" --trust-remote-code --data "${IN}" \
  --output "${OUT}" --token-freq-path "${OUT}/token_freq.pt" --no-skip-token-freq --seq-length 8192
cat "${OUT}/dataset_info.json" 2> /dev/null | head -c 400 || true
echo

if [ "${FULL:-0}" = 1 ]; then
  stamp "held-out split"
  python3 "${CORPUS_8B_DIR}/val_ids.py" "${IN}" "${OUT}" "${DSPARK_DIR}"
  if cmp -s "${DSPARK_DIR}/val_ids.txt" "${CORPUS_8B_DIR}/val_ids.txt"; then
    stamp "SPLIT_MATCHES_8B"
  else
    stamp "SPLIT_DIFFERS_FROM_8B ($(comm -12 <(sort "${DSPARK_DIR}/val_ids.txt") <(sort "${CORPUS_8B_DIR}/val_ids.txt") | wc -l) ids shared)"
  fi
  mkdir -p "${STORE_DIR}"
  cp "${IN}" "${DSPARK_DIR}/val_ids.txt" "${DSPARK_DIR}/val_rows.tsv" "${DSPARK_DIR}/prompts/MANIFEST.json" "${STORE_DIR}/"
  chmod -R g+rwX "${STORE_DIR}"
  stamp "corpus copied to ${STORE_DIR}"
fi
stamp "PREPARE_OK"
