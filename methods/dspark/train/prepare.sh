#!/usr/bin/env bash
# Build the speculators training dataset inside the container (Yu's
# apertus_opb_prepare.sh): prepare-data packs at 8,192 tokens.
#   STAGE=8b MODE=prepare OUT=dir submit-train.sh   # Yu's 8B corpus, split checked
#   IN=answers.jsonl OUT=dir bash prepare.sh        # any regenerated jsonl (pilot)
#   FULL=1 bash prepare.sh                          # the whole corpus
# FULL=1 first merges the parts so that the trainer holds out the 8B run's
# validation conversations (merge_corpus.py), checks the prepared dataset with
# Yu's val_ids.py, and copies the corpus to the shared store.
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
    --val-ids "${CORPUS_8B_DIR}/val_ids.txt" --out "${IN}" --missing "${DSPARK_DIR}/missing-ids.txt" \
    ${ALLOW_MISSING:+--allow-missing}
fi
: "${IN:?}" "${OUT:?}"
stamp "prepare-data ${IN} -> ${OUT}"
python3 -m speculators prepare-data --model "${TARGET_MODEL}" --trust-remote-code --data "${IN}" \
  --output "${OUT}" --token-freq-path "${OUT}/token_freq.pt" --no-skip-token-freq --seq-length 8192
cat "${OUT}/dataset_info.json" 2> /dev/null | head -c 400 || true
echo

# A corpus that already has a held-out list (Yu's 8B one): the prepared
# dataset must hold out exactly those rows, as in his run.
if [ "${CHECK_SPLIT:-0}" = 1 ] && [ "${FULL:-0}" != 1 ]; then
  stamp "held-out split"
  python3 "$(dirname "${IN}")/val_ids.py" "${IN}" "${OUT}" "${OUT}"
  if cmp -s <(sort "${OUT}/val_ids.txt") <(sort "$(dirname "${IN}")/val_ids.txt"); then
    stamp "SPLIT_VERIFIED: $(wc -l < "${OUT}/val_ids.txt") held out, as in $(dirname "${IN}")/val_ids.txt"
  else
    stamp "SPLIT_MISMATCH: the prepared dataset does not hold out the rows of $(dirname "${IN}")/val_ids.txt"
    exit 1
  fi
fi

if [ "${FULL:-0}" = 1 ]; then
  stamp "held-out split"
  python3 "${CORPUS_8B_DIR}/val_ids.py" "${IN}" "${OUT}" "${DSPARK_DIR}"
  shared=$(comm -12 <(sort "${DSPARK_DIR}/val_ids.txt") <(sort "${CORPUS_8B_DIR}/val_ids.txt") | wc -l)
  if cmp -s <(sort "${DSPARK_DIR}/val_ids.txt") <(sort "${IN}.heldout.txt"); then
    stamp "SPLIT_VERIFIED: $(wc -l < "${DSPARK_DIR}/val_ids.txt") held out, ${shared} of them in the 8B run's held-out set"
  else
    stamp "SPLIT_MISMATCH: the prepared dataset does not hold out the planned rows"
    exit 1
  fi
  mkdir -p "${STORE_DIR}"
  cp "${IN}" "${DSPARK_DIR}/val_ids.txt" "${DSPARK_DIR}/val_rows.tsv" "${DSPARK_DIR}/missing-ids.txt" \
    "${DSPARK_DIR}/prompts/MANIFEST.json" "${STORE_DIR}/"
  chmod -R g+rwX "${STORE_DIR}"
  stamp "corpus copied to ${STORE_DIR}"
fi
stamp "PREPARE_OK"
