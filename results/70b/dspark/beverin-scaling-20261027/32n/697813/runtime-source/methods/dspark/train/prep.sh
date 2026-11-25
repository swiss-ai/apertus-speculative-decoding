#!/usr/bin/env bash
# One-off preparation inside the container: speculators side packages, the
# open-perfectblend revision, the prompt shards (checked against Yu's manifest)
# and the regeneration parts. Rerunning skips what exists.
#   REPO_ROOT=... NPARTS=32 bash methods/dspark/train/prep.sh
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/train/common.sh"
NPARTS="${NPARTS:-32}"
PROMPTS="${DSPARK_DIR}/prompts"

stamp "side deps into ${SIDE_DEPS}"
install_side_deps
check_side_deps

if [ ! -f "${PROMPTS}/MANIFEST.json" ]; then
  stamp "download ${OPB_REPO}@${OPB_REVISION}"
  OPB_DIR="$(HF_HUB_OFFLINE=0 python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${OPB_REPO}', repo_type='dataset', revision='${OPB_REVISION}'))")"
  stamp "shards from ${OPB_DIR}"
  python3 "${REPO_ROOT}/methods/dspark/train/make_opb_shards.py" --opb-dir "${OPB_DIR}" \
    --out "${PROMPTS}" --shards 4 --parts "${NPARTS}" \
    --manifest "${CORPUS_8B_DIR}/open-perfectblend_prompts_MANIFEST.json"
fi
ls "${PROMPTS}/parts" | wc -l
stamp "PREP_OK"
