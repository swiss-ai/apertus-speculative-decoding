#!/usr/bin/env bash
# Domain drafters (A3): rebuild e10's open-perfectblend held-out eval rows after the
# scratch cleaner removed apertus-eagle/data/dspark-opb-thinking-off/heldout.jsonl.
# speculators_corpus split writes held-out rows in corpus order, so the first N
# held-out rows are the first N corpus rows whose id is in val_ids.txt. Reading
# stops after N matches (~19k rows for N=2000), light enough for a login node.
set -euo pipefail
SRC="${OPB_SRC:-/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-thinking-off}"
OUT="${OPB_HELDOUT_OUT:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters/opb/heldout.jsonl}"
N="${OPB_HELDOUT_ROWS:-2000}"
mkdir -p "$(dirname "${OUT}")"
python3 - "${SRC}/open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl" "${SRC}/val_ids.txt" "${OUT}" "${N}" <<'PY'
import json, sys
corpus, ids_path, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ids = {line.strip() for line in open(ids_path) if line.strip()}
kept = scanned = 0
with open(corpus) as src, open(out, "w") as dst:
    for line in src:
        scanned += 1
        if str(json.loads(line).get("id")) in ids:
            dst.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            if kept == n:
                break
print(json.dumps({"kept": kept, "scanned": scanned, "out": out}))
PY
sha256sum "${OUT}"
