#!/usr/bin/env bash
# Compute-node entry for the expansion corpus. Login-node Python is not used.
set -euo pipefail
: "${REPO_ROOT:?}" "${WORK:?}"
# shellcheck disable=SC1091
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
OUT="${EXPANSION_OUT:-${WORK}/expansion/alps-1epoch-20261031}"
BENCH="${OUT}/benchmarks"
mkdir -p "${BENCH}"
PREV_BENCH="${WORK}/expansion/alps-1epoch-20261031/benchmarks/prompts.jsonl"
PREV_PLAN="${WORK}/expansion/alps-1epoch-20261031/sample-plan.json"
if [ -s "${PREV_BENCH}" ]; then
  cp "${PREV_BENCH}" "${BENCH}/prompts.jsonl"
  echo "reused benchmark prompts $(wc -l < "${BENCH}/prompts.jsonl")"
else
python3 - "${BENCH}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
prompts = []
def take(row):
    for key in ("question", "prompt", "text", "instruction", "problem"):
        value = row.get(key) if isinstance(row, dict) else None
        if isinstance(value, str) and value.strip():
            prompts.append(value.strip())
            return
roots = [out, Path("/users/faruk_zahiragic/.cache/huggingface/hub")]
files = []
for root in roots:
    if root.is_dir():
        files.extend(root.rglob("*.jsonl"))
        files.extend(root.rglob("*.parquet"))
needles = ("gsm8k", "mbpp", "humaneval", "math500", "mt-bench", "aime", "benchmark")
for path in files:
    if not any(needle in str(path).lower() for needle in needles):
        continue
    try:
        if path.suffix == ".jsonl":
            for line in path.open(encoding="utf-8", errors="replace"):
                if line.strip():
                    take(json.loads(line))
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    take(row)
    except Exception as exc:
        print("skip", path, exc)
dest = out / "prompts.jsonl"
with dest.open("w", encoding="utf-8") as handle:
    for index, text in enumerate(prompts):
        handle.write(json.dumps({"id": "bench-%d" % index, "prompt": text}) + "\n")
print("benchmark prompts", len(prompts))
if not prompts:
    raise SystemExit(2)
PY
fi
build_args=(
  --sft-dir /capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1/data
  --opb-jsonl "/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/open-perfectblend_apertus15-70b_thinking-off_seed0.jsonl"
  --finepdf-root /capstor/store/cscs/swissai/infra01/datasets
  --tokenizer /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B
  --output "${OUT}"
  --require-benchmarks
  --benchmark-jsonl "${BENCH}/prompts.jsonl"
)
if [ -s "${PREV_PLAN}" ]; then
  build_args+=(--plan-json "${PREV_PLAN}")
fi
python3 "${REPO_ROOT}/methods/dspark/beverin/data_expansion/build.py" "${build_args[@]}"
echo "EXPANSION_BUILD_DONE ${OUT}"
