#!/usr/bin/env bash
# On-node driver of the workload distribution-shape study (serving image, 4 GPUs).
#   experiments/workload-shape/run-shape.sh RUN_DIR
# Stages (each skipped when its output exists, so a rerun resumes):
#   inputs   rebuild coverage.jsonl (ddb921fe...) from the pinned raw sources and the
#            28-slice task-speed corpus (7d8e60ef...); both are checked against frozen digests;
#   capture  plain vLLM greedy continuations, C=1, 512-token cap, one corpus shard per GPU;
#   shape    HF teacher forcing: target + DSpark e10 + EAGLE e10 distributions per token;
#   aggregate per-workload quantiles, entropy bins, greedy-chain replay at k=1/3/5/7.
# Env: RAW (coverage raw sources), EAGLE_HEAD, DSPARK, LIMIT (rows per capture shard, 0 = all),
#      STAGES (default "inputs capture shape aggregate").
# Other targets (defaults reproduce the Apertus 8B run; see docs/workload-distribution-shape-qwen3-8b.md):
#   TARGET, APERTUS_EAGLE_CONTRACT   target checkpoint and its contract (aux layers, EOS, render)
#   CORPUS_FROM   directory with an already built corpus.jsonl / corpus-measured.jsonl; copied
#                 and checked against the frozen corpus digest instead of rebuilding it
#   SHAPE_ARGS    extra apertus_tree.shape args (e.g. --eagle-format speculators
#                 --block-format dflash --names dflash,eagle3 --depth 15 --dspark-hf-layers ...)
#   AGG_ARGS      extra shape_aggregate args (e.g. --sampled)
#   MAX_MODEL_LEN capture engine context (32768)
set -euo pipefail
RUN_DIR="$1"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
TREE_PYDEPS="${TREE_PYDEPS:-/users/faruk_zahiragic/apertus-spec-decoding/scratch/pydeps}"
export PYTHONPATH="${REPO_ROOT}/methods/tree:${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src:${TREE_PYDEPS}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export APERTUS_EAGLE_CONTRACT="${APERTUS_EAGLE_CONTRACT:-${REPO_ROOT}/targets/8b/contract.json}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf_home}"
export VLLM_CACHE_ROOT=/tmp/shape-cache/vllm TRITON_CACHE_DIR=/tmp/shape-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/shape-cache/inductor XDG_CACHE_HOME=/tmp/shape-cache/xdg
# Home is over quota: FlashInfer's JIT locks and vLLM's usage stats default to it.
export FLASHINFER_WORKSPACE_BASE=/tmp/shape-cache/flashinfer VLLM_CONFIG_ROOT=/tmp/shape-cache/vllm-config
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
W=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape
RAW="${RAW:-${W}/raw}"
EAGLE_HEAD="${EAGLE_HEAD:-${W}/heads/eagle-e10}"
DSPARK="${DSPARK:-/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final}"
WORKLOADS="${WORKLOADS:-/users/faruk_zahiragic/apertus-spec-decoding/workloads/8b}"
TARGET="${TARGET:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B}"
CORPUS_FROM="${CORPUS_FROM:-}"
SHAPE_ARGS="${SHAPE_ARGS:-}"
AGG_ARGS="${AGG_ARGS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
INPUTS="${REPO_ROOT}/results/8b/workload-shape/inputs"
LIMIT="${LIMIT:-0}"
STAGES="${STAGES:-inputs capture shape aggregate}"
NG=4
mkdir -p "${RUN_DIR}"/{inputs,captures,shape,logs}
stamp() { echo "=== [$(date -u +%FT%TZ)] $*"; }

python3 - "${RUN_DIR}" <<'PY'
import json, os, subprocess, sys, time
run = sys.argv[1]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
job = os.environ.get("SLURM_JOB_ID", "local")
json.dump({
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_revision": git("rev-parse", "HEAD"),
    # Cluster copies made with git archive carry the source commit in SOURCE_REVISION.
    "source_revision": open("SOURCE_REVISION").read().strip() if os.path.exists("SOURCE_REVISION") else None,
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
    "env": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
            "SLURM_JOB_ACCOUNT", "EAGLE_HEAD", "DSPARK", "RAW", "LIMIT", "STAGES", "SHAPE_IMAGE")},
    **({"target_env": {k: os.environ.get(k) for k in ("TARGET", "APERTUS_EAGLE_CONTRACT",
        "CORPUS_FROM", "SHAPE_ARGS", "AGG_ARGS", "MAX_MODEL_LEN")}}
       if os.environ.get("CORPUS_FROM") or os.environ.get("SHAPE_ARGS") else {}),
}, open(os.path.join(run, f"provenance-{job}.json"), "w"), indent=1)
PY
if git rev-parse --git-dir > /dev/null 2>&1; then
  git diff HEAD > "${RUN_DIR}/logs/uncommitted-${SLURM_JOB_ID:-local}.patch" || true
fi
if [ -n "${SHAPE_IMAGE:-}" ] && [ ! -s "${RUN_DIR}/logs/image.sha256" ]; then
  (sha256sum "${SHAPE_IMAGE}" > "${RUN_DIR}/logs/image.sha256.tmp" \
    && mv "${RUN_DIR}/logs/image.sha256.tmp" "${RUN_DIR}/logs/image.sha256") &
fi
python3 -m pytest -q tests/test_shape.py > "${RUN_DIR}/logs/pytest-${SLURM_JOB_ID:-local}.txt" 2>&1 \
  || { cat "${RUN_DIR}/logs/pytest-${SLURM_JOB_ID:-local}.txt"; exit 1; }

if [[ " ${STAGES} " == *" inputs "* ]] && [ ! -s "${RUN_DIR}/inputs/corpus-measured.jsonl" ] \
    && [ -n "${CORPUS_FROM}" ]; then
  stamp "stage inputs (copy ${CORPUS_FROM})"
  cp "${CORPUS_FROM}/corpus.jsonl" "${CORPUS_FROM}/corpus-measured.jsonl" "${RUN_DIR}/inputs/"
  python3 - "${INPUTS}/corpus-manifest.json" "${RUN_DIR}/inputs" "${NG}" <<'PY'
import hashlib, json, sys
from pathlib import Path
frozen, d, n = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), int(sys.argv[3])
digest = hashlib.sha256((d / "corpus.jsonl").read_bytes()).hexdigest()
if digest != frozen["corpus_sha256"]:
    sys.exit(f"corpus digest {digest} != frozen {frozen['corpus_sha256']}")
# The measured rows are the corpus without its pre-warm rows, in order.
rows = (d / "corpus.jsonl").read_text().splitlines(keepends=True)
measured = "".join(r for r in rows if json.loads(r)["workload"] != "prewarm")
if (d / "corpus-measured.jsonl").read_text() != measured:
    sys.exit("corpus-measured.jsonl is not the corpus without its pre-warm rows")
lines = measured.splitlines(keepends=True)
for i in range(n):
    (d / f"corpus-shard-{i}.jsonl").write_text("".join(lines[i::n]))
print(json.dumps({"corpus_sha256": digest, "measured_rows": len(lines)}))
PY
fi

if [[ " ${STAGES} " == *" inputs "* ]] && [ ! -s "${RUN_DIR}/inputs/corpus-measured.jsonl" ]; then
  stamp "stage inputs"
  python3 -m apertus_bench.coverage_workloads --raw "${RAW}" --tokenizer "${TARGET}" \
    --out "${RUN_DIR}/inputs/coverage.jsonl" --manifest "${RUN_DIR}/inputs/coverage-manifest.json" \
    > "${RUN_DIR}/logs/coverage-build.log" 2>&1
  for f in opb-heldout.jsonl eagle-8b-test.jsonl probe-speculator-benchmarks.jsonl arena-hard.jsonl; do
    cp "${WORKLOADS}/${f}" "${RUN_DIR}/inputs/${f}"
  done
  # build_corpus checks every source digest (coverage ddb921fe... included) and the corpus digest.
  python3 experiments/workload-shape/build_corpus.py "${INPUTS}/suite.yaml" \
    "${INPUTS}/corpus-manifest.json" "${RUN_DIR}/inputs" "${RUN_DIR}/inputs/corpus.jsonl"
  python3 - "${RUN_DIR}/inputs" "${NG}" <<'PY'
import sys
from pathlib import Path
d, n = Path(sys.argv[1]), int(sys.argv[2])
lines = (d / "corpus-measured.jsonl").read_text().splitlines(keepends=True)
for i in range(n):
    (d / f"corpus-shard-{i}.jsonl").write_text("".join(lines[i::n]))
PY
fi

if [[ " ${STAGES} " == *" capture "* ]]; then
  stamp "stage capture"
  for i in $(seq 0 $((NG - 1))); do
    out="${RUN_DIR}/captures/shard-${i}.json"
    [ -s "${out}" ] && { echo "exists: ${out}"; continue; }
    CUDA_VISIBLE_DEVICES=${i} python3 -m apertus_eagle.verify_offline capture --mode plain \
      --prompts "${RUN_DIR}/inputs/corpus-shard-${i}.jsonl" --limit "${LIMIT}" --max-tokens 512 \
      --cap-sweep 0 --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization 0.85 \
      --output "${out}" > "${RUN_DIR}/logs/capture-${i}.log" 2>&1 &
  done
  wait
  for i in $(seq 0 $((NG - 1))); do
    test -s "${RUN_DIR}/captures/shard-${i}.json" || { tail -40 "${RUN_DIR}/logs/capture-${i}.log"; exit 1; }
  done
fi

if [[ " ${STAGES} " == *" shape "* ]]; then
  stamp "stage shape"
  pids=()
  for i in $(seq 0 $((NG - 1))); do
    out="${RUN_DIR}/shape/shard-${i}.npz"
    [ -s "${RUN_DIR}/shape/shard-${i}.summary.json" ] && continue
    CUDA_VISIBLE_DEVICES=${i} python3 -m apertus_tree.shape \
      --capture "${RUN_DIR}"/captures/shard-*.json --prompts "${RUN_DIR}/inputs/corpus-measured.jsonl" \
      --eagle-head "${EAGLE_HEAD}" --dspark "${DSPARK}" --shard "${i}" --num-shards "${NG}" \
      ${SHAPE_ARGS} --output "${out}" > "${RUN_DIR}/logs/shape-${i}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [ "${fail}" = 0 ] || { tail -40 "${RUN_DIR}"/logs/shape-*.log; exit 1; }
fi

if [[ " ${STAGES} " == *" aggregate "* ]]; then
  stamp "stage aggregate"
  python3 -m apertus_tree.shape_aggregate "${RUN_DIR}"/shape/shard-*.npz --suite "${INPUTS}/suite.yaml" \
    --output "${RUN_DIR}/aggregate.json" --sequences "${RUN_DIR}/sequences.csv" ${AGG_ARGS}
fi
wait
stamp "done"
