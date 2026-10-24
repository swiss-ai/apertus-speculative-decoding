#!/usr/bin/env bash
# On-node driver for the tree-drafter simulation (inside the serving image, 4 GPUs).
#   methods/tree/run-candidates.sh RUN_DIR
# Stages, each skipped when its output exists (a rerun resumes):
#   1. plain vLLM greedy captures of 128 opb-heldout and 128 chat prompts
#      (apertus_eagle.verify_offline capture, C=1, 512-token cap);
#   2. candidates: target hidden states + DSpark/EAGLE/n-gram tries per anchor,
#      one shard per GPU (methods/tree/apertus_tree/candidates.py);
#   3. replay of every policy and budget (apertus_bench.tree_sim) on the CPU.
# Env: LIMIT (sequences per workload, 0 = all), MAX_ANCHORS (per sequence, 0 = all),
# DSPARK_HF_LAYERS, STAGES (default "capture candidates analyze").
set -euo pipefail
RUN_DIR="$1"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
# accelerate (needed by the teacher's device_map) is not in the serving image; the
# EAGLE trainer's side packages provide it. Appended, so the image's own packages win.
TREE_PYDEPS="${TREE_PYDEPS:-${REPO_ROOT}/scratch/pydeps}"
export PYTHONPATH="${REPO_ROOT}/methods/tree:${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src:${TREE_PYDEPS}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# verify_offline and candidates.py read the target from the 8B contract.
export APERTUS_EAGLE_CONTRACT="${APERTUS_EAGLE_CONTRACT:-${REPO_ROOT}/targets/8b/contract.json}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf_home}"
export VLLM_CACHE_ROOT=/tmp/tree-cache/vllm TRITON_CACHE_DIR=/tmp/tree-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/tree-cache/inductor XDG_CACHE_HOME=/tmp/tree-cache/xdg
EAGLE_HEAD="${EAGLE_HEAD:-/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/heads/e31-opb-thinkoff-stage1-se}"
DSPARK="${DSPARK:-/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final}"
COST_TABLE="${COST_TABLE:-results/8b/sweeps/kv-cost-a-20261015T230650Z/dspark-e10-k7-paper-cut/cost-table.json}"
LIMIT="${LIMIT:-0}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"
DSPARK_HF_LAYERS="${DSPARK_HF_LAYERS:-0,7,14,21,28}"
STAGES="${STAGES:-capture candidates analyze}"
declare -A PROMPTS=([opb]=workloads/8b/opb-heldout.jsonl [chat]=workloads/8b/eagle-8b-test.jsonl)
mkdir -p "${RUN_DIR}"/{captures,candidates,logs}
stamp() { echo "=== [$(date -u +%FT%TZ)] $*"; }

python3 - "${RUN_DIR}" "$@" <<'PY'
import json, os, subprocess, sys, time
run = sys.argv[1]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
path = os.path.join(run, f"provenance-{os.environ.get('SLURM_JOB_ID', 'local')}.json")
json.dump({
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_revision": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
    "command": sys.argv[1:],
    "env": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
            "EAGLE_HEAD", "DSPARK", "COST_TABLE", "LIMIT", "MAX_ANCHORS", "DSPARK_HF_LAYERS", "STAGES",
            "TREE_IMAGE")},
}, open(path, "w"), indent=1)
PY
git diff HEAD > "${RUN_DIR}/logs/uncommitted-${SLURM_JOB_ID:-local}.patch" || true
# Image digest, computed while the GPUs work.
if [ -n "${TREE_IMAGE:-}" ] && [ ! -s "${RUN_DIR}/logs/image.sha256" ]; then
  (sha256sum "${TREE_IMAGE}" > "${RUN_DIR}/logs/image.sha256.tmp" && mv "${RUN_DIR}/logs/image.sha256.tmp" "${RUN_DIR}/logs/image.sha256") &
fi

# Does the image's vLLM have token-tree verification? (evidence for the design step)
python3 -c "import os, vllm; print(vllm.__version__, os.path.dirname(vllm.__file__))" > "${RUN_DIR}/logs/vllm-tree-grep.txt" 2>&1 || true
VLLM_DIR="$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
[ -n "${VLLM_DIR}" ] && grep -rlE "speculative_token_tree|TreeAttention|tree_attn" "${VLLM_DIR}" >> "${RUN_DIR}/logs/vllm-tree-grep.txt" 2>/dev/null || true

if [[ " ${STAGES} " == *" capture "* ]]; then
  stamp "stage 1: plain greedy captures"
  gpu=0
  for wl in opb chat; do
    out="${RUN_DIR}/captures/${wl}-plain.json"
    if [ -s "${out}" ]; then echo "exists: ${out}"; continue; fi
    CUDA_VISIBLE_DEVICES=${gpu} python3 -m apertus_eagle.verify_offline capture --mode plain \
      --prompts "${PROMPTS[$wl]}" --limit 128 --max-tokens 512 --cap-sweep 0 \
      --output "${out}" > "${RUN_DIR}/logs/capture-${wl}.log" 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  for wl in opb chat; do test -s "${RUN_DIR}/captures/${wl}-plain.json" || { tail -30 "${RUN_DIR}/logs/capture-${wl}.log"; exit 1; }; done
fi

if [[ " ${STAGES} " == *" candidates "* ]]; then
  stamp "stage 2: candidates"
  pids=()
  for shard in 0 1 2 3; do
    (
      for wl in opb chat; do
        out="${RUN_DIR}/candidates/${wl}-${shard}.jsonl.gz"
        [ -s "${RUN_DIR}/candidates/${wl}-${shard}.summary.json" ] && continue
        CUDA_VISIBLE_DEVICES=${shard} python3 -m apertus_tree.candidates \
          --capture "${RUN_DIR}/captures/${wl}-plain.json" --prompts "${PROMPTS[$wl]}" --workload "${wl}" \
          --eagle-head "${EAGLE_HEAD}" --dspark "${DSPARK}" --dspark-hf-layers "${DSPARK_HF_LAYERS}" \
          --shard ${shard} --num-shards 4 --limit "${LIMIT}" --max-anchors "${MAX_ANCHORS}" \
          --output "${out}" --summary "${RUN_DIR}/candidates/${wl}-${shard}.summary.json"
      done
    ) > "${RUN_DIR}/logs/candidates-${shard}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [ "${fail}" = 0 ] || { tail -40 "${RUN_DIR}"/logs/candidates-*.log; exit 1; }
fi

if [[ " ${STAGES} " == *" analyze "* ]]; then
  stamp "stage 3: replay"
  python3 -m apertus_bench.tree_sim "${RUN_DIR}"/candidates/*.jsonl.gz --cost-table "${COST_TABLE}" \
    --processes 64 --output "${RUN_DIR}/tree-sim.json"
fi
stamp "done"
