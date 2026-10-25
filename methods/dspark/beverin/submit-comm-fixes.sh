#!/usr/bin/env bash
# Two diagnostic lanes at lower scheduling priority than the one-node run.
# Supply the baseline profiling job; each trial changes one setting and has
# its own run/checkpoints. No Clariden operations are performed.
set -euo pipefail
: "${1:?usage: submit-comm-fixes.sh BASELINE_JOB_ID}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
PREFIX="${PREFIX:-comm-2n-$(date -u +%Y%m%dT%H%M%SZ)}"
export NODES=2 TIME=01:30:00 CHAIN=1 SEQ=8192 ANCHORS=512 LR=1.5e-4 EPOCHS=10
export EXTRACT_MEM=0.56 WORKERS=1 PREFETCH=1 CKPT_FREQ=0.015625
export BEVERIN_SEGMENT_STEPS=6 BEVERIN_EMPTY_CACHE_ON_SAVE=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,REG,COLL
export SBATCH_EXTRA=--nice=10000
manifest="${WORK}/logs/${PREFIX}-jobs.tsv"
printf 'variant\tjob_id\tdependency\trun_id\n' > "${manifest}"

submit() {
  local variant="$1" dep="$2" output
  shift 2
  output=$(env RUN_ID="${PREFIX}-${variant}" DEPENDENCY="${dep}" "$@" \
    bash "${ROOT}/methods/dspark/beverin/submit-train.sh")
  printf '%s\n' "${output}" >&2
  local id
  id=$(awk '/^submitted / {print $2}' <<< "${output}")
  [[ "${id}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\t%s\t%s\t%s\n' "${variant}" "${id}" "${dep}" "${PREFIX}-${variant}" >> "${manifest}"
  printf '%s' "${id}"
}

bf16=$(submit bf16 '' BEVERIN_REDUCE_DTYPE=bf16)
simple=$(submit simple "afterany:$1" BEVERIN_NCCL_PROTO=Simple)
channels=$(submit channels16 "afterany:${bf16}" BEVERIN_NCCL_MIN_NCHANNELS=16)
submit peers8 "afterany:${simple}" BEVERIN_NCCL_NCHANNELS_PER_NET_PEER=8 > /dev/null
submit host-rocm6 "afterany:${channels}" EDF="${ROOT}/methods/dspark/beverin/env-vllm031-host-rocm6.toml" > /dev/null
cat "${manifest}"
