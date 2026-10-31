#!/usr/bin/env bash
# Container-side Beverin wrapper. Overrides follow the enroot hooks, which
# otherwise replace NCCL tuning supplied in the submit environment.
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
export BEVERIN_NUMERICAL_GUARD=1
if [ -s "${DSPARK_DIR}/runs/${RUN_ID}/checkpoints/numerical-failure.json" ]; then
  echo 'BEVERIN_NUMERICAL_FAILURE: refusing to restart a numerically failed run' >&2
  exit 65
fi
[ -z "${BEVERIN_NCCL_PROTO:-}" ] || export NCCL_PROTO="${BEVERIN_NCCL_PROTO}"
[ -z "${BEVERIN_NCCL_MIN_NCHANNELS:-}" ] || export NCCL_MIN_NCHANNELS="${BEVERIN_NCCL_MIN_NCHANNELS}"
[ -z "${BEVERIN_NCCL_NCHANNELS_PER_NET_PEER:-}" ] || export NCCL_NCHANNELS_PER_NET_PEER="${BEVERIN_NCCL_NCHANNELS_PER_NET_PEER}"
[ -z "${BEVERIN_FI_MR_CACHE_MONITOR:-}" ] || export FI_MR_CACHE_MONITOR="${BEVERIN_FI_MR_CACHE_MONITOR}"
export EXTRACT_START_ATTEMPTS="${EXTRACT_START_ATTEMPTS:-3}"
export BEVERIN_PROFILE_DIR="${BEVERIN_DEPLOYMENT}/profiles"
node_provenance="${BEVERIN_DEPLOYMENT}/node${SLURM_NODEID}"
mkdir -p "${node_provenance}"
env | sort | grep -E '^(NCCL_|FI_|OFI_|BEVERIN_|SLURM_JOB_|SLURM_NODE|HWLOC_)' \
  > "${node_provenance}/resolved-environment.txt"
hostname > "${node_provenance}/hostname.txt"
cat /proc/meminfo > "${node_provenance}/initial-meminfo.txt"
if [ -f /aws-ofi-nccl-hook.log ]; then
  cp /aws-ofi-nccl-hook.log "${node_provenance}/"
fi
( while sleep 10; do
    echo "$(date -u +%FT%TZ) $(free -g | awk '/Mem:/ {print "used_gb=" $3 " avail_gb=" $7}')"
  done ) > "${WORK}/logs/mem-${SLURM_JOB_ID}-node${SLURM_NODEID}.log" &
memory_monitor=$!
trap 'kill "${memory_monitor}" 2>/dev/null || true' EXIT
bash "${REPO_ROOT}/methods/dspark/train/train-node.sh"
