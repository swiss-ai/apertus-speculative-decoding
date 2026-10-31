#!/usr/bin/env bash
# Container-side Beverin wrapper. Overrides follow the enroot hooks, which
# otherwise replace NCCL tuning supplied in the submit environment.
set -euo pipefail
# shellcheck disable=SC1091
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
export BEVERIN_NUMERICAL_GUARD=1
if [ "${BEVERIN_ROLLBACK:-0}" != 1 ] && [ -s "${DSPARK_DIR}/runs/${RUN_ID}/checkpoints/numerical-failure.json" ]; then
  echo 'BEVERIN_NUMERICAL_FAILURE: refusing to restart a numerically failed run' >&2
  exit 65
fi
if [ "${BEVERIN_ALPS:-0}" = 1 ] && [ "${ALPS_EXTENDED_IMAGES_ENV_SOURCED:-}" != 1 ]; then
  echo "ALPS environment was not sourced (BASH_ENV=${BASH_ENV-})" >&2
  exit 1
fi
[ -z "${BEVERIN_NCCL_PROTO:-}" ] || export NCCL_PROTO="${BEVERIN_NCCL_PROTO}"
[ -z "${BEVERIN_NCCL_MIN_NCHANNELS:-}" ] || export NCCL_MIN_NCHANNELS="${BEVERIN_NCCL_MIN_NCHANNELS}"
[ -z "${BEVERIN_NCCL_NCHANNELS_PER_NET_PEER:-}" ] || export NCCL_NCHANNELS_PER_NET_PEER="${BEVERIN_NCCL_NCHANNELS_PER_NET_PEER}"
[ -z "${BEVERIN_FI_MR_CACHE_MONITOR:-}" ] || export FI_MR_CACHE_MONITOR="${BEVERIN_FI_MR_CACHE_MONITOR}"
export EXTRACT_START_ATTEMPTS="${EXTRACT_START_ATTEMPTS:-3}"
export BEVERIN_PROFILE_DIR="${BEVERIN_DEPLOYMENT}/profiles"
node_provenance="${BEVERIN_DEPLOYMENT}/node${SLURM_NODEID}"
mkdir -p "${node_provenance}"
env | sort | grep -E '^(NCCL_|FI_|OFI_|BEVERIN_|SLURM_JOB_|SLURM_NODE|HWLOC_|BASH_ENV|ALPS_)' \
  > "${node_provenance}/resolved-environment.txt"
python3 - <<'PY' > "${node_provenance}/library-versions.json"
import json, os, sys
out = {"python": sys.version.split()[0]}
for name in ("torch", "vllm", "triton", "transformers"):
    try:
        module = __import__(name)
        out[name] = getattr(module, "__version__", None)
        out[name + "_file"] = getattr(module, "__file__", None)
    except Exception as exc:
        out[name + "_error"] = str(exc)
try:
    import torch
    out["torch_git"] = getattr(torch.version, "git_version", None)
    out["hip"] = getattr(torch.version, "hip", None)
except Exception:
    pass
for key in ("ALPS_EXTENDED_IMAGES_ENV_SOURCED", "BASH_ENV", "NCCL_P2P_DISABLE", "NCCL_PROTO", "FI_PROVIDER"):
    out[key] = os.environ.get(key)
json.dump(out, sys.stdout, indent=2, sort_keys=True)
print()
PY
ldconfig -p 2>/dev/null | grep -Ei 'rccl|rocblas|hipblas|amdhip' > "${node_provenance}/ldconfig-math.txt" || true
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
