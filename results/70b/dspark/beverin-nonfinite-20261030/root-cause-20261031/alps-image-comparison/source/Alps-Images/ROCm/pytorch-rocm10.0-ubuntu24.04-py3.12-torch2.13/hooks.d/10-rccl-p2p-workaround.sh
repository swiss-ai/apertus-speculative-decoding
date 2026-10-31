#!/usr/bin/env bash
set -euo pipefail

runtime_env="/opt/alps/env/alps-runtime.env"
[[ -f "${runtime_env}" ]] || { echo "ERROR: missing runtime env: ${runtime_env}" >&2; exit 1; }

cat >> "${runtime_env}" <<'EOF'

# ROCm 10.0 bundled RCCL (2.30.4) is unreliable for 2-node MI300A collectives with P2P
# enabled under enroot on Beverin's current kernel (6.4): RCCL cuMem support needs
# kernel >= 6.8, and the IPC-handle fallback either crashes during communicator setup
# ("HIP failure: invalid device pointer", ~20-30% of launches with one GPU per rank) or
# hangs outright (multi-GPU-per-rank). Only NCCL_NET=Socket (no aws-ofi-rccl
# registration) or disabling P2P avoids it; no RCCL knob (nvlink-centric scheduler,
# read-enable, forced cuMem, HMEM or dmabuf disable) helps. Same failure class as the
# rocm7.14 variant.
#
# Users can still override by exporting NCCL_P2P_DISABLE (any value, including empty)
# before the runtime env loads; defvar semantics keep pre-set values untouched.
defvar NCCL_P2P_DISABLE "1"
EOF
