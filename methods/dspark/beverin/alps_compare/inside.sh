#!/bin/bash
set -euo pipefail
tools=${1:?}
cell=${2:?}
deployment=${3:?}
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES BEVERIN_BLAS_BACKEND
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
python3 "$tools/records.py" diagnose "$cell"
ldconfig -p > "$cell/ldconfig.txt" 2>&1 || true
if [[ -d /opt/alps/env ]]; then cp -a /opt/alps/env "$cell/alps-environment"; fi
for log in /cxi-hook.log /aws-ofi-nccl-hook.log; do
    if [[ -f "$log" ]]; then cp "$log" "$cell/"; fi
done
case "$cell" in
    *-rocblas) probe="$deployment/probe-rocblas.py" ;;
    *) probe="$deployment/probe.py" ;;
esac
exec torchrun --standalone --nproc_per_node=2 "$probe"
