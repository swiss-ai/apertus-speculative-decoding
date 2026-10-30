#!/bin/bash
set -euo pipefail
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
export TRITON_CACHE_DIR=/tmp/bev-num-guard-${SLURM_JOB_ID}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/bev-num-guard-${SLURM_JOB_ID}/inductor
export TORCH_EXTENSIONS_DIR=/tmp/bev-num-guard-${SLURM_JOB_ID}/extensions
python3 - <<'ENVIRONMENT'
import json, os, torch
from pathlib import Path
prefixes=("SLURM_", "CUDA_", "HIP_", "ROCR_", "GPU_", "NCCL_", "RCCL_", "FI_", "OFI_", "BEVERIN_", "PYTHON", "TORCH", "TRITON_")
data={"torch":torch.__version__,"hip":torch.version.hip,"environment":{k:v for k,v in os.environ.items() if k.startswith(prefixes)}}
root=Path(os.environ["BEVERIN_PROBE_DIR"])
root.mkdir(parents=True,exist_ok=True)
(root/"resolved-environment.json").write_text(json.dumps(data,indent=2))
ENVIRONMENT
timeout --signal=TERM --kill-after=30s 600s torchrun --standalone --nproc-per-node=2 /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-e6c84030/methods/dspark/beverin/check_numerical_guard.py
