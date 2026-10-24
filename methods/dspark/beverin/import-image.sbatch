#!/bin/bash
# Import AMD's public ROCm vLLM image on a Beverin compute node (they reach
# Docker Hub) and probe it: host driver, GPUs from torch, vLLM version, RCCL
# network plugin.  sbatch methods/dspark/beverin/import-image.sbatch [TAG] [OUT]
#SBATCH -A infra01 -p mi300 -N 1 -t 01:00:00 --job-name=bev-import
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
TAG="${1:-rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0}"
OUT="${2:-/capstor/scratch/cscs/${USER}/container-images/rocm-vllm-0.23.0-cdna.sqsh}"
EDF="${EDF:-${SLURM_SUBMIT_DIR}/methods/dspark/beverin/env-vllm023.toml}"
echo "job ${SLURM_JOB_ID} node $(hostname) tag ${TAG}"
echo "amdgpu driver: $(cat /sys/module/amdgpu/version 2>/dev/null || modinfo -F version amdgpu 2>/dev/null)"
if [ ! -f "${OUT}" ]; then
  t0=$(date +%s)
  enroot import -o "${OUT}.part" "docker://rocm/vllm:${TAG}"
  mv "${OUT}.part" "${OUT}"
  echo "import took $(( $(date +%s) - t0 )) s"
fi
ls -la "${OUT}"; sha256sum "${OUT}" | tee "${OUT}.sha256"
srun --environment="${EDF}" bash -c '
  set -x
  python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_properties(i).total_memory/2**30 for i in range(torch.cuda.device_count())])"
  python3 -c "import vllm, transformers; print(\"vllm\", vllm.__version__, vllm.__file__, \"transformers\", transformers.__version__)"
  pip list 2>/dev/null | grep -iE "^(vllm|torch|transformers|triton|aiter|amdsmi|flash|safetensors|openai|pydantic) " 
  ls /opt/rocm*/ -d; cat /opt/rocm/.info/version 2>/dev/null
  ls -la $(python3 -c "import vllm,os; print(os.path.dirname(vllm.__file__))")/*.so
  env | grep -iE "^(NCCL|FI_|RCCL|HSA|HIP|ROCM|VLLM)" | sort
  ldconfig -p | grep -iE "rccl-net|nccl-net|libfabric" 
  fi_info -p cxi 2>&1 | head -3
'
