# Sourced inside the Beverin container by the feasibility scripts.
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
# The apertus15_text vLLM plugin (and its entry-point dist-info) and, when
# present, speculators + its side deps built for this image's Python.
export PYTHONPATH="${REPO_ROOT}/methods/dspark/beverin${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Slurm sets ROCR_VISIBLE_DEVICES=0,1,2,3; with CUDA_VISIBLE_DEVICES=2,3 on top,
# ROCm torch counts 4 devices but creates 2, and the trainer's manual_seed_all
# fails (job 693749). The scripts select GPUs with CUDA_VISIBLE_DEVICES alone
# (CSCS's vLLM-on-AMD tutorial likewise unsets ROCR_VISIBLE_DEVICES).
unset ROCR_VISIBLE_DEVICES
export HF_HOME="${WORK}/hf_home" HF_HUB_OFFLINE=1
# JIT caches node-local, as on Clariden (/tmp is RAM-backed tmpfs on MI300A,
# i.e. it shares the 4x128 GB HBM with the GPUs; keep it small).
export VLLM_CACHE_ROOT=/tmp/bev-cache/vllm TRITON_CACHE_DIR=/tmp/bev-cache/triton \
  TORCHINDUCTOR_CACHE_DIR=/tmp/bev-cache/inductor XDG_CACHE_HOME=/tmp/bev-cache/xdg \
  TORCH_EXTENSIONS_DIR=/tmp/bev-cache/torch_extensions MIOPEN_USER_DB_PATH=/tmp/bev-cache/miopen \
  MIOPEN_CUSTOM_CACHE_DIR=/tmp/bev-cache/miopen
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "${WORK}/logs" /tmp/bev-cache
# curl in the ROCm image fails to load (libldap_r-2.4.so.2 missing; job 693716),
# so probe the server with Python.
health() {
  python3 -c "import sys, urllib.request; urllib.request.urlopen('http://localhost:$1/health', timeout=5)" 2> /dev/null
}
wait_health() {
  local i
  for i in $(seq 1 $(( $3 / 10 ))); do
    health "$1" && return 0
    kill -0 "$2" 2> /dev/null || return 1
    sleep 10
  done
  return 1
}
