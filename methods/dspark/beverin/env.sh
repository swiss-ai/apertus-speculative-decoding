# Sourced inside the Beverin container by the feasibility scripts.
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
# The apertus15_text vLLM plugin (and its entry-point dist-info) and, when
# present, speculators + its side deps built for this image's Python.
export PYTHONPATH="${REPO_ROOT}/methods/dspark/beverin${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="${WORK}/hf_home" HF_HUB_OFFLINE=1
# JIT caches node-local, as on Clariden (/tmp is RAM-backed tmpfs on MI300A,
# i.e. it shares the 4x128 GB HBM with the GPUs; keep it small).
export VLLM_CACHE_ROOT=/tmp/bev-cache/vllm TRITON_CACHE_DIR=/tmp/bev-cache/triton \
  TORCHINDUCTOR_CACHE_DIR=/tmp/bev-cache/inductor XDG_CACHE_HOME=/tmp/bev-cache/xdg \
  TORCH_EXTENSIONS_DIR=/tmp/bev-cache/torch_extensions MIOPEN_USER_DB_PATH=/tmp/bev-cache/miopen \
  MIOPEN_CUSTOM_CACHE_DIR=/tmp/bev-cache/miopen
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "${WORK}/logs" /tmp/bev-cache
wait_health() {
  local i
  for i in $(seq 1 $(( $3 / 10 ))); do
    curl -sf "localhost:$1/health" > /dev/null && return 0
    kill -0 "$2" 2> /dev/null || return 1
    sleep 10
  done
  return 1
}
