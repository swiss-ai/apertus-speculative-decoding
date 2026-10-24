# Sourced inside the container by the DSpark 70B jobs (port of Yu's
# methods/dspark/jobs on branch DSpark_Apertus_1.5_8B). Expects REPO_ROOT.
# speculators (7046ac6 + Yu's patch) lives untracked in the checkout's scratch/;
# it is prepended to PYTHONPATH, never replacing it: the image's /workspace/vllm
# entry carries the vLLM overlays.

DSPARK_DIR="${DSPARK_DIR:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
SPECULATORS="${SPECULATORS:-${REPO_ROOT}/scratch/speculators}"
SIDE_DEPS="${DSPARK_DIR}/pydeps"
OPB_REPO="mlabonne/open-perfectblend"
OPB_REVISION="af60f3c18201652a83a93f46fcfee1b646ba3df7"
# Yu's 8B corpus: its row order and val_ids.txt define the held-out split.
CORPUS_8B_DIR="/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-thinking-off"

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf_home}"
export PYTHONPATH="${SPECULATORS}/src:${SPECULATORS}/hs_connectors/src:${SIDE_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
# JIT and compile caches default to home, which was at its 50 GB quota when the
# first pilot (job 3680143) wrote there. Keep them node-local so no job depends
# on home space or on another job's half-written build; a few minutes per job.
export VLLM_CACHE_ROOT=/tmp/dspark-cache/vllm
export TRITON_CACHE_DIR=/tmp/dspark-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/dspark-cache/inductor
export FLASHINFER_WORKSPACE_BASE=/tmp/dspark-cache/flashinfer
export TORCH_EXTENSIONS_DIR=/tmp/dspark-cache/torch_extensions
export CUDA_CACHE_PATH=/tmp/dspark-cache/nv
export XDG_CACHE_HOME=/tmp/dspark-cache/xdg
# Loading the 70B from capstor takes ~6 min, compile and graphs ~4 more.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export PYTHONUNBUFFERED=1

stamp() { echo "=== [$(date -u +%FT%TZ)] $(hostname) $*"; }

# Install, without dependencies, whatever `import speculators` reports missing
# (Yu's side_deps.sh). Only the prep job calls it: parallel installs into one
# directory race.
install_side_deps() {
  declare -A pkg=([dateutil]=python-dateutil [yaml]=pyyaml [PIL]=pillow [sklearn]=scikit-learn [attr]=attrs
    [google]=protobuf [aiohttp]=aiohttp [typer]=typer [click]=click [rich]=rich [loguru]=loguru
    [pydantic_settings]=pydantic-settings [datasets]="datasets>=4.0.0,<=5.0.1" [dotenv]=python-dotenv
    [markdown_it]=markdown-it-py [multiprocess]=multiprocess [xxhash]=xxhash [dill]=dill
    [shellingham]=shellingham [annotated_types]=annotated-types [pydantic_core]=pydantic-core)
  mkdir -p "${SIDE_DEPS}"
  # speculators reads its own version from package metadata, which a PYTHONPATH
  # checkout lacks; record the version Yu's checkpoints carry.
  local meta="${SIDE_DEPS}/speculators-0.9.0.dev36.dist-info"
  mkdir -p "${meta}"
  printf 'Metadata-Version: 2.1\nName: speculators\nVersion: 0.9.0.dev36\n' > "${meta}/METADATA"
  local round err name
  for round in $(seq 1 25); do
    err="$(python3 -c "import speculators, hs_connectors, speculators.train.data, speculators.cli" 2>&1 \
      | grep -oE "No module named '[A-Za-z0-9_.]+'" | head -1 | sed "s/No module named '//; s/'//" | cut -d. -f1 || true)"
    if [ -z "${err}" ]; then echo "side deps complete after $((round - 1)) installs"; break; fi
    name="${pkg[${err}]:-${err}}"
    echo "missing ${err} -> installing ${name}"
    pip install -q --no-deps --target "${SIDE_DEPS}" "${name}" 2>&1 | tail -1
  done
  python3 -c "import transformers, torch, vllm; print('image stack intact: transformers', transformers.__version__, 'torch', torch.__version__, 'vllm', vllm.__version__)"
}

check_side_deps() {
  python3 -c "import speculators, hs_connectors, speculators.train.data, speculators.cli" \
    || { echo "speculators does not import: run the prep job first" >&2; return 1; }
}

# wait_health PORT PID SECONDS: the 70B takes ~10 min to load from capstor.
wait_health() {
  local i
  for i in $(seq 1 $(( $3 / 10 ))); do
    # curl does not load in the Beverin ROCm image (libldap_r missing): Python.
    { curl -sf "localhost:$1/health" > /dev/null 2>&1 \
      || python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$1/health', timeout=5)" 2> /dev/null; } \
      && return 0
    kill -0 "$2" 2> /dev/null || return 1
    sleep 10
  done
  return 1
}

# stop_server PID: vLLM forks engine and worker processes; started with setsid,
# the whole group goes, and the GPUs are free again before the next server
# (job 3680758: TP2 workers that outlived their server left TP4 6.8 GB free).
stop_server() {
  kill -TERM -- "-$1" 2> /dev/null || true
  local i
  for i in $(seq 1 30); do
    kill -0 -- "-$1" 2> /dev/null || break
    sleep 2
  done
  kill -KILL -- "-$1" 2> /dev/null || true
  for i in $(seq 1 30); do
    command -v nvidia-smi > /dev/null || return 0
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && return 0
    sleep 2
  done
}
