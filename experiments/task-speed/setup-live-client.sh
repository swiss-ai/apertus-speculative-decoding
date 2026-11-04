#!/usr/bin/env bash
# Build the x86_64 client environment for run-live.py on Clariden's xfer partition
# (CPU nodes, shared, not billed, reach the gateway). The project venvs and the
# staged kubectl are aarch64, so the live client gets its own copies on scratch:
#   experiments/task-speed/setup-live-client.sh
# Pins: uv 0.8.0 (managed CPython 3.12), the repo's uv.lock versions, kubectl v1.35.9.
set -euo pipefail
ROOT="${TASK_SPEED_CLIENT_ROOT:-/iopsstor/scratch/cscs/${USER}/apertus-task-speed-client}"
mkdir -p "${ROOT}"
srun -p xfer -A "${TASK_SPEED_ACCOUNT:-infra01}" -t 00:30:00 -n1 -c4 bash -s -- "${ROOT}" <<'EOF'
set -euo pipefail
root="$1"
cd "${root}"
test "$(uname -m)" = x86_64
if [ ! -x uv/uv ]; then
  base=https://github.com/astral-sh/uv/releases/download/0.8.0/uv-x86_64-unknown-linux-gnu.tar.gz
  curl -fsSL -o uv.tar.gz "${base}"
  curl -fsSL -o uv.tar.gz.sha256 "${base}.sha256"
  echo "$(cut -d' ' -f1 uv.tar.gz.sha256)  uv.tar.gz" | sha256sum -c -
  mkdir -p uv && tar -xzf uv.tar.gz -C uv --strip-components=1 && rm uv.tar.gz
fi
export UV_PYTHON_INSTALL_DIR="${root}/python" UV_CACHE_DIR="${root}/uv-cache"
[ -x venv/bin/python ] || uv/uv venv -q --python 3.12 venv
uv/uv pip install -q --python venv/bin/python httpx==0.28.1 httpcore==1.0.9 \
  prometheus-client==0.26.0 pyyaml==6.0.3
if [ ! -x kubectl ]; then
  base=https://dl.k8s.io/release/v1.35.9/bin/linux/amd64/kubectl
  curl -fsSL -o kubectl "${base}"
  echo "$(curl -fsSL "${base}.sha256")  kubectl" | sha256sum -c -
  chmod +x kubectl
fi
venv/bin/python -c "import httpx, prometheus_client, yaml, sys; print(sys.version.split()[0])"
./kubectl version --client | head -1
venv/bin/python -m pip freeze 2> /dev/null > requirements-frozen.txt || uv/uv pip freeze --python venv/bin/python > requirements-frozen.txt
EOF
echo "client=${ROOT}"
