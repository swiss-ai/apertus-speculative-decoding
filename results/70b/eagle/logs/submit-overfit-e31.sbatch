#!/bin/bash
#SBATCH --job-name=eagle-t4-overfit-e31
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --exclude=nid007129,nid005422
#SBATCH --environment=/users/faruk_zahiragic/.sml/env_resolved_arm64_kwZJ6B.toml
#SBATCH --output=/users/faruk_zahiragic/apertus-bench-repo/results/eagle/logs/%j.out
#SBATCH --error=/users/faruk_zahiragic/apertus-bench-repo/results/eagle/logs/%j.err
#SBATCH --chdir=/users/faruk_zahiragic/apertus-bench-repo

set -euo pipefail
export I_AM_ON_AN_ALLOCATED_GPU_JOB=1
export EAGLE_TRAIN_STAGE=overfit-e31
export MODEL_LAUNCH_ROOT=/users/faruk_zahiragic/model-launch-apertus
export TORCHSPEC_ROOT=/users/faruk_zahiragic/apertus-bench-repo/scratch/TorchSpec
export PYTHONPATH=/users/faruk_zahiragic/apertus-bench-repo/training:/users/faruk_zahiragic/apertus-bench-repo/scratch/TorchSpec
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf_home}
export TMPDIR=${TMPDIR:-/tmp}

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) job=${SLURM_JOB_ID}"
python3 - <<'PY'
import sys
print("python", sys.version)
mods = ("torch", "transformers", "vllm", "ray", "omegaconf")
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "imported"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, exc)
PY

# TorchSpec extras that the serving image may not ship. Skip mooncake on
# overfit: the one-node path uses an HF teacher, then overfit_direct.
python3 -m pip install --user --quiet --no-deps -e "/users/faruk_zahiragic/apertus-bench-repo/scratch/TorchSpec" || true
python3 -m pip install --user --quiet omegaconf pydantic ninja packaging || true
if [ "overfit-e31" != "overfit-e31" ] && [ "overfit-e31" != "overfit-e3" ]; then
  python3 -m pip install --user --quiet ray datasets accelerate numba pyzmq || true
  python3 -m pip install --user --quiet "mooncake-transfer-engine>=0.3.10.post1" || echo "mooncake-transfer-engine not installed"
fi

bash "/users/faruk_zahiragic/apertus-bench-repo/launch/train-eagle-on-node.sh"
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
