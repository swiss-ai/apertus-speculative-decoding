#!/bin/bash
# Build the x86_64 Python side deps for EAGLE training on Beverin (MI300A).
#   sbatch methods/eagle/beverin/build-pydeps.sbatch OUT_DIR
# 1. Lists package versions in the Clariden training image (amd64 build of
#    vllm_apertus_1.5_release) and in the Beverin ROCm image.
# 2. Copies the Swiss AI transformers fork (pure Python) from the Clariden image.
# 3. Copies the other packages Clariden's scratch/pydeps adds (pure Python),
#    and pip-installs (no deps) the compiled ones at the same versions.
# Writes OUT_DIR/manifest.txt (versions, sha256 of every copied file tree).
#SBATCH -A infra01 -p mi300 -N 1 -t 00:40:00 --job-name=eagle-bev-pydeps
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/logs/%x-%j.out
set -euo pipefail
OUT="${1:?OUT_DIR}"
HERE="${SLURM_SUBMIT_DIR}/methods/eagle/beverin"
CLARIDEN_PYDEPS="${CLARIDEN_PYDEPS:-/users/${USER}/apertus-spec-decoding/scratch/pydeps}"
mkdir -p "${OUT}/pydeps" "${OUT}/meta"
echo "job ${SLURM_JOB_ID} node $(hostname) $(date -u -Iseconds)"
sha256sum /capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-amd64.sqsh \
  > "${OUT}/meta/clariden-amd64-image.sha256" &
hash_pid=$!
# Clariden image (x86_64 build): versions and the transformers fork.
srun -N1 -n1 --environment="${HERE}/env-clariden-amd64-files.toml" bash -c '
  set -euo pipefail
  OUT='"${OUT}"'
  python3 -V > "${OUT}/meta/clariden-image-python.txt"
  pip list --format=freeze 2>/dev/null > "${OUT}/meta/clariden-image-pip-freeze.txt" || true
  python3 - "${OUT}/pydeps" <<PY
import importlib.metadata as md, pathlib, shutil, sys, transformers
out = pathlib.Path(sys.argv[1])
src = pathlib.Path(transformers.__file__).parent
print("transformers", transformers.__version__, src)
shutil.copytree(src, out / "transformers", symlinks=True)
dist = md.distribution("transformers")
info = pathlib.Path(dist._path)
shutil.copytree(info, out / info.name, symlinks=True)
PY
'
# Beverin ROCm image: versions only.
srun -N1 -n1 --environment="${HERE}/../configs/train-env-beverin.toml" bash -c '
  set -euo pipefail
  OUT='"${OUT}"'
  python3 -V > "${OUT}/meta/beverin-image-python.txt"
  pip list --format=freeze 2>/dev/null > "${OUT}/meta/beverin-image-pip-freeze.txt" || true
'
wait "${hash_pid}"
# Pure-Python packages from Clariden scratch/pydeps (wandb is replaced by the
# offline stub at job start); compiled ones (protobuf) reinstalled for x86_64.
for p in accelerate accelerate-1.15.0.dist-info antlr4 antlr4_python3_runtime-4.9.3.dist-info \
         click click-8.5.0.dist-info omegaconf omegaconf-2.3.1.dist-info \
         platformdirs platformdirs-4.11.11.dist-info; do
  [ -e "${CLARIDEN_PYDEPS}/${p}" ] && cp -a "${CLARIDEN_PYDEPS}/${p}" "${OUT}/pydeps/"
done
srun -N1 -n1 --environment="${HERE}/../configs/train-env-beverin.toml" bash -c '
  set -euo pipefail
  OUT='"${OUT}"'
  pv=$(ls -d '"${CLARIDEN_PYDEPS}"'/protobuf-*.dist-info 2>/dev/null | sed "s|.*/protobuf-||; s|.dist-info||")
  if [ -n "${pv}" ]; then
    pip install --no-deps --only-binary=:all: --target "${OUT}/pydeps" "protobuf==${pv}" 2>&1 | tail -2
  fi
  # The fork pins tokenizers<=0.23.0 (the ROCm image has 0.23.2); match the
  # Clariden image exactly for tokenizers and huggingface_hub.
  pins=$(grep -iE "^(tokenizers|huggingface_hub)==" "${OUT}/meta/clariden-image-pip-freeze.txt")
  pip install --no-deps --only-binary=:all: --target "${OUT}/pydeps" ${pins} 2>&1 | tail -2
  PYTHONPATH="${OUT}/pydeps" python3 -c "import transformers, tokenizers, huggingface_hub, torch; print(\"import ok\", transformers.__version__, tokenizers.__version__, huggingface_hub.__version__, torch.__version__)"
'
( cd "${OUT}/pydeps" && ls -1 ) > "${OUT}/meta/pydeps-listing.txt"
( cd "${OUT}/pydeps" && find . -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) \
  > "${OUT}/meta/pydeps-files.sha256"
sha256sum "${OUT}/meta/pydeps-files.sha256"
echo "DONE $(date -u -Iseconds)"
