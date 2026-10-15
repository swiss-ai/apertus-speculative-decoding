#!/usr/bin/env bash
# Build a container image from serving/images/<name>/ on a Clariden compute node
# (podman, then enroot to a squashfs pyxis can mount) and submit it as a job.
#   ./serving/images/build-image.sh vllm-apertus15-upstream
# Login nodes are for orchestration only, so the build runs in the job. The
# squashfs lands in IMAGE_DIR as <name>-arm64.sqsh; the job log next to it.
# Only submits; check the job with squeue.
set -euo pipefail

IMAGES_DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?usage: build-image.sh NAME (a directory of serving/images)}"
CONTEXT="${IMAGES_DIR}/${NAME}"
[ -f "${CONTEXT}/Dockerfile" ] || { echo "no Dockerfile in ${CONTEXT}" >&2; exit 1; }
IMAGE_DIR="${IMAGE_DIR:-/capstor/scratch/cscs/${USER}/container-images}"
PARTITION="${BUILD_PARTITION:-normal}"
TIME_LIMIT="${BUILD_TIME:-02:00:00}"
ACCOUNT="${BUILD_ACCOUNT:-infra01}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${IMAGE_DIR}/${NAME}-arm64.sqsh"
LOG="${IMAGE_DIR}/${NAME}-arm64-${STAMP}.log"
TAG="localhost/${NAME}:${STAMP}"
mkdir -p "${IMAGE_DIR}"

JOB="$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=build-${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${LOG}
set -euo pipefail
# podman's storage (~/.config/containers/storage.conf) is node-local /dev/shm:
# start clean and leave nothing behind.
trap 'podman system reset -f > /dev/null 2>&1 || true' EXIT
podman build --network=host --platform linux/arm64 -t "${TAG}" "${CONTEXT}"
# Write next to the target, then move: a running job may be reading the old one.
rm -f "${OUT}.partial"
enroot import -x mount -o "${OUT}.partial" "podman://${TAG}"
mv -f "${OUT}.partial" "${OUT}"
ls -la "${OUT}"
EOF
)"
echo "job=${JOB} image=${OUT} log=${LOG}"
