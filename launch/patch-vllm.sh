#!/usr/bin/env bash
# Materialize patched copies of files taken from a container image and print the
# pyxis mount specs that overlay them, for use as EXTRA_MOUNTS.
#
# The image is read-only, so a source-level fix can only be applied by binding a
# corrected file over the one inside it. Drop this once a rebuilt image carries
# the fix: pass no EXTRA_MOUNTS and the launch is back to the stock image.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 IMAGE_SQSH PATCH_FILE [PATCH_FILE...]" >&2
  exit 2
fi

IMAGE="$1"
shift
PATCHES=("$@")
# Must be on a shared filesystem: pyxis resolves the mount on the compute node.
OUT_DIR="${VLLM_PATCH_DIR:-${HOME}/.sml/vllm-patch}"

if [ ! -f "${IMAGE}" ]; then
  echo "not found: ${IMAGE}" >&2
  exit 1
fi
for patch in "${PATCHES[@]}"; do
  if [ ! -f "${patch}" ]; then
    echo "not found: ${patch}" >&2
    exit 1
  fi
done

mapfile -t TARGETS < <(sed -n 's|^+++ b/||p' "${PATCHES[@]}" | awk 'NF && !seen[$0]++')
if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "no target files found in patches" >&2
  exit 1
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
unsquashfs -q -d "${OUT_DIR}" "${IMAGE}" "${TARGETS[@]}" > /dev/null
for patch in "${PATCHES[@]}"; do
  patch -s -p1 -d "${OUT_DIR}" < "${patch}"
done

for t in "${TARGETS[@]}"; do
  printf '%s/%s:/%s\n' "${OUT_DIR}" "${t}" "${t}"
done
