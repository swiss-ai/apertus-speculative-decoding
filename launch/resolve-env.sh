#!/usr/bin/env bash
# Print the path of an environment toml whose image path is ready for pyxis.
#
# The packaged asset ships a literal {arch} in its image path. Only some sml
# builds substitute it on the node; one without that substitution passes the
# placeholder through and pyxis rejects it ("Invalid argument: ...-{arch}.sqsh"),
# killing the job seconds after it starts. Resolving here keeps the launch
# independent of which sml is on PATH.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 ENV_TOML" >&2
  exit 2
fi

ENV_SOURCE="$1"
# Clariden compute nodes are GH200 (aarch64).
SML_ARCH="${SML_ARCH:-arm64}"
# The SLURM launcher hands this path to the compute node as-is instead of
# copying it, so it has to live on a shared filesystem: a node-local /tmp copy
# is invisible to the job. Home is shared on Alps.
SML_ENV_DIR="${SML_ENV_DIR:-${HOME}/.sml}"
# Additional "host:container" pyxis mounts, whitespace- or comma-separated.
# See launch/patch-vllm.sh for overlaying a file inside the read-only image.
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

if [ ! -f "${ENV_SOURCE}" ]; then
  echo "environment toml not found: ${ENV_SOURCE}" >&2
  exit 1
fi

mkdir -p "${SML_ENV_DIR}"
RESOLVED="$(mktemp "${SML_ENV_DIR}/env_resolved_${SML_ARCH}_XXXXXX.toml")"
sed "s|{arch}|${SML_ARCH}|g" "${ENV_SOURCE}" > "${RESOLVED}"

if [ -n "${EXTRA_MOUNTS}" ]; then
  awk -v specs="${EXTRA_MOUNTS}" '
    { print }
    /^mounts *= *\[/ && !injected {
      n = split(specs, a, /[,[:space:]]+/)
      for (i = 1; i <= n; i++) {
        if (a[i] != "") printf "  \"%s\",\n", a[i]
      }
      injected = 1
    }
    END { if (!injected) exit 1 }
  ' "${RESOLVED}" > "${RESOLVED}.new" || {
    echo "no mounts array to extend in ${ENV_SOURCE}" >&2
    exit 1
  }
  mv "${RESOLVED}.new" "${RESOLVED}"
fi

IMAGE="$(sed -n 's|^ *image *= *"\(/[^"]*\)".*|\1|p' "${RESOLVED}" | head -1)"
# Only enforce existence where the image store is actually mounted, so this
# still works when launching from a machine without Capstor.
if [ -n "${IMAGE}" ] && [ -d "$(dirname "${IMAGE}")" ] && [ ! -e "${IMAGE}" ]; then
  echo "container image not found: ${IMAGE}" >&2
  exit 1
fi

echo "${RESOLVED}"
