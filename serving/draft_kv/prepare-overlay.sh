#!/usr/bin/env bash
# Add drafter-only sources/helpers to an already-built base compatibility overlay.
set -euo pipefail
IMAGE="${1:?image}"
BASE="${2:?base overlay directory}"
OUT="${3:?results directory}"
EXISTING="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${BASE}/draft-kv-source"
OVERLAY="${BASE}/draft-kv-overlay"
[ ! -e "${SOURCE}" ] && [ ! -e "${OVERLAY}" ] || {
  echo "refusing to overwrite drafter KV overlay" >&2; exit 2;
}
mapfile -t TARGETS < <(python3 - "${HERE}/install.py" <<'PY'
import runpy,sys
for name in runpy.run_path(sys.argv[1])["TRANSFORMS"]:
    print("workspace/vllm/" + name)
PY
)
[ "${#TARGETS[@]}" -gt 0 ] || { echo "no drafter overlay targets" >&2; exit 2; }
unsquashfs -q -d "${SOURCE}" "${IMAGE}" "${TARGETS[@]}" > /dev/null
for target in "${TARGETS[@]}"; do
  if [ -f "${BASE}/${target}" ]; then cp "${BASE}/${target}" "${SOURCE}/${target}"; fi
done
NEW="$(python3 "${HERE}/install.py" --source-root "${SOURCE}/workspace/vllm" \
  --output-root "${OVERLAY}" --patch-output "${OUT}/drafter-kv-applied.patch")"
cp "${OVERLAY}/draft-kv-overlay-manifest.json" "${OUT}/draft-kv-overlay-manifest.json"
python3 - "${EXISTING}" "${NEW}" <<'PY'
import sys
mounts = {}
for group in sys.argv[1:]:
    for line in group.splitlines():
        if line.strip():
            mounts[line.split(":", 1)[1]] = line
print("\n".join(mounts.values()))
PY
