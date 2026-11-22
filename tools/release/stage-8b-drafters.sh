#!/usr/bin/env bash
# Assemble the upload folders of the 8B drafter release on Clariden scratch and
# verify every file against release/apertus15-8b-drafters/<head>/MANIFEST.json.
# It does NOT upload anything. Run it only after the leads approve the release
# (docs/release/hf-release-checklist.md).
#
#   tools/release/stage-8b-drafters.sh [OUT_DIR]
#
# Run from a checkout of this branch on Clariden. Weights are copied from the
# shared store (DSpark) and fetched read-only from the private pre-upload repo
# (EAGLE: the only complete copy left after the 2026-11-20 scratch purge).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RELEASE_DIR="${REPO_ROOT}/release/apertus15-8b-drafters"
OUT_DIR="${1:-/iopsstor/scratch/cscs/faruk_zahiragic/hf-release-8b-drafters-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-${HOME}/miniconda/bin/python}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${OUT_DIR}/.hf-cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-${OUT_DIR}/.hf-cache/xet}"

if [ -e "${OUT_DIR}/dspark-e10-thinking-off" ] || [ -e "${OUT_DIR}/eagle31-e10-thinking-off" ]; then
  echo "refusing to overwrite an existing staging dir: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

for head in dspark-e10-thinking-off eagle31-e10-thinking-off; do
  "${PYTHON}" - "${REPO_ROOT}" "${RELEASE_DIR}/${head}" "${OUT_DIR}/${head}" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

repo, src, out = (Path(p) for p in sys.argv[1:4])
manifest = json.loads((src / "MANIFEST.json").read_text())
out.mkdir(parents=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


lines = []
for entry in manifest["files"]:
    rel, source = entry["path"], entry["source"]
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("repo:"):
        shutil.copyfile(repo / source[5:].split(" ")[0], dest)
    elif source.startswith("clariden:"):
        shutil.copyfile(source[9:].split(" ")[0], dest)
    elif source.startswith("hf-private:"):
        from huggingface_hub import hf_hub_download  # read-only download

        repo_rev, _, file_name = source[11:].split(" ")[0].rpartition("/")
        repo_id, _, revision = repo_rev.partition("@")
        cached = hf_hub_download(repo_id, file_name, revision=revision)
        shutil.copyfile(cached, dest)
    else:
        sys.exit(f"unknown source for {rel}: {source}")
    actual = sha256(dest)
    if entry["sha256"] is not None and actual != entry["sha256"]:
        sys.exit(f"{rel}: sha256 {actual} != manifest {entry['sha256']}")
    lines.append(f"{actual}  {rel}")
    print(f"ok {out.name}/{rel}")
(out / "SHA256SUMS").write_text("\n".join(sorted(lines, key=lambda s: s.split("  ")[1])) + "\n")
PY
done

echo "staged in ${OUT_DIR}; nothing was uploaded"
