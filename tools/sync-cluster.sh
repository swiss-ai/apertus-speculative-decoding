#!/usr/bin/env bash
# Fast-forward the Clariden checkout to this branch (the cluster cannot reach GitHub).
#   tools/sync-cluster.sh [ssh-host] [remote-checkout]
# Ships the missing commits as a git bundle. Untracked cluster copies of files
# that the new commits add (run records pulled back and committed here) are
# set aside, compared after the merge, and kept only if they differ.
set -euo pipefail
HOST="${1:-clariden1}"
REMOTE="${2:-apertus-spec-decoding}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BASE="$(ssh -o BatchMode=yes "${HOST}" "git -C ${REMOTE} rev-parse HEAD")"
if [ "${BASE}" = "$(git rev-parse HEAD)" ]; then echo "cluster already at ${BASE:0:7}"; exit 0; fi
BUNDLE="$(mktemp --suffix=.bundle)"
trap 'rm -f "${BUNDLE}"' EXIT
git bundle create -q "${BUNDLE}" "${BASE}..${BRANCH}"
scp -q -o BatchMode=yes "${BUNDLE}" "${HOST}:.sml/sync.bundle"
ssh -o BatchMode=yes "${HOST}" bash -s -- "${REMOTE}" "${BRANCH}" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$1"
git fetch -q ~/.sml/sync.bundle "+refs/heads/$2:refs/remotes/bundle/$2"
aside="$(mktemp -d)"
git diff --name-only --diff-filter=A HEAD "bundle/$2" | while read -r f; do
  if [ -e "$f" ]; then mkdir -p "${aside}/$(dirname "$f")"; mv "$f" "${aside}/$f"; fi
done
git merge -q --ff-only "bundle/$2"
(cd "${aside}" && find . -type f) | while read -r f; do
  if ! cmp -s "${aside}/$f" "$f"; then echo "kept differing cluster copy: $f.cluster"; mv "${aside}/$f" "$f.cluster"; fi
done
rm -rf "${aside}" ~/.sml/sync.bundle
git log --oneline -1
REMOTE_SCRIPT
