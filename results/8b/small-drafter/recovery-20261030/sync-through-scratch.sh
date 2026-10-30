#!/usr/bin/env bash
set -euo pipefail
TASK_LOCAL=/home/projects/apertus-speculative-decoding/.claude/worktrees/small-drafter
TASK_REMOTE=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter
TASK_HEAD="$(git -C "${TASK_LOCAL}" rev-parse HEAD)"
TASK_BUNDLE="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/source-bundles/small-drafter-${TASK_HEAD}.bundle"
test "$(git -C "${TASK_LOCAL}" branch --show-current)" = small-drafter
TASK_BASE="$(ssh -o BatchMode=yes clariden1 git -C "${TASK_REMOTE}" rev-parse HEAD)"
TASK_TMP="$(mktemp --suffix=.bundle)"
trap 'rm -f "${TASK_TMP}"' EXIT
git -C "${TASK_LOCAL}" bundle create -q "${TASK_TMP}" "${TASK_BASE}..small-drafter"
ssh -o BatchMode=yes clariden1 mkdir -p /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/source-bundles
scp -q -o BatchMode=yes "${TASK_TMP}" "clariden1:${TASK_BUNDLE}"
ssh -o BatchMode=yes clariden1 bash -s -- "${TASK_REMOTE}" "${TASK_BUNDLE}" "${TASK_BASE}" "${TASK_HEAD}" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(git branch --show-current)" = small-drafter
test "$(git rev-parse HEAD)" = "$3"
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q "$2" '+refs/heads/small-drafter:refs/remotes/recovery-bundle/small-drafter'
git merge -q --ff-only refs/remotes/recovery-bundle/small-drafter
test "$(git rev-parse HEAD)" = "$4"
git rev-parse HEAD
sha256sum "$2"
REMOTE
