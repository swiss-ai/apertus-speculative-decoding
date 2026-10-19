#!/usr/bin/env bash
# Build the trainer checkout and ship it to the cluster (which cannot reach GitHub).
#   methods/dspark/train/setup-speculators.sh [ssh-host] [remote-checkout]
# speculators at Yu's commit 7046ac6 with his patch (pruned output head,
# process-group timeout) and ours (stop and save before the job's time limit;
# stack dumps on SIGUSR1),
# into scratch/speculators here and in the remote checkout.
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${TRAIN_DIR}/../../.." && pwd)"
HOST="${1:-clariden1}"
REMOTE="${2:-/iopsstor/scratch/cscs/$(ssh -o BatchMode=yes "${HOST}" 'echo $USER')/apertus-spec-decoding-dspark70b}"
COMMIT=7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00
DEST="${REPO_ROOT}/scratch/speculators"

if [ ! -d "${DEST}/.git" ]; then
  git init -q "${DEST}"
  git -C "${DEST}" remote add origin https://github.com/vllm-project/speculators.git
fi
git -C "${DEST}" fetch -q --depth 1 origin "${COMMIT}"
git -C "${DEST}" checkout -q --force FETCH_HEAD
git -C "${DEST}" clean -qfd
for p in speculators_7046ac6_daint.patch speculators-deadline-stop.patch speculators-stack-dump.patch; do
  git -C "${DEST}" apply "${TRAIN_DIR}/patches/${p}"
done
git -C "${DEST}" diff --stat | tail -1
ssh -o BatchMode=yes "${HOST}" "mkdir -p ${REMOTE}/scratch"
rsync -a --delete -e "ssh -o BatchMode=yes" "${DEST}/" "${HOST}:${REMOTE}/scratch/speculators/"
echo "speculators ${COMMIT:0:7} + patches at ${HOST}:${REMOTE}/scratch/speculators"
