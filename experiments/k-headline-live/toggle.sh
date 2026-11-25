#!/usr/bin/env bash
# Build (and optionally merge) one GitOps toggle of the DSpark e10 Deployment for the
# live k-headline windows. Only files under the dspark-e10 directory change.
#   toggle.sh build <window> <mode> <pin-node|-> <avoid-node>...   # branch + push + PR
#   toggle.sh merge <window>                                       # rebase check, merge
# Modes:
#   k3-on    the deployed k=3 manifest (ConfigMap unchanged), replicas 1, pinned,
#            off the avoided nodes, plus the client pod
#   plain | dspark-k5 | dspark-k7   prepare.py render of that variant, same pin
#   restore  the deployed k=3 manifest as at rob-poc 9452772, replicas 1, unpinned,
#            off the avoided nodes, client pod removed
set -euo pipefail
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
ROBPOC="${ROBPOC:-/home/projects/rob-poc-k-live}"
REPO=breithorn-poc/rob-poc
D=swissai-research-platform/models/vllm-apertus-dspark-e10
BASE_REV=9452772   # last rob-poc commit before this campaign touched the Deployment
DEPLOYED_SOURCE=a23699122a9c6d2c108e2120a8f81e57da6d7b7c
PY="${PY:-${CODE}/../apertus-speculative-decoding/.venv/bin/python}"

affinity_block() {  # YAML lines for a NotIn hostname rule, 6-space pod-spec indent
  printf '      affinity:\n        nodeAffinity:\n          requiredDuringSchedulingIgnoredDuringExecution:\n'
  printf '            nodeSelectorTerms:\n            - matchExpressions:\n'
  printf '              - key: kubernetes.io/hostname\n                operator: NotIn\n                values:\n'
  for node in "$@"; do printf '                - %s\n' "${node}"; done
}

case "${1:?build|merge}" in
  build)
    WINDOW="${2:?window}" MODE="${3:?mode}" PIN="${4:?pin node or -}"
    shift 4
    [ "$#" -gt 0 ] || { echo "give the nodes to avoid" >&2; exit 2; }
    BRANCH="serving/dspark-e10-k-live-${WINDOW,,}"
    git -C "${ROBPOC}" fetch -q origin
    git -C "${ROBPOC}" checkout -q -B "${BRANCH}" origin/main
    tmp="$(mktemp -d)"
    pin=(); [ "${PIN}" = - ] || pin=(--pin-node "${PIN}")
    avoid=(); for node in "$@"; do avoid+=(--avoid-node "${node}"); done
    case "${MODE}" in
      k3-on)
        PYTHONPATH="${CODE}/src" "${PY}" "${CODE}/serving/kubernetes/prepare.py" "${tmp}/r" \
          --replicas 1 --variant dspark --source-revision "${DEPLOYED_SOURCE}" "${pin[@]}" \
          "${avoid[@]}" --client-pod
        git -C "${ROBPOC}" show "${BASE_REV}:${D}/configmap.yaml" | cmp - "${tmp}/r/configmap.yaml" \
          || { echo "k3 render no longer reproduces the deployed ConfigMap" >&2; exit 2; }
        cp "${tmp}/r/configmap.yaml" "${tmp}/r/deployment.yaml" "${tmp}/r/task-speed-client.yaml" \
          "${ROBPOC}/${D}/"
        ;;
      plain|dspark-k5|dspark-k7)
        git -C "${CODE}" merge-base --is-ancestor HEAD "@{upstream}" \
          || { echo "push the source revision first" >&2; exit 2; }
        PYTHONPATH="${CODE}/src" "${PY}" "${CODE}/serving/kubernetes/prepare.py" "${tmp}/r" \
          --replicas 1 --variant "${MODE}" "${pin[@]}" "${avoid[@]}" --client-pod
        cp "${tmp}/r/configmap.yaml" "${tmp}/r/deployment.yaml" "${tmp}/r/task-speed-client.yaml" \
          "${ROBPOC}/${D}/"
        ;;
      restore)
        [ "${PIN}" = - ] || { echo "restore is unpinned" >&2; exit 2; }
        git -C "${ROBPOC}" show "${BASE_REV}:${D}/configmap.yaml" > "${ROBPOC}/${D}/configmap.yaml"
        git -C "${ROBPOC}" show "${BASE_REV}:${D}/deployment.yaml" \
          | sed 's/^  replicas: 0  # off 2026-11-23:.*$/  replicas: 1/' \
          | awk -v block="$(affinity_block "$@")" \
              '{print} /^        kubernetes.io\/arch: arm64$/ && !done {print block; done=1}' \
          > "${ROBPOC}/${D}/deployment.yaml"
        grep -q '^  replicas: 1$' "${ROBPOC}/${D}/deployment.yaml"
        git -C "${ROBPOC}" rm -q --ignore-unmatch "${D}/task-speed-client.yaml"
        ;;
      *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
    esac
    rm -rf "${tmp}"
    printf '\n%s\n' "Live k-headline window ${WINDOW} ($(date -u +%F)): mode ${MODE}, pin ${PIN}, kept off $*; source swiss-ai/apertus-speculative-decoding $(git -C "${CODE}" rev-parse --short HEAD) (experiment/dspark-e10-k-headline-live)." \
      >> "${ROBPOC}/${D}/README.md"
    git -C "${ROBPOC}" add -A -- "${D}"
    outside="$(git -C "${ROBPOC}" status --porcelain | grep -v " ${D}/" || true)"
    [ -z "${outside}" ] || { echo "changes outside ${D}: ${outside}" >&2; exit 2; }
    git -C "${ROBPOC}" commit -q -m "DSpark e10: live k-headline window ${WINDOW} (${MODE})

Mode ${MODE}, node pin ${PIN}, NotIn affinity for $*. Only the dspark-e10
Deployment, its ConfigMap and the temporary client pod change.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${D}"
    [ -n "${NO_PUSH:-}" ] || git -C "${ROBPOC}" push -q -f -u origin "${BRANCH}" 2>&1 | grep -v '^remote' || true
    git -C "${ROBPOC}" diff --stat origin/main "${BRANCH}"
    ;;
  merge)
    WINDOW="${2:?window}"
    BRANCH="serving/dspark-e10-k-live-${WINDOW,,}"
    git -C "${ROBPOC}" fetch -q origin
    git -C "${ROBPOC}" checkout -q "${BRANCH}"
    # Never revert anyone's commits: the branch must sit on the latest main.
    if ! git -C "${ROBPOC}" merge-base --is-ancestor origin/main HEAD; then
      git -C "${ROBPOC}" rebase -q origin/main || { git -C "${ROBPOC}" rebase --abort; echo "conflict" >&2; exit 5; }
      git -C "${ROBPOC}" push -q -f origin "${BRANCH}" 2>&1 | grep -v '^remote' || true
    fi
    outside="$(git -C "${ROBPOC}" diff --name-only origin/main HEAD | grep -v "^${D}/" || true)"
    [ -z "${outside}" ] || { echo "changes outside ${D}: ${outside}" >&2; exit 2; }
    number="$(gh pr list -R "${REPO}" --head "${BRANCH}" --state open --json number -q '.[0].number')"
    [ -n "${number}" ] || number="$(gh pr create -R "${REPO}" --base main --head "${BRANCH}" \
      --title "DSpark e10: live k-headline window ${WINDOW}" \
      --body "$(git -C "${ROBPOC}" log -1 --format=%b)

Part of the live k=3/5/7 check (swiss-ai/apertus-speculative-decoding, branch \`experiment/dspark-e10-k-headline-live\`). Touches only \`${D}/\`; the fixed-nov23 and Chonk Deployments are not modified, and the NotIn affinity keeps this pod off their nodes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)" | grep -o '[0-9]*$')"
    gh pr merge -R "${REPO}" "${number}" --merge
    gh pr view -R "${REPO}" "${number}" --json number,mergeCommit,mergedAt
    ;;
esac
