#!/usr/bin/env bash
# Run a list of Stage A deployments strictly one after another.
# Each line of PLAN: "ARM PHASE BLOCK_ID [DEPTH] [PROFILE_TRACE]".
#   PLAN=methods/eagle/configs/8b/a5-plan.txt EAGLE_HEAD=... setsid nohup ./methods/eagle/launch/stagea-campaign.sh &
set -uo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
PLAN="${PLAN:?set PLAN}"
LOG="${CAMPAIGN_LOG:-${REPO_ROOT}/results/8b/eagle/campaign-$(date -u +%Y%m%dT%H%M%SZ).log}"
while read -r arm phase block depth trace; do
  case "${arm}" in ""|\#*) continue ;; esac
  while squeue -h -u "${USER}" -p debug -o '%j' | grep -q '^sml_'; do sleep 30; done
  echo "$(date -u -Iseconds) start ${arm} ${phase} ${block} ${depth:-} ${trace:-}" >> "${LOG}"
  env ARM="${arm}" PHASE="${phase}" BLOCK_ID="${block}" \
    ${depth:+DEPTH="${depth}"} ${trace:+PROFILE_TRACE="${trace}"} \
    "${LAUNCH_DIR}/eagle8b-measure.sh" >> "${LOG}" 2>&1 < /dev/null  # srun would eat the plan
  echo "$(date -u -Iseconds) end ${arm} ${phase} ${block} rc=$?" >> "${LOG}"
done < "${PLAN}"
echo "$(date -u -Iseconds) campaign done" >> "${LOG}"
