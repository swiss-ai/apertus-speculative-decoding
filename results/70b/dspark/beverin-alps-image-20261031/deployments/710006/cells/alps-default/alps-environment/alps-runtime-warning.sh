#!/usr/bin/env bash

_alps_warn_rdzv_get_enabled() {
  # Only warn in a Slurm task context
  [[ -n "${SLURM_STEP_ID:-}" || -n "${SLURM_JOB_ID:-}" ]] || return 0

  # Only once per process
  [[ -n "${ALPS_WARN_RDZV_DONE:-}" ]] && return 0
  ALPS_WARN_RDZV_DONE=1
  export ALPS_WARN_RDZV_DONE

  # Only relevant when cxi + alt_read
  [[ "${FI_PROVIDER:-}" == "cxi" ]] || return 0
  [[ "${FI_CXI_RDZV_PROTO:-}" == "alt_read" ]] || return 0

  # If SLURM_NETWORK is missing or does not contain disable_rdzv_get, warn.
  local net="${SLURM_NETWORK-}"
  case "$net" in
    *disable_rdzv_get*) return 0 ;;
  esac

  # Only rank 0 warns
  if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
    return 0
  fi

  : "${ALPS_WARNING_TXT:=/opt/alps/env/alps-runtime-warning.txt}"

  {
    # Static part (if present), then context lines.
    if [ -r "${ALPS_WARNING_TXT-}" ]; then
      cat "${ALPS_WARNING_TXT-}" >&2
    else
      echo "WARNING(alps-runtime): Alps runtime warning text missing: ${ALPS_WARNING_TXT-<unset>}" >&2
    fi

    echo "" >&2
    echo "Context:" >&2
    echo "  SLURM_NETWORK=${SLURM_NETWORK-<unset>}" >&2
    echo "  FI_PROVIDER=${FI_PROVIDER-<unset>}" >&2
    echo "  FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO-<unset>}" >&2
    echo "" >&2
  } || true

  return 0
}

# Call immediately on sourcing.
_alps_warn_rdzv_get_enabled
unset -f _alps_warn_rdzv_get_enabled
