#!/usr/bin/env python3
import os
import sys

# Single source of truth for the static part (shared with Bash).
WARNING_TXT = os.environ.get("ALPS_WARNING_TXT", "/opt/alps/env/alps-runtime-warning.txt")


def _slurm_context() -> bool:
    return bool(os.environ.get("SLURM_STEP_ID") or os.environ.get("SLURM_JOB_ID"))


def _relevant() -> bool:
    return (
        os.environ.get("FI_PROVIDER", "") == "cxi"
        and os.environ.get("FI_CXI_RDZV_PROTO", "") == "alt_read"
    )


def _rank_is_zero() -> bool:
    # Only rank 0 warns when rank is known; otherwise allow warning.
    procid = os.environ.get("SLURM_PROCID")
    return procid is None or procid == "" or procid == "0"


def _read_static() -> str:
    try:
        with open(WARNING_TXT, "r", encoding="utf-8") as f:
            return f.read().rstrip()
    except Exception:
        return f"WARNING(alps-runtime): missing static text file: {WARNING_TXT}"


def maybe_warn() -> None:
    # Only warn in Slurm context and only for the specific config.
    if not _slurm_context():
        return
    if not _relevant():
        return

    # Only once per process.
    if os.environ.get("ALPS_WARN_RDZV_DONE") == "1":
        return
    os.environ["ALPS_WARN_RDZV_DONE"] = "1"

    # Only rank 0 warns (when rank info exists).
    if not _rank_is_zero():
        return

    slurm_network = os.environ.get("SLURM_NETWORK", "<unset>")
    if "disable_rdzv_get" in slurm_network:
        return

    # Print static header text + common context block (aligned with Bash).
    msg = _read_static()
    sys.stderr.write(msg + "\n\n")
    sys.stderr.write("Context:\n")
    sys.stderr.write(f"  SLURM_NETWORK={slurm_network}\n")
    sys.stderr.write(f"  FI_PROVIDER={os.environ.get('FI_PROVIDER','')}\n")
    sys.stderr.write(f"  FI_CXI_RDZV_PROTO={os.environ.get('FI_CXI_RDZV_PROTO','')}\n\n")


if __name__ == "__main__":
    maybe_warn()
