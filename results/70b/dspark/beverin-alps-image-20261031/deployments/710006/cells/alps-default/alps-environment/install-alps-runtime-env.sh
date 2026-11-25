#!/usr/bin/env bash
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

install_alps_runtime_env() {
    [[ "$#" -gt 0 ]] || die "at least one runtime env fragment is required"

    local runtime_dir="/opt/alps/env"
    local source_hook="${runtime_dir}/source-alps-env.sh"
    local runtime_env="${runtime_dir}/alps-runtime.env"
    local fragment

    # The source hook is copied by the Containerfile. It is the only file linked
    # into shell startup locations; it then sources the generated runtime env.
    [[ -f "${source_hook}" ]] || die "missing source hook: ${source_hook}"

    # Fragments are passed in family-specific order by the Containerfile. The
    # order matters because later defaults may intentionally override earlier
    # ones through the shared defvar semantics.
    for fragment in "$@"; do
        [[ -f "${fragment}" ]] || die "missing runtime env fragment: ${fragment}"
    done

    mkdir -p "${runtime_dir}" /etc/profile.d

    # Build the single runtime file consumed by source-alps-env.sh. Keeping the
    # merge here avoids duplicating cat/chmod/link logic in each base family.
    cat "$@" > "${runtime_env}"
    chmod 0644 "$@" "${runtime_env}" "${source_hook}"

    # /etc/profile.d covers interactive login shells and is safe for both CUDA
    # and ROCm images.
    ln -sf "${source_hook}" /etc/profile.d/99-alps-env.sh

    # NVIDIA base images also source /opt/nvidia/entrypoint.d hooks. Keep this
    # opt-in so ROCm images do not create NVIDIA-specific paths.
    if [[ "${ALPS_NVIDIA_ENTRYPOINT_HOOK:-0}" == "1" ]]; then
        mkdir -p /opt/nvidia/entrypoint.d
        ln -sf "${source_hook}" /opt/nvidia/entrypoint.d/99-alps-env.sh
    fi
}

install_alps_runtime_env "$@"
