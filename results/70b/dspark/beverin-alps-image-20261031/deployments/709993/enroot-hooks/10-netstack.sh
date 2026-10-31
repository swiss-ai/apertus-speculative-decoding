#! /usr/bin/env bash

set -euo pipefail
shopt -s lastpipe nullglob

source "${ENROOT_LIBRARY_PATH}/common.sh"

netstack_src=${OCI_ANNOTATION_com__hooks__netstack__source:-artifact}

# Do a basic sanitization on the "netstack.host" value.
if [ "${netstack_src}" != "host" ] && [ "${netstack_src}" != "artifact" ]; then
    common::err "Invalid netstack source: ${netstack_src}"
fi

# Exit if netstack is not from the artifact.
if [ "${netstack_src}" != "artifact" ]; then
    exit 0
fi

# Check required commands
common::checkcmd awk

# Get netstack info.
# NOTE: the main logic of "netstack_init_info" is the same in all hooks.
netstack_init_info() {
    # Derive the netstack version.
    # Precedence: i) EDF annotation, ii) TF manifest definition, iii) <error>.
    export netstack_ver=""
    if [[ -n "${OCI_ANNOTATION_com__hooks__netstack__version:-}" ]]; then
        export netstack_ver=${OCI_ANNOTATION_com__hooks__netstack__version}
    elif [[ -n "26.08.1" ]]; then
        export netstack_ver="26.08.1"
    fi

    if [[ -z "$netstack_ver" ]]; then
      common::log ERROR "[NETSTACK hook] Netstack version not specified."
      exit 1
    fi

    # Define the host netstack base path.
    host_netstack_base_path=/capstor/store/cscs/cscs/public/containers/netstack/$(uname -m)/${netstack_ver}

    # Derive the netstack name.
    # Precedence: i) EDF annotation, ii) TF manifest definition, iii) <error>.
    netstack_name=""
    if [[ -n "${OCI_ANNOTATION_com__hooks__netstack__name:-}" ]]; then
        netstack_name="${OCI_ANNOTATION_com__hooks__netstack__name}"
    elif [[ -n "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0" ]]; then
        netstack_name="gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
    fi

    if [[ -z "$netstack_name" ]]; then
      common::log ERROR "[NETSTACK hook] Netstack name not specified."
      exit 1
    fi

    # Define netstack paths.
    export host_netstack_path=${host_netstack_base_path}/${netstack_name}
    export container_netstack_path=/opt/cscs/netstack

    # Check if the host netstack path is valid.
    if [[ ! -d "${host_netstack_path}" ]]; then
        common::log ERROR "[NETSTACK hook] Invalid netstack path '${host_netstack_path}'."
        if [[ -d "${host_netstack_base_path}" ]]; then
            common::log ERROR "[NETSTACK hook] Valid netstack versions: $(cd ${host_netstack_base_path}; ls * -d | tr '\n' ' ')"
        fi
        exit 1
    fi
}
netstack_init_info

# Mount dependency directory.
cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
${host_netstack_path} ${container_netstack_path} none x-create=dir,bind,rw,nosuid,exec,nodev,private
EOF

