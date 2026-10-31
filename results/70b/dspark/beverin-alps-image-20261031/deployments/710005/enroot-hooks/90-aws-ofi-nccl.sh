#! /usr/bin/env bash

set -euo pipefail
shopt -s lastpipe nullglob

# Add sbin to the path, as ldconfig is often in it, and sbin often not in the path
export PATH="${PATH}:/usr/sbin:/sbin"
source "${ENROOT_LIBRARY_PATH}/common.sh"

netstack_src=${OCI_ANNOTATION_com__hooks__netstack__source:-artifact}

# Exit if this hook was disabled.
if [ "${OCI_ANNOTATION_com__hooks__aws_ofi_nccl__enabled:-}" != "true" ]; then
    exit 0
fi

# Check required commands
common::checkcmd ldconfig awk

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
}

if [ "${netstack_src}" == "artifact" ]; then
  netstack_init_info
fi

plugin_host_path=""
if [ "${netstack_src}" == "host" ]; then
    case "${OCI_ANNOTATION_com__hooks__aws_ofi_nccl__variant:-}" in
        cuda-dl)
            plugin_host_path="/opt/cscs/aws-ofi-ccl-plugin/cuda-dl/libnccl-net.so" ;;
        cuda12)
            plugin_host_path="/opt/cscs/aws-ofi-ccl-plugin/cuda12/libnccl-net.so" ;;
        cuda13)
            plugin_host_path="/opt/cscs/aws-ofi-ccl-plugin/cuda13/libnccl-net.so" ;;
        rocm5)
            plugin_host_path="/opt/cscs/aws-ofi-ccl-plugin/rocm5/librccl-net.so" ;;
        rocm6)
            plugin_host_path="/opt/cscs/aws-ofi-ccl-plugin/rocm6/librccl-net.so" ;;
        *)
            common::err "[AWS-OFI-NCCL hook] Unknown plugin variant specified: ${OCI_ANNOTATION_com__hooks__aws_ofi_nccl__variant}"
    esac
elif [ "${netstack_src}" == "artifact" ]; then
    plugin_host_path="${host_netstack_path}/libnccl-net.so"
    
    if [ -f "${host_netstack_path}/librccl-net.so" ]; then
        plugin_host_path="${host_netstack_path}/librccl-net.so"
    fi
fi

echo "Using host plugin: ${plugin_host_path}" >> "${ENROOT_ROOTFS}/aws-ofi-nccl-hook.log"

# Obtain basename whether we are working with libnccl-net or librccl-net
plugin_basename=$(basename "${plugin_host_path}" ".so")

# Search for all nccl-net plugins in the image and replace them
ldconfig="$(set -- "" "${ENROOT_ROOTFS}"/sbin/ldconfig*; echo "${@: -1}")"
plugin_array=()
readarray -t plugin_array < <(${ldconfig:-ldconfig} -r "${ENROOT_ROOTFS}" -p | grep "${plugin_basename}" | sed 's/.*=> *\(.*\)/\1/')

# Also mount a copy of the plugin with custom suffix in case there are no plugins to replace and as fallback
plugin_array+=("${OCI_ANNOTATION_com__hooks__aws_ofi_nccl__container_dir:-/usr/lib}/${plugin_basename}-ofi.so")

for container_plugin in "${plugin_array[@]}"; do
    echo "Performing mount to: ${container_plugin}" >> "${ENROOT_ROOTFS}/aws-ofi-nccl-hook.log"
    cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
${plugin_host_path} ${container_plugin} none x-create=file,bind,ro,nosuid,nodev,private
EOF
done

# Mount dependencies (only for the "host" source)
if [ "${netstack_src}" == "host" ]; then
    cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
/usr/lib64/libhwloc.so.15 /usr/lib/libhwloc.so.15 none x-create=file,bind,rw,nosuid,nodev,private
/usr/lib64/libpciaccess.so.0 /usr/lib/libpciaccess.so.0 none x-create=file,bind,rw,nosuid,nodev,private
/usr/lib64/libxml2.so.2 /usr/lib/libxml2.so.2 none x-create=file,bind,rw,nosuid,nodev,private

EOF
fi

# Forces NCCL to use the libfabric plugin
# See: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net
printf "NCCL_NET='AWS Libfabric'\n"  >> "${ENROOT_ENVIRON}"

# Since we replace all existing NCCL plugins with (or just add) our own, and we invoke ldconfig which will
# register the plugin in ldcache by its internal SONAME, we have to force NCCL to load our custom plugin
printf "NCCL_NET_PLUGIN=ofi\n"  >> "${ENROOT_ENVIRON}"

# Common variables for optimal OFI/NCCL behavior
printf "CUDA_CACHE_DISABLE=1\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_DEFAULT_CQ_SIZE=131072\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_DEFAULT_TX_SIZE=16384\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_DISABLE_HOST_REGISTER=1\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_RDZV_EAGER_SIZE=0\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_RDZV_GET_MIN=0\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_RDZV_PROTO=alt_read\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_RDZV_THRESHOLD=0\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_RX_MATCH_MODE=software\n"  >> "${ENROOT_ENVIRON}"
printf "FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216\n"  >> "${ENROOT_ENVIRON}"
printf "FI_MR_CACHE_MAX_COUNT=524288\n"  >> "${ENROOT_ENVIRON}"
printf "FI_MR_CACHE_MAX_SIZE=-1\n"  >> "${ENROOT_ENVIRON}"
printf "FI_MR_CACHE_MONITOR=userfaultfd\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_CROSS_NIC=0\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_NCHANNELS_PER_NET_PEER=4\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_NET_GDR_C2C=1\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_NET_GDR_LEVEL=PHB\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_NET_GDR_READ=1\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_P2P_LEVEL=NVL\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_PROTO=^LL128\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_PXN_DISABLE=1\n"  >> "${ENROOT_ENVIRON}"
printf "NCCL_SOCKET_IFNAME=hsn\n"  >> "${ENROOT_ENVIRON}"

# Refresh the dynamic linker cache to include newly mounted libs and plugins
cat << EOF > "${ENROOT_ROOTFS}/etc/ld.so.conf.d/aws-ofi-nccl-hook.conf"
/lib64
/usr/lib64
EOF
if ! ${ldconfig:-ldconfig} -r "${ENROOT_ROOTFS}" >> "${ENROOT_ROOTFS}/aws-ofi-nccl-hook.log" 2>&1; then
    common::err "Failed to refresh the dynamic linker cache"
fi

ldcache_libs=$(${ldconfig:-ldconfig} -C "${ENROOT_ROOTFS}/etc/ld.so.cache" -p | grep -E '[nr]ccl-net' || echo NONE)
echo "nccl-net plugins from ldcache after refresh: ${ldcache_libs}" >> "${ENROOT_ROOTFS}/aws-ofi-nccl-hook.log"

