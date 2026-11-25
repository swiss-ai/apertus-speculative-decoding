#! /usr/bin/env bash

set -euo pipefail
shopt -s lastpipe nullglob

# Add sbin to the path, as ldconfig is often in it, and sbin often not in the path
export PATH="${PATH}:/usr/sbin:/sbin"
source "${ENROOT_LIBRARY_PATH}/common.sh"

netstack_src=${OCI_ANNOTATION_com__hooks__netstack__source:-artifact}

# Exit if this hook was disabled.
if   [ "${OCI_ANNOTATION_com__hooks__cxi__enabled:-true}" != "true" ] \
  && [ "${OCI_ANNOTATION_com__hooks__aws_ofi_nccl__enabled:-}" != "true" ]
then
    exit 0
fi

# Check required commands
common::checkcmd grep sed ldd ldconfig awk

# Get netstack info.
# NOTE: tha main logic of "netstack_init_info" is the same in all hooks.
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

libfabric_host_path=""
if [ "${netstack_src}" == "host" ]; then
  libfabric_host_path=/opt/cray/libfabric/host/lib64/libfabric.so.1
elif [ "${netstack_src}" == "artifact" ]; then
  libfabric_host_path=${host_netstack_path}/libfabric.so.1
fi

echo "Using host libfabric: ${libfabric_host_path}" >> "${ENROOT_ROOTFS}/cxi-hook.log"

# Try to obtain paths to libfabric libs inside container from the dynamic linker cache.
ldconfig="$(set -- "" "${ENROOT_ROOTFS}"/sbin/ldconfig*; echo "${@: -1}")"
libfabric_array=()
readarray -t libfabric_array < <(${ldconfig:-ldconfig} -r "${ENROOT_ROOTFS}" -p | grep libfabric.so.1 | sed 's/.*=> *\(.*\)/\1/' || true)

# Also mount libfabric in a default location (useful if no replacement happened and as fallback)
libfabric_array+=("/usr/lib64/libfabric.so.1")

# Replace libfabric libs found inside the container
for container_libfabric in "${libfabric_array[@]}"; do
    echo "Performing mount to: ${container_libfabric}" >> "${ENROOT_ROOTFS}/cxi-hook.log"
    cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
${libfabric_host_path} ${container_libfabric} none x-create=file,bind,ro,nosuid,nodev,private
EOF
done

# Mount dependency libraries (only for the "host" source)
if [ "${netstack_src}" == "host" ]; then 
    cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
/lib64/libcom_err.so.2 /lib64/libcom_err.so.2 none x-create=file,bind,ro,nosuid,nodev,private
/opt/xpmem/lib64/libxpmem.so.0 /usr/lib64/libxpmem.so.0 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libatomic.so.1 /usr/lib64/libatomic.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libbrotlicommon.so.1 /usr/lib64/libbrotlicommon.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libbrotlidec.so.1 /usr/lib64/libbrotlidec.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libcrypto.so.1.1 /usr/lib64/libcrypto.so.1.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libcurl.so.4 /usr/lib64/libcurl.so.4 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libcxi.so.1 /usr/lib64/libcxi.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libgssapi_krb5.so.2 /usr/lib64/libgssapi_krb5.so.2 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libidn2.so.0 /usr/lib64/libidn2.so.0 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libjitterentropy.so.3 /usr/lib64/libjitterentropy.so.3 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libjson-c.so.5 /usr/lib64/libjson-c.so.5 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libk5crypto.so.3 /usr/lib64/libk5crypto.so.3 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libkeyutils.so.1 /usr/lib64/libkeyutils.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libkrb5.so.3 /usr/lib64/libkrb5.so.3 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libkrb5support.so.0 /usr/lib64/libkrb5support.so.0 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/liblber-2.4.so.2 /usr/lib64/liblber-2.4.so.2 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libldap_r-2.4.so.2 /usr/lib64/libldap_r-2.4.so.2 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libnghttp2.so.14 /usr/lib64/libnghttp2.so.14 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libnl-3.so.200 /usr/lib64/libnl-3.so.200 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libpcre.so.1 /usr/lib64/libpcre.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libpsl.so.5 /usr/lib64/libpsl.so.5 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libsasl2.so.3 /usr/lib64/libsasl2.so.3 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libselinux.so.1 /usr/lib64/libselinux.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libssh.so.4 /usr/lib64/libssh.so.4 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libssl.so.1.1 /usr/lib64/libssl.so.1.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libunistring.so.2 /usr/lib64/libunistring.so.2 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libz.so.1 /usr/lib64/libz.so.1 none x-create=file,bind,ro,nosuid,nodev,private
/usr/lib64/libzstd.so.1 /usr/lib64/libzstd.so.1 none x-create=file,bind,ro,nosuid,nodev,private

EOF
fi

# Mount additional generic files/dirs
cat << EOF | enroot-mount --root "${ENROOT_ROOTFS}" -
/var/spool/slurmd /var/spool/slurmd none x-create=dir,bind,rw,nosuid,noexec,nodev,private
/var/lib/hugetlbfs /var/lib/hugetlbfs none x-create=dir,rbind,rw,nosuid,nodev,private

EOF

# Refresh the dynamic linker cache to include newly mounted libs
cat << EOF > "${ENROOT_ROOTFS}/etc/ld.so.conf.d/enroot-cxi-hook.conf"
/lib64
/usr/lib64
EOF

## Set the correct value for striping in mpich
#
# Only with message sized > 100MB  the overhead of striping is offset by the extra
# bandwith gained by using multiple nics with slingshot.
# The default of 1MB is much too low and leads to a dip in performance that can be 
# seen for example in the osu_bw benchmark
echo "MPIR_CVAR_CH4_OFI_MULTI_NIC_STRIPING_THRESHOLD=100000000" >> "${ENROOT_ENVIRON}"

if ! ${ldconfig:-ldconfig} -r "${ENROOT_ROOTFS}" >> "${ENROOT_ROOTFS}/cxi-hook.log" 2>&1; then
    common::err "Failed to refresh the dynamic linker cache"
fi

ldcache_libs=$(${ldconfig:-ldconfig} -C "${ENROOT_ROOTFS}/etc/ld.so.cache" -p | grep libfabric)
echo "Libfabric libs from ldcache after refresh: ${ldcache_libs}" >> "${ENROOT_ROOTFS}/cxi-hook.log"

