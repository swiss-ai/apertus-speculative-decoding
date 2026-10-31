if [ -z "${BASH_VERSION:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

if [ -n "${ALPS_EXTENDED_IMAGES_ENV_SOURCED:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
export ALPS_EXTENDED_IMAGES_ENV_SOURCED=1

if [ -f /opt/alps/env/alps-versions.env ]; then
    . /opt/alps/env/alps-versions.env
fi

if [ -f /opt/alps/env/alps-runtime.env ]; then
    . /opt/alps/env/alps-runtime.env
fi

if [ -f /opt/alps/env/alps-runtime-warning.sh ]; then
    . /opt/alps/env/alps-runtime-warning.sh
fi
