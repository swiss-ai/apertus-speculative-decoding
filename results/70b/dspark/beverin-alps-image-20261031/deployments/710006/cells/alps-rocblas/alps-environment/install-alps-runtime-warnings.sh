#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR(alps-runtime-install): $*" >&2; exit 1; }
info() { echo "INFO(alps-runtime-install): $*" >&2; }

# ----------------------------
# Args
# ----------------------------
PYTHON_BIN="python3"
BASH_BIN="bash"

WARNING_TXT="/opt/alps/env/alps-runtime-warning.txt"
WARN_SH="/opt/alps/env/alps-runtime-warning.sh"
WARN_PY="/opt/alps/env/alps-runtime-warning.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:?}"; shift 2;;
    --bash)   BASH_BIN="${2:?}"; shift 2;;
    --txt)    WARNING_TXT="${2:?}"; shift 2;;
    --sh)     WARN_SH="${2:?}"; shift 2;;
    --py)     WARN_PY="${2:?}"; shift 2;;
    -h|--help)
      cat <<EOF
Usage:
  $0 [--python python3] [--bash bash] [--txt /opt/alps/env/alps-runtime-warning.txt] \\
     [--sh /opt/alps/env/alps-runtime-warning.sh] [--py /opt/alps/env/alps-runtime-warning.py]
EOF
      exit 0
      ;;
    *) die "Unknown argument: $1";;
  esac
done

[[ -r "$WARNING_TXT" ]] || die "missing/unreadable static warning text: $WARNING_TXT"
[[ -r "$WARN_SH" ]]     || die "missing/unreadable bash warning script: $WARN_SH"
[[ -r "$WARN_PY" ]]     || die "missing/unreadable python warning script: $WARN_PY"
command -v "$BASH_BIN" >/dev/null || die "bash not found: $BASH_BIN"
command -v "$PYTHON_BIN" >/dev/null || die "python not found: $PYTHON_BIN"

# ----------------------------
# Discover BASH_ENV
# ----------------------------
bash_env="$("$BASH_BIN" -c 'printf "%s" "${BASH_ENV-}"' 2>/dev/null || true)"
[[ -n "$bash_env" ]] || die "BASH_ENV is empty/unset in non-interactive bash; cannot install bash warnings"
[[ -f "$bash_env" ]] || die "BASH_ENV points to non-existent file: $bash_env"
[[ -w "$bash_env" ]] || die "BASH_ENV is not writable: $bash_env"
info "Detected BASH_ENV=$bash_env"

# ----------------------------
# Patch BASH_ENV
# ----------------------------
bash_marker="# ALPS_RUNTIME_WARNING_HOOK"
bash_hook="$(cat <<EOF
$bash_marker
# Source Alps runtime warning
if [[ -r "$WARN_SH" ]]; then
  # shellcheck disable=SC1090
  source "$WARN_SH"
fi

EOF
)"

if grep -Fqx "$bash_marker" "$bash_env"; then
  info "Bash hook already present in $bash_env"
elif grep -Fq "$WARN_SH" "$bash_env"; then
  info "Bash env already sources $WARN_SH"
else
  # Prepend the fallback hook so it still runs if the existing BASH_ENV returns
  # early for non-interactive shells. source-alps-env.sh already sources the
  # warning after runtime env setup, and is handled by the branch above.
  tmp="$(mktemp)"
  echo "$bash_hook" > "$tmp"
  cat "$bash_env" >> "$tmp"
  cat "$tmp" > "$bash_env"
  rm -f "$tmp"
  grep -Fqx "$bash_marker" "$bash_env" || die "failed to patch $bash_env"
  info "Prepended bash warning hook to $bash_env"
fi

# ----------------------------
# Patch Python stdlib sitecustomize.py
# ----------------------------
stdlib_dir="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])' 2>/dev/null || true)"
[[ -n "$stdlib_dir" ]] || die "could not determine stdlib dir via $PYTHON_BIN"
[[ -d "$stdlib_dir" ]] || die "stdlib dir does not exist: $stdlib_dir"

sitecustomize="${stdlib_dir}/sitecustomize.py"
py_marker="# ALPS_RUNTIME_WARNING_HOOK"

py_hook="$(cat <<EOF
$py_marker
# Loads external Alps runtime warning logic (kept outside stdlib for maintainability).
def _alps_runtime_warn():
    try:
        import importlib.util
        _p = r"${WARN_PY}"
        _spec = importlib.util.spec_from_file_location("alps_runtime_warnings", _p)
        if _spec is None or _spec.loader is None:
            return
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)  # type: ignore[attr-defined]
        if hasattr(_m, "maybe_warn"):
            _m.maybe_warn()
    except Exception:
        # Never break python startup
        return

_alps_runtime_warn()
EOF
)"

# Create if missing
if [[ ! -f "$sitecustomize" ]]; then
  info "Creating $sitecustomize"
  install -m 0644 /dev/null "$sitecustomize"
fi
[[ -w "$sitecustomize" ]] || die "sitecustomize.py not writable: $sitecustomize"

if grep -Fqx "$py_marker" "$sitecustomize"; then
  info "Python hook already present in $sitecustomize"
else
  printf "\n%s\n" "$py_hook" >> "$sitecustomize"
  grep -Fqx "$py_marker" "$sitecustomize" || die "failed to patch $sitecustomize"
  info "Patched python stdlib file: $sitecustomize"
fi

# ----------------------------
# Sanity checks
# ----------------------------
# Verify hooks are reachable
info "Sanity: bash non-interactive sources BASH_ENV"
"$BASH_BIN" -c 'true' >/dev/null 2>&1 || die "bash sanity check failed"
info "Sanity: python imports sitecustomize"
"$PYTHON_BIN" -c 'import site; print("ok")' >/dev/null 2>&1 || die "python sanity check failed"

info "Done."
