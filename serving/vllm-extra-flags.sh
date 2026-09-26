#!/usr/bin/env bash
# Print additional vLLM CLI flags from environment variables.
# Unset/empty values mean "do not pass the flag; keep the engine default".
# Launchers execute this and append the printed tokens to --framework-args.
set -euo pipefail

extra=()

if [ "${ENABLE_PREFIX_CACHING:-}" = "0" ]; then
  extra+=(--no-enable-prefix-caching)
elif [ "${ENABLE_PREFIX_CACHING:-}" = "1" ]; then
  extra+=(--enable-prefix-caching)
fi

if [ "${ASYNC_SCHEDULING:-}" = "0" ]; then
  extra+=(--no-async-scheduling)
elif [ "${ASYNC_SCHEDULING:-}" = "1" ]; then
  extra+=(--async-scheduling)
fi

if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
  extra+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [ -n "${MAX_NUM_SEQS:-}" ]; then
  extra+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if [ -n "${MAX_NUM_SCHEDULED_TOKENS:-}" ]; then
  extra+=(--max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}")
fi

if [ "${ENFORCE_EAGER:-}" = "1" ]; then
  extra+=(--enforce-eager)
fi

if [ -n "${PROFILER_DIR:-}" ]; then
  extra+=(--profiler-config.profiler torch)
  extra+=(--profiler-config.torch_profiler_dir "${PROFILER_DIR}")
fi

if [ "${#extra[@]}" -gt 0 ]; then
  printf '%q ' "${extra[@]}"
fi
