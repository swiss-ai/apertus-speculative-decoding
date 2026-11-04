#!/usr/bin/env bash
# torchrun --no-python wrapper (Yu's train_rank_wrapper.sh): every trainer rank
# of a node reads target features from that node's extraction server, or with
# DSPARK_PER_RANK=1 from its own (port DSPARK_PORT + LOCAL_RANK).
port=${DSPARK_PORT:-8001}
[ "${DSPARK_PER_RANK:-0}" = 1 ] && port=$(( port + ${LOCAL_RANK:-0} ))
exec python3 -m speculators.train "$@" --vllm-endpoint "http://localhost:${port}/v1"
