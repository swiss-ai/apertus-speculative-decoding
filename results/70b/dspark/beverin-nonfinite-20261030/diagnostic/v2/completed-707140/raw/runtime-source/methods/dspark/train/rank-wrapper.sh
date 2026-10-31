#!/usr/bin/env bash
# torchrun --no-python wrapper (Yu's train_rank_wrapper.sh): every trainer rank
# of a node reads target features from that node's extraction server.
exec python3 ${PY_PRELUDE:-} -m speculators.train "$@" --vllm-endpoint "http://localhost:${DSPARK_PORT:-8001}/v1"
