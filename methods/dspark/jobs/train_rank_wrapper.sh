#!/bin/bash
# torchrun --no-python wrapper: give each local trainer rank its own extraction server (port 8001 + LOCAL_RANK).
exec python3 -m speculators.train "$@" --vllm-endpoint http://localhost:$((8001 + ${LOCAL_RANK:-0}))/v1
