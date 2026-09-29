#!/bin/bash
# Download swiss-ai/Apertus-v1.5-8B (gated) into the spec HF cache on capstor.
# Needs an HF token whose account accepted the Apertus 1.5 terms:
#   hf auth login            (stores ~/.cache/huggingface/token)   or   export HF_TOKEN=hf_...
export HF_HOME=/capstor/scratch/cscs/zyu/spec/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
LOG=/capstor/scratch/cscs/zyu/spec/logs/download_apertus15_8b.log
echo "start $(date)" >> "$LOG"
hf download swiss-ai/Apertus-v1.5-8B --max-workers 8 >> "$LOG" 2>&1
rc=$?
echo "end $(date) rc=$rc" >> "$LOG"
[ $rc -eq 0 ] && du -sh "$HF_HOME/hub/models--swiss-ai--Apertus-v1.5-8B" >> "$LOG"
exit $rc
