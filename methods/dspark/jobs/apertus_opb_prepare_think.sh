#!/bin/bash
# Merge the regenerated shards and build the speculators training dataset (packed at 8192).
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; R=$SPEC/runs/apertus_opb; MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
TAG=ghcr; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v speculators >/dev/null || speculators() { python -m speculators "$@"; }
cd $SPEC/repos/speculators; stamp() { echo "=== [$(date +%T)] $(date +%s) $*"; }
grep THINK $R/regen/status.txt; ls -la $R/regen/opb_think_shard*.jsonl | awk '{print $5, $9}'
cat $R/regen/opb_think_shard0.jsonl $R/regen/opb_think_shard1.jsonl $R/regen/opb_think_shard2.jsonl $R/regen/opb_think_shard3.jsonl > $R/opb_think_all.jsonl
stamp "merged: $(wc -l < $R/opb_think_all.jsonl) rows"
t0=$(date +%s); speculators prepare-data --model $MODEL --trust-remote-code --data $R/opb_think_all.jsonl --output $R/data_think --token-freq-path $R/data_think/token_freq.pt --no-skip-token-freq --seq-length 8192 > $R/prepare_think.log 2>&1; rc=$?
stamp "prepare rc=$rc in $(( $(date +%s)-t0 ))s"; tail -3 $R/prepare_think.log | cut -c1-200; ls -la $R/data_think | head; cat $R/data_think/dataset_info.json 2>/dev/null | head -c 400; echo
[ $rc -eq 0 ] && echo PREPARE_THINK_OK >> $R/status.txt || echo PREPARE_THINK_FAILED >> $R/status.txt
P=/capstor/store/cscs/userlab/sm94/datasets/apertus15_8b_regenerated; mkdir -p $P && cp $R/opb_think_all.jsonl $P/open-perfectblend_apertus15-8b_thinking-on_seed0.jsonl && cp $R/prompts/MANIFEST.json $P/open-perfectblend_prompts_MANIFEST.json && chmod -R g+rwX $P && stamp "copied the regenerated corpus to $P for the project"
