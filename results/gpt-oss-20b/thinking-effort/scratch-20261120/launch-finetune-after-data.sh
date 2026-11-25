#!/bin/bash
# Canvas #18 control: start the matched released-head fine-tune campaign once the
# scratch campaign data stage is complete (it reuses that exact corpus).
set -euo pipefail
T=/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort
C=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-effort-scratch
FULL=$T/scratch18/scratch-20261120-f7898022
FT=$T/scratch18/finetune-20261120-f7898022
P1=/users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/campaign-20261027-f409b9da
Q=/users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/phase2-prerequisites/prereq-20261028-d
until [ -f $FULL/data-status.json ] && grep -q "\"complete\"" $FULL/data-status.json; do
  if grep -q "\"failed\"" $FULL/controller-status.json 2>/dev/null; then echo "scratch campaign failed; not starting fine-tune"; exit 1; fi
  sleep 300
done
cd $C
exec env PYTHONPATH=$C/src ~/venvs/apertus-bench/bin/python -m apertus_bench.effort_training_submit --output $FT --phase1-reference $P1 --upstream $T/upstream/speculators-e3041ddb --qualification $Q --qualification-job 3782492 --data-reference $FULL --config $C/experiments/thinking-effort/phase2-scratch-finetune.yaml --partition normal --time 04:00:00
