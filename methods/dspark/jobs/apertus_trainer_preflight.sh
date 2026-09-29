#!/bin/bash
# Trainer-side preflight for Apertus 1.5 8B in a given image: can transformers load the config/tokenizer, can speculators.train build (dry-run) an Eagle3 / DFlash draft against it?
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; TAG=${TAG:-img}; OUT=$SPEC/runs/apertus_trainer_preflight_$TAG; mkdir -p $OUT; rm -f $OUT/status.txt
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
EX=$SPEC/venvs/apertus-extra-$TAG; mkdir -p $EX; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
python - <<PY
import torch, transformers, vllm; print("torch", torch.__version__, "| transformers", transformers.__version__, "| vllm", vllm.__version__)
from transformers import AutoConfig, AutoTokenizer
M="$MODEL"
try:
    c=AutoConfig.from_pretrained(M, trust_remote_code=True); tc=getattr(c,"text_config",c)
    print("config:", type(c).__name__, "| text:", type(tc).__name__, "| layers", tc.num_hidden_layers, "hidden", tc.hidden_size, "vocab", tc.vocab_size, "out_vocab", getattr(tc,"output_vocab_size",None))
    open("$OUT/status.txt","a").write("CONFIG_OK\\n")
except Exception as e: print("config FAILED:", repr(e)[:300]); open("$OUT/status.txt","a").write("CONFIG_FAILED\\n")
try:
    t=AutoTokenizer.from_pretrained(M, trust_remote_code=True); print("tokenizer:", type(t).__name__, "| len", len(t), "| think tokens:", t.encode("<|inner_prefix|>", add_special_tokens=False))
    open("$OUT/status.txt","a").write("TOKENIZER_OK\\n")
except Exception as e: print("tokenizer FAILED:", repr(e)[:300]); open("$OUT/status.txt","a").write("TOKENIZER_FAILED\\n")
try:
    import speculators; print("speculators", getattr(speculators,"__version__","?"), "imported"); open("$OUT/status.txt","a").write("SPEC_IMPORT_OK\\n")
except Exception as e: print("speculators import FAILED:", repr(e)[:300]); open("$OUT/status.txt","a").write("SPEC_IMPORT_FAILED\\n")
PY
cd $SPEC/repos/speculators
for typ in eagle3 dflash; do
  echo "== dry-run build: $typ"; t0=$(date +%s)
  python -m speculators.train --verifier-name-or-path $MODEL --speculator-type $typ --draft-vocab-size 32000 --target-layer-ids 2 16 29 --save-path $OUT/dryrun_$typ --dry-run --data-path /nonexistent --on-missing raise > $OUT/dryrun_$typ.log 2>&1; rc=$?
  echo "   rc=$rc in $(( $(date +%s)-t0 ))s"; [ $rc -eq 0 ] && { ls $OUT/dryrun_$typ | head -5 | tr "\n" " "; echo; echo "DRYRUN_${typ}_OK" >> $OUT/status.txt; } || { grep -E "Error|error|Traceback|raise" $OUT/dryrun_$typ.log | grep -v INFO | tail -6 | cut -c1-220; echo "DRYRUN_${typ}_FAILED" >> $OUT/status.txt; }
done
echo "== status: $(cat $OUT/status.txt | tr '\n' ' ')"
