# usage: source fork_patch.sh  -> shadows the image's vLLM fork with a patched copy on scratch (aux-hidden-state hooks unwrap the language model only if needed)
PV=$SPEC/vllm_fork_patched_$TAG
if [ ! -f $PV/vllm/__init__.py ]; then
  SRC=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"); echo "   copying $SRC -> $PV ($(du -sh $SRC | cut -f1))"; mkdir -p $PV && cp -r $SRC $PV/ || { echo "FORK_COPY_FAILED" >> $OUT/status.txt; return 1; }
fi
python - "$PV/vllm/model_executor/models/interfaces.py" <<'PY'
import sys,re; p=sys.argv[1]; s=open(p).read()
if "unwrap only if needed (patched)" not in s:
    old1='''        assert hasattr(parent_ref, "model"), (
            "Model instance must have 'model' attribute to set number of layers"
        )
        assert isinstance(parent_ref.model, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        parent_ref.model._set_aux_hidden_state_layers(layers)'''
    new1='''        # unwrap only if needed (patched): get_language_model() may already return the decoder
        holder = parent_ref if isinstance(parent_ref, EagleModelMixin) else getattr(parent_ref, "model", None)
        assert isinstance(holder, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        holder._set_aux_hidden_state_layers(layers)'''
    old2='''        assert hasattr(parent_ref, "model"), (
            "Model instance must have 'model' attribute to get number of layers"
        )
        assert hasattr(parent_ref.model, "layers"), (
            "Model instance must have 'layers' attribute to get number of layers"
        )
        num_layers = len(parent_ref.model.layers)'''
    new2='''        holder = parent_ref if hasattr(parent_ref, "layers") else getattr(parent_ref, "model", None)  # unwrap only if needed (patched)
        assert hasattr(holder, "layers"), (
            "Model instance must have 'layers' attribute to get number of layers"
        )
        num_layers = len(holder.layers)'''
    assert old1 in s and old2 in s, "fork interfaces.py text differs; patch not applied"
    s=s.replace(old1,new1).replace(old2,new2); open(p,"w").write(s); print("   fork interfaces.py patched")
else: print("   fork interfaces.py already patched")
PY
export PYTHONPATH=$PV:${PYTHONPATH:-}
python -c "import vllm, os; assert vllm.__file__.startswith('$PV'), vllm.__file__; print('   vllm now from', os.path.dirname(vllm.__file__))" || echo "FORK_SHADOW_FAILED" >> $OUT/status.txt
# V2-runner speculator loaders: unwrap the language model only if it still wraps a .model
for f in eagle dflash dspark; do python - "$PV/vllm/v1/worker/gpu/spec_decode/$f/utils.py" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
old="    target_inner = target_language_model.model\n"
new="    target_inner = target_language_model.model if hasattr(target_language_model, \"model\") else target_language_model  # unwrap only if needed (patched)\n"
if old in s: s=s.replace(old,new); open(p,"w").write(s); print("   patched", p.split("spec_decode/")[1])
elif "unwrap only if needed (patched)" in s: print("   already patched", p.split("spec_decode/")[1])
else: print("   WARNING: pattern not found in", p)
PY
done
# --- DSpark anchor layout (added 2026-09-25): the fork's speculators->HF translation hard-codes dspark_bonus_anchor=True (1+N block),
#     but speculators trains DSpark with sample_from_anchor=True (anchor-as-first, N slots) by default -> off-by-one in every draft slot.
AL=$PV/vllm/transformers_utils/configs/speculators/algos.py
if grep -q 'pre_trained_config\["dspark_bonus_anchor"\] = True' $AL; then
  sed -i 's|    pre_trained_config\["dspark_bonus_anchor"\] = True|    _sfa = config_dict.get("sample_from_anchor"); _sfa = True if _sfa is None else bool(_sfa)  # patched: speculators dspark default is anchor-as-first\n    pre_trained_config["dspark_bonus_anchor"] = not _sfa  # patched: follow the checkpoint instead of assuming the 1+N layout|' $AL
  python3 -m py_compile $AL && echo "   fork algos.py patched: dspark_bonus_anchor follows the checkpoint's sample_from_anchor" || echo "FORK_ALGOS_PATCH_BROKEN" >> $OUT/status.txt
else echo "   fork algos.py: dspark anchor patch already applied"; fi
