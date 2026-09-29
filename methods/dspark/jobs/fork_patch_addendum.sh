# --- DSpark anchor layout (added 2026-09-25): the fork's speculators->HF translation hard-codes dspark_bonus_anchor=True (1+N block),
#     but speculators trains DSpark with sample_from_anchor=True (anchor-as-first, N slots) by default -> off-by-one in every draft slot.
AL=$PV/vllm/transformers_utils/configs/speculators/algos.py
if grep -q 'pre_trained_config\["dspark_bonus_anchor"\] = True' $AL; then
  sed -i 's|    pre_trained_config\["dspark_bonus_anchor"\] = True|    _sfa = config_dict.get("sample_from_anchor"); _sfa = True if _sfa is None else bool(_sfa)  # patched: speculators dspark default is anchor-as-first\n    pre_trained_config["dspark_bonus_anchor"] = not _sfa  # patched: follow the checkpoint instead of assuming the 1+N layout|' $AL
  python3 -m py_compile $AL && echo "   fork algos.py patched: dspark_bonus_anchor follows the checkpoint's sample_from_anchor" || echo "FORK_ALGOS_PATCH_BROKEN" >> $OUT/status.txt
else echo "   fork algos.py: dspark anchor patch already applied"; fi
