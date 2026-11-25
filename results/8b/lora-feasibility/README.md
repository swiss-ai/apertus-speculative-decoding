# Drafter LoRA size inputs (2026-11-20)

- `head-tensor-shapes.json`: tensor dtypes/shapes read with
  `huggingface_hub.get_safetensors_metadata` (header only) from the private
  exports `faruk-zahiragic/Apertus-v1.5-8B-DSpark-e10-thinking-off`
  (revision f1741d68803a4837499dcefc65aca114eaa0837b) and
  `faruk-zahiragic/Apertus-v1.5-8B-EAGLE3.1-e10-thinking-off`
  (revision 4c9d4c9e250a83ebeee43563ecf71c1118b735cd).
- `lora-sizes.txt`: adapter parameter counts, rank r times (in + out) per
  q/k/v/o/gate/up/down matrix in every layer, at bf16. Same convention as
  vLLM RFC #52038: no lm_head or embeddings.
- vLLM source inspected: swiss-ai/vllm a601a9d998ddeb488f0c17e8512874b116aa7658
  (raw files from GitHub). Analysis: `docs/drafter-lora-feasibility.md`.
