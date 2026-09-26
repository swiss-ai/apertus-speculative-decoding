#!/usr/bin/env bash
# Apply the Apertus TargetLMHead / HFTargetModel patches to a TorchSpec checkout.
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
TORCHSPEC_ROOT="${TORCHSPEC_ROOT:-${REPO_ROOT}/scratch/TorchSpec}"
if [ ! -d "${TORCHSPEC_ROOT}/torchspec" ]; then
  echo "TorchSpec checkout not found: ${TORCHSPEC_ROOT}" >&2
  exit 1
fi
if grep -q "def output_vocab_size" \
  "${TORCHSPEC_ROOT}/torchspec/models/target/target_utils.py"; then
  echo "TargetLMHead already patched"
else
  patch -p1 --forward -d "${TORCHSPEC_ROOT}" \
    < "${REPO_ROOT}/methods/eagle/patches/torchspec-apertus-target-lm-head.patch"
fi
if grep -q "_walk_language_model" \
  "${TORCHSPEC_ROOT}/torchspec/models/target/eagle3_target_model.py"; then
  echo "HFTargetModel language_model walk already patched"
else
  patch -p1 --forward -d "${TORCHSPEC_ROOT}" \
    < "${REPO_ROOT}/methods/eagle/patches/torchspec-apertus-hf-target-layers.patch"
fi
if grep -q "AutoModelForImageTextToText" \
  "${TORCHSPEC_ROOT}/torchspec/models/target/eagle3_target_model.py"; then
  echo "HFTargetModel AutoModel fallback already patched"
else
  python3 - "${TORCHSPEC_ROOT}/torchspec/models/target/eagle3_target_model.py" <<'PY' \
    || echo "AutoModel / forward adapter not applied; language_model walk is still present"
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old_load = '''        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
'''
new_load = '''        load_kwargs = {
            "torch_dtype": torch_dtype,
            "cache_dir": cache_dir,
            **device_kwargs,
            **kwargs,
        }
        try:
            target_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, **load_kwargs
            )
        except (ValueError, OSError, AttributeError, TypeError):
            # Apertus 1.5 is Apertus1p5ForConditionalGeneration. Do not relabel as Llama.
            try:
                from transformers import AutoModelForImageTextToText

                target_model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path, **load_kwargs
                )
            except Exception:
                from transformers import AutoModel

                target_model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path, **load_kwargs
                )
'''
old_fwd = '''            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
'''
new_fwd = '''            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": False,
                "output_attentions": False,
                "use_cache": False,
            }
            try:
                self.model(**forward_kwargs, output_router_logits=False)
            except TypeError:
                self.model(**forward_kwargs)
'''
if old_load not in text:
    raise SystemExit("could not find AutoModelForCausalLM.from_pretrained block")
if old_fwd not in text:
    raise SystemExit("could not find HFTargetModel.generate_eagle3_data forward")
path.write_text(text.replace(old_load, new_load, 1).replace(old_fwd, new_fwd, 1), encoding="utf-8")
print("applied AutoModel / forward kwargs adapter")
PY
fi
RENDERER_SRC="${REPO_ROOT}/methods/eagle/apertus_eagle/torchspec_renderer.py"
RENDERER_DST="${TORCHSPEC_ROOT}/torchspec/data/renderers/apertus.py"
if [ -f "${RENDERER_SRC}" ]; then
  cp "${RENDERER_SRC}" "${RENDERER_DST}"
fi
INIT_PY="${TORCHSPEC_ROOT}/torchspec/data/renderers/__init__.py"
if ! grep -q "ApertusRenderer" "${INIT_PY}"; then
  python3 -c '
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "RENDERER_REGISTRY.register(\"kimi-k3\", K3Renderer)\n"
insert = (
    needle
    + "from torchspec.data.renderers.apertus import ApertusRenderer\n"
    + "RENDERER_REGISTRY.register(\"apertus\", ApertusRenderer)\n"
)
if needle not in text:
    raise SystemExit(f"could not find kimi-k3 register line in {path}")
if "ApertusRenderer" not in text:
    path.write_text(text.replace(needle, insert, 1), encoding="utf-8")
' "${INIT_PY}"
fi
MOONCAKE_INIT="${TORCHSPEC_ROOT}/torchspec/transfer/mooncake/__init__.py"
if [ -f "${MOONCAKE_INIT}" ] && grep -q "from torchspec.transfer.mooncake.utils import" "${MOONCAKE_INIT}"; then
  python3 - "${MOONCAKE_INIT}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = '''from torchspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size
from torchspec.transfer.mooncake.utils import (
    MooncakeMaster,
    check_mooncake_master_available,
    launch_mooncake_master,
    resolve_mooncake_master_bin,
)


def __getattr__(name):
    # Lazy imports to avoid circular dependency with config.mooncake_config
    if name == "MooncakeConfig":
'''
new = '''from torchspec.transfer.mooncake.helpers import calculate_eagle3_buffer_size

# utils.py imports ray. One-node overfit never starts Mooncake, and the
# serving image has no ray. Keep these names lazy so LlamaForCausalLMEagle3
# can be imported. Job 3446638 died on this import after the 70B load.
_UTILS_EXPORTS = {
    "MooncakeMaster",
    "check_mooncake_master_available",
    "launch_mooncake_master",
    "resolve_mooncake_master_bin",
}


def __getattr__(name):
    # Lazy imports to avoid circular dependency with config.mooncake_config
    if name in _UTILS_EXPORTS:
        from torchspec.transfer.mooncake import utils

        return getattr(utils, name)
    if name == "MooncakeConfig":
'''
if old not in text:
    raise SystemExit("mooncake/__init__.py eager utils import block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("lazy-loaded mooncake utils (no import-time ray)")
PY
fi
echo "patched ${TORCHSPEC_ROOT}"
