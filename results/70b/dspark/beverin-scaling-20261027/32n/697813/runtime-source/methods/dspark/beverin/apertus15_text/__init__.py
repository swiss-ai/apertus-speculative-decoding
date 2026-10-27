"""Text-only Apertus 1.5 for stock vLLM (Beverin / ROCm).

The Clariden image serves Apertus-v1.5-70B through the Swiss AI vLLM fork
(swiss-ai/vllm a601a9d9, apertus_mm.py), which needs the fork's transformers
for its vision tokenizer and is CUDA-only. Hidden-state extraction feeds token
ids and never media, so only the language model matters. This general plugin
registers the checkpoint's config and architecture with an unmodified vLLM:

- config: model_type "apertus1p5" -> ApertusConfig with text_config flattened
  into it (as the fork's configs/apertus1p5.py does);
- model: Apertus1p5ForConditionalGeneration -> upstream ApertusForCausalLM
  with the fork's pruned head (output_vocab_size rows, logits padded with -inf
  to the input vocabulary) and the vision/audio tokenizer weights skipped.

Same decoder (vLLM's ApertusModel, Python xIELU) as the fork, so the captured
hidden states should match. The fork's processor also prepends BOS to token
prompts that lack one; the regenerated rows already start with BOS.

Loaded via the vllm.general_plugins entry point in the dist-info next to this
package (vLLM loads general plugins in every engine and worker process).
"""


def register_config() -> None:
    from transformers import AutoConfig

    from .config import Apertus1p5Config

    try:
        AutoConfig.register("apertus1p5", Apertus1p5Config)
    except ValueError:
        pass  # already registered (e.g. a transformers that knows Apertus 1.5)


def register() -> None:
    register_config()
    from vllm import ModelRegistry

    if "Apertus1p5ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Apertus1p5ForConditionalGeneration",
            "apertus15_text.model:Apertus1p5TextForCausalLM",
        )
