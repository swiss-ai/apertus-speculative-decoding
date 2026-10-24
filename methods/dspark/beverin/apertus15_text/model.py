from collections.abc import Iterable

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.apertus import ApertusForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

_SKIP = ("model.vision_tokenizer.", "model.audio_tokenizer.")


class Apertus1p5TextForCausalLM(ApertusForCausalLM):
    """Language model of Apertus1p5ForConditionalGeneration (see __init__.py)."""

    _mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
    hf_to_vllm_mapper = (
        ApertusForCausalLM.hf_to_vllm_mapper | _mapper
        if getattr(ApertusForCausalLM, "hf_to_vllm_mapper", None) is not None
        else _mapper
    )
    # No allow_patterns_overrides for the language-model shards (the fork's):
    # vLLM 0.31 then refuses an index that names the tower files (job 693753).
    # The towers are 1.2 GB of 144.6; load_weights drops their tensors.

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        output_vocab_size = getattr(config, "output_vocab_size", config.vocab_size)
        if output_vocab_size > config.vocab_size:
            raise ValueError("Output vocabulary cannot exceed input vocabulary.")
        self._input_vocab_size = config.vocab_size
        self._pad_logits = False
        if (
            get_pp_group().is_last_rank
            and not config.tie_word_embeddings
            and output_vocab_size != config.vocab_size
        ):
            self.lm_head = ParallelLMHead(
                output_vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(
                output_vocab_size, scale=getattr(config, "logit_scale", 1.0)
            )
            self._pad_logits = True

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None or not self._pad_logits:
            return logits
        return F.pad(logits, (0, self._input_vocab_size - logits.shape[-1]), value=float("-inf"))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # AutoWeightsLoader's skip_prefixes is gone in newer vLLM; filter here.
        skip = _SKIP + (("lm_head.",) if self.config.tie_word_embeddings else ())
        weights = ((n, w) for n, w in weights if not n.startswith(skip))
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
