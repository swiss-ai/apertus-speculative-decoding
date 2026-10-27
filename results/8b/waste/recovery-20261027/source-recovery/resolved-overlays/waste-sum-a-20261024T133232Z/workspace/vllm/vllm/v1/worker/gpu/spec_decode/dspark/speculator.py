# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import json
import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


def dspark_draft_lengths(max_length: int) -> list[int]:
    """Draft lengths the scheduler may pick per step
    (VLLM_DSPARK_DRAFT_LENGTHS, comma separated; default 1..max_length)."""
    raw = os.environ.get("VLLM_DSPARK_DRAFT_LENGTHS")
    lengths = (
        [int(part) for part in raw.split(",") if part.strip()]
        if raw
        else list(range(1, max_length + 1))
    )
    return sorted({n for n in lengths if 1 <= n <= max_length} | {max_length})


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Anchor-as-first (N slots) unless the checkpoint uses the 1+N fill-in
        # block, where the anchor is a separate bonus token.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Draft length per step (VLLM_DSPARK_DRAFT_CUT=1, with the paper
        # scheduler): the scheduler picks how many slots to draft, D, and the
        # drafter runs as a k=D drafter for that step (a block of D query
        # tokens, D sequential Markov steps). Not in the paper, which drafts
        # every slot. One CUDA graph set and anchor index per allowed length.
        self.draft_lengths = [self.num_speculative_steps]
        if os.environ.get("VLLM_DSPARK_DRAFT_CUT", "0") == "1":
            self.draft_lengths = dspark_draft_lengths(self.num_speculative_steps)
        self._draft_len = self.num_speculative_steps
        self._anchor_idx_by_len = {
            length: torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self._query_len(length)
            for length in self.draft_lengths
        }
        self._cg_managers: dict[int, Any] = {}

        # Confidence-scheduled verification (VLLM_DSPARK_VERIFY_SCHED=1): the
        # confidence head's per-slot acceptance probabilities and their running
        # product (prefix survival), written inside the captured draft step.
        self.verify_sched = os.environ.get("VLLM_DSPARK_VERIFY_SCHED", "0") == "1"
        self._use_confidence = False
        self.draft_confidence = torch.ones(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_survival = torch.ones_like(self.draft_confidence)
        # Raw head logits, and the inverse Sequential Temperature Scaling
        # temperature per slot (DSpark section 3.2.1; VLLM_DSPARK_STS, a JSON
        # file with "temperatures", one per slot; 1 = the head as trained).
        self.draft_conf_logits = torch.zeros_like(self.draft_confidence)
        self.conf_inv_temperature = torch.ones(
            self.num_speculative_steps, dtype=torch.float32, device=device
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

    def _query_len(self, draft_len: int) -> int:
        return draft_len if self.sample_from_anchor else 1 + draft_len

    def set_draft_len(self, draft_len: int) -> None:
        """Draft ``draft_len`` slots in the next draft steps (one of
        ``draft_lengths``)."""
        if draft_len not in self._anchor_idx_by_len:
            raise ValueError(
                f"draft length {draft_len} not in {self.draft_lengths}"
            )
        self._draft_len = draft_len
        self.num_query_per_req = self._query_len(draft_len)
        self.sample_stride = draft_len
        self._anchor_idx = self._anchor_idx_by_len[draft_len]
        # Absent while init_cudagraph_manager is still creating them.
        if draft_len in self._cg_managers:
            self.query_cudagraph_manager = self._cg_managers[draft_len]

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if len(self.draft_lengths) == 1:
            super().init_cudagraph_manager(cudagraph_mode)
            return
        for length in self.draft_lengths:
            self.set_draft_len(length)
            super().init_cudagraph_manager(cudagraph_mode)
            self._cg_managers[length] = self.query_cudagraph_manager
        self.set_draft_len(self.num_speculative_steps)

    def capture(self, attn_states: dict | None = None) -> None:
        if len(self.draft_lengths) == 1:
            super().capture(attn_states)
            return
        for length in self.draft_lengths:
            self.set_draft_len(length)
            super().capture(attn_states)
        self.set_draft_len(self.num_speculative_steps)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        self._use_confidence = self.verify_sched and model.has_confidence_head()
        if self.verify_sched and not self._use_confidence:
            raise ValueError(
                "VLLM_DSPARK_VERIFY_SCHED=1 needs a DSpark checkpoint with "
                "enable_confidence_head"
            )
        sts_path = os.environ.get("VLLM_DSPARK_STS")
        if self._use_confidence and sts_path:
            with open(sts_path) as handle:
                temperatures = json.load(handle)["temperatures"]
            if len(temperatures) < self.num_speculative_steps:
                raise ValueError(
                    f"{sts_path} calibrates {len(temperatures)} slots, the "
                    f"drafter proposes {self.num_speculative_steps}"
                )
            self.conf_inv_temperature.copy_(
                1.0
                / torch.tensor(
                    temperatures[: self.num_speculative_steps], dtype=torch.float32
                )
            )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self._draft_len
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        if self._use_confidence:
            slot_hidden = sample_hidden.view(num_reqs, n_spec, -1)
        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            if self._use_confidence:
                self.draft_conf_logits[:num_reqs, i] = self.model.confidence_logit(
                    slot_hidden[:, i], markov_embed
                )
            logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i
        if self._use_confidence:
            # Calibrated per-slot probabilities (STS: sigmoid(logit / T_j)),
            # then their running product, the prefix survival.
            if n_spec == self.num_speculative_steps:
                torch.sigmoid(
                    self.draft_conf_logits[:num_reqs] * self.conf_inv_temperature,
                    out=self.draft_confidence[:num_reqs],
                )
                torch.cumprod(
                    self.draft_confidence[:num_reqs],
                    dim=1,
                    out=self.draft_survival[:num_reqs],
                )
            else:
                # Shortened draft: slots not drafted have survival 0.
                confidence = torch.sigmoid(
                    self.draft_conf_logits[:num_reqs, :n_spec]
                    * self.conf_inv_temperature[:n_spec]
                )
                self.draft_confidence[:num_reqs, :n_spec] = confidence
                self.draft_survival[:num_reqs, :n_spec] = torch.cumprod(
                    confidence, dim=1
                )
                self.draft_survival[:num_reqs, n_spec:] = 0.0

    def spec_survival(self, num_reqs: int) -> torch.Tensor | None:
        """Prefix survival of the drafts just proposed, [num_reqs, n_spec]."""
        if not self._use_confidence:
            return None
        return self.draft_survival[:num_reqs]

    def spec_conf_logits(self, num_reqs: int) -> torch.Tensor | None:
        """Raw confidence-head logits of the drafts just proposed."""
        if not self._use_confidence:
            return None
        return self.draft_conf_logits[:num_reqs]

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
