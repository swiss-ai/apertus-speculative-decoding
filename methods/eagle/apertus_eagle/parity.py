"""Training/serving parity checks. Numeric comparison needs a GPU job."""

from __future__ import annotations

from typing import Any

from apertus_eagle.contract import aux_layer_ids
from apertus_eagle.renderer import BOS_ID, serving_tokenize_kwargs

CHECKS = (
    "token_shift",
    "bos_doubling",
    "layer_index_offset",
    "residual_addition",
    "position_ids",
    "final_norm_once",
    "output_vocab_mask",
    "tp_gather_order",
)


def serving_bos_policy(token_ids: list[int]) -> dict[str, Any]:
    doubled = len(token_ids) >= 2 and token_ids[0] == BOS_ID and token_ids[1] == BOS_ID
    return {
        "starts_with_bos": bool(token_ids) and token_ids[0] == BOS_ID,
        "double_bos": doubled,
        "tokenize_kwargs": serving_tokenize_kwargs(),
        "pass": (not doubled) and (not token_ids or token_ids[0] == BOS_ID),
    }


def mix_omits_leading_bos(rendered_ids: list[int], stored_ids: list[int]) -> bool:
    """True when mix text is the chat-template sequence minus a single leading BOS.

    The SFT-mix ``text`` column may omit the BOS that ``chat_template.jinja``
    emits via ``{{ bos_token }}``. That does not block training if the training
    renderer is the serving chat template with ``add_special_tokens=False``.
    """
    if not rendered_ids or rendered_ids[0] != BOS_ID:
        return False
    if rendered_ids == stored_ids:
        return False
    if stored_ids and stored_ids[0] == BOS_ID:
        return False
    return rendered_ids[1:] == stored_ids


def layer_offset_hypothesis(contract: dict[str, Any]) -> dict[str, Any]:
    vllm = aux_layer_ids(contract, "vllm")
    hf = aux_layer_ids(contract, "hf")
    return {
        "vllm_aux_layer_ids": vllm,
        "hf_decoder_layer_aux_ids": hf,
        "putative_match": f"HF/TorchSpec decoder-layer {hf} vs vLLM mixin {tuple(vllm)}",
        "vllm_engine_shift": "TorchSpec vllm_engine adds +1 before sending ids to vLLM",
        "status": "unproven",
        "blocks_training_until": "A2 numeric feature comparison on a fixed prefix",
    }


def report_template() -> dict[str, Any]:
    return {
        "checks": {name: {"status": "not_run"} for name in CHECKS},
        "tolerance": "use repeated same-path numeric error as the control",
        "pass_rule": "no systematic offset or vocabulary mismatch",
    }
