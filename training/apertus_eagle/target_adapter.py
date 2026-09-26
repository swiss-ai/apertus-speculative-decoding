"""Documented TorchSpec traps for Apertus 1.5. No silent Llama relabel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apertus_eagle.contract import aux_layer_ids, num_decoder_layers, target_keys

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES = (
    REPO_ROOT / "patches/torchspec-apertus-target-lm-head.patch",
    REPO_ROOT / "patches/torchspec-apertus-hf-target-layers.patch",
)


def torchspec_model_overrides(contract: dict[str, Any]) -> dict[str, Any]:
    keys = target_keys(contract)
    hf_layers = aux_layer_ids(contract, "hf")
    vllm_layers = aux_layer_ids(contract, "vllm")
    return {
        "embedding_key": keys["embedding_key"],
        "lm_head_key": keys["lm_head_key"],
        "norm_key": keys["norm_key"],
        "output_vocab_size": keys["output_vocab_size"],
        "input_vocab_size": keys["input_vocab_size"],
        "hidden_size": keys["hidden_size"],
        "do_not_relabel_as_llama": True,
        "target_lm_head_out_features": keys["lm_head_shape"][0],
        "aux_hidden_states_layers_torchspec": hf_layers,
        "aux_hidden_states_layers_vllm": vllm_layers,
        "vllm_engine_adds_one": (
            "TorchSpec vllm_engine shifts decoder-layer ids +1 to mixin indices, "
            f"so inference.aux_hidden_states_layers={hf_layers} becomes vLLM {vllm_layers}."
        ),
        "notes": [
            keys["torchspec_vocab_trap"],
            keys["torchspec_norm_trap"],
            keys["torchspec_embed_trap"],
            "HFTargetModel must walk language_model.layers, not model.model.layers.",
        ],
        "required_patches": [str(path) for path in PATCHES],
    }


def assert_lm_head_allocation(module: Any, contract: dict[str, Any]) -> None:
    """Call after constructing TargetLMHead. Fails closed on the vocab trap."""
    expected = tuple(target_keys(contract)["lm_head_shape"])
    observed = tuple(module.lm_head.weight.shape)
    if observed != expected:
        raise ValueError(
            f"TargetLMHead weight shape {observed} != checkpoint {expected}. "
            "The loader used input vocab_size or a generic tensor key. "
            "Do not continue training."
        )


def assert_language_model_walk(module: Any, contract: dict[str, Any]) -> None:
    """Call after constructing HFTargetModel. Fails if layers are not under language_model."""
    layers = module._get_transformer_layers()
    norm = module._get_final_norm()
    if layers is None or norm is None:
        raise ValueError("HFTargetModel did not resolve Apertus language_model layers/norm")
    expected = num_decoder_layers(contract)
    if len(layers) != expected:
        raise ValueError(f"expected {expected} decoder layers, found {len(layers)}")
