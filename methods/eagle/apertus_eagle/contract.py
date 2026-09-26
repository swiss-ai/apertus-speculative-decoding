"""Checkpoint-contract helpers for Apertus 1.5 EAGLE training.

Every caller selects a target contract explicitly (argument or the
``APERTUS_EAGLE_CONTRACT`` environment variable). There is no silent default:
the 8B pilot and the 70B experiment have different layer counts, widths and
auxiliary-layer tuples, and a fallback to either one would train or validate
against the wrong target.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    """The checkout root (holds pyproject.toml), wherever this package sits in it."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"no pyproject.toml above {__file__}")


REPO_ROOT = _find_repo_root()
CONTRACT_ENV = "APERTUS_EAGLE_CONTRACT"

# Target contracts are method-agnostic and live under targets/<stage>/.
# targets/70b/contract.json is the 2026-09-19 record; Stage B re-verifies it.
STAGE_CONTRACTS = {
    "8b": REPO_ROOT / "targets/8b/contract.json",
    "70b": REPO_ROOT / "targets/70b/contract.json",
}

TARGET_LM_HEAD_KEY = "lm_head.weight"
TARGET_NORM_KEY = "model.language_model.norm.weight"
TARGET_EMBED_KEY = "model.language_model.embed_tokens.weight"
TORCHSPEC_DEFAULT_NORM_KEY = "model.norm.weight"
TORCHSPEC_DEFAULT_EMBED_KEY = "model.embed_tokens.weight"


class ContractError(ValueError):
    """Raised when the selected Apertus contract is missing or inconsistent."""


def resolve_contract_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(CONTRACT_ENV)
    if env:
        return Path(env)
    raise ContractError(
        f"no target contract selected; pass --contract or set {CONTRACT_ENV} "
        f"(8B: {STAGE_CONTRACTS['8b']})"
    )


def load_contract(path: Path | str | None = None) -> dict[str, Any]:
    contract_path = resolve_contract_path(path)
    if not contract_path.is_file():
        raise ContractError(f"missing contract: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    target = payload.get("target")
    if not isinstance(target, dict):
        raise ContractError(f"{contract_path} has no target object")
    payload.setdefault("_path", str(contract_path))
    return payload


def derive_aux_layers(num_layers: int) -> dict[str, list[int]]:
    """Both auxiliary-layer conventions for an ``num_layers``-deep target.

    vLLM ``EagleModelMixin`` index ``k`` is the residual stream after decoder
    layer ``k-1`` (0-based): ``ApertusModel.forward`` records index 0 before
    the loop and ``idx + 1`` after layer ``idx``. The pinned default is
    ``(2, L // 2, L - 3)``. A Hugging Face forward hook on decoder layer ``i``
    sees the same tensor as vLLM index ``i + 1``, so the trainer uses
    ``(1, L // 2 - 1, L - 4)``. That is also TorchSpec's HF default.
    """
    if num_layers < 8:
        raise ContractError(f"implausible decoder depth {num_layers}")
    vllm = [2, num_layers // 2, num_layers - 3]
    return {"vllm": vllm, "hf_decoder_layer": [layer - 1 for layer in vllm]}


def num_decoder_layers(contract: dict[str, Any]) -> int:
    return int(contract["target"]["text"]["num_hidden_layers"])


def aux_layer_ids(contract: dict[str, Any], convention: str = "vllm") -> list[int]:
    """Recorded aux ids, cross-checked against the depth-derived rule."""
    eagle = contract["target"]["eagle"]
    derived = derive_aux_layers(num_decoder_layers(contract))
    if convention == "vllm":
        recorded = list(eagle["default_aux_hidden_state_layer_ids"])
        expected = derived["vllm"]
    elif convention in ("hf", "hf_decoder_layer", "torchspec"):
        recorded = list(
            eagle.get("hf_decoder_layer_aux_ids")
            or eagle.get("torchspec_hf_default_aux_layers_80")
            or derived["hf_decoder_layer"]
        )
        expected = derived["hf_decoder_layer"]
    else:
        raise ContractError(f"unknown aux-layer convention {convention!r}")
    if recorded != expected:
        raise ContractError(
            f"contract {convention} aux layers {recorded} != depth-derived {expected}"
        )
    return recorded


def target_identity(contract: dict[str, Any]) -> dict[str, Any]:
    """Fields that identify the exact target a head or cache was built for."""
    identity = contract["target"].get("identity")
    if isinstance(identity, dict):
        return dict(identity)
    source = contract.get("source") or {}
    # The 2026-09-19 70B record predates the identity block.
    return {
        "model_id": source.get("hf_id"),
        "path": source.get("authorized_checkpoint"),
        "revision": source.get("revision"),
        "config_sha256": source.get("config_sha256"),
        "tokenizer_sha256": source.get("tokenizer_json_sha256"),
        "chat_template_sha256": source.get("chat_template_sha256"),
        "weight_manifest_sha256": source.get("weight_manifest_sha256"),
    }


def target_keys(contract: dict[str, Any]) -> dict[str, Any]:
    text = contract["target"]["text"]
    tensors = contract["target"]["tensors"]
    return {
        "lm_head_key": tensors["lm_head"]["key"],
        "norm_key": tensors["final_norm"]["key"],
        "embedding_key": tensors["embed_tokens"]["key"],
        "lm_head_shape": list(tensors["lm_head"]["shape"]),
        "embed_shape": list(tensors["embed_tokens"]["shape"]),
        "norm_shape": list(tensors["final_norm"]["shape"]),
        "input_vocab_size": int(text["vocab_size"]),
        "output_vocab_size": int(text["output_vocab_size"]),
        "hidden_size": int(text["hidden_size"]),
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "torchspec_vocab_trap": (
            "TargetLMHead allocates nn.Linear(hidden, text_config.vocab_size="
            f"{text['vocab_size']}) but lm_head.weight is {tensors['lm_head']['shape']}. "
            "Use output_vocab_size and do not copy Llama/Apertus-v1 defaults."
        ),
        "torchspec_norm_trap": (
            f"TorchSpec default norm key is {TORCHSPEC_DEFAULT_NORM_KEY}; "
            f"Apertus 1.5 uses {tensors['final_norm']['key']}."
        ),
        "torchspec_embed_trap": (
            f"TorchSpec default embedding key is {TORCHSPEC_DEFAULT_EMBED_KEY}; "
            f"Apertus 1.5 uses {tensors['embed_tokens']['key']}."
        ),
    }


def validate_draft_config_path(
    config_path: Path, contract_path: Path | str | None = None
) -> dict[str, Any]:
    from apertus_bench.eagle import validate_eagle_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = load_contract(contract_path)
    return validate_eagle_config(config, target=contract["target"], source=str(config_path))
