"""Apertus-specific EAGLE training adapters. Not a substitute for TorchSpec."""

from __future__ import annotations

from apertus_eagle.contract import (
    CONTRACT_ENV,
    STAGE_CONTRACTS,
    aux_layer_ids,
    derive_aux_layers,
    load_contract,
    resolve_contract_path,
    target_identity,
    target_keys,
    validate_draft_config_path,
)
from apertus_eagle.renderer import (
    ASSISTANT_END_ID,
    ASSISTANT_START_ID,
    BOS_ID,
    USER_END_ID,
    USER_START_ID,
    assistant_loss_mask,
)

__all__ = [
    "ASSISTANT_END_ID",
    "ASSISTANT_START_ID",
    "BOS_ID",
    "CONTRACT_ENV",
    "STAGE_CONTRACTS",
    "USER_END_ID",
    "USER_START_ID",
    "assistant_loss_mask",
    "aux_layer_ids",
    "derive_aux_layers",
    "load_contract",
    "resolve_contract_path",
    "target_identity",
    "target_keys",
    "validate_draft_config_path",
]
