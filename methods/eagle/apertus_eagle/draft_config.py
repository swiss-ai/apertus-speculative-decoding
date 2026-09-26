"""Derive EAGLE-3 / EAGLE 3.1 draft configs from a target contract.

The draft architecture is a single Llama-style ``LlamaForCausalLMEagle3``
layer with target-aligned width. The two algorithms differ only in the
trained architecture flags; each gets its own file and its own checkpoint.

    python3 -m apertus_eagle.draft_config --contract targets/8b/contract.json \
      --algorithm eagle31 --output methods/eagle/configs/8b/draft-e31-config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import aux_layer_ids, load_contract, target_identity

ARCHITECTURE_FLAGS = {
    "eagle31": {"fc_norm": True, "norm_output": True},
    "eagle3": {"fc_norm": False, "norm_output": False},
}


def contract_sha256(contract: dict[str, Any]) -> str:
    path = contract.get("_path")
    if path and Path(path).is_file():
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = {key: value for key, value in contract.items() if key != "_path"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_draft_config(contract: dict[str, Any], algorithm: str) -> dict[str, Any]:
    if algorithm not in ARCHITECTURE_FLAGS:
        raise ValueError(f"algorithm must be one of {sorted(ARCHITECTURE_FLAGS)}")
    target = contract["target"]
    text = target["text"]
    draft = contract.get("draft") or {}
    tokens = target["tokens"]
    hidden = int(text["hidden_size"])
    output_vocab = int(text["output_vocab_size"])
    layers = aux_layer_ids(contract, "vllm")
    eos = tokens.get("eos_token_id")
    config: dict[str, Any] = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": hidden,
        "target_hidden_size": hidden,
        "intermediate_size": int(draft.get("intermediate_size") or round(hidden * 3.5)),
        "num_hidden_layers": 1,
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "vocab_size": output_vocab,
        "draft_vocab_size": output_vocab,
        "rms_norm_eps": float(text["rms_norm_eps"]),
        "hidden_act": "silu",
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "bos_token_id": int(tokens.get("bos_token_id", 1)),
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "pad_token_id": int(tokens.get("pad_token_id", 3)),
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        **ARCHITECTURE_FLAGS[algorithm],
        "torch_dtype": "bfloat16",
        "num_aux_hidden_states": len(layers),
        "eagle_aux_hidden_state_layer_ids": layers,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": layers, "use_aux_hidden_state": True},
        "rope_parameters": dict(text["rope_parameters"]),
        # TorchSpec's draft RoPE builder reads rope_scaling; vLLM reads either.
        "rope_scaling": dict(text["rope_parameters"]),
        "rope_theta": text["rope_parameters"].get("rope_theta"),
        "apertus_target": {
            **target_identity(contract),
            "contract_sha256": contract_sha256(contract),
            "stage": contract.get("stage"),
            "algorithm": algorithm,
            "aux_layer_convention": "vLLM EagleModelMixin index (residual after layer k-1)",
            "hf_decoder_layer_aux_ids": aux_layer_ids(contract, "hf"),
        },
    }
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-draft-config")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--algorithm", choices=sorted(ARCHITECTURE_FLAGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = build_draft_config(load_contract(args.contract), args.algorithm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
