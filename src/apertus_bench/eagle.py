"""EAGLE-3 / EAGLE 3.1 checkpoint validation used by launchers and tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ENV = "APERTUS_EAGLE_CONTRACT"

# Public EAGLE-3 head for Apertus-8B-Instruct-2509. It is not an Apertus 1.5 head.
FORBIDDEN_HEAD_MARKERS = (
    "EAGLE3-Apertus-8B-Instruct-2509",
    "thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509",
    "Apertus-8B-Instruct-2509",
)
# Identity fields a head must share with the selected target contract.
PROVENANCE_FIELDS = ("revision", "config_sha256", "tokenizer_sha256", "chat_template_sha256")

REQUIRED_CONFIG_FIELDS = (
    "architectures",
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
)


class EagleHeadError(ValueError):
    """Raised when an EAGLE checkpoint cannot be used for the selected Apertus target."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_contract_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(CONTRACT_ENV)
    if env:
        return Path(env)
    raise EagleHeadError(
        f"no target contract selected; pass --contract or set {CONTRACT_ENV}. "
        "There is no default: 8B and 70B heads are not interchangeable."
    )


def load_target_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = resolve_contract_path(path)
    if not contract_path.is_file():
        raise EagleHeadError(f"missing target contract: {contract_path}")
    payload = _load_json(contract_path)
    target = payload.get("target")
    if not isinstance(target, dict):
        raise EagleHeadError(f"{contract_path} has no target object")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def infer_algorithm(config: dict[str, Any]) -> str:
    fc_norm = _as_bool(config.get("fc_norm", False))
    norm_output = _as_bool(config.get("norm_output", False))
    if fc_norm and norm_output:
        return "eagle31"
    if not fc_norm and not norm_output:
        return "eagle3"
    raise EagleHeadError(
        f"inconsistent EAGLE flags: fc_norm={fc_norm!r} norm_output={norm_output!r}; "
        "E3 requires both false, E3.1 requires both true"
    )


def count_draft_parameters(config: dict[str, Any]) -> dict[str, int]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    draft_vocab = int(config.get("draft_vocab_size") or vocab)
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    n_aux = len(
        config.get("eagle_aux_hidden_state_layer_ids")
        or (config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        or (None, None, None)
    )
    qkv_in = 2 * hidden
    qkv_out = (heads + 2 * kv_heads) * head_dim
    parts = {
        "embed_tokens": vocab * hidden,
        "lm_head": draft_vocab * hidden,
        "fc": n_aux * hidden * hidden,
        "fc_norm": n_aux * hidden if _as_bool(config.get("fc_norm", False)) else 0,
        "qkv_proj": qkv_in * qkv_out,
        "o_proj": heads * head_dim * hidden,
        "gate_up_proj": hidden * intermediate * 2,
        "down_proj": intermediate * hidden,
        "norms": hidden * 4,
    }
    parts["total"] = sum(parts.values())
    parts["excluding_embed_and_lm_head"] = (
        parts["total"] - parts["embed_tokens"] - parts["lm_head"]
    )
    return parts


def validate_eagle_config(
    config: dict[str, Any],
    *,
    target: dict[str, Any],
    expected_algorithm: str | None = None,
    source: str = "config.json",
) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise EagleHeadError(f"{source} missing fields: {', '.join(missing)}")

    architectures = config.get("architectures") or []
    if "LlamaForCausalLMEagle3" not in architectures:
        raise EagleHeadError(
            f"{source} architectures {architectures!r} is not LlamaForCausalLMEagle3"
        )
    if int(config["num_hidden_layers"]) != 1:
        raise EagleHeadError("this experiment uses a single-layer Llama-style EAGLE head")

    text = target["text"]
    target_hidden = int(text["hidden_size"])
    head_hidden = int(config["hidden_size"])
    target_hidden_size = int(config.get("target_hidden_size") or head_hidden)
    if target_hidden_size != target_hidden:
        raise EagleHeadError(
            f"target_hidden_size {target_hidden_size} != target hidden {target_hidden}"
        )
    if head_hidden != target_hidden:
        raise EagleHeadError(
            f"head hidden_size {head_hidden} != target hidden {target_hidden}; "
            "the first experiment keeps target-aligned draft width"
        )
    for field in ("head_dim", "num_attention_heads", "num_key_value_heads"):
        if int(config[field]) != int(text[field]):
            raise EagleHeadError(
                f"{field} {config[field]} must match the target ({text[field]})"
            )

    output_vocab = int(text["output_vocab_size"])
    vocab = int(config["vocab_size"])
    draft_vocab = int(config.get("draft_vocab_size") or vocab)
    if vocab > int(text["vocab_size"]):
        raise EagleHeadError("draft vocab_size exceeds target input vocabulary")
    if draft_vocab > output_vocab:
        raise EagleHeadError("draft_vocab_size exceeds target output_vocab_size")
    if vocab < output_vocab:
        raise EagleHeadError(
            f"draft vocab_size {vocab} is smaller than target output support {output_vocab}; "
            "the initial recipe uses full text output support"
        )

    algorithm = infer_algorithm(config)
    if expected_algorithm and algorithm != expected_algorithm:
        raise EagleHeadError(
            f"checkpoint algorithm is {algorithm}, expected {expected_algorithm}"
        )

    layer_ids = config.get("eagle_aux_hidden_state_layer_ids")
    if not layer_ids:
        layer_ids = (config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    expected_layers = target["eagle"]["default_aux_hidden_state_layer_ids"]
    if layer_ids is None:
        raise EagleHeadError(
            "config must record eagle_aux_hidden_state_layer_ids; "
            f"runtime default if omitted would be {expected_layers}"
        )
    if list(layer_ids) != list(expected_layers):
        raise EagleHeadError(
            f"aux layers {list(layer_ids)} != recorded default {expected_layers}"
        )

    provenance = check_provenance(config, target=target, source=source)

    return {
        "algorithm": algorithm,
        "provenance": provenance,
        "fc_norm": _as_bool(config.get("fc_norm", False)),
        "norm_output": _as_bool(config.get("norm_output", False)),
        "parameter_count": count_draft_parameters(config),
        "aux_hidden_state_layer_ids": list(layer_ids),
        "draft_vocab_size": draft_vocab,
        "vocab_size": vocab,
        "hidden_size": head_hidden,
    }


def check_provenance(
    config: dict[str, Any], *, target: dict[str, Any], source: str = "config.json"
) -> dict[str, Any]:
    """Refuse a head whose recorded target differs from the selected contract.

    Shapes alone cannot tell an 8B head from another 4096-wide checkpoint, or a
    head trained on a different revision of the same model.
    """
    identity = target.get("identity")
    if not isinstance(identity, dict):
        # The 2026-09-19 70B record predates identity blocks.
        return {"checked": False, "reason": "contract has no target.identity"}
    recorded = config.get("apertus_target")
    if not isinstance(recorded, dict):
        raise EagleHeadError(
            f"{source} has no apertus_target provenance; cannot prove it was built for "
            f"{identity.get('model_id')}@{identity.get('revision')}"
        )
    mismatches = {
        field: {"head": recorded.get(field), "target": identity.get(field)}
        for field in PROVENANCE_FIELDS
        if identity.get(field) is not None and recorded.get(field) != identity.get(field)
    }
    if recorded.get("model_id") and identity.get("model_id"):
        if recorded["model_id"] != identity["model_id"]:
            mismatches["model_id"] = {"head": recorded["model_id"], "target": identity["model_id"]}
    if mismatches:
        raise EagleHeadError(f"{source} was built for a different target: {mismatches}")
    return {
        "checked": True,
        "model_id": identity.get("model_id"),
        "revision": identity.get("revision"),
        "tokenizer_sha256": identity.get("tokenizer_sha256"),
    }


def checkpoint_manifest_sha256(head_dir: Path, weight_files: list[Path]) -> str:
    """Digest over config.json and every weight file, not the config alone."""
    entries = [
        {"name": path.name, "sha256": sha256_file(path)}
        for path in [head_dir / "config.json", *weight_files]
    ]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def validate_eagle_head(
    head_dir: Path,
    *,
    expected_algorithm: str | None = None,
    require_weights: bool = True,
    target_contract_path: Path | None = None,
) -> dict[str, Any]:
    head_dir = head_dir.resolve()
    if not head_dir.is_dir():
        raise EagleHeadError(f"EAGLE head directory not found: {head_dir}")

    path_text = str(head_dir)
    for marker in FORBIDDEN_HEAD_MARKERS:
        if marker in path_text:
            raise EagleHeadError(
                "refusing the public Apertus-8B-Instruct-2509 EAGLE-3 head; "
                "it is not a substitute for an Apertus 1.5 head"
            )

    config_path = head_dir / "config.json"
    if not config_path.is_file():
        raise EagleHeadError(f"missing {config_path}")
    config_text = config_path.read_text(encoding="utf-8")
    for marker in FORBIDDEN_HEAD_MARKERS:
        if marker in config_text:
            raise EagleHeadError(f"{config_path} references {marker}; not an Apertus 1.5 head")
    config = json.loads(config_text)
    target = load_target_contract(target_contract_path)
    summary = validate_eagle_config(
        config,
        target=target,
        expected_algorithm=expected_algorithm,
        source=str(config_path),
    )
    summary["path"] = str(head_dir)
    summary["config_sha256"] = sha256_file(config_path)

    weight_files = sorted(
        [*head_dir.glob("*.safetensors"), *head_dir.glob("*.bin"), *head_dir.glob("*.pt")]
    )
    if require_weights and not weight_files:
        raise EagleHeadError(f"no weight files in {head_dir}")
    summary["weight_files"] = [path.name for path in weight_files]
    summary["weights_present"] = bool(weight_files)
    summary["trained_head"] = bool(weight_files)
    if weight_files:
        summary["checkpoint_manifest_sha256"] = checkpoint_manifest_sha256(
            head_dir, weight_files
        )
    return summary


def check_target_model(target_model: str, contract_path: Path | None = None) -> str:
    """The loaded target must be the contract's checkpoint; a served name is not enough."""
    payload = _load_json(resolve_contract_path(contract_path))
    recorded = (payload.get("source") or {}).get("authorized_checkpoint")
    if recorded is None:
        raise EagleHeadError("contract records no authorized_checkpoint")
    if os.path.normpath(recorded) != os.path.normpath(target_model):
        raise EagleHeadError(
            f"TARGET_MODEL {target_model} is not the contract checkpoint {recorded}"
        )
    return recorded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apertus-bench-eagle-validate")
    parser.add_argument("head", type=Path)
    parser.add_argument("--algorithm", choices=("eagle3", "eagle31"))
    parser.add_argument("--allow-config-only", action="store_true")
    parser.add_argument("--contract", type=Path, help=f"target contract (or ${CONTRACT_ENV})")
    parser.add_argument("--target-model", help="refuse unless this is the contract checkpoint")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.target_model:
            check_target_model(args.target_model, args.contract)
        report = validate_eagle_head(
            args.head,
            expected_algorithm=args.algorithm,
            require_weights=not args.allow_config_only,
            target_contract_path=args.contract,
        )
    except EagleHeadError as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
