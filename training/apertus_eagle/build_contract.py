"""A0: record a target contract from the checkpoint on disk.

Standard library only, so it runs on a login node without Transformers or a
GPU. Reads config/tokenizer/template files and safetensors headers (never the
tensor payloads, except to hash files when ``--hash-weights`` is given).

    python3 -m apertus_eagle.build_contract \
      --model /capstor/.../swiss-ai/Apertus-v1.5-8B --stage 8b \
      --vllm-source /iopsstor/.../vllm-src/root/opt/venv/lib/python3.12/site-packages/vllm \
      --hash-weights
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import socket
import struct
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT, derive_aux_layers

SPECIAL_TOKEN_NAMES = {
    "user_start_id": "<|user_start|>",
    "user_end_id": "<|user_end|>",
    "assistant_start_id": "<|assistant_start|>",
    "assistant_end_id": "<|assistant_end|>",
    "think_start_id": "<|inner_prefix|>",
    "think_end_id": "<|inner_suffix|>",
    "system_start_id": "<|system_start|>",
    "system_end_id": "<|system_end|>",
    "developer_start_id": "<|developer_start|>",
    "developer_end_id": "<|developer_end|>",
    "tools_suffix_id": "<|tools_suffix|>",
}
TEXT_TENSORS = {
    "embed_tokens": "model.language_model.embed_tokens.weight",
    "final_norm": "model.language_model.norm.weight",
    "lm_head": "lm_head.weight",
}
# Draft MLP width: Llama-family SwiGLU ratio (3.5 x hidden), not Apertus xIELU.
DRAFT_INTERMEDIATE_RATIO = 3.5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def optional_sha(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def tokenizer_special_ids(tokenizer_json: Path) -> tuple[dict[str, int], dict[str, str]]:
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    added = payload.get("added_tokens", [])
    by_content = {item["content"]: int(item["id"]) for item in added}
    control = {str(item["id"]): item["content"] for item in added if int(item["id"]) < 128}
    vocab = (payload.get("model") or {}).get("vocab") or {}
    found: dict[str, int] = {}
    for field, content in SPECIAL_TOKEN_NAMES.items():
        if content in by_content:
            found[field] = by_content[content]
        elif isinstance(vocab, dict) and content in vocab:
            found[field] = int(vocab[content])
    return found, control


def vllm_source_checks(vllm_root: Path | None, num_layers: int) -> dict[str, Any]:
    """Read the pinned vLLM source instead of assuming its conventions."""
    if vllm_root is None:
        return {"verified": False, "reason": "no --vllm-source given"}
    report: dict[str, Any] = {"root": str(vllm_root)}
    interfaces = vllm_root / "model_executor/models/interfaces.py"
    apertus = vllm_root / "model_executor/models/apertus.py"
    eagle_head = vllm_root / "model_executor/models/llama_eagle3.py"
    text = interfaces.read_text(encoding="utf-8") if interfaces.is_file() else ""
    rule = re.search(r"return \(2, num_layers // 2, num_layers - 3\)", text)
    report["default_rule"] = "(2, num_layers // 2, num_layers - 3)" if rule else None
    report["default_rule_sha256"] = optional_sha(interfaces)
    atext = apertus.read_text(encoding="utf-8") if apertus.is_file() else ""
    report["apertus_capture_before_loop_index0"] = (
        "_maybe_add_hidden_state([], 0, hidden_states, residual)" in atext
    )
    report["apertus_capture_after_layer_idx_plus_1"] = bool(
        re.search(r"_maybe_add_hidden_state\(\s*aux_hidden_states,\s*idx \+ 1", atext)
    )
    report["apertus_sha256"] = optional_sha(apertus)
    htext = eagle_head.read_text(encoding="utf-8") if eagle_head.is_file() else ""
    report["eagle_head_supports_fc_norm"] = 'getattr(self.config, "fc_norm", False)' in htext
    report["eagle_head_supports_norm_output"] = (
        'getattr(self.config, "norm_output", False)' in htext
    )
    report["eagle_head_midlayer_mapping"] = '"midlayer.": "layers.0."' in htext
    report["eagle_head_sha256"] = optional_sha(eagle_head)
    version = vllm_root / "_version.py"
    if version.is_file():
        match = re.search(r"__version__ = version = '([^']+)'", version.read_text())
        report["vllm_version"] = match.group(1) if match else None
    report["verified"] = bool(
        rule
        and report["apertus_capture_before_loop_index0"]
        and report["apertus_capture_after_layer_idx_plus_1"]
        and report["eagle_head_supports_fc_norm"]
        and report["eagle_head_supports_norm_output"]
    )
    report["derived_default_for_target"] = derive_aux_layers(num_layers)["vllm"]
    return report


def weight_manifest(model_dir: Path, *, hash_weights: bool) -> dict[str, Any]:
    files = []
    for path in sorted(model_dir.glob("*.safetensors")):
        entry: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
        entry["sha256"] = sha256_file(path) if hash_weights else None
        files.append(entry)
    config_sha = sha256_file(model_dir / "config.json")
    manifest_payload = json.dumps(
        {"config_sha256": config_sha, "files": files}, sort_keys=True
    ).encode()
    return {
        "files": files,
        "hashed": hash_weights,
        "manifest_sha256": sha256_bytes(manifest_payload) if hash_weights else None,
        "manifest_rule": "sha256(json.dumps({config_sha256, files[name,bytes,sha256]}, sort_keys=True))",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    model_dir: Path = args.model
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or config
    num_layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or hidden // heads)

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    headers: dict[str, dict[str, Any]] = {}

    def tensor_info(key: str) -> dict[str, Any]:
        file_name = weight_map[key]
        if file_name not in headers:
            headers[file_name] = read_safetensors_header(model_dir / file_name)
        meta = headers[file_name][key]
        return {"key": key, "shape": meta["shape"], "dtype": meta["dtype"], "file": file_name}

    tensors = {name: tensor_info(key) for name, key in TEXT_TENSORS.items()}
    layer_keys = sorted(
        {
            int(match.group(1))
            for key in weight_map
            if (match := re.match(r"model\.language_model\.layers\.(\d+)\.", key))
        }
    )
    if layer_keys != list(range(num_layers)):
        raise SystemExit(
            f"decoder layers in weight map {layer_keys[:3]}..{layer_keys[-3:]} "
            f"do not match num_hidden_layers={num_layers}"
        )
    dtypes = sorted({info["dtype"] for info in tensors.values()})

    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    generation_config_path = model_dir / "generation_config.json"
    generation_config = (
        json.loads(generation_config_path.read_text()) if generation_config_path.is_file() else {}
    )
    chat_template_path = model_dir / "chat_template.jinja"
    chat_template_text = (
        chat_template_path.read_text(encoding="utf-8")
        if chat_template_path.is_file()
        else tokenizer_config.get("chat_template", "")
    )
    revision_path = model_dir / ".revision"
    revision = revision_path.read_text().strip() if revision_path.is_file() else None

    manifest = weight_manifest(model_dir, hash_weights=args.hash_weights)
    aux = derive_aux_layers(num_layers)
    vllm_checks = vllm_source_checks(args.vllm_source, num_layers)
    special, control_tokens = tokenizer_special_ids(tokenizer_json)
    output_vocab = int(text.get("output_vocab_size") or text["vocab_size"])
    if tensors["lm_head"]["shape"][0] != output_vocab:
        raise SystemExit(
            f"lm_head rows {tensors['lm_head']['shape'][0]} != output_vocab_size {output_vocab}"
        )
    for field, token_id in special.items():
        if token_id >= output_vocab:
            raise SystemExit(f"{field}={token_id} is outside the text output vocabulary")

    identity = {
        "model_id": args.hf_id,
        "path": str(model_dir),
        "revision": revision,
        "config_sha256": sha256_file(model_dir / "config.json"),
        "tokenizer_sha256": sha256_file(tokenizer_json),
        "chat_template_sha256": sha256_bytes(chat_template_text.encode("utf-8")),
        "weight_manifest_sha256": manifest["manifest_sha256"],
    }
    draft_intermediate = int(round(hidden * DRAFT_INTERMEDIATE_RATIO / 256.0)) * 256
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    return {
        "schema_version": 2,
        "stage": args.stage,
        "recorded_at": now,
        "recorded_on_host": socket.gethostname(),
        "status": "contract_recorded",
        "source": {
            "authorized_checkpoint": str(model_dir),
            "hf_id": args.hf_id,
            "revision": revision,
            "config_sha256": identity["config_sha256"],
            "tokenizer_json_sha256": identity["tokenizer_sha256"],
            "tokenizer_json_bytes": tokenizer_json.stat().st_size,
            "tokenizer_config_sha256": optional_sha(model_dir / "tokenizer_config.json"),
            "chat_template_sha256": identity["chat_template_sha256"],
            "chat_template_source": (
                "chat_template.jinja" if chat_template_path.is_file() else "tokenizer_config.json"
            ),
            "generation_config_sha256": optional_sha(generation_config_path),
            "special_tokens_map_sha256": optional_sha(model_dir / "special_tokens_map.json"),
            "weight_index_sha256": sha256_file(index_path),
            "weight_total_size_bytes": (index.get("metadata") or {}).get("total_size"),
            "weight_tensor_count": len(weight_map),
            "weight_manifest": manifest,
            "weight_manifest_sha256": manifest["manifest_sha256"],
        },
        "target": {
            "identity": identity,
            "architecture": (config.get("architectures") or [None])[0],
            "model_type": config.get("model_type"),
            "precision": "bfloat16" if text.get("dtype") == "bfloat16" else text.get("dtype"),
            "precision_source": (
                "text_config.dtype plus safetensors header dtypes "
                f"{dtypes} for embed/norm/lm_head"
            ),
            "quantization": config.get("quantization_config"),
            "text": {
                "model_type": text.get("model_type"),
                "hidden_size": hidden,
                "intermediate_size": int(text["intermediate_size"]),
                "num_hidden_layers": num_layers,
                "num_attention_heads": heads,
                "num_key_value_heads": int(text["num_key_value_heads"]),
                "head_dim": head_dim,
                "head_dim_source": (
                    "config field" if "head_dim" in text else "derived hidden_size / num_heads"
                ),
                "vocab_size": int(text["vocab_size"]),
                "output_vocab_size": output_vocab,
                "rms_norm_eps": float(text["rms_norm_eps"]),
                "hidden_act": text.get("hidden_act"),
                "qk_norm": text.get("qk_norm"),
                "post_norm": text.get("post_norm"),
                "mlp_bias": text.get("mlp_bias"),
                "attention_bias": text.get("attention_bias"),
                "tie_word_embeddings": text.get("tie_word_embeddings"),
                "max_position_embeddings": text.get("max_position_embeddings"),
                "rope_parameters": text.get("rope_parameters") or text.get("rope_scaling"),
            },
            "tensors": {
                **tensors,
                "decoder_layer_count_in_weight_map": len(layer_keys),
                "note": (
                    "lm_head rows are the text output vocabulary; serving pads logits to "
                    "input vocab_size with -inf. Input ids above output_vocab_size are "
                    "multimodal and outside this text-only experiment."
                ),
            },
            "tokens": {
                "bos_token_id": text.get("bos_token_id", 1),
                "eos_token_id": text.get("eos_token_id"),
                "generation_eos_token_id": generation_config.get("eos_token_id"),
                "pad_token_id": text.get("pad_token_id"),
                "bos_token": tokenizer_config.get("bos_token"),
                "eos_token": tokenizer_config.get("eos_token"),
                **special,
                "control_tokens_below_128": control_tokens,
                "chat_template_owns_bos": "bos_token" in chat_template_text,
                "serving_add_special_tokens": False,
                "image_token_id": config.get("image_token_id"),
                "audio_token_id": config.get("audio_token_id"),
            },
            "eagle": {
                "engine_method": "eagle3",
                "default_aux_hidden_state_layer_ids": aux["vllm"],
                "hf_decoder_layer_aux_ids": aux["hf_decoder_layer"],
                "torchspec_aux_hidden_states_layers": aux["hf_decoder_layer"],
                "vllm_default_rule": "(2, L // 2, L - 3)",
                "vllm_source_checks": vllm_checks,
                "indexing_convention": (
                    "vLLM EagleModelMixin index k is hidden_states+residual after decoder "
                    "layer k-1 (index 0 is the embedding stream). The HF trainer hooks "
                    "decoder layer k-1 outputs. Final RMSNorm is not part of the tuple."
                ),
                "captured_features_include_residual": True,
                "captured_features_include_final_norm": False,
                "trainer_last_hidden_state": "final RMSNorm output (post-norm)",
                "status": (
                    "rule verified in pinned vLLM source"
                    if vllm_checks.get("verified")
                    else "derived rule; vLLM source not verified"
                ),
            },
        },
        "draft": {
            "architecture": "LlamaForCausalLMEagle3",
            "num_hidden_layers": 1,
            "hidden_size": hidden,
            "intermediate_size": draft_intermediate,
            "intermediate_size_rule": f"round({DRAFT_INTERMEDIATE_RATIO} x hidden / 256) x 256",
            "vocab_size": output_vocab,
            "draft_vocab_size": output_vocab,
            "vocab_note": (
                "Draft embedding covers the text output vocabulary rows of the target "
                "input embedding; every control token id is below output_vocab_size."
            ),
        },
        "serving": dict(args.serving),
        "internal_head_search": args.internal_head_search,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-build-contract")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", choices=("8b", "70b"), required=True)
    parser.add_argument("--hf-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--internal-head-search", type=Path, help="JSON from a Capstor search")
    args = parser.parse_args(argv)
    presets = {
        "8b": {
            "hf_id": "swiss-ai/Apertus-v1.5-8B",
            "serving": {
                "target_tensor_parallel_size": 1,
                "draft_tensor_parallel_size": 1,
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.8,
                "prefix_caching": False,
            },
        },
        "70b": {
            "hf_id": "swiss-ai/Apertus-v1.5-70B",
            "serving": {
                "target_tensor_parallel_size": 4,
                "draft_tensor_parallel_size": 4,
                "max_model_len": 131072,
                "gpu_memory_utilization": 0.8,
                "prefix_caching": False,
            },
        },
    }
    preset = presets[args.stage]
    args.hf_id = args.hf_id or preset["hf_id"]
    args.serving = preset["serving"]
    args.internal_head_search = (
        json.loads(args.internal_head_search.read_text()) if args.internal_head_search else None
    )
    output = args.output or REPO_ROOT / f"results/eagle/{args.stage}/preflight/compatibility.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}; move it aside first")
    contract = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract": str(output), "identity": contract["target"]["identity"],
                      "aux": contract["target"]["eagle"]["default_aux_hidden_state_layer_ids"],
                      "vllm_verified": contract["target"]["eagle"]["vllm_source_checks"].get("verified")},
                     indent=2))


if __name__ == "__main__":
    main()
