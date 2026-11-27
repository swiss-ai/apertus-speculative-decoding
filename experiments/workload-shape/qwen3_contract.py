"""Write a target contract for a Qwen3 dense target (workload-shape replay only).

    python3 experiments/workload-shape/qwen3_contract.py MODEL_DIR HF_ID REVISION OUT.json [STAGE]

MODEL_DIR is a pinned snapshot with SHA256SUMS (``sha256sum`` lines over every
file, written by the download job). The contract holds what the shape pipeline
reads (checkpoint path, identity, decoder depth, EAGLE aux layers in both
conventions, generation EOS ids, prompt rendering) in the layout of
``targets/8b/contract.json``; it is not a training contract.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    model, hf_id, revision, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    sums = {}
    for line in (model / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.removeprefix("./")] = digest
    config = json.loads((model / "config.json").read_text())
    generation = json.loads((model / "generation_config.json").read_text())
    tok_cfg = json.loads((model / "tokenizer_config.json").read_text())
    template = tok_cfg["chat_template"]
    weights = sorted(n for n in sums if n.endswith(".safetensors"))
    files = [
        {"name": n, "bytes": (model / n).stat().st_size, "sha256": sums[n]} for n in weights
    ]
    manifest_sha = hashlib.sha256(
        json.dumps({"config_sha256": sums["config.json"], "files": files}, sort_keys=True).encode()
    ).hexdigest()
    layers = int(config["num_hidden_layers"])
    vllm_aux = [2, layers // 2, layers - 3]
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    identity = {
        "model_id": hf_id,
        "path": str(model),
        "revision": revision,
        "config_sha256": sums["config.json"],
        "tokenizer_sha256": sums["tokenizer.json"],
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "weight_manifest_sha256": manifest_sha,
    }
    contract = {
        "schema_version": 2,
        "stage": sys.argv[5] if len(sys.argv) > 5 else out.parent.name,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorded_on_host": socket.gethostname(),
        "status": "contract_recorded",
        "purpose": "workload-shape replay (apertus_tree.shape); not a training contract",
        "source": {
            "authorized_checkpoint": str(model),
            "hf_id": hf_id,
            "revision": revision,
            "config_sha256": sums["config.json"],
            "tokenizer_json_sha256": sums["tokenizer.json"],
            "tokenizer_config_sha256": sums["tokenizer_config.json"],
            "chat_template_sha256": identity["chat_template_sha256"],
            "chat_template_source": "tokenizer_config.json",
            "generation_config_sha256": sums["generation_config.json"],
            "weight_manifest": {"files": files, "hashed": True, "manifest_sha256": manifest_sha},
            "weight_manifest_sha256": manifest_sha,
        },
        "target": {
            "identity": identity,
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "precision": config.get("torch_dtype") or config.get("dtype"),
            "prompt_render": "chat_template",
            "chat_template_kwargs": {"enable_thinking": False},
            "text": {
                k: config.get(k)
                for k in (
                    "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                    "rms_norm_eps", "rope_theta", "rope_scaling", "tie_word_embeddings",
                    "max_position_embeddings",
                )
            },  # fmt: skip
            "eagle": {
                "default_aux_hidden_state_layer_ids": vllm_aux,
                "hf_decoder_layer_aux_ids": [i - 1 for i in vllm_aux],
                "convention": "vLLM id k = residual stream entering decoder layer k "
                "= output of HF decoder layer k-1 (Qwen2Model.forward); vLLM's "
                "default for EAGLE-3 heads without eagle_aux_hidden_state_layer_ids",
            },
            "tokens": {
                "bos_token_id": generation.get("bos_token_id"),
                "eos_token_id": eos,
                "generation_eos_token_id": eos,
                "pad_token_id": generation.get("pad_token_id"),
            },
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(identity, indent=1))


if __name__ == "__main__":
    main()
