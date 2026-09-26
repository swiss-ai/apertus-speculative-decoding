"""A3: export a trained draft to a vLLM-loadable directory, then reload it.

Serving key names come from TorchSpec's own ``to_export_keys``
(``midlayer.`` -> ``layers.0.``), the mapping ``tools/convert_to_hf.py`` uses
and ``Eagle3LlamaForCausalLM.hf_to_vllm_mapper`` inverts. The config carries
the architecture flags, aux-layer convention and target/training provenance.

Completion requires more than a saved file: the exported directory is
re-read into a fresh ``LlamaForCausalLMEagle3`` and must reproduce the
training weights' rollout metrics and first-step logits on held-out samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import load_contract
from apertus_eagle.train_rollout import FeatureCache, _prepare_imports, build_batch, device_name


def _build_draft(config_raw: dict[str, Any]):
    import torch
    from torchspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel

    config = AutoDraftModelConfig.from_dict(config_raw)
    if getattr(config, "rope_scaling", None) is None:
        config.rope_scaling = dict(config_raw["rope_parameters"])
    return AutoEagle3DraftModel.from_config(
        config, attention_backend="sdpa", torch_dtype=torch.bfloat16
    )


def export(
    checkpoint: Path,
    run_dir: Path,
    output: Path,
    contract: dict[str, Any],
    *,
    drop_embed_tokens: bool = False,
) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file
    from torchspec.models.draft.keymap import to_export_keys

    from apertus_bench.eagle import validate_eagle_head

    provenance = json.loads((run_dir / "provenance.json").read_text())
    summary_path = run_dir / "train-summary.json"
    train_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    config = json.loads(Path(provenance["draft_config"]).read_text())
    state = load_file(str(checkpoint / "draft.safetensors"))
    state = {k: v for k, v in state.items() if k not in ("t2d", "d2t")}
    if drop_embed_tokens:
        # Serve with the target's full input embedding (vLLM shares it when the
        # checkpoint has none). The draft's own table covers only the text rows,
        # and a prompt with a literal multimodal token (e.g. <|image|> = 131079)
        # indexed past it: CUDA device-side assert, job 3512315 / 3512427.
        # The frozen draft rows are bit-identical copies of the target's.
        state = {k: v for k, v in state.items() if k != "embed_tokens.weight"}
    tensors = to_export_keys(state)
    torchspec_version = (
        Path(provenance["torchspec_root"], "version.txt").read_text().strip()
        if Path(provenance["torchspec_root"], "version.txt").is_file()
        else "unknown"
    )
    output.mkdir(parents=True, exist_ok=False)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(output / "model.safetensors"),
        metadata={"torchspec_version": torchspec_version, "format": "pt"},
    )
    checkpoint_state = json.loads((checkpoint / "state.json").read_text())
    config["torch_dtype"] = str(next(iter(tensors.values())).dtype).replace("torch.", "")
    config["_torchspec_version"] = torchspec_version
    config["apertus_training"] = {
        "objective": provenance["objective"],
        "effective_batch": provenance["effective_batch"],
        "optimizer": provenance["optimizer"],
        "freeze": provenance["freeze"],
        "caches": provenance["caches"],
        "torchspec_revision": provenance["torchspec_revision"],
        "harness_revision": provenance["harness_revision"],
        "slurm_job_id": provenance["slurm_job_id"],
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_state.get("step"),
        "eval_at_checkpoint": checkpoint_state.get("eval"),
        "train_summary_gates": (train_summary or {}).get("gates"),
    }
    config["apertus_training"]["embedding"] = (
        "not exported; vLLM shares the target input embedding (all input ids)"
        if drop_embed_tokens
        else "exported (text rows of the target embedding)"
    )
    config.pop("note", None)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = validate_eagle_head(
        output,
        expected_algorithm=provenance["algorithm"],
        target_contract_path=Path(contract["_path"]),
    )
    return {"export": str(output), "keys": sorted(tensors), "validation": report}


def reload_check(
    output: Path, checkpoint: Path, eval_cache: FeatureCache, contract: dict[str, Any], samples: int
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from torchspec.models.draft.base import load_tensor_from_pretrained
    from torchspec.models.draft.keymap import to_internal_keys
    from torchspec.models.eagle3 import Eagle3Model, compute_lazy_target_padded

    config = json.loads((output / "config.json").read_text())
    device = torch.device(device_name())
    trained = _build_draft(config)
    trained.load_state_dict(load_file(str(checkpoint / "draft.safetensors")), strict=True)
    reloaded = _build_draft(config)
    exported = load_file(str(output / "model.safetensors"))
    if "embed_tokens.weight" not in exported:
        embed_key = contract["target"]["tensors"]["embed_tokens"]["key"]
        rows = reloaded.embed_tokens.weight.shape[0]
        exported["embed_tokens.weight"] = load_tensor_from_pretrained(
            contract["source"]["authorized_checkpoint"], embed_key
        )[:rows].to(reloaded.embed_tokens.weight.dtype)
    internal = to_internal_keys(exported, reloaded.state_dict().keys())
    missing, unexpected = reloaded.load_state_dict(internal, strict=False)
    if missing or unexpected:
        raise SystemExit(f"reload key mismatch: missing={missing} unexpected={unexpected}")
    trained_state = trained.state_dict()
    tensors_equal = all(
        torch.equal(value, trained_state[key]) for key, value in reloaded.state_dict().items()
    )
    lm_head_key = contract["target"]["tensors"]["lm_head"]["key"]
    target_lm_head = load_tensor_from_pretrained(
        contract["source"]["authorized_checkpoint"], lm_head_key
    ).to(device, torch.bfloat16)
    length = int(config["apertus_training"]["objective"]["ttt_length"])
    results = []
    # Third pass: the trained weights again, as the run-to-run noise control for
    # the compiled fp32 loss reduction.
    for model in (trained, reloaded, trained):
        model.to(device).eval()
        wrapper = Eagle3Model(model, length=length, attention_backend="sdpa")
        per_sample = []
        with torch.no_grad():
            for index in range(min(samples, len(eval_cache))):
                batch = build_batch(eval_cache.load(index), device)
                target = compute_lazy_target_padded(
                    batch["target_hidden_states"], target_lm_head, length
                )
                _p, vlosses, acces, _c, _a = wrapper(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target=target,
                    loss_mask=batch["loss_mask"],
                    hidden_states=batch["hidden_states"],
                )
                projected = model.project_hidden_states(batch["hidden_states"])
                embeds = model.embed_input_ids(batch["input_ids"]).to(projected.dtype)
                mask = model.prepare_decoder_attention_mask(
                    attention_mask=batch["attention_mask"],
                    hidden_states=projected,
                    batch_size=1,
                    seq_length=projected.shape[1],
                    past_key_values_length=0,
                )
                positions = torch.arange(projected.shape[1], device=device).unsqueeze(0)
                hidden, _k, _v = model.backbone(embeds, projected, mask, positions, use_cache=False)
                logits = model.compute_logits(hidden).float()
                per_sample.append(
                    {
                        "vlosses": [float(v) for v in vlosses],
                        "acces": [float(a) for a in acces],
                        "logits": logits.cpu(),
                    }
                )
        results.append(per_sample)
        model.to("cpu")
    max_logit_diff = max(
        float((a["logits"] - b["logits"]).abs().max())
        for a, b in zip(results[0], results[1], strict=True)
    )
    acc_equal = all(a["acces"] == b["acces"] for a, b in zip(results[0], results[1], strict=True))

    def max_loss_diff(left, right):
        return max(
            abs(x - y)
            for a, b in zip(left, right, strict=True)
            for x, y in zip(a["vlosses"], b["vlosses"], strict=True)
        )

    loss_diff = max_loss_diff(results[0], results[1])
    control = max_loss_diff(results[0], results[2])
    tolerance = max(control, 1e-5)
    return {
        "samples": len(results[0]),
        "exported_tensors_bitwise_equal": tensors_equal,
        "max_first_step_logit_abs_diff": max_logit_diff,
        "rollout_acc_identical": acc_equal,
        "rollout_loss_max_abs_diff": loss_diff,
        "same_weights_repeat_loss_max_abs_diff": control,
        "loss_tolerance": tolerance,
        "pass": tensors_equal and max_logit_diff == 0.0 and acc_equal and loss_diff <= tolerance,
        "reloaded_acc_first_sample": results[1][0]["acces"] if results[1] else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-export-head")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", help="checkpoint tag under run-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--reload-samples", type=int, default=4)
    parser.add_argument("--drop-embed-tokens", action="store_true")
    args = parser.parse_args(argv)
    _prepare_imports()
    contract = load_contract(args.contract)
    checkpoint = args.run_dir / "checkpoints" / args.checkpoint
    if not (checkpoint / "draft.safetensors").is_file():
        checkpoint = args.run_dir / "checkpoints" / "final"
    result = export(
        checkpoint, args.run_dir, args.output, contract, drop_embed_tokens=args.drop_embed_tokens
    )
    provenance = json.loads((args.run_dir / "provenance.json").read_text())
    eval_cache = FeatureCache(Path(provenance["caches"]["eval"]["path"]), contract)
    result["reload"] = reload_check(
        args.output, checkpoint, eval_cache, contract, args.reload_samples
    )
    (args.output / "export-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))
    if not result["reload"]["pass"]:
        raise SystemExit("exported head does not reproduce the training weights")


if __name__ == "__main__":
    main()
