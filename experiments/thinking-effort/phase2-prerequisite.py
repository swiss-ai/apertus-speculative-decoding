"""Validate the existing Speculators DFlash training/export path on a GPU.

The synthetic update is a compatibility test, never one of the four research heads.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--speculators", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "prerequisite-source.py")
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "node": platform.node(),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "source_sha256": digest(Path(__file__)),
        "speculators_revision": subprocess.check_output(
            ["git", "-C", str(args.speculators), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": "running",
    }
    try:
        import torch
        import yaml
        from safetensors.torch import load_file, save_file
        from speculators.convert.dflash.converter import DFlashConverter
        from speculators.losses import resolve_loss_config
        from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig

        spec = yaml.safe_load(args.spec.read_text())
        source = Path(spec["drafter_path"])
        from apertus_bench.effort import model_manifest
        from apertus_bench.workloads import file_sha256

        report["target"] = model_manifest(Path(spec["target_path"]))
        report["initialization"] = model_manifest(source)
        report["container_sha256"] = file_sha256(Path(spec["image"]))
        converted = args.output / "converted"
        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "vllm", "triton", "safetensors")
        }
        report["gpu"] = torch.cuda.get_device_name(0)
        report["original_config_sha256"] = digest(source / "config.json")
        converter = DFlashConverter()
        converter.convert(source, converted, base_model=spec["target_path"])
        original = load_file(str(source / "model.safetensors"))
        draft_config = DFlashSpeculatorConfig.from_pretrained(str(converted))
        draft_config.transformer_layer_config._attn_implementation = "sdpa"
        model = DFlashDraftModel.from_pretrained(
            str(converted), config=draft_config, dtype=torch.float32
        )
        report["draft_attention_implementation"] = "sdpa"
        model.load_verifier_weights()
        state = model.state_dict()
        for key, tensor in original.items():
            torch.testing.assert_close(state[key].to(tensor.dtype), tensor, rtol=0, atol=0)
        report["identical_original_tensors"] = len(original)
        report["target_layers"] = model.target_layer_ids
        report["block_size"] = model.block_size
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        assert not any(
            name.startswith(("embed_tokens.", "lm_head.", "verifier_")) for name in trainable
        )
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model = model.cuda().train()
        torch.manual_seed(0)
        sequence = 32
        width = model.config.transformer_layer_config.hidden_size
        batch = {
            "hidden_states": torch.randn(
                1,
                sequence,
                len(model.target_layer_ids) * width,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "input_ids": torch.randint(
                0, model.config.target_vocab_size, (1, sequence), device="cuda"
            ),
            "loss_mask": torch.ones(1, sequence, device="cuda"),
            "verifier_last_hidden_states": torch.randn(
                1, sequence, width, device="cuda", dtype=torch.bfloat16
            ),
            "document_ids": torch.zeros(1, sequence, device="cuda", dtype=torch.long),
        }
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        before = model.fc.weight.detach().clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(
                **batch,
                loss_config=resolve_loss_config("ce", "fused"),
                max_anchors=8,
                per_position_loss_weight="fixed-exp-decay",
                gamma=4.0,
            )
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        report["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        assert not torch.equal(before, model.fc.weight)
        report["loss"] = float(loss.detach())
        export = args.output / "smoke-trained-native"
        export.mkdir()
        state = model.state_dict()
        save_file(
            {
                key: state[key].detach().cpu().to(tensor.dtype).contiguous()
                for key, tensor in original.items()
            },
            str(export / "model.safetensors"),
        )
        shutil.copyfile(source / "config.json", export / "config.json")
        for path in source.glob("*.py"):
            shutil.copyfile(path, export / path.name)
        report["export"] = str(export)
        report["export_sha256"] = digest(export / "model.safetensors")
        report["export_config_unchanged"] = digest(export / "config.json") == digest(
            source / "config.json"
        )
        del model, optimizer, state, batch, gradients, before
        gc.collect()
        torch.cuda.empty_cache()
        smoke_spec = {**spec, "drafter_path": str(export)}
        (args.output / "serving-smoke.yaml").write_text(yaml.safe_dump(smoke_spec))
        report["status"] = "training_export_passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
