"""A3: train an EAGLE-3 / EAGLE 3.1 head with TorchSpec's multi-step rollout.

Single-process driver for one GPU. It reuses the pinned TorchSpec pieces that
``Eagle3Trainer`` is built from, unchanged:

- ``AutoEagle3DraftModel`` / ``LlamaForCausalLMEagle3`` (``fc_norm`` and
  ``norm_output`` select E3.1 versus E3)
- ``Eagle3Model.forward``: ``ttt_length`` simulated draft steps, each feeding
  the draft's own (optionally normalized) hidden state back in
- ``compute_lazy_target_padded`` + forward-KL against the target ``lm_head``
- ``BF16Optimizer`` (fp32 master weights, fused AdamW, cosine schedule)
- per-position loss weights ``0.8 ** i`` and the token-weighted metrics

What it replaces is TorchSpec's Ray/Mooncake data plane: batches come from the
bounded feature caches written by ``apertus_eagle.features`` and are shaped
exactly as ``Eagle3Trainer._forward`` expects (input ids and target hidden
states shifted left by one, loss mask and aux features unshifted).

Effective batch = micro_batch (1) x accumulation x data-parallel ranks (1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT, load_contract, target_identity, target_keys
from apertus_eagle.features import check_cache


def device_name() -> str:
    """``cuda`` on the cluster; ``APERTUS_EAGLE_DEVICE=cpu`` for local smoke tests."""
    return os.environ.get("APERTUS_EAGLE_DEVICE", "cuda")


def peak_gpu_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_imports() -> Path:
    from apertus_eagle.import_stubs import ensure_import_stubs

    ensure_import_stubs()
    root = Path(os.environ.get("TORCHSPEC_ROOT", REPO_ROOT / "scratch/TorchSpec"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


class FeatureCache:
    def __init__(self, root: Path, contract: dict[str, Any]):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        check_cache(self.manifest, contract)
        self.samples = self.manifest["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, index: int) -> dict[str, Any]:
        from safetensors.torch import load_file

        return load_file(str(self.root / self.samples[index]["file"]))


def build_batch(record: dict[str, Any], device) -> dict[str, Any]:
    """One microbatch in ``Eagle3Trainer._forward`` layout (batch size 1)."""
    import torch
    from torchspec.utils.tensor import padding

    input_ids = record["input_ids"].long().unsqueeze(0).to(device)
    loss_mask = record["loss_mask"].long().unsqueeze(0).to(device)
    hidden = record["hidden_states"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    last = record["last_hidden_states"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    return {
        "input_ids": padding(input_ids, left=False),
        "target_hidden_states": padding(last, left=False),
        "loss_mask": loss_mask,
        "attention_mask": torch.ones_like(input_ids),
        "hidden_states": hidden,
    }


class RolloutTrainer:
    def __init__(self, cfg: dict[str, Any], contract: dict[str, Any], output: Path):
        import torch
        from torchspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
        from torchspec.models.draft.base import load_tensor_from_pretrained
        from torchspec.models.eagle3 import Eagle3Model
        from torchspec.training.optimizer import BF16Optimizer

        from apertus_bench.eagle import validate_eagle_config

        self.cfg = cfg
        self.contract = contract
        self.output = output
        train = cfg["training"]
        self.device = torch.device(device_name())
        self.ttt_length = int(train["ttt_length"])
        self.accum = int(train["draft_accumulation_steps"])
        self.max_steps = int(train["num_train_steps"])
        self.weights = [0.8**i for i in range(self.ttt_length)]
        self.weight_sum = sum(self.weights)

        draft_config_path = Path(cfg["model"]["draft_model_config"])
        if not draft_config_path.is_absolute():
            draft_config_path = REPO_ROOT / draft_config_path
        self.draft_config_raw = json.loads(draft_config_path.read_text())
        self.draft_config_path = draft_config_path
        self.config_report = validate_eagle_config(
            self.draft_config_raw, target=contract["target"], source=str(draft_config_path)
        )
        algorithm = cfg["model"]["algorithm"]
        if self.config_report["algorithm"] != algorithm:
            raise SystemExit(
                f"draft config is {self.config_report['algorithm']}, run says {algorithm}"
            )

        torch.manual_seed(int(train["seed"]))
        draft_config = AutoDraftModelConfig.from_dict(self.draft_config_raw)
        # Transformers 5 keeps llama3 RoPE under rope_parameters; TorchSpec's
        # draft reads rope_scaling. Same values as the serving config.
        if getattr(draft_config, "rope_scaling", None) is None:
            draft_config.rope_scaling = dict(self.draft_config_raw["rope_parameters"])
        for flag in ("fc_norm", "norm_output"):
            if bool(getattr(draft_config, flag, False)) != bool(self.draft_config_raw[flag]):
                raise SystemExit(
                    f"LlamaConfig dropped {flag}; refusing to train the wrong architecture"
                )
        draft = AutoEagle3DraftModel.from_config(
            draft_config, attention_backend=train["attention_backend"], torch_dtype=torch.bfloat16
        )
        keys = target_keys(contract)
        target_path = contract["source"]["authorized_checkpoint"]
        embed = load_tensor_from_pretrained(target_path, keys["embedding_key"])
        rows = draft.embed_tokens.weight.shape[0]
        with torch.no_grad():
            # Draft embedding covers the text output rows of the target input embedding.
            draft.embed_tokens.weight.copy_(embed[:rows].to(torch.bfloat16))
        del embed
        target_lm_head = load_tensor_from_pretrained(target_path, keys["lm_head_key"])
        if list(target_lm_head.shape) != keys["lm_head_shape"]:
            raise SystemExit(f"target lm_head {list(target_lm_head.shape)} != contract")
        self.freeze_lm_head = bool(cfg["model"].get("freeze_lm_head", False))
        if self.freeze_lm_head:
            # TorchSpec seeds a frozen head from the target (Eagle3Trainer.init_model).
            with torch.no_grad():
                draft.lm_head.weight.copy_(target_lm_head.to(torch.bfloat16))
        draft.freeze_embedding()
        if self.freeze_lm_head:
            draft.freeze_lm_head()
        self.draft = draft.to(self.device)
        self.target_lm_head_weight = target_lm_head.to(self.device, torch.bfloat16)
        self.target_lm_head_weight.requires_grad_(False)

        self.model = Eagle3Model(
            draft_model=self.draft,
            length=self.ttt_length,
            attention_backend=train["attention_backend"],
            gradient_checkpointing=bool(train.get("gradient_checkpointing", False)),
            loss_type=train.get("loss_type", "forward_kl"),
        ).to(self.device)
        if self.model.vocab_pruning:
            raise SystemExit(
                "vocabulary pruning is a later ablation; initial recipe uses full output"
            )
        self.optimizer = BF16Optimizer(
            self.draft,
            lr=float(train["learning_rate"]),
            weight_decay=float(train.get("weight_decay", 0.0)),
            max_grad_norm=float(train["max_grad_norm"]),
            total_steps=int(train.get("lr_total_steps") or self.max_steps),
            warmup_ratio=float(train["warmup_ratio"]),
            decay_style=train.get("lr_decay_style", "cosine"),
            min_lr=float(train.get("min_lr", 0.0)),
        )
        self.param_counts = {
            "trainable": sum(p.numel() for p in self.draft.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in self.draft.parameters() if not p.requires_grad),
            "by_module": {
                name: sum(p.numel() for p in module.parameters())
                for name, module in self.draft.named_children()
            },
        }

    def forward(self, batch: dict[str, Any]):
        from torchspec.models.eagle3 import compute_lazy_target_padded

        target = compute_lazy_target_padded(
            batch["target_hidden_states"], self.target_lm_head_weight, self.ttt_length
        )
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target=target,
            loss_mask=batch["loss_mask"],
            hidden_states=batch["hidden_states"],
        )

    def metrics(self, stats: list[tuple[list, list, list]]) -> dict[str, Any]:
        """Token-weighted per-position metrics, as Eagle3Trainer aggregates them."""
        n = self.ttt_length
        loss = [0.0] * n
        correct = [0.0] * n
        counts = [0.0] * n
        for vlosses, acces, acc_counts in stats:
            for i in range(n):
                loss[i] += float(vlosses[i])
                correct[i] += float(acces[i]) * float(acc_counts[i])
                counts[i] += float(acc_counts[i])
        mean_loss = [value / max(len(stats), 1) for value in loss]
        acc = [c / max(k, 1.0) for c, k in zip(correct, counts, strict=True)]
        cumulative = 1.0
        sim_len = 0.0
        for value in acc:
            cumulative *= value
            sim_len += cumulative
        return {
            "avg_loss": sum(w * v for w, v in zip(self.weights, mean_loss, strict=True))
            / self.weight_sum,
            "avg_acc": sum(acc) / n,
            "simulated_acc_len": sim_len,
            "acc": acc,
            "ploss": mean_loss,
            "tokens": counts[0],
        }

    def evaluate(self, cache: FeatureCache) -> dict[str, Any]:
        import torch

        self.model.eval()
        stats = []
        with torch.no_grad():
            for index in range(len(cache)):
                _p, vlosses, acces, acc_counts, _alphas = self.forward(
                    build_batch(cache.load(index), self.device)
                )
                stats.append(
                    (
                        [v.item() for v in vlosses],
                        [a.item() for a in acces],
                        [c.item() for c in acc_counts],
                    )
                )
        self.model.train()
        return self.metrics(stats)

    def save_checkpoint(self, tag: str, extra: dict[str, Any]) -> Path:
        import torch
        from safetensors.torch import save_file

        directory = self.output / "checkpoints" / tag
        directory.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.draft.state_dict().items()}
        save_file(state, str(directory / "draft.safetensors"))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.optimizer.lr_scheduler.state_dict(),
            },
            directory / "optimizer.pt",
        )
        (directory / "state.json").write_text(json.dumps(extra, indent=2) + "\n")
        return directory


def _probe_param(draft) -> Any:
    return draft.fc.weight.detach().float().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    torchspec_root = _prepare_imports()
    import torch

    cfg = _load_yaml(args.config)
    for override in args.set or []:
        dotted, _, value = override.partition("=")
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml_scalar(value)
    contract = load_contract(args.contract or cfg["model"].get("contract"))
    output = Path(args.output or cfg["output_dir"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if (output / "train-summary.json").exists() and not args.resume_ok:
        raise SystemExit(f"{output} already holds a finished run; use a new output_dir")
    output.mkdir(parents=True, exist_ok=True)

    def cache_path(key: str) -> Path:
        path = Path(cfg["dataset"][key])
        return path if path.is_absolute() else REPO_ROOT / path

    train_cache = FeatureCache(cache_path("train_cache"), contract)
    eval_cache = FeatureCache(cache_path("eval_cache"), contract)
    trainer = RolloutTrainer(cfg, contract, output)
    train = cfg["training"]
    eval_interval = int(train["eval_interval"])
    save_interval = int(train.get("save_interval") or 0)
    patience = int(train.get("stop_after_evals_without_improvement") or 0)
    max_epochs = int(train.get("num_epochs") or 10**9)
    max_non_finite = int(train.get("max_non_finite_steps", 3))

    provenance = {
        "target": target_identity(contract),
        "contract": contract["_path"],
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "resolved_config": cfg,
        "draft_config": str(trainer.draft_config_path),
        "algorithm": cfg["model"]["algorithm"],
        "objective": {
            "implementation": "torchspec.models.eagle3.Eagle3Model",
            "ttt_length": trainer.ttt_length,
            "loss_type": train.get("loss_type", "forward_kl"),
            "position_weights": trainer.weights,
            "target": "compute_lazy_target_padded(target lm_head)",
        },
        "effective_batch": {
            "micro_batch_size": 1,
            "draft_accumulation_steps": trainer.accum,
            "data_parallel_ranks": 1,
            "sequences_per_optimizer_step": trainer.accum,
            "audit": "TorchSpec global_batch = per_dp_rank_batch(1) x dp(1) x accumulation",
        },
        "optimizer": {
            "class": "torchspec.training.optimizer.BF16Optimizer (fp32 master, fused AdamW)",
            "lr": float(train["learning_rate"]),
            "warmup_ratio": float(train["warmup_ratio"]),
            "max_grad_norm": float(train["max_grad_norm"]),
            "weight_decay": float(train.get("weight_decay", 0.0)),
            "decay_style": train.get("lr_decay_style", "cosine"),
            "min_lr": float(train.get("min_lr", 0.0)),
            "lr_total_steps": int(train.get("lr_total_steps") or trainer.max_steps),
            "adamw_betas_eps": "torch defaults (0.9, 0.999), 1e-8",
        },
        "freeze": {"embedding": True, "lm_head": trainer.freeze_lm_head},
        "parameter_counts": trainer.param_counts,
        "caches": {
            "train": {
                "path": str(train_cache.root),
                "corpus_sha256": train_cache.manifest["corpus_sha256"],
                "samples": len(train_cache),
            },
            "eval": {
                "path": str(eval_cache.root),
                "corpus_sha256": eval_cache.manifest["corpus_sha256"],
                "samples": len(eval_cache),
            },
        },
        "torchspec_root": str(torchspec_root),
        "torchspec_revision": _git_revision(torchspec_root),
        "harness_revision": _git_revision(REPO_ROOT),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"parameter_counts": trainer.param_counts}), flush=True)

    history: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    initial = trainer.evaluate(eval_cache)
    evals.append({"step": 0, **initial})
    print(json.dumps({"eval": evals[-1]}), flush=True)
    best = initial["simulated_acc_len"]
    best_step = 0
    evals_without_improvement = 0
    non_finite = 0
    rng = random.Random(int(train["seed"]))
    step = 0
    epoch = 0
    started = time.time()
    tokens_seen = 0
    stop_reason = "max_steps"
    probe = _probe_param(trainer.draft)
    window: list[tuple[list, list, list]] = []
    trainer.model.train()
    while step < trainer.max_steps:
        if epoch >= max_epochs:
            stop_reason = "max_epochs"
            break
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        # Drop the ragged tail so every optimizer step sees exactly `accum` sequences.
        order = order[: len(order) - len(order) % trainer.accum] or order
        for position, index in enumerate(order):
            batch = build_batch(train_cache.load(index), trainer.device)
            tokens_seen += int(batch["input_ids"].shape[1])
            plosses, vlosses, acces, acc_counts, _alphas = trainer.forward(batch)
            loss = sum(w * p for w, p in zip(trainer.weights, plosses, strict=True)) / trainer.accum
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step} sample {index}")
            loss.backward()
            window.append(
                (
                    [v.item() for v in vlosses],
                    [a.item() for a in acces],
                    [c.item() for c in acc_counts],
                )
            )
            if (position + 1) % trainer.accum:
                continue
            # TorchSpec warmup starts at init_lr=0: the first step's update is zero by design.
            lr_used = trainer.optimizer.get_learning_rate()
            grad_norm = float(trainer.optimizer.step())
            step += 1
            if not math.isfinite(grad_norm):
                non_finite += 1
                if non_finite > max_non_finite:
                    raise SystemExit(
                        f"{non_finite} non-finite grad norms; optimizer skipped updates"
                    )
            current = _probe_param(trainer.draft)
            update_norm = float((current - probe).norm())
            probe = current
            record = {
                "step": step,
                "epoch": epoch,
                "lr_used": lr_used,
                "lr_next": trainer.optimizer.get_learning_rate(),
                "grad_norm": grad_norm,
                "fc_update_norm": update_norm,
                "elapsed_s": round(time.time() - started, 1),
                "train_tokens_per_s": round(tokens_seen / max(time.time() - started, 1e-6), 1),
                "peak_gpu_gb": peak_gpu_gb(),
                **{f"train_{k}": v for k, v in trainer.metrics(window).items()},
            }
            window = []
            history.append(record)
            print(json.dumps(record), flush=True)
            if lr_used > 0 and update_norm == 0.0 and math.isfinite(grad_norm):
                raise SystemExit(f"optimizer step {step} at lr {lr_used} did not change fc.weight")
            if step % eval_interval == 0 or step == trainer.max_steps:
                result = trainer.evaluate(eval_cache)
                evals.append({"step": step, **result})
                print(json.dumps({"eval": evals[-1]}), flush=True)
                if result["simulated_acc_len"] > best:
                    best = result["simulated_acc_len"]
                    best_step = step
                    evals_without_improvement = 0
                    trainer.save_checkpoint("best", {"step": step, "eval": result})
                else:
                    evals_without_improvement += 1
                (output / "evals.json").write_text(json.dumps(evals, indent=2) + "\n")
                if patience and evals_without_improvement >= patience:
                    stop_reason = f"{patience} evals without improvement"
                    break
            if save_interval and step % save_interval == 0:
                trainer.save_checkpoint(f"step-{step:06d}", {"step": step})
            if step >= trainer.max_steps:
                break
        else:
            epoch += 1
            continue
        break

    final_dir = trainer.save_checkpoint("final", {"step": step})
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "evals.json").write_text(json.dumps(evals, indent=2) + "\n")
    first_losses = [h["train_avg_loss"] for h in history[: max(1, len(history) // 10)]]
    last_losses = [h["train_avg_loss"] for h in history[-max(1, len(history) // 10) :]]
    summary = {
        "steps": step,
        "epochs_completed": epoch,
        "stop_reason": stop_reason,
        "sequences_seen": step * trainer.accum,
        "tokens_seen": tokens_seen,
        "wall_seconds": round(time.time() - started, 1),
        "peak_gpu_gb": peak_gpu_gb(),
        "non_finite_grad_steps": non_finite,
        "train_loss_first_decile": sum(first_losses) / len(first_losses) if first_losses else None,
        "train_loss_last_decile": sum(last_losses) / len(last_losses) if last_losses else None,
        "eval_initial": evals[0],
        "eval_best": max(evals, key=lambda e: e["simulated_acc_len"]),
        "eval_final": evals[-1],
        "best_step": best_step,
        "gates": {
            "finite_losses": True,
            "real_updates": all(h["fc_update_norm"] > 0 for h in history if h["lr_used"] > 0)
            and any(h["lr_used"] > 0 for h in history),
            "loss_decreased": bool(history)
            and (sum(last_losses) / len(last_losses)) < (sum(first_losses) / len(first_losses)),
            "heldout_acceptance_improved": best > initial["simulated_acc_len"],
        },
        "final_checkpoint": str(final_dir),
        "best_checkpoint": str(output / "checkpoints/best") if best_step else None,
    }
    (output / "train-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    return summary


def yaml_scalar(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-train-rollout")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", action="append", help="dotted.key=value override")
    parser.add_argument("--resume-ok", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args)
    raise SystemExit(0 if all(summary["gates"].values()) else 3)


if __name__ == "__main__":
    main()
