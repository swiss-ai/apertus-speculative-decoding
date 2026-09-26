"""One-step import/overfit debug tool for an Apertus 1.5 EAGLE head.

Trains a single draft step against frozen target features. It does not run
the multi-step rollout objective and is not a substitute for
``apertus_eagle.train_rollout``. The target, aux layers and tensor keys come
from the selected ``--contract``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from apertus_eagle.contract import REPO_ROOT, aux_layer_ids, load_contract, target_keys
from apertus_eagle.import_stubs import ensure_import_stubs
from apertus_eagle.renderer import ApertusRenderer


def _torchspec_root() -> Path:
    env = os.environ.get("TORCHSPEC_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "scratch/TorchSpec"


def _ensure_torchspec_on_path() -> None:
    root = _torchspec_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _ensure_import_stubs() -> None:
    ensure_import_stubs()
    _ensure_torchspec_on_path()


def load_target(model_path: Path, dtype: torch.dtype) -> nn.Module:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "dtype": dtype,
    }
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        try:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    except Exception:
        from transformers import AutoModel

        return AutoModel.from_pretrained(str(model_path), **kwargs)


def embed_device(model: nn.Module) -> torch.device:
    language_model = walk_language_model(model)
    return language_model.embed_tokens.weight.device


def walk_language_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "language_model"):
        return model.language_model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    if inner is not None:
        return inner
    return model


def capture_features(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    aux_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    language_model = walk_language_model(model)
    layers = language_model.layers
    captured: dict[int, torch.Tensor] = {}
    last_hidden: dict[str, torch.Tensor] = {}

    def layer_hook(idx: int):
        def hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach()

        return hook

    def norm_hook(_module, _inp, output):
        last_hidden["value"] = (output[0] if isinstance(output, tuple) else output).detach()

    handles = [layers[idx].register_forward_hook(layer_hook(idx)) for idx in aux_layers]
    handles.append(language_model.norm.register_forward_hook(norm_hook))
    try:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        try:
            model(**kwargs)
        except TypeError:
            kwargs.pop("output_attentions", None)
            model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    hidden = torch.cat(
        [captured[idx].to(last_hidden["value"].device) for idx in aux_layers], dim=-1
    )
    lm_head = model.lm_head
    head_device = next(lm_head.parameters()).device
    logits = lm_head(last_hidden["value"].to(head_device))
    return hidden, last_hidden["value"], logits


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def conversations_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    if row.get("conversations"):
        return list(row["conversations"])
    messages = list(row.get("messages") or [])
    if row.get("original_assistant") is not None:
        messages = [*messages, {"role": "assistant", "content": row["original_assistant"]}]
    return messages


def import_draft_class():
    """Import the EAGLE draft class before the 70B weight load.

    TorchSpec's package init reaches Mooncake. ``mooncake/utils.py`` imports
    ``ray``, which the serving image does not ship. One-node overfit never
    starts Mooncake; ``apply-torchspec-patches.sh`` lazy-loads that import.
    Stubs cover ray plus datasets/pydantic, which ``train_config`` imports
    next and which the serving image may also omit.
    """
    _ensure_import_stubs()
    from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3

    print(
        json.dumps({"preflight": "draft_import_ok", "cls": LlamaForCausalLMEagle3.__name__}),
        flush=True,
    )
    return LlamaForCausalLMEagle3


def build_draft(config_path: Path, device: torch.device, dtype: torch.dtype):
    LlamaForCausalLMEagle3 = import_draft_class()
    from transformers import LlamaConfig

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw.pop("note", None)
    config = LlamaConfig.from_dict(raw)
    for key in (
        "fc_norm",
        "norm_output",
        "target_hidden_size",
        "draft_vocab_size",
        "eagle_aux_hidden_state_layer_ids",
        "eagle_config",
        "num_aux_hidden_states",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    if getattr(config, "num_aux_hidden_states", None) is None:
        config.num_aux_hidden_states = 3
    # Transformers 5 keeps llama3 RoPE under rope_parameters. The draft
    # builder reads rope_scaling. Copy only when that block is absent.
    # Do not change fc_norm or norm_output.
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_parameters = raw.get("rope_parameters")
    if rope_scaling is None and isinstance(rope_parameters, dict):
        config.rope_scaling = dict(rope_parameters)
    draft = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
    return draft.to(device=device, dtype=dtype), raw


def seed_shared_weights(draft: nn.Module, target_path: Path, keys: dict[str, Any]) -> None:
    _ensure_import_stubs()
    from torchspec.models.draft.base import load_tensor_from_pretrained

    embed = load_tensor_from_pretrained(str(target_path), keys["embedding_key"])
    lm_head = load_tensor_from_pretrained(str(target_path), keys["lm_head_key"])
    n_embed = draft.embed_tokens.weight.shape[0]
    if embed.shape[0] < n_embed:
        raise ValueError(f"target embed rows {embed.shape[0]} < draft {n_embed}")
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(embed[:n_embed].to(draft.embed_tokens.weight.dtype))
        if lm_head.shape != draft.lm_head.weight.shape:
            raise ValueError(
                f"lm_head {tuple(lm_head.shape)} != draft {tuple(draft.lm_head.weight.shape)}"
            )
        draft.lm_head.weight.copy_(lm_head.to(draft.lm_head.weight.dtype))
    draft.freeze_embedding()
    draft.freeze_lm_head()


def remap_export_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state.items():
        new_key = "layers.0." + key[len("midlayer.") :] if key.startswith("midlayer.") else key
        remapped[new_key] = value.detach().cpu().contiguous()
    return remapped


def save_head(draft: nn.Module, raw_config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = remap_export_keys(draft.state_dict())
    save_file(tensors, str(output_dir / "model.safetensors"))
    export = dict(raw_config)
    export["_trained_by"] = "apertus_eagle.overfit_direct"
    export["torch_dtype"] = "bfloat16"
    (output_dir / "config.json").write_text(json.dumps(export, indent=2) + "\n")


def train(args: argparse.Namespace) -> dict[str, Any]:
    # Import before the teacher load. Job 3446638 spent ~14 min loading 70B
    # and then died on ``import ray`` inside the draft-class import.
    import_draft_class()
    from transformers import AutoTokenizer

    contract = load_contract(args.contract)
    aux_layers = aux_layer_ids(contract, "hf")
    keys = target_keys(contract)
    if args.model is None:
        args.model = Path(contract["source"]["authorized_checkpoint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if args.chat_template.is_file():
        tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    renderer = ApertusRenderer(tokenizer)
    rows = load_jsonl(args.input)[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.input}")

    target = load_target(args.model, dtype)
    target.eval()
    for param in target.parameters():
        param.requires_grad_(False)
    teacher_device = embed_device(target)

    samples: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for row in rows:
            input_ids, loss_mask = renderer.render(
                conversations_from_row(row), max_seq_length=args.max_seq_length
            )
            if sum(loss_mask) < 2:
                continue
            ids = torch.tensor(input_ids, device=teacher_device).unsqueeze(0)
            mask = torch.ones_like(ids)
            hidden, _last, logits = capture_features(target, ids, mask, aux_layers)
            samples.append(
                {
                    "input_ids": ids.cpu(),
                    "loss_mask": torch.tensor(loss_mask),
                    "hidden": hidden.cpu(),
                    "target_logits": logits.cpu(),
                }
            )
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not samples:
        raise SystemExit("teacher produced no supervised samples")

    draft, raw_config = build_draft(args.draft_config, device, dtype)
    seed_shared_weights(draft, args.model, keys)
    trainable = [param for param in draft.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    history: list[dict[str, float]] = []
    draft.train()
    step = 0
    while step < args.steps:
        epoch_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(samples):
            input_ids = sample["input_ids"].to(device)
            hidden = sample["hidden"].to(device=device, dtype=dtype)
            loss_mask = sample["loss_mask"].to(device)
            target_logits = sample["target_logits"].to(device=device, dtype=torch.float32)
            projected = draft.project_hidden_states(hidden)
            embeds = draft.embed_input_ids(input_ids)
            position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            hidden_out, _, _ = draft.backbone(
                embeds, projected, None, position_ids, use_cache=False
            )
            logits = draft.compute_logits(hidden_out).float()
            shift_logits = logits[:, :-1].contiguous()
            shift_target = target_logits[:, 1:].contiguous()
            shift_mask = loss_mask[1:].to(dtype=shift_logits.dtype)
            log_p = F.log_softmax(shift_logits, dim=-1)
            target_p = F.softmax(shift_target, dim=-1)
            kl = F.kl_div(log_p, target_p, reduction="none").sum(dim=-1)
            denom = shift_mask.sum().clamp_min(1.0)
            loss = (kl * shift_mask.unsqueeze(0)).sum() / denom
            (loss / args.accum).backward()
            epoch_loss += float(loss.detach())
            seen += 1
            if (index + 1) % args.accum == 0 or index == len(samples) - 1:
                nn.utils.clip_grad_norm_(trainable, 0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                history.append({"step": step, "loss": epoch_loss / max(seen, 1)})
                print(json.dumps(history[-1]), flush=True)
                if step >= args.steps:
                    break
        if seen and history and history[-1]["loss"] < args.overfit_loss:
            break

    save_head(draft, raw_config, args.output)
    summary = {
        "n_samples": len(samples),
        "steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "overfit": bool(history) and history[-1]["loss"] < args.overfit_loss,
        "output": str(args.output),
        "draft_config": str(args.draft_config),
        "teacher_generation": "not_run_mix_assistant_teacher_forced",
        "aux_layers_hf_decoder": aux_layers,
        "contract": contract["_path"],
        "objective": "one-step debug; not the multi-step rollout objective",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overfit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-overfit-direct")
    parser.add_argument(
        "--contract", type=Path, help="target contract (or $APERTUS_EAGLE_CONTRACT)"
    )
    parser.add_argument("--model", type=Path, help="default: the contract's checkpoint")
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(__file__).resolve().parent / "chat_template.jinja",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit-loss", type=float, default=0.05)
    args = parser.parse_args(argv)
    summary = train(args)
    raise SystemExit(0 if summary.get("overfit") else 1)


if __name__ == "__main__":
    main()
