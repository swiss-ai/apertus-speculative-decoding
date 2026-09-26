"""A2/A3: bounded target-feature caches from the frozen HF teacher on one GPU.

One safetensors file per conversation holds exactly what TorchSpec's
``Eagle3Trainer._forward`` consumes:

- ``input_ids`` [T] int32: ``prompt_ids + generated_ids`` (the served stream)
- ``loss_mask`` [T] uint8: 1 on generated tokens, last position forced to 0
  (as TorchSpec preprocessing does)
- ``hidden_states`` [T, 3H] bf16: decoder-layer outputs at the contract's HF
  aux ids, concatenated low-to-high (vLLM aux order)
- ``last_hidden_states`` [T, H] bf16: final RMSNorm output (post-norm)

``manifest.json`` names the target identity, feature contract and corpus
digest. ``check_cache`` refuses reuse when any of them changes.

The first ``--parity-samples`` conversations also get numeric checks:
reconstructed logits ``lm_head(last_hidden)`` versus the model's own logits,
and a repeated same-path forward as the noise control. Their first
``--parity-prefix`` tokens are saved for ``apertus_eagle.parity_vllm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

from apertus_eagle.contract import aux_layer_ids, load_contract, target_identity, target_keys

FEATURE_FORMAT = "apertus-eagle-features-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": FEATURE_FORMAT,
        "hf_decoder_layer_aux_ids": aux_layer_ids(contract, "hf"),
        "vllm_aux_ids": aux_layer_ids(contract, "vllm"),
        "hidden_states": "concat of HF decoder-layer outputs (residual included), low to high",
        "last_hidden_states": "final RMSNorm output (post-norm); trainer applies no norm",
        "dtype": "bfloat16",
        "loss_mask": "1 on generated tokens; last position 0",
        "hidden_size": int(contract["target"]["text"]["hidden_size"]),
    }


def check_cache(
    manifest: dict[str, Any], contract: dict[str, Any], corpus_sha256: str | None = None
) -> None:
    """Refuse a cache built for another target, feature layout or corpus."""
    problems = []
    if manifest.get("target") != target_identity(contract):
        problems.append("target identity differs")
    if manifest.get("feature_contract") != feature_contract(contract):
        problems.append("feature contract differs")
    if corpus_sha256 is not None and manifest.get("corpus_sha256") != corpus_sha256:
        problems.append("corpus digest differs")
    if manifest.get("status") != "complete":
        problems.append(f"cache status is {manifest.get('status')!r}")
    if problems:
        raise ValueError(f"refusing feature cache {manifest.get('path')}: {', '.join(problems)}")


def load_teacher(model_path: str):
    import torch

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, "low_cpu_mem_usage": True}
    errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers

            cls = getattr(transformers, name)
            model = cls.from_pretrained(model_path, **kwargs)
            return model, name
        except Exception as error:  # noqa: BLE001 - record and try the next class
            errors.append(f"{name}: {type(error).__name__}: {error}")
    raise RuntimeError("could not load teacher:\n" + "\n".join(errors))


def language_model(model):
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        module = model
        for attr in path:
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None and hasattr(module, "layers") and hasattr(module, "norm"):
            return module
    raise RuntimeError("could not locate Apertus language_model layers/norm")


class FeatureCapture:
    def __init__(self, model, aux_layers: list[int], num_layers: int):
        self.model = model
        self.lm = language_model(model)
        if len(self.lm.layers) != num_layers:
            raise RuntimeError(f"teacher has {len(self.lm.layers)} layers, contract {num_layers}")
        self.aux_layers = aux_layers
        self.captured: dict[int, Any] = {}
        self.final: dict[str, Any] = {}
        self.handles = []
        for idx in aux_layers:
            self.handles.append(self.lm.layers[idx].register_forward_hook(self._layer_hook(idx)))
        self.handles.append(self.lm.norm.register_forward_hook(self._norm_hook))

    def _layer_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self.captured[idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    def _norm_hook(self, _module, _inputs, output):
        self.final["value"] = (output[0] if isinstance(output, tuple) else output).detach()

    def run(self, input_ids, *, full_logits: bool = False):
        import torch

        self.captured.clear()
        self.final.clear()
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if not full_logits:
            kwargs["logits_to_keep"] = 1
        try:
            output = self.model(**kwargs)
        except TypeError:
            kwargs.pop("logits_to_keep", None)
            output = self.model(**kwargs)
        hidden = torch.cat([self.captured[idx][0] for idx in self.aux_layers], dim=-1)
        last = self.final["value"][0]
        return hidden, last, getattr(output, "logits", None)


def training_sequence(row: dict[str, Any], max_seq_length: int) -> tuple[list[int], list[int]]:
    prompt = list(row["prompt_ids"])
    generated = list(row["generated_ids"])
    ids = (prompt + generated)[:max_seq_length]
    mask = ([0] * len(prompt) + [1] * len(generated))[:max_seq_length]
    if mask:
        mask[-1] = 0
    return ids, mask


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-features")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--input", type=Path, required=True, help="generate_targets output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-cache-gb", type=float, default=400.0)
    parser.add_argument("--measure-samples", type=int, default=100)
    parser.add_argument("--parity-samples", type=int, default=0)
    parser.add_argument("--parity-prefix", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import save_file

    torch.backends.cuda.matmul.allow_tf32 = False
    contract = load_contract(args.contract)
    keys = target_keys(contract)
    aux = aux_layer_ids(contract, "hf")
    draft_vocab = int(contract["draft"]["vocab_size"])
    corpus_sha = file_sha256(args.input)
    out = args.output_dir
    manifest_path = out / "manifest.json"
    resumable: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "partial":
            # A timed-out run: keep its samples only if target, layout and corpus match.
            partial = {**existing, "status": "complete"}
            try:
                check_cache(partial, contract, corpus_sha)
            except ValueError as error:
                raise SystemExit(f"{error}; move the partial cache aside") from error
            resumable = {entry["file"]: entry for entry in existing["samples"]}
        else:
            try:
                check_cache(existing, contract, corpus_sha)
            except ValueError as error:
                raise SystemExit(f"{error}; move it aside instead of overwriting") from error
            print(json.dumps({"cache": str(out), "status": "reused", "samples": len(existing["samples"])}))
            return
    elif out.exists() and any(out.glob("sample-*.safetensors")):
        raise SystemExit(f"{out} has samples but no manifest; move it aside")
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    load_started = time.time()
    model, loader = load_teacher(contract["source"]["authorized_checkpoint"])
    model.eval()
    capture = FeatureCapture(model, aux, keys["num_hidden_layers"])
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    if tuple(lm_head.weight.shape) != tuple(keys["lm_head_shape"]):
        raise SystemExit(f"teacher lm_head {tuple(lm_head.weight.shape)} != {keys['lm_head_shape']}")
    load_seconds = time.time() - load_started

    manifest: dict[str, Any] = {
        "path": str(out),
        "status": "partial",
        "target": target_identity(contract),
        "feature_contract": feature_contract(contract),
        "corpus": str(args.input),
        "corpus_sha256": corpus_sha,
        "code_sha256": file_sha256(Path(__file__)),
        "teacher_loader": loader,
        "max_seq_length": args.max_seq_length,
        "samples": [],
    }
    parity: dict[str, Any] = {"samples": [], "prefix_tokens": args.parity_prefix}
    parity_tensors: dict[str, Any] = {}
    total_bytes = 0
    total_tokens = 0
    started = time.time()
    with torch.no_grad():
        for index, row in enumerate(rows):
            ids, mask = training_sequence(row, args.max_seq_length)
            if sum(mask) < 2:
                continue
            too_big = sorted({token for token in ids if token >= draft_vocab})
            if too_big:
                # e.g. a literal <|image|> (131079) in prompt text. TorchSpec clamps
                # draft input ids to the text rows, so this row would train on the
                # wrong embedding row; serving uses the target's full embedding.
                manifest.setdefault("skipped_non_text_ids", []).append(
                    {"id": row["id"], "ids": too_big[:8]}
                )
                continue
            name = f"sample-{index:06d}.safetensors"
            want_parity = len(parity["samples"]) < args.parity_samples
            previous = resumable.get(name)
            if previous is not None and previous["id"] == row["id"] and not want_parity:
                manifest["samples"].append(previous)
                total_bytes += previous["bytes"]
                total_tokens += previous["tokens"]
                continue
            input_ids = torch.tensor([ids], device="cuda")
            hidden, last, logits = capture.run(input_ids, full_logits=want_parity)
            if want_parity:
                output_vocab = keys["output_vocab_size"]
                recon = lm_head(last.unsqueeze(0))[0].float()
                served = logits[0, :, :output_vocab].float()
                beyond = logits[0, :, output_vocab:]
                top2 = torch.topk(recon, 2, dim=-1)
                hidden2, last2, _ = capture.run(input_ids, full_logits=False)
                n = min(args.parity_prefix, len(ids))
                parity["samples"].append(
                    {
                        "id": row["id"],
                        "tokens": len(ids),
                        "recon_vs_model_max_abs": float((recon - served).abs().max()),
                        "recon_vs_model_argmax_agree": float(
                            (recon.argmax(-1) == served.argmax(-1)).float().mean()
                        ),
                        "model_logits_beyond_output_vocab": (
                            "all -inf" if beyond.numel() == 0 or bool(torch.isinf(beyond).all())
                            else f"max {float(beyond.float().max())}"
                        ),
                        "repeat_hidden_max_abs": float((hidden - hidden2).abs().max()),
                        "repeat_last_max_abs": float((last - last2).abs().max()),
                        "hidden_rms_per_layer": [
                            float(chunk.float().pow(2).mean().sqrt())
                            for chunk in hidden.chunk(len(aux), dim=-1)
                        ],
                    }
                )
                slot = len(parity["samples"]) - 1
                parity_tensors[f"s{slot}.input_ids"] = input_ids[0, :n].int().cpu()
                parity_tensors[f"s{slot}.hidden_states"] = hidden[:n].cpu()
                parity_tensors[f"s{slot}.last_hidden_states"] = last[:n].cpu()
                parity_tensors[f"s{slot}.top2_ids"] = top2.indices[:n].int().cpu()
                parity_tensors[f"s{slot}.top2_logits"] = top2.values[:n].cpu()
                del logits, recon, served, beyond
            tensors = {
                "input_ids": torch.tensor(ids, dtype=torch.int32),
                "loss_mask": torch.tensor(mask, dtype=torch.uint8),
                "hidden_states": hidden.to(torch.bfloat16).cpu().contiguous(),
                "last_hidden_states": last.to(torch.bfloat16).cpu().contiguous(),
            }
            save_file(tensors, str(out / name), metadata={"id": str(row["id"])})
            size = (out / name).stat().st_size
            total_bytes += size
            total_tokens += len(ids)
            manifest["samples"].append(
                {"file": name, "id": row["id"], "tokens": len(ids), "loss_tokens": sum(mask),
                 "bytes": size, "domain": row.get("domain")}
            )
            if len(manifest["samples"]) == args.measure_samples or index == len(rows) - 1:
                per_token = total_bytes / max(total_tokens, 1)
                mean_tokens = total_tokens / len(manifest["samples"])
                projected = per_token * mean_tokens * len(rows) / 1e9
                estimate = {
                    "after_samples": len(manifest["samples"]),
                    "bytes_per_token": round(per_token, 1),
                    "mean_tokens": round(mean_tokens, 1),
                    "projected_gb": round(projected, 2),
                    "seconds": round(time.time() - started, 1),
                    "teacher_tokens_per_second": round(total_tokens / (time.time() - started), 1),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                manifest.setdefault("estimates", []).append(estimate)
                print(json.dumps({"estimate": estimate}), flush=True)
                if projected > args.max_cache_gb:
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    raise SystemExit(
                        f"projected cache {projected:.1f} GB exceeds --max-cache-gb {args.max_cache_gb}"
                    )
            if index % 50 == 0:
                print(json.dumps({"progress": index, "of": len(rows)}), flush=True)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    elapsed = time.time() - started
    manifest.update(
        {
            "status": "complete",
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "bytes_per_token": total_bytes / max(total_tokens, 1),
            "teacher_load_seconds": round(load_seconds, 1),
            "extract_seconds": round(elapsed, 1),
            "teacher_tokens_per_second": round(total_tokens / max(elapsed, 1e-6), 1),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "peak_host_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    if parity["samples"]:
        save_file(parity_tensors, str(out / "parity-prefixes.safetensors"))
        parity["summary"] = {
            "argmax_agree_min": min(s["recon_vs_model_argmax_agree"] for s in parity["samples"]),
            "recon_max_abs_max": max(s["recon_vs_model_max_abs"] for s in parity["samples"]),
            "repeat_hidden_max_abs_max": max(s["repeat_hidden_max_abs"] for s in parity["samples"]),
        }
        (out / "hf-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        manifest["hf_parity"] = parity["summary"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
