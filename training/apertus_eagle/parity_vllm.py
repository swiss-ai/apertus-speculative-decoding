"""A2: compare HF-extracted features and logits with the vLLM serving path.

Loads ``parity-prefixes.safetensors`` written by ``apertus_eagle.features``,
runs the same prefixes through vLLM (TP=1, eager, in-process engine) with
forward hooks on every decoder layer, and reports for each HF aux layer the
error against vLLM index ``i + 1`` (the contract's claim) and its neighbours
``i`` and ``i + 2``. A correct offset gives the smallest error by a wide
margin. Each prefix runs twice so vLLM's own run-to-run noise is the control.
Final-token agreement uses ``prompt_logprobs``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from apertus_eagle.contract import aux_layer_ids, load_contract  # noqa: E402

_CAPTURE: dict[int, Any] = {}
_HANDLES: list[Any] = []


def _decoder_layers(model):
    lm = model.get_language_model() if hasattr(model, "get_language_model") else model
    for candidate in (lm, getattr(lm, "model", None)):
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise RuntimeError(f"no decoder layers under {type(lm).__name__}")


def _install_hooks(model):
    import torch

    _CAPTURE.clear()
    for handle in _HANDLES:
        handle.remove()
    _HANDLES.clear()
    layers = _decoder_layers(model)

    def make(idx):
        def hook(_module, _inputs, output):
            if idx in _CAPTURE:
                return
            hidden, residual = output if isinstance(output, tuple) else (output, None)
            value = hidden if residual is None else hidden + residual
            _CAPTURE[idx] = value.detach().to(torch.bfloat16).cpu()

        return hook

    for idx, layer in enumerate(layers):
        _HANDLES.append(layer.register_forward_hook(make(idx)))
    return len(layers)


def _reset(_model):
    _CAPTURE.clear()
    return True


def _collect(_model):
    return dict(_CAPTURE)


def rel_err(a, b) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-6))


def cosine(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean())


OFFSET_RATIO = 0.2


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Offset gate, judged per sample and per layer.

    Errors grow with depth, so pooling (worst claimed error over all layers
    versus best neighbour error over all layers) compares a deep layer with a
    shallow layer's neighbours. The claimed vLLM index must beat both of its
    own neighbours by ``1 / OFFSET_RATIO`` in every sample.
    """
    ratios = []
    for sample in samples:
        for layer in sample["layers"]:
            neighbours = [layer[k] for k in ("minus1_rel_err", "plus1_rel_err") if k in layer]
            if neighbours:
                ratios.append(layer["claimed_rel_err"] / min(neighbours))
    return {
        "claimed_rel_err_max": max(l["claimed_rel_err"] for s in samples for l in s["layers"]),
        "claimed_over_nearest_neighbour_max": max(ratios) if ratios else None,
        "offset_ratio_threshold": OFFSET_RATIO,
        "offset_confirmed": bool(ratios) and max(ratios) <= OFFSET_RATIO,
        "vllm_repeat_rel_err_max": max(l["vllm_repeat_rel_err"] for s in samples for l in s["layers"]),
        "argmax_agree_min": min(s["argmax_agree"] for s in samples),
        "max_disagreement_margin": max(s["max_disagreement_margin"] for s in samples),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-parity-vllm")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args(argv)

    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    contract = load_contract(args.contract)
    hf_aux = aux_layer_ids(contract, "hf")
    vllm_aux = aux_layer_ids(contract, "vllm")
    tensors = load_file(str(args.prefixes))
    slots = sorted({int(key.split(".")[0][1:]) for key in tensors})[: args.max_samples]
    max_len = max(int(tensors[f"s{s}.input_ids"].numel()) for s in slots)

    llm_kwargs: dict[str, Any] = dict(
        model=contract["source"]["authorized_checkpoint"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=max(4096, max_len + 16),
        max_num_batched_tokens=max(8192, max_len + 16),
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        seed=0,
    )
    try:
        llm = LLM(**llm_kwargs, compilation_config={"mode": 0})
    except TypeError:
        llm = LLM(**llm_kwargs, compilation_config={"level": 0})
    num_layers = llm.apply_model(_install_hooks)[0]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    report: dict[str, Any] = {
        "hf_aux": hf_aux,
        "vllm_aux": vllm_aux,
        "num_layers_vllm": num_layers,
        "samples": [],
    }
    for slot in slots:
        ids = tensors[f"s{slot}.input_ids"].tolist()
        hf_hidden = tensors[f"s{slot}.hidden_states"]
        hf_chunks = hf_hidden.chunk(len(hf_aux), dim=-1)
        runs = []
        for _ in range(2):
            llm.apply_model(_reset)
            out = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0]
            captured = llm.apply_model(_collect)[0]
            runs.append((out, captured))
        (out1, cap1), (_out2, cap2) = runs
        sample: dict[str, Any] = {"slot": slot, "tokens": len(ids), "layers": []}
        for chunk, hf_layer, vllm_index in zip(hf_chunks, hf_aux, vllm_aux):
            if vllm_index - 1 != hf_layer:
                raise SystemExit(f"contract offset broken: vLLM {vllm_index} vs HF {hf_layer}")
            entry: dict[str, Any] = {"hf_layer": hf_layer, "vllm_index": vllm_index}
            # vLLM index k is the stream after decoder layer k-1 = hook on layer k-1.
            for label, layer in (("claimed", hf_layer), ("minus1", hf_layer - 1), ("plus1", hf_layer + 1)):
                if 0 <= layer < num_layers and layer in cap1:
                    vl = cap1[layer][: len(ids)]
                    entry[f"{label}_rel_err"] = rel_err(vl, chunk)
                    entry[f"{label}_cosine"] = cosine(vl, chunk)
            entry["vllm_repeat_rel_err"] = rel_err(cap1[hf_layer][: len(ids)], cap2[hf_layer][: len(ids)])
            sample["layers"].append(entry)
        # prompt_logprobs[t] scores token t given tokens < t; HF top1 at t-1 predicts token t.
        hf_top1 = tensors[f"s{slot}.top2_ids"][:, 0].tolist()
        hf_logits = tensors[f"s{slot}.top2_logits"].float()
        agree = 0
        compared = 0
        disagreements = []
        for t, entry in enumerate(out1.prompt_logprobs or []):
            if t == 0 or entry is None:
                continue
            best = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
            compared += 1
            if best == hf_top1[t - 1]:
                agree += 1
            else:
                margin = float(hf_logits[t - 1, 0] - hf_logits[t - 1, 1])
                disagreements.append({"pos": t - 1, "hf_margin": round(margin, 4)})
        sample["argmax_agree"] = agree / max(compared, 1)
        sample["argmax_compared"] = compared
        sample["disagreement_hf_margins"] = disagreements[:20]
        sample["max_disagreement_margin"] = max((d["hf_margin"] for d in disagreements), default=0.0)
        report["samples"].append(sample)
        print(json.dumps(sample), flush=True)

    report["summary"] = summarize(report["samples"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
