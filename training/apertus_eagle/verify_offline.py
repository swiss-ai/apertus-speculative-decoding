"""A4: token-level greedy verification of an EAGLE head in the vLLM engine.

Each ``capture`` is its own process and its own engine load (TP=1,
``max_num_seqs=1`` so every request runs alone as at C=1, prefix caching off):

    capture --mode plain  --output plain-r1.json     # repeat for r2
    capture --mode eagle  --head DIR --output eagle-r1.json
    compare plain-r1.json eagle-r1.json --output cmp.json
    margins cmp.json --output margins.json           # plain engine, forced prefixes

Captures record prompt/generated token ids, finish reasons and the engine's
per-position acceptance counters. ``--cap-sweep`` adds requests whose output
cap lands on every position of a draft window (1..k+2 and a few longer), so
the output limit is exercised mid-round, at the bonus token and after it.
``compare`` reports exact token equality and the first differing token;
``margins`` scores both continuations on the identical forced prefix with the
plain target, so a divergence at a near-tie can be told apart from a
systematic verification error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from apertus_eagle.contract import load_contract, target_identity
from apertus_eagle.generate_targets import prompt_messages, render_prompt


def load_prompts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def wait_for_free_gpu(min_free_fraction: float, timeout_s: float = 300.0) -> dict[str, float]:
    """A just-exited step can still hold GPU memory for a few seconds."""
    import torch

    started = time.time()
    while True:
        free, total = torch.cuda.mem_get_info(0)
        if free / total >= min_free_fraction or time.time() - started > timeout_s:
            return {"free_gib": free / 2**30, "total_gib": total / 2**30,
                    "waited_s": round(time.time() - started, 1)}
        time.sleep(5)


def build_engine(contract: dict[str, Any], args: argparse.Namespace):
    from vllm import LLM

    print(json.dumps({"gpu_before_engine": wait_for_free_gpu(args.gpu_memory_utilization + 0.05)}),
          flush=True)

    kwargs: dict[str, Any] = dict(
        model=contract["source"]["authorized_checkpoint"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        # C=1 for greedy captures; sampling may batch (exercises the batched
        # rejection sampler and keeps n=1024 draws affordable).
        max_num_seqs=getattr(args, "max_num_seqs", 1),
        seed=0,
        disable_log_stats=False,
    )
    if args.mode == "eagle":
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": str(args.head),
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }
    return LLM(**kwargs)


def spec_counters(llm) -> dict[str, Any]:
    try:
        metrics = llm.get_metrics()
    except Exception as error:  # noqa: BLE001 - counters are optional evidence
        return {"available": False, "error": str(error)}
    out: dict[str, Any] = {"available": True}
    for metric in metrics:
        name = getattr(metric, "name", "")
        if "spec_decode" not in name:
            continue
        value = getattr(metric, "value", None)
        if value is None and hasattr(metric, "values"):
            value = list(metric.values)
        out[name] = value
    return out


def capture(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    contract = load_contract(args.contract)
    eos = list(contract["target"]["tokens"]["generation_eos_token_id"])
    started = time.time()
    llm = build_engine(contract, args)
    load_seconds = time.time() - started
    tokenizer = llm.get_tokenizer()
    rows = load_prompts(args.prompts, args.limit)
    requests: list[tuple[str, list[int], int]] = []
    for row in rows:
        ids = row.get("prompt_ids") or render_prompt(tokenizer, prompt_messages(row))
        requests.append((str(row["id"]), list(ids), args.max_tokens))
    sweep_caps = list(range(1, args.num_speculative_tokens + 3)) + [2 * args.num_speculative_tokens + 3, 17]
    for row_id, ids, _cap in requests[: args.cap_sweep]:
        for cap in sweep_caps:
            requests.append((f"{row_id}@cap{cap}", ids, cap))
    params = [
        SamplingParams(temperature=0.0, max_tokens=cap, stop_token_ids=[] if args.ignore_eos else eos,
                       ignore_eos=args.ignore_eos, skip_special_tokens=False)
        for _id, _ids, cap in requests
    ]
    started = time.time()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for _id, ids, _cap in requests], params, use_tqdm=True
    )
    generate_seconds = time.time() - started
    cases = []
    for (case_id, ids, cap), output in zip(requests, outputs):
        completion = output.outputs[0]
        cases.append(
            {
                "id": case_id,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "prompt_tokens": len(ids),
                "max_tokens": cap,
                "token_ids": list(completion.token_ids),
                "finish_reason": str(completion.finish_reason),
                "stop_reason": completion.stop_reason,
            }
        )
    report = {
        "mode": args.mode,
        "head": str(args.head) if args.head else None,
        "num_speculative_tokens": args.num_speculative_tokens if args.mode == "eagle" else None,
        "target": target_identity(contract),
        "prompts": str(args.prompts),
        "ignore_eos": args.ignore_eos,
        "engine": {"tensor_parallel_size": 1, "max_num_seqs": 1, "prefix_caching": False,
                   "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_memory_utilization},
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(generate_seconds, 1),
        "output_tokens": sum(len(case["token_ids"]) for case in cases),
        "spec_counters": spec_counters(llm),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


def sample(args: argparse.Namespace) -> None:
    """A4.5: many temperature samples of the first ``--horizon`` tokens per prompt.

    Seeds for different runs must differ by far more than ``n``: vLLM derives the
    per-sample streams of one request from consecutive seeds, so seeds 1, 2, 3
    share almost all streams (first run, 2026-09-24: identical first-token counts
    across all three runs).

    Same-seed sequence equality is not expected under speculation (the
    rejection sampler consumes RNG differently); the distributions are what
    must agree. ``dist`` compares them against a plain-vs-plain control.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    contract = load_contract(args.contract)
    llm = build_engine(contract, args)
    tokenizer = llm.get_tokenizer()
    rows = load_prompts(args.prompts, args.limit)
    prompts = [list(r.get("prompt_ids") or render_prompt(tokenizer, prompt_messages(r))) for r in rows]
    params = SamplingParams(
        temperature=args.temperature, top_p=1.0, max_tokens=args.horizon, n=args.n,
        seed=args.seed, skip_special_tokens=False,
    )
    started = time.time()
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params, use_tqdm=True)
    report = {
        "mode": args.mode, "head": str(args.head) if args.head else None,
        "temperature": args.temperature, "horizon": args.horizon, "n": args.n, "seed": args.seed,
        "target": target_identity(contract), "prompts": str(args.prompts),
        "generate_seconds": round(time.time() - started, 1),
        "spec_counters": spec_counters(llm),
        "cases": [
            {"id": str(row["id"]), "samples": [list(c.token_ids) for c in out.outputs]}
            for row, out in zip(rows, outputs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


def _tv(left: list[list[int]], right: list[list[int]], depth: int) -> float:
    from collections import Counter

    a = Counter(tuple(s[:depth]) for s in left)
    b = Counter(tuple(s[:depth]) for s in right)
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def dist(args: argparse.Namespace) -> None:
    """Total-variation distance of sampled prefixes, candidate versus control."""
    ref = json.loads(args.reference.read_text())
    control = json.loads(args.control.read_text())
    candidate = json.loads(args.candidate.read_text())
    by_id = lambda rep: {c["id"]: c["samples"] for c in rep["cases"]}  # noqa: E731
    r, c, k = by_id(ref), by_id(control), by_id(candidate)
    horizon = ref["horizon"]
    per_depth = {}
    for depth in range(1, horizon + 1):
        tv_control = [_tv(r[i], c[i], depth) for i in r]
        tv_candidate = [_tv(r[i], k[i], depth) for i in r]
        per_depth[str(depth)] = {
            "mean_tv_reference_vs_control": sum(tv_control) / len(tv_control),
            "mean_tv_reference_vs_candidate": sum(tv_candidate) / len(tv_candidate),
            "candidate_minus_control": (sum(tv_candidate) - sum(tv_control)) / len(tv_control),
            "prompts_candidate_above_control": sum(x > y for x, y in zip(tv_candidate, tv_control)),
            "prompts": len(tv_control),
        }
    from collections import Counter

    varied = [i for i in r if len({s[0] for s in r[i]}) > 1]
    equal = sum(Counter(s[0] for s in r[i]) == Counter(s[0] for s in c[i]) for i in varied)
    correlated = bool(varied) and equal / len(varied) > 0.5
    report = {"reference": str(args.reference), "control": str(args.control),
              "seeds": [ref.get("seed"), control.get("seed"), candidate.get("seed")],
              "prompts_with_varied_first_token": len(varied),
              "identical_first_token_counts_ref_vs_control": equal,
              "correlated_streams_suspected": correlated,
              "candidate": str(args.candidate), "n": ref["n"], "temperature": ref["temperature"],
              "per_depth": per_depth}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def compare(args: argparse.Namespace) -> None:
    left = json.loads(args.baseline.read_text())
    right = json.loads(args.candidate.read_text())
    right_cases = {case["id"]: case for case in right["cases"]}
    divergences = []
    exact = 0
    finish_mismatch = 0
    for case in left["cases"]:
        other = right_cases[case["id"]]
        if case["prompt_ids_sha256"] != other["prompt_ids_sha256"]:
            raise SystemExit(f"prompt ids differ for {case['id']}")
        a, b = case["token_ids"], other["token_ids"]
        if a == b and case["finish_reason"] == other["finish_reason"]:
            exact += 1
            continue
        if case["finish_reason"] != other["finish_reason"]:
            finish_mismatch += 1
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        divergences.append(
            {
                "id": case["id"],
                "first_diff": first,
                "baseline_token": a[first] if first < len(a) else None,
                "candidate_token": b[first] if first < len(b) else None,
                "baseline_len": len(a),
                "candidate_len": len(b),
                "baseline_finish": case["finish_reason"],
                "candidate_finish": other["finish_reason"],
                "prefix": a[:first],
            }
        )
    k = right.get("num_speculative_tokens") or left.get("num_speculative_tokens") or 1
    stop_positions: dict[str, int] = {}
    for case in right["cases"]:
        if case["finish_reason"] == "stop":
            key = str(len(case["token_ids"]) % (k + 1))
            stop_positions[key] = stop_positions.get(key, 0) + 1
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_mode": left["mode"],
        "candidate_mode": right["mode"],
        "cases": len(left["cases"]),
        "exact": exact,
        "exact_rate": exact / max(len(left["cases"]), 1),
        "finish_reason_mismatches": finish_mismatch,
        "eos_position_mod_k_plus_1": stop_positions,
        "candidate_spec_counters": right.get("spec_counters"),
        "divergences": divergences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "divergences"}, indent=2), flush=True)


def margins(args: argparse.Namespace) -> None:
    """Score both continuations on the identical forced prefix with the plain target."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    report = json.loads(args.comparison.read_text())
    contract = load_contract(args.contract)
    args.mode = "plain"
    llm = build_engine(contract, args)
    base = json.loads(Path(report["baseline"]).read_text())
    prompt_by_id = {case["id"]: case for case in base["cases"]}
    rows = {str(r["id"]): r for r in load_prompts(args.prompts, None)}
    tokenizer = llm.get_tokenizer()
    results = []
    for item in report["divergences"]:
        if item["baseline_token"] is None or item["candidate_token"] is None:
            results.append({**item, "kind": "length_only"})
            continue
        row = rows[item["id"].split("@cap")[0]]
        ids = row.get("prompt_ids") or render_prompt(tokenizer, prompt_messages(row))
        prefix = list(ids) + list(item["prefix"])
        if hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest() != prompt_by_id[item["id"]]["prompt_ids_sha256"]:
            raise SystemExit(f"prompt mismatch for {item['id']}")
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=prefix)],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=20),
            use_tqdm=False,
        )[0].outputs[0]
        top = out.logprobs[0] if out.logprobs else {}
        scored = {int(t): float(lp.logprob) for t, lp in top.items()}
        b, c = item["baseline_token"], item["candidate_token"]
        results.append(
            {
                "id": item["id"],
                "first_diff": item["first_diff"],
                "repeat_argmax": int(out.token_ids[0]),
                "baseline_token_logprob": scored.get(b),
                "candidate_token_logprob": scored.get(c),
                "margin": (scored[b] - scored[c]) if b in scored and c in scored else None,
                "candidate_in_top20": c in scored,
                "baseline_is_repeat_argmax": int(out.token_ids[0]) == b,
            }
        )
    near_ties = [r for r in results if r.get("margin") is not None and abs(r["margin"]) < args.tie_margin]
    summary = {
        "divergences": len(results),
        "near_ties": len(near_ties),
        "tie_margin_logprob": args.tie_margin,
        "not_near_tie": [r for r in results if r not in near_ties and r.get("kind") != "length_only"],
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-verify-offline")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--contract", type=Path)
    cap.add_argument("--mode", choices=("plain", "eagle"), required=True)
    cap.add_argument("--head", type=Path)
    cap.add_argument("--prompts", type=Path, required=True)
    cap.add_argument("--limit", type=int, default=128)
    cap.add_argument("--max-tokens", type=int, default=512)
    cap.add_argument("--num-speculative-tokens", type=int, default=3)
    cap.add_argument("--cap-sweep", type=int, default=16)
    cap.add_argument("--max-model-len", type=int, default=8192)
    # Offline C=1 check: weights plus a small KV cache. Serving keeps 0.8.
    cap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--ignore-eos", action="store_true")
    smp = sub.add_parser("sample")
    smp.add_argument("--contract", type=Path)
    smp.add_argument("--mode", choices=("plain", "eagle"), required=True)
    smp.add_argument("--head", type=Path)
    smp.add_argument("--prompts", type=Path, required=True)
    smp.add_argument("--limit", type=int, default=32)
    smp.add_argument("--n", type=int, default=1024)
    smp.add_argument("--max-num-seqs", type=int, default=256)
    smp.add_argument("--horizon", type=int, default=4)
    smp.add_argument("--temperature", type=float, default=0.8)
    smp.add_argument("--seed", type=int, default=1)
    smp.add_argument("--num-speculative-tokens", type=int, default=3)
    smp.add_argument("--max-model-len", type=int, default=8192)
    smp.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    smp.add_argument("--output", type=Path, required=True)
    dst = sub.add_parser("dist")
    dst.add_argument("reference", type=Path)
    dst.add_argument("control", type=Path)
    dst.add_argument("candidate", type=Path)
    dst.add_argument("--output", type=Path, required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("baseline", type=Path)
    cmp_.add_argument("candidate", type=Path)
    cmp_.add_argument("--output", type=Path, required=True)
    mar = sub.add_parser("margins")
    mar.add_argument("comparison", type=Path)
    mar.add_argument("--contract", type=Path)
    mar.add_argument("--prompts", type=Path, required=True)
    mar.add_argument("--tie-margin", type=float, default=0.05)
    mar.add_argument("--max-model-len", type=int, default=8192)
    mar.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    mar.add_argument("--num-speculative-tokens", type=int, default=3)
    mar.add_argument("--head", type=Path)
    mar.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        if args.mode == "eagle" and args.head is None:
            raise SystemExit("--mode eagle needs --head")
        capture(args)
    elif args.command == "sample":
        if args.mode == "eagle" and args.head is None:
            raise SystemExit("--mode eagle needs --head")
        sample(args)
    elif args.command == "dist":
        dist(args)
    elif args.command == "compare":
        compare(args)
    else:
        margins(args)


if __name__ == "__main__":
    main()
