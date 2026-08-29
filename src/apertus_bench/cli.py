from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from apertus_bench.analysis import add_speedups, collect_rows, write_csv
from apertus_bench.client import StreamingChatClient
from apertus_bench.correctness import (
    capture_greedy_outputs,
    compare_capture_files,
    compare_greedy_outputs,
)
from apertus_bench.runner import CellSettings, GenerationSettings, Variant, run_cell
from apertus_bench.workloads import available_workloads, load_prompts


def _add_auth(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key-env",
        default="CSCS_SERVING_API",
        help="environment variable containing the bearer token (default: CSCS_SERVING_API)",
    )


def _add_generation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)


def _add_variant(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", required=True, help="stable label for the server variant")
    parser.add_argument("--method", choices=("none", "draft_model", "ngram"), required=True)
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--draft-tensor-parallel-size", type=int)
    parser.add_argument("--prompt-lookup-max", type=int)


def _add_benchmark_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name sent in requests")
    parser.add_argument("--workloads", type=Path, required=True, help="JSONL prompt corpus")
    parser.add_argument("--requests", type=int, default=128, help="measured requests per cell")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="warmup count; 0 uses one request per concurrent worker",
    )
    parser.add_argument(
        "--metrics-url", help="vLLM Prometheus endpoint; defaults to BASE_URL/metrics"
    )
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra non-secret provenance field; may be repeated",
    )
    _add_auth(parser)
    _add_generation(parser)
    _add_variant(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apertus-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-workloads", help="validate a workload JSONL file")
    validate.add_argument("path", type=Path)

    run = subparsers.add_parser("run", help="run one workload/concurrency/repeat cell")
    _add_benchmark_common(run)
    run.add_argument("--workload", required=True)
    run.add_argument("--concurrency", type=int, required=True)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--output", type=Path, required=True)

    matrix = subparsers.add_parser("matrix", help="run a matrix against one server variant")
    _add_benchmark_common(matrix)
    matrix.add_argument("--workload", action="append", dest="workload_names")
    matrix.add_argument("--concurrencies", type=int, nargs="+", required=True)
    matrix.add_argument("--repeats", type=int, default=3)
    matrix.add_argument("--output", type=Path, required=True)

    correctness = subparsers.add_parser(
        "correctness", help="compare baseline and speculative greedy outputs"
    )
    correctness.add_argument("--baseline-url", required=True)
    correctness.add_argument("--baseline-model", required=True)
    correctness.add_argument("--candidate-url", required=True)
    correctness.add_argument("--candidate-model", required=True)
    correctness.add_argument("--workloads", type=Path, required=True)
    correctness.add_argument("--workload")
    correctness.add_argument("--max-tokens", type=int)
    correctness.add_argument("--seed", type=int, default=1)
    correctness.add_argument("--timeout-seconds", type=float, default=600.0)
    correctness.add_argument("--output", type=Path, required=True)
    _add_auth(correctness)

    capture = subparsers.add_parser(
        "capture", help="capture greedy outputs from one deployment for sequential comparison"
    )
    capture.add_argument("--base-url", required=True)
    capture.add_argument("--model", required=True)
    capture.add_argument("--workloads", type=Path, required=True)
    capture.add_argument("--workload")
    capture.add_argument("--max-tokens", type=int)
    capture.add_argument("--seed", type=int, default=1)
    capture.add_argument("--timeout-seconds", type=float, default=600.0)
    capture.add_argument("--output", type=Path, required=True)
    _add_auth(capture)

    compare = subparsers.add_parser(
        "compare-captures", help="compare two sequential greedy-output captures"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="flatten result cells and add speedups")
    analyze.add_argument("results", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("analysis/results.csv"))
    return parser


def _metadata(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must have KEY=VALUE form: {value!r}")
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError(f"metadata must have non-empty KEY and VALUE: {value!r}")
        parsed[key] = item
    return parsed


def _variant(args: argparse.Namespace) -> Variant:
    return Variant(
        name=args.variant,
        method=args.method,
        num_speculative_tokens=args.num_speculative_tokens,
        draft_tensor_parallel_size=args.draft_tensor_parallel_size,
        prompt_lookup_max=args.prompt_lookup_max,
    )


def _generation(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )


def _metrics_url(args: argparse.Namespace) -> str | None:
    if args.no_metrics:
        return None
    return args.metrics_url or f"{args.base_url.rstrip('/')}/metrics"


def _api_key(args: argparse.Namespace) -> str | None:
    return os.getenv(args.api_key_env)


async def _run_one(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.workloads, args.workload)
    settings = CellSettings(
        workload=args.workload,
        concurrency=args.concurrency,
        requests=args.requests,
        warmup_requests=args.warmup_requests or args.concurrency,
        repeat=args.repeat,
        generation=_generation(args),
    )
    async with StreamingChatClient(
        args.base_url, args.model, _api_key(args), timeout_seconds=args.timeout_seconds
    ) as client:
        summary = await run_cell(
            client,
            prompts,
            args.workloads,
            args.output,
            settings,
            _variant(args),
            _metrics_url(args),
            args.model,
            _metadata(args.metadata),
        )
    print(json.dumps(summary, indent=2))


async def _run_matrix(args: argparse.Namespace) -> None:
    all_prompts = load_prompts(args.workloads)
    workload_names = args.workload_names or available_workloads(all_prompts)
    async with StreamingChatClient(
        args.base_url, args.model, _api_key(args), timeout_seconds=args.timeout_seconds
    ) as client:
        for workload in workload_names:
            prompts = [prompt for prompt in all_prompts if prompt.workload == workload]
            if not prompts:
                raise ValueError(f"workload {workload!r} does not exist in {args.workloads}")
            for concurrency in args.concurrencies:
                for repeat in range(1, args.repeats + 1):
                    output = args.output / workload / f"c{concurrency}" / f"repeat-{repeat:02d}"
                    print(f"running {workload=} {concurrency=} {repeat=} -> {output}")
                    await run_cell(
                        client,
                        prompts,
                        args.workloads,
                        output,
                        CellSettings(
                            workload=workload,
                            concurrency=concurrency,
                            requests=args.requests,
                            warmup_requests=args.warmup_requests or concurrency,
                            repeat=repeat,
                            generation=_generation(args),
                        ),
                        _variant(args),
                        _metrics_url(args),
                        args.model,
                        _metadata(args.metadata),
                    )


async def _run_correctness(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.workloads, args.workload)
    api_key = _api_key(args)
    async with (
        StreamingChatClient(
            args.baseline_url,
            args.baseline_model,
            api_key,
            timeout_seconds=args.timeout_seconds,
        ) as baseline,
        StreamingChatClient(
            args.candidate_url,
            args.candidate_model,
            api_key,
            timeout_seconds=args.timeout_seconds,
        ) as candidate,
    ):
        report = await compare_greedy_outputs(
            baseline,
            candidate,
            prompts,
            args.output,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


async def _run_capture(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.workloads, args.workload)
    async with StreamingChatClient(
        args.base_url,
        args.model,
        _api_key(args),
        timeout_seconds=args.timeout_seconds,
    ) as client:
        report = await capture_greedy_outputs(
            client,
            prompts,
            args.output,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    print(json.dumps({"model": report["model"], "cases": len(prompts)}, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "validate-workloads":
            prompts = load_prompts(args.path)
            print(
                json.dumps(
                    {"prompts": len(prompts), "workloads": available_workloads(prompts)}, indent=2
                )
            )
        elif args.command == "run":
            asyncio.run(_run_one(args))
        elif args.command == "matrix":
            asyncio.run(_run_matrix(args))
        elif args.command == "correctness":
            asyncio.run(_run_correctness(args))
        elif args.command == "capture":
            asyncio.run(_run_capture(args))
        elif args.command == "compare-captures":
            report = compare_capture_files(args.baseline, args.candidate, args.output)
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "cases"}, indent=2
                )
            )
        elif args.command == "analyze":
            rows = collect_rows(args.results)
            add_speedups(rows)
            write_csv(rows, args.output)
            print(f"wrote {len(rows)} rows to {args.output}")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
