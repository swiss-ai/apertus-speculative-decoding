from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from apertus_bench.analysis import add_inferred_round_costs, add_speedups, collect_rows, write_csv
from apertus_bench.client import StreamingChatClient
from apertus_bench.correctness import (
    DEFAULT_TOLERANCE,
    GateTolerance,
    capture_greedy_outputs,
    compare_capture_files,
    compare_greedy_outputs,
    run_correctness_gate,
)
from apertus_bench.diagnostics import break_even_report, format_env_exports, write_corpus
from apertus_bench.eagle import EagleHeadError, validate_eagle_head
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
    parser.add_argument(
        "--method",
        choices=("none", "draft_model", "ngram", "eagle3", "dspark"),
        required=True,
        help=(
            "engine speculative method; E3/E3.1/P-EAGLE all use eagle3 "
            "and are distinguished by --algorithm; dspark is the colleague's "
            "DSpark drafter"
        ),
    )
    parser.add_argument(
        "--algorithm",
        help=(
            "logical algorithm label (eagle3, eagle31, peagle). "
            "Required metadata when --method eagle3"
        ),
    )
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--draft-tensor-parallel-size", type=int)
    parser.add_argument("--prompt-lookup-max", type=int)
    parser.add_argument(
        "--parallel-drafting",
        action="store_true",
        help="record that the live server was launched with parallel_drafting; does not enable it",
    )


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
        "--metrics-interval",
        type=float,
        help="also sample KV-cache usage, queue and preemptions every N seconds during a cell",
    )
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

    loadtest = subparsers.add_parser(
        "loadtest",
        help="step through rising concurrency on one workload, sampling server load",
    )
    _add_benchmark_common(loadtest)
    loadtest.add_argument("--workload", required=True)
    loadtest.add_argument("--concurrencies", type=int, nargs="+", required=True)
    loadtest.add_argument(
        "--requests-per-slot",
        type=int,
        default=4,
        help=(
            "measured requests per level = max(--requests, this x concurrency), "
            "rounded up to whole passes over the workload's prompts"
        ),
    )
    loadtest.add_argument("--output", type=Path, required=True)

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

    gate = subparsers.add_parser(
        "correctness-gate",
        help="judge a greedy comparison against same-configuration calibration controls",
    )
    gate.add_argument(
        "--treatment",
        type=Path,
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        required=True,
        help="the two captures whose losslessness is in question",
    )
    gate.add_argument(
        "--control",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("BASELINE", "CANDIDATE"),
        required=True,
        help="two captures of one configuration, measuring nondeterminism; may be repeated",
    )
    gate.add_argument(
        "--common-prefix-fraction-margin",
        type=float,
        default=DEFAULT_TOLERANCE.common_prefix_fraction_margin,
    )
    gate.add_argument(
        "--normalized-edit-distance-margin",
        type=float,
        default=DEFAULT_TOLERANCE.normalized_edit_distance_margin,
    )
    gate.add_argument(
        "--completion-token-difference-margin",
        type=int,
        default=DEFAULT_TOLERANCE.completion_token_difference_margin,
    )
    gate.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="flatten result cells and add speedups")
    analyze.add_argument("results", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("analysis/results.csv"))
    analyze.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="do not fail when a speculative cell has no compatible operational baseline",
    )

    even = subparsers.add_parser(
        "break-even", help="compare inferred round cost against g * t0 for one speculative variant"
    )
    even.add_argument("results", type=Path)
    even.add_argument("--t0-variant", required=True)
    even.add_argument("--spec-variant", required=True)
    even.add_argument("--workload")
    even.add_argument("--concurrency", type=int)
    even.add_argument("--output", type=Path)

    envp = subparsers.add_parser(
        "diagnostics-env", help="print bash exports for a diagnostic configuration id"
    )
    envp.add_argument("config_id")

    corpus = subparsers.add_parser(
        "write-diagnostics-corpus", help="write the 32-prompt fixed-output diagnostic JSONL"
    )
    corpus.add_argument(
        "--output", type=Path, default=Path("workloads/diagnostics-fixed-256.jsonl")
    )

    for action in ("start-profile", "stop-profile"):
        profile = subparsers.add_parser(
            action, help=f"POST /{action.replace('-', '_')} on a vLLM server"
        )
        profile.add_argument("--base-url", required=True)
        profile.add_argument("--timeout-seconds", type=float, default=600.0)
        _add_auth(profile)

    eagle = subparsers.add_parser(
        "validate-eagle-head",
        help="validate an EAGLE-3/3.1 checkpoint against a selected Apertus 1.5 target contract",
    )
    eagle.add_argument("head", type=Path)
    eagle.add_argument("--algorithm", choices=("eagle3", "eagle31"))
    eagle.add_argument("--allow-config-only", action="store_true")
    eagle.add_argument("--contract", type=Path)
    return parser


EAGLE_REQUIRED_METADATA = (
    "deployment_id",
    "checkpoint_sha256",
    "target_model",
    "target_revision",
    "target_tensor_parallel_size",
)


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


def _require_eagle_provenance(method: str, extra: dict[str, str]) -> None:
    if method != "eagle3":
        return
    missing = [key for key in EAGLE_REQUIRED_METADATA if not extra.get(key)]
    if missing:
        raise ValueError(
            "method eagle3 requires --metadata " + ", ".join(f"{key}=..." for key in missing)
        )


def _variant(args: argparse.Namespace) -> Variant:
    method = args.method
    algorithm = args.algorithm
    tokens = args.num_speculative_tokens
    if method == "eagle3":
        if tokens is None or tokens < 1:
            raise ValueError(
                "--num-speculative-tokens is required and must be >= 1 for method eagle3"
            )
        if not algorithm:
            raise ValueError(
                "--algorithm is required for method eagle3 "
                "(engine method is eagle3 for E3, E3.1, and P-EAGLE)"
            )
        if algorithm not in {"eagle3", "eagle31", "peagle"}:
            raise ValueError(f"unsupported algorithm {algorithm!r} for method eagle3")
        if algorithm == "peagle" and not args.parallel_drafting:
            raise ValueError(
                "algorithm=peagle requires --parallel-drafting in the recorded variant"
            )
        if algorithm != "peagle" and args.parallel_drafting:
            raise ValueError("--parallel-drafting is only valid with --algorithm peagle")
    elif args.parallel_drafting:
        raise ValueError("--parallel-drafting is only valid with --method eagle3")
    if method == "dspark" and (tokens is None or tokens < 1):
        raise ValueError("--num-speculative-tokens is required and must be >= 1 for method dspark")
    return Variant(
        name=args.variant,
        method=method,
        num_speculative_tokens=tokens,
        draft_tensor_parallel_size=args.draft_tensor_parallel_size,
        prompt_lookup_max=args.prompt_lookup_max,
        algorithm=algorithm,
        parallel_drafting=bool(args.parallel_drafting),
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
    extra = _metadata(args.metadata)
    variant = _variant(args)
    _require_eagle_provenance(variant.method, extra)
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
            variant,
            _metrics_url(args),
            args.model,
            extra,
            metrics_interval=args.metrics_interval,
        )
    print(json.dumps(summary, indent=2))


async def _run_matrix(args: argparse.Namespace) -> None:
    extra = _metadata(args.metadata)
    variant = _variant(args)
    _require_eagle_provenance(variant.method, extra)
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
                        variant,
                        _metrics_url(args),
                        args.model,
                        extra,
                        metrics_interval=args.metrics_interval,
                    )


async def _run_loadtest(args: argparse.Namespace) -> None:
    """One cell per concurrency level, lowest first, then a per-level summary."""
    extra = _metadata(args.metadata)
    variant = _variant(args)
    _require_eagle_provenance(variant.method, extra)
    prompts = load_prompts(args.workloads, args.workload)
    interval = args.metrics_interval or 1.0
    levels = []
    async with StreamingChatClient(
        args.base_url, args.model, _api_key(args), timeout_seconds=args.timeout_seconds
    ) as client:
        for concurrency in sorted(args.concurrencies):
            requests = loadtest_requests(
                args.requests, args.requests_per_slot, concurrency, len(prompts)
            )
            output = args.output / args.workload / f"c{concurrency}"
            print(f"loadtest {args.workload} concurrency={concurrency} requests={requests}")
            summary = await run_cell(
                client,
                prompts,
                args.workloads,
                output,
                CellSettings(
                    workload=args.workload,
                    concurrency=concurrency,
                    requests=requests,
                    warmup_requests=args.warmup_requests or concurrency,
                    repeat=1,
                    generation=_generation(args),
                ),
                variant,
                _metrics_url(args),
                args.model,
                extra,
                metrics_interval=interval,
                tolerate_warmup_failures=True,
            )
            levels.append(loadtest_level(concurrency, summary))
    report = {"workload": args.workload, "variant": variant.name, "levels": levels}
    (args.output / "loadtest-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def loadtest_requests(minimum: int, per_slot: int, concurrency: int, prompts: int) -> int:
    """Requests for one level: max(minimum, per_slot x concurrency), rounded up to
    whole passes over the prompts. Every level then sees the same prompt mix, so
    acceptance and throughput are comparable across levels; with a partial pass
    the low levels only saw the first prompts of the file."""
    requests = max(minimum, per_slot * concurrency)
    return -(-requests // prompts) * prompts


def loadtest_level(concurrency: int, summary: dict[str, Any]) -> dict[str, Any]:
    """The per-level row of a load test: throughput, latency tails, KV pressure,
    and acceptance when the server speculates."""
    latency = summary["latency_ms"]
    load = summary.get("load") or {}
    spec = summary.get("speculative_decoding") or {}

    def pick(block: dict[str, Any] | None, key: str) -> Any:
        return (block or {}).get(key)

    return {
        "concurrency": concurrency,
        "requests": summary["requests"]["attempted"],
        "success_rate": summary["requests"]["success_rate"],
        "warmup_failed": (summary.get("warmup") or {}).get("failed"),
        "output_tokens_per_second": summary["tokens"]["output_tokens_per_second"],
        "ttft_p50_ms": pick(latency.get("ttft"), "p50"),
        "ttft_p95_ms": pick(latency.get("ttft"), "p95"),
        "tpot_p50_ms": pick(latency.get("tpot"), "p50"),
        "tpot_p95_ms": pick(latency.get("tpot"), "p95"),
        "e2e_p95_ms": pick(latency.get("e2e"), "p95"),
        "kv_cache_usage_mean": pick(load.get("kv_cache_usage"), "mean"),
        "kv_cache_usage_max": pick(load.get("kv_cache_usage"), "max"),
        "kv_cache_tokens_peak": load.get("kv_cache_tokens_peak"),
        "kv_cache_size_tokens": load.get("kv_cache_size_tokens"),
        "running_max": pick(load.get("running"), "max"),
        "waiting_max": pick(load.get("waiting"), "max"),
        "preemptions": load.get("preemptions"),
        "mean_acceptance_length": spec.get("mean_acceptance_length"),
        "acceptance_rate": spec.get("acceptance_rate"),
    }


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
        elif args.command == "loadtest":
            asyncio.run(_run_loadtest(args))
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
        elif args.command == "correctness-gate":
            gate_report = run_correctness_gate(
                (args.treatment[0], args.treatment[1]),
                [(baseline, candidate) for baseline, candidate in args.control],
                args.output,
                GateTolerance(
                    normalized_edit_distance_margin=args.normalized_edit_distance_margin,
                    common_prefix_fraction_margin=args.common_prefix_fraction_margin,
                    completion_token_difference_margin=args.completion_token_difference_margin,
                ),
            )
            print(json.dumps(gate_report, indent=2))
            if not gate_report["passed"]:
                raise SystemExit(1)
        elif args.command == "analyze":
            rows = collect_rows(args.results)
            add_speedups(rows, require_baseline=not args.allow_missing_baseline)
            add_inferred_round_costs(rows)
            write_csv(rows, args.output)
            print(f"wrote {len(rows)} rows to {args.output}")
        elif args.command == "break-even":
            rows = collect_rows(args.results)
            add_speedups(rows, require_baseline=False)
            add_inferred_round_costs(rows)
            report = break_even_report(
                rows,
                t0_variant=args.t0_variant,
                spec_variant=args.spec_variant,
                workload=args.workload,
                concurrency=args.concurrency,
            )
            text = json.dumps(report, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n")
            print(text)
        elif args.command == "diagnostics-env":
            print(format_env_exports(args.config_id), end="")
        elif args.command == "write-diagnostics-corpus":
            print(json.dumps(write_corpus(args.output), indent=2))
        elif args.command in {"start-profile", "stop-profile"}:
            action = args.command.replace("-", "_")
            headers = {}
            api_key = _api_key(args)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            url = f"{args.base_url.rstrip('/')}/{action}"
            timeout = args.timeout_seconds
            with httpx.Client(headers=headers, timeout=timeout) as client:
                response = client.post(url)
                response.raise_for_status()
                print(response.text or json.dumps({"ok": True, "url": url}))
        elif args.command == "validate-eagle-head":
            report = validate_eagle_head(
                args.head,
                expected_algorithm=args.algorithm,
                require_weights=not args.allow_config_only,
                target_contract_path=args.contract,
            )
            print(json.dumps(report, indent=2))
    except (OSError, RuntimeError, ValueError, EagleHeadError, httpx.HTTPError) as error:
        raise SystemExit(f"error: {error}") from error
