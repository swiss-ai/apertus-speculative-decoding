from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apertus_bench import __version__
from apertus_bench.client import RequestMeasurement, StreamingChatClient
from apertus_bench.prometheus import (
    SpeculativeSnapshot,
    parse_speculative_snapshot,
    speculative_delta,
)
from apertus_bench.stats import distribution
from apertus_bench.workloads import Prompt, file_sha256


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 1
    max_tokens: int | None = None
    ignore_eos: bool = False


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    num_speculative_tokens: int | None = None
    draft_tensor_parallel_size: int | None = None
    prompt_lookup_max: int | None = None


@dataclass(frozen=True)
class CellSettings:
    workload: str
    concurrency: int
    requests: int
    warmup_requests: int
    repeat: int
    generation: GenerationSettings = field(default_factory=GenerationSettings)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def _fetch_metrics(client: StreamingChatClient, metrics_url: str | None) -> str:
    if not metrics_url:
        return ""
    response = await client.http.get(metrics_url)
    response.raise_for_status()
    return response.text


async def _execute_requests(
    client: StreamingChatClient,
    prompts: list[Prompt],
    count: int,
    concurrency: int,
    generation: GenerationSettings,
    *,
    store_output: bool,
) -> list[RequestMeasurement]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for request_index in range(count):
        queue.put_nowait(request_index)
    results: list[RequestMeasurement] = []
    result_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                request_index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            prompt = prompts[request_index % len(prompts)]
            measurement = await client.request(
                prompt,
                request_index,
                temperature=generation.temperature,
                top_p=generation.top_p,
                seed=generation.seed + request_index,
                max_tokens=generation.max_tokens,
                ignore_eos=generation.ignore_eos,
                store_output=store_output,
            )
            async with result_lock:
                results.append(measurement)
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(min(concurrency, count))))
    return sorted(results, key=lambda result: result.request_index)


def summarize(
    measurements: list[RequestMeasurement],
    wall_seconds: float,
    spec_metrics: dict[str, object],
) -> dict[str, object]:
    successes = [measurement for measurement in measurements if measurement.success]
    completion_tokens = sum(measurement.completion_tokens or 0 for measurement in successes)
    prompt_tokens = sum(measurement.prompt_tokens or 0 for measurement in successes)
    event_gaps = [gap for measurement in successes for gap in measurement.stream_event_gaps_ms]
    return {
        "requests": {
            "attempted": len(measurements),
            "successful": len(successes),
            "failed": len(measurements) - len(successes),
            "success_rate": len(successes) / len(measurements) if measurements else None,
            "requests_per_second": len(successes) / wall_seconds if wall_seconds else None,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "output_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else None,
            "total_tokens_per_second": (prompt_tokens + completion_tokens) / wall_seconds
            if wall_seconds
            else None,
        },
        "latency_ms": {
            "e2e": distribution(measurement.e2e_ms for measurement in successes),
            "ttft": distribution(
                measurement.ttft_ms for measurement in successes if measurement.ttft_ms is not None
            ),
            "tpot": distribution(
                measurement.tpot_ms for measurement in successes if measurement.tpot_ms is not None
            ),
            "stream_event_gap": distribution(event_gaps),
        },
        "wall_seconds": wall_seconds,
        "speculative_decoding": spec_metrics,
    }


async def run_cell(
    client: StreamingChatClient,
    prompts: list[Prompt],
    workload_path: Path,
    output_dir: Path,
    settings: CellSettings,
    variant: Variant,
    metrics_url: str | None,
    model: str,
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    if settings.concurrency <= 0 or settings.requests <= 0:
        raise ValueError("concurrency and requests must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite result cell: {output_dir}")

    if settings.warmup_requests:
        warmup = await _execute_requests(
            client,
            prompts,
            settings.warmup_requests,
            settings.concurrency,
            settings.generation,
            store_output=False,
        )
        warmup_failures = [measurement for measurement in warmup if not measurement.success]
        if warmup_failures:
            first_error = warmup_failures[0].error or "unknown warmup failure"
            raise RuntimeError(
                f"{len(warmup_failures)}/{len(warmup)} warmup requests failed: {first_error}"
            )

    raw_before = await _fetch_metrics(client, metrics_url)
    before = (
        parse_speculative_snapshot(raw_before, model=model) if raw_before else SpeculativeSnapshot()
    )
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    measurements = await _execute_requests(
        client,
        prompts,
        settings.requests,
        settings.concurrency,
        settings.generation,
        store_output=False,
    )
    wall_seconds = time.perf_counter() - wall_started
    ended_at = datetime.now(UTC)
    raw_after = await _fetch_metrics(client, metrics_url)
    after = (
        parse_speculative_snapshot(raw_after, model=model) if raw_after else SpeculativeSnapshot()
    )
    spec_metrics = speculative_delta(before, after)
    summary = summarize(measurements, wall_seconds, spec_metrics)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "harness_version": __version__,
        "harness_git_revision": _git_revision(),
        "python": platform.python_version(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "base_url": client.base_url,
        "metrics_url": metrics_url,
        "model": model,
        "variant": asdict(variant),
        "cell": asdict(settings),
        "workload_file": str(workload_path),
        "workload_file_sha256": file_sha256(workload_path),
        "prompt_ids": [prompt.id for prompt in prompts],
        "extra": extra_metadata or {},
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(json.dumps(measurement.to_dict(), sort_keys=True) + "\n")
    if raw_before:
        (output_dir / "metrics_before.prom").write_text(raw_before)
        (output_dir / "metrics_after.prom").write_text(raw_after)
    return summary
