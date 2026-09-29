"""Server-side load and memory signals sampled while a cell runs.

vLLM reserves its GPU memory at start-up (weights, activation workspace, CUDA
graphs and a KV-cache pool sized by ``gpu_memory_utilization``), so device
memory is nearly flat under load. What load changes is how full the KV pool is
and what happens when it is full: requests wait or are preempted. This module
samples those gauges from ``/metrics`` during a cell:

- ``kv_cache_usage``: fraction of the KV pool in use
- ``running`` / ``waiting``: scheduled and queued requests
- ``preemptions``: cumulative preempted requests (recomputed later)
- ``prompt_tokens`` / ``generation_tokens``: cumulative token counters

and summarizes them next to the static pool size from ``cache_config_info``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from prometheus_client.parser import text_string_to_metric_families

from apertus_bench.prometheus import _matches_model

# Names verified against the pinned image (confirm-run metrics_after.prom).
GAUGES = {
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
}
CACHE_INFO = "vllm:cache_config_info"


def parse_load_sample(text: str, model: str | None = None) -> dict[str, Any]:
    """Load gauges from one ``/metrics`` scrape (missing gauges are absent, not 0)."""
    names = {name: key for key, name in GAUGES.items()}
    sample: dict[str, Any] = {}
    for family in text_string_to_metric_families(text):
        for metric in family.samples:
            if not _matches_model(metric.labels, model):
                continue
            key = names.get(metric.name)
            if key is not None:
                sample[key] = sample.get(key, 0.0) + float(metric.value)
            elif metric.name == CACHE_INFO:
                sample["kv_cache_size_tokens"] = int(metric.labels.get("kv_cache_size_tokens", 0))
                sample["num_gpu_blocks"] = int(metric.labels.get("num_gpu_blocks", 0))
                sample["block_size"] = int(metric.labels.get("block_size", 0))
                sample["gpu_memory_utilization"] = float(
                    metric.labels.get("gpu_memory_utilization", 0)
                )
    return sample


class MetricsSampler:
    """Poll ``/metrics`` every ``interval`` seconds until stopped."""

    def __init__(self, fetch, interval: float, model: str | None = None) -> None:
        self.fetch = fetch  # async () -> str
        self.interval = interval
        self.model = model
        self.samples: list[dict[str, Any]] = []
        self.errors = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            tick = time.monotonic()
            try:
                sample = parse_load_sample(await self.fetch(), self.model)
                sample["t"] = round(tick - started, 3)
                self.samples.append(sample)
            except Exception:  # noqa: BLE001 - a missed scrape must not fail the cell
                self.errors += 1
            delay = self.interval - (time.monotonic() - tick)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=max(delay, 0.0))

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def summarize_load(samples: list[dict[str, Any]], errors: int = 0) -> dict[str, Any]:
    """Per-cell summary of the sampled load signals."""
    if not samples:
        return {"samples": 0, "scrape_errors": errors}

    def series(key: str) -> list[float]:
        return [s[key] for s in samples if key in s]

    def delta(key: str) -> float | None:
        values = series(key)
        return values[-1] - values[0] if values else None

    kv = series("kv_cache_usage")
    size = next((s["kv_cache_size_tokens"] for s in samples if "kv_cache_size_tokens" in s), None)
    kv_stats = _stats(kv)
    return {
        "samples": len(samples),
        "scrape_errors": errors,
        "duration_s": samples[-1]["t"] - samples[0]["t"],
        "kv_cache_size_tokens": size,
        "kv_cache_usage": kv_stats,
        "kv_cache_tokens_peak": round(kv_stats["max"] * size) if kv_stats and size else None,
        "seconds_kv_above_90pct": sum(1 for v in kv if v >= 0.9) * _interval(samples),
        "running": _stats(series("running")),
        "waiting": _stats(series("waiting")),
        "preemptions": delta("preemptions"),
        "prompt_tokens": delta("prompt_tokens"),
        "generation_tokens": delta("generation_tokens"),
    }


def _interval(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        return 0.0
    return (samples[-1]["t"] - samples[0]["t"]) / (len(samples) - 1)
