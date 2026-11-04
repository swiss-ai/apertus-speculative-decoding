"""Nsight Systems profiles of a sweep arm (docs/waste-profiling.md).

A sweep arm with a ``profile`` block is served under ``nsys profile`` with a
CUDA-profiler capture range: nothing is collected until the load test POSTs
``/start_profile`` (vLLM's ``--profiler-config.profiler cuda`` turns that into
``cudaProfilerStart`` in the worker), and each start/stop pair becomes its own
report (``--capture-range-end=repeat``). The client opens one window of
``window_s`` seconds, ``delay_s`` into the measured phase of each chosen
concurrency level, and records the server counters at both ends of it
(``profile-window.json`` in the cell). After the load tests the arm waits for
nsys to write the reports, stops the server and exports ``nsys stats`` CSVs.
The ``.nsys-rep`` and ``.sqlite`` files stay where they were written; only the
CSVs and the manifest belong in a committed record.

    profile:
      tool: nsys
      window_s: 15        # capture length inside each level
      delay_s: 10         # from the start of the level's measured requests
      levels: [64, 128]   # concurrency levels to capture (default: all)
      gpu_metrics: false  # --gpu-metrics-devices=cuda-visible (needs counters)
      eager_draft: false  # run the draft step without CUDA graphs (ratios only)

The profiled numbers are never headline numbers: run an unprofiled twin for
throughput (docs/waste-profiling.md, "Overhead").
"""

from __future__ import annotations

import asyncio
import csv
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Nsight Systems 2025.6.3 (CLI of the sbsa .deb), extracted on scratch; see
# docs/waste-profiling.md for the download URL and digest. /iopsstor is mounted
# in the serving container by methods/eagle/configs/train-env.toml.
DEFAULT_NSYS = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nsight-systems-2025.6.3/"
    "opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys"
)
DEFAULT_STATS = [
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_api_sum",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "nvtx_gpu_proj_trace",
    "osrt_sum",
]
PROFILE_KEYS = {
    "tool", "window_s", "delay_s", "levels", "loadtests", "nsys", "trace",
    "cuda_graph_trace", "gpu_metrics", "gpu_metrics_frequency", "nvtx",
    "eager_draft", "sample", "extra_args", "stats", "report_timeout_s",
}  # fmt: skip
WINDOW_FILE = "profile-window.json"


@dataclass
class ProfileSpec:
    tool: str = "nsys"
    window_s: float = 15.0
    delay_s: float = 10.0
    levels: list[int] | None = None
    loadtests: list[str] | None = None
    nsys: str = DEFAULT_NSYS
    trace: str = "cuda,nvtx,osrt"
    cuda_graph_trace: str = "node"
    gpu_metrics: bool = False
    gpu_metrics_frequency: int = 10000
    nvtx: bool = True
    eager_draft: bool = False
    sample: str = "none"
    extra_args: list[str] = field(default_factory=list)
    stats: list[str] = field(default_factory=lambda: list(DEFAULT_STATS))
    report_timeout_s: float = 900.0

    def captures(self, label: str, concurrency: int) -> bool:
        if self.loadtests is not None and label not in self.loadtests:
            return False
        return self.levels is None or concurrency in self.levels


def parse_profile(raw: Any, arm: str) -> ProfileSpec | None:
    """The ``profile`` block of an arm, validated; None when the arm has none."""
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"arm {arm}: profile must be a mapping")
    unknown = set(raw) - PROFILE_KEYS
    if unknown:
        raise ValueError(f"arm {arm}: unknown profile settings {sorted(unknown)}")
    spec = ProfileSpec(**raw)
    spec.window_s, spec.delay_s = float(spec.window_s), float(spec.delay_s)
    if spec.tool != "nsys":
        raise ValueError(f"arm {arm}: profile tool must be nsys, got {spec.tool!r}")
    if spec.window_s <= 0 or spec.delay_s < 0:
        raise ValueError(f"arm {arm}: profile needs window_s > 0 and delay_s >= 0")
    if spec.levels is not None:
        spec.levels = [int(level) for level in spec.levels]
    if spec.cuda_graph_trace not in {"node", "graph"}:
        raise ValueError(f"arm {arm}: cuda_graph_trace must be node or graph")
    spec.extra_args = [str(arg) for arg in spec.extra_args]
    return spec


def nsys_argv(spec: ProfileSpec, serve_argv: list[str], output: Path) -> list[str]:
    """``serve_argv`` under ``nsys profile``, collecting only between the CUDA
    profiler start/stop calls, one report per window (``output.N.nsys-rep``)."""
    argv = [
        spec.nsys, "profile",
        "--output", str(output),
        "--force-overwrite", "true",
        "--trace", spec.trace,
        "--cuda-graph-trace", spec.cuda_graph_trace,
        "--capture-range", "cudaProfilerApi",
        "--capture-range-end", "repeat",
        "--sample", spec.sample,
        "--cpuctxsw", "none",
    ]  # fmt: skip
    if spec.gpu_metrics:
        argv += [
            "--gpu-metrics-devices", "cuda-visible",
            "--gpu-metrics-frequency", str(spec.gpu_metrics_frequency),
        ]  # fmt: skip
    return [*argv, *spec.extra_args, *serve_argv]


def server_args(spec: ProfileSpec) -> list[str]:
    """vLLM flags that route /start_profile to cudaProfilerStart in the worker."""
    del spec
    return ["--profiler-config.profiler", "cuda"]


def server_env(spec: ProfileSpec) -> dict[str, str]:
    env = {}
    if spec.nvtx:
        env["VLLM_APERTUS_NVTX"] = "1"
    if spec.eager_draft:
        env["VLLM_APERTUS_NVTX_EAGER_DRAFT"] = "1"
    return env


def bench_args(spec: ProfileSpec, label: str, concurrencies: list[int]) -> list[str]:
    """Load-test client flags for the levels of one load test that are captured."""
    levels = [c for c in concurrencies if spec.captures(label, c)]
    if not levels:
        return []
    return [
        "--profile-levels", *[str(level) for level in levels],
        "--profile-window", str(spec.window_s),
        "--profile-delay", str(spec.delay_s),
    ]  # fmt: skip


def expected_windows(spec: ProfileSpec, loadtests: list[tuple[str, list[int]]]) -> int:
    return sum(spec.captures(label, c) for label, levels in loadtests for c in levels)


# --- client side -----------------------------------------------------------


@dataclass(frozen=True)
class WindowSettings:
    delay_s: float
    window_s: float


def _counters(text: str, model: str | None) -> dict[str, Any]:
    # Imported here: the login node plans sweeps without prometheus_client.
    from apertus_bench.load import parse_load_sample
    from apertus_bench.prometheus import parse_speculative_snapshot

    load = parse_load_sample(text, model=model)
    spec = parse_speculative_snapshot(text, model=model)
    return {
        "prompt_tokens": load.get("prompt_tokens"),
        "generation_tokens": load.get("generation_tokens"),
        "spec": spec,
    }


def window_counters(before: str, after: str, model: str | None) -> dict[str, Any]:
    """Server counter deltas over a capture window (tokens, drafts, acceptance)."""
    from apertus_bench.prometheus import speculative_delta

    first, last = _counters(before, model), _counters(after, model)
    delta: dict[str, Any] = {}
    for key in ("prompt_tokens", "generation_tokens"):
        if first[key] is not None and last[key] is not None:
            delta[key] = last[key] - first[key]
    delta["speculative_decoding"] = speculative_delta(first["spec"], last["spec"])
    return delta


async def profile_window(
    http: httpx.AsyncClient,
    base_url: str,
    metrics_url: str | None,
    model: str | None,
    settings: WindowSettings,
    record: dict[str, Any],
    stop: asyncio.Event,
) -> None:
    """Open one capture window ``delay_s`` after the call, for ``window_s``, or
    until ``stop`` is set (the level ran out of requests). Fills ``record``."""
    record.update(asdict(settings))
    try:
        await asyncio.wait_for(stop.wait(), settings.delay_s)
        record["skipped"] = "level ended before the window opened"
        return
    except TimeoutError:
        pass

    async def scrape() -> str:
        if not metrics_url:
            return ""
        response = await http.get(metrics_url)
        response.raise_for_status()
        return response.text

    before = await scrape()
    record["start_requested_at"] = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    response = await http.post(f"{base_url}/start_profile", timeout=600)
    response.raise_for_status()
    record["start_latency_s"] = round(time.perf_counter() - started, 3)
    opened = time.perf_counter()
    try:
        await asyncio.wait_for(stop.wait(), settings.window_s)
        record["truncated"] = True
    except TimeoutError:
        record["truncated"] = False
    finally:
        response = await http.post(f"{base_url}/stop_profile", timeout=600)
        response.raise_for_status()
        record["open_s"] = round(time.perf_counter() - opened, 3)
        record["stopped_at"] = datetime.now(UTC).isoformat()
        after = await scrape()
        if before and after:
            record["counters"] = window_counters(before, after, model)


# --- after the run ---------------------------------------------------------


def reports(directory: Path) -> list[Path]:
    """Finished reports, oldest first (the capture order)."""
    return sorted(directory.glob("*.nsys-rep"), key=lambda path: path.stat().st_mtime)


def pending(directory: Path) -> list[Path]:
    """nsys intermediates that are not yet a report."""
    return [path for path in directory.glob("*") if path.suffix in {".qdstrm", ".tmp"}]


async def wait_for_reports(directory: Path, expected: int, timeout: float) -> list[Path]:
    """Wait until ``expected`` reports exist, none is still being written, and
    their sizes are stable for two polls."""
    deadline = time.monotonic() + timeout
    sizes: dict[Path, int] = {}
    stable = 0
    while time.monotonic() < deadline:
        found = reports(directory)
        now = {path: path.stat().st_size for path in found}
        if len(found) >= expected and not pending(directory):
            stable = stable + 1 if now == sizes else 0
            if stable >= 2:
                return found
        sizes = now
        await asyncio.sleep(5)
    return reports(directory)


def stats_argv(nsys: str, report: Path, out_prefix: Path, stats: list[str]) -> list[str]:
    return [
        nsys, "stats",
        "--report", ",".join(stats),
        "--format", "csv",
        "--output", str(out_prefix),
        "--force-export", "true",
        "--force-overwrite", "true",
        str(report),
    ]  # fmt: skip


def gpu_metrics_summary(database: Path) -> list[dict[str, Any]]:
    """Mean, min and max of each sampled GPU metric in an exported report
    (empty when the report has no GPU-metrics tables)."""
    connection = sqlite3.connect(database)
    try:
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = {row[0] for row in connection.execute(query)}
        if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
            return []
        rows = connection.execute(
            """
            SELECT info.metricName, COUNT(*), AVG(m.value), MIN(m.value), MAX(m.value)
            FROM GPU_METRICS AS m
            JOIN (SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS) AS info
              ON m.metricId = info.metricId
            GROUP BY info.metricName ORDER BY info.metricName
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {"metric": name, "samples": count, "mean": mean, "min": low, "max": high}
        for name, count, mean, low, high in rows
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def export_reports(
    spec: ProfileSpec, found: list[Path], stats_dir: Path, windows: list[dict[str, Any]]
) -> dict[str, Any]:
    """``nsys stats`` CSVs per report, GPU-metric summaries, and a manifest that
    pairs each report with its capture window (in capture order)."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, report in enumerate(found):
        stem = report.name.removesuffix(".nsys-rep")
        prefix = stats_dir / stem
        argv = stats_argv(spec.nsys, report, prefix, spec.stats)
        log = stats_dir / f"{stem}.stats.log"
        with log.open("wb") as handle:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=handle, stderr=asyncio.subprocess.STDOUT
            )
            code = await process.wait()
        database = report.with_suffix(".sqlite")
        metrics = gpu_metrics_summary(database) if database.exists() else []
        write_csv(stats_dir / f"{stem}_gpu_metrics_summary.csv", metrics)
        entries.append(
            {
                "report": str(report),
                "report_bytes": report.stat().st_size,
                "sqlite": str(database) if database.exists() else None,
                "stats_exit_code": code,
                "csv": sorted(path.name for path in stats_dir.glob(f"{stem}_*.csv")),
                "gpu_metrics": bool(metrics),
                "window": windows[index] if index < len(windows) else None,
            }
        )
    manifest = {"profile": asdict(spec), "reports": entries}
    (stats_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def collect_windows(arm_dir: Path, loadtests: list[tuple[str, str, list[int]]]) -> list[dict]:
    """The recorded capture windows of an arm, in the order they were opened.

    ``loadtests`` lists (label, workload, concurrencies) in run order."""
    windows = []
    for label, workload, levels in loadtests:
        for concurrency in sorted(levels):
            path = arm_dir / label / workload / f"c{concurrency}" / WINDOW_FILE
            if path.exists():
                record = json.loads(path.read_text())
                if "start_requested_at" in record:
                    windows.append({"loadtest": label, "concurrency": concurrency, **record})
    return windows
