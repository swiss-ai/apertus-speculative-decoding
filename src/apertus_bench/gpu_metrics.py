"""Node-wide GPU hardware counters of a sweep, per load-test cell.

serving/gpu-metrics-sampler.sh samples every GPU of the node with Nsight
Systems GPU metrics (no CUDA tracing) for the whole job, in chunks
(``gpu-metrics/chunk-NNN.nsys-rep``). This module turns those reports into

- ``gpu-metrics/per-second.csv.gz``: one row per (second, GPU, metric), the mean
  of the samples in that second, keyed by the physical GPU index the arms were
  given (CUDA_VISIBLE_DEVICES);
- ``gpu-metrics/cells.json``: for every load-test cell of every arm, the mean of
  each metric on each of the arm's GPUs (by role: ``serve`` for a colocated
  arm, ``prefill``/``decode`` for a disaggregated one) over the cell's measured
  window and over its steady part (seconds in which the serving instance had at
  least 90% of the cell's peak running requests).

    python3 -m apertus_bench.gpu_metrics export SWEEP_DIR   (in the job, needs nsys)
    python3 -m apertus_bench.gpu_metrics cells SWEEP_DIR    (per-second CSV -> cells.json)

Metric values are what nsys reports: throughputs in percent of peak (``SMs
Active``, ``Tensor Active``, ``DRAM Read Bandwidth``, ...), clocks in MHz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_NSYS = os.environ.get(
    "NSYS",
    "/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nsight-systems-2025.6.3/"
    "opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys",
)
STEADY_FRACTION = 0.9


def bus_key(bus: str) -> tuple[int, int, int]:
    """(domain, bus, device) of a PCI address in nvidia-smi or nsys spelling."""
    bus = bus.strip()
    parts = bus.split(":")
    if len(parts) == 2:  # bus:device.function without a domain
        parts = ["0", *parts]
    domain, number, rest = parts[-3], parts[-2], parts[-1]
    return int(domain, 16), int(number, 16), int(rest.split(".")[0], 16)


def physical_indices(gpus_csv: Path) -> dict[tuple[int, int, int], int]:
    """PCI address -> nvidia-smi index, from the sampler's gpus.csv."""
    rows = list(csv.reader(gpus_csv.read_text().splitlines()))
    header = [cell.strip() for cell in rows[0]]
    index, bus = header.index("index"), header.index("pci.bus_id")
    return {bus_key(row[bus]): int(row[index]) for row in rows[1:] if row}


def _columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]


def reduce_report(sqlite_path: Path, indices: dict[tuple[int, int, int], int]) -> tuple[
    dict[tuple[int, int, str], list[float]], dict[str, Any]
]:
    """Per (unix second, physical GPU, metric): [sum, count] of the samples."""
    db = sqlite3.connect(sqlite_path)
    info: dict[str, Any] = {"report": sqlite_path.name}
    start_ns = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    info["session_start_utc_ns"] = start_ns
    names = {
        (type_id, metric_id): name
        for type_id, metric_id, name in db.execute(
            "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        )
    }
    # nsys GPU id -> physical index, through the PCI address.
    gpu_columns = _columns(db, "TARGET_INFO_GPU")
    bus_column = next(c for c in ("busLocation", "busId", "pciBusId") if c in gpu_columns)
    nsys_to_physical = {}
    for gpu_id, bus in db.execute(f"SELECT id, {bus_column} FROM TARGET_INFO_GPU"):
        nsys_to_physical[int(gpu_id)] = indices.get(bus_key(str(bus)), int(gpu_id))
    info["gpu_map"] = {str(k): v for k, v in sorted(nsys_to_physical.items())}
    sums: dict[tuple[int, int, str], list[float]] = defaultdict(lambda: [0.0, 0])
    rows = 0
    first = last = None
    for type_id, timestamp, value, metric_id in db.execute(
        "SELECT typeId, timestamp, value, metricId FROM GPU_METRICS"
    ):
        name = names.get((type_id, metric_id))
        if name is None:
            continue
        gpu = nsys_to_physical.get(type_id & 0xFF, type_id & 0xFF)
        second = (start_ns + timestamp) // 1_000_000_000
        cell = sums[(second, gpu, name)]
        cell[0] += value
        cell[1] += 1
        rows += 1
        first = timestamp if first is None else min(first, timestamp)
        last = timestamp if last is None else max(last, timestamp)
    info["samples"] = rows
    if first is not None:
        info["first_utc_s"] = (start_ns + first) / 1e9
        info["last_utc_s"] = (start_ns + last) / 1e9
    info["metrics"] = sorted(set(names.values()))
    db.close()
    return sums, info


def export(sweep: Path, nsys: str = DEFAULT_NSYS) -> None:
    metrics_dir = sweep / "gpu-metrics"
    indices = physical_indices(metrics_dir / "gpus.csv")
    manifest = []
    rows: dict[tuple[int, int, str], list[float]] = {}
    for report in sorted(metrics_dir.glob("chunk-*.nsys-rep")):
        sqlite_path = report.with_suffix(".sqlite")
        if not sqlite_path.exists():
            code = subprocess.run(
                [nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                 "--output", str(sqlite_path), str(report)],
                check=False,
            ).returncode  # fmt: skip
            if code != 0 or not sqlite_path.exists():
                manifest.append({"report": report.name, "export_exit_code": code})
                continue
        sums, info = reduce_report(sqlite_path, indices)
        info["report_bytes"] = report.stat().st_size
        manifest.append(info)
        for key, (total, count) in sums.items():
            merged = rows.setdefault(key, [0.0, 0])
            merged[0] += total
            merged[1] += count
    with gzip.open(metrics_dir / "per-second.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unix_s", "gpu", "metric", "mean", "samples"])
        for (second, gpu, name), (total, count) in sorted(rows.items()):
            writer.writerow([second, gpu, name, round(total / count, 3), count])
    (metrics_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cells(sweep)


def load_per_second(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    """gpu -> unix second -> metric -> mean."""
    data: dict[int, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            data[int(row["gpu"])][int(row["unix_s"])][row["metric"]] = float(row["mean"])
    return data


def arm_roles(status: dict[str, Any]) -> dict[str, list[int]]:
    """Role -> physical GPUs of an arm, as node_sweep.run_arm assigned them."""
    gpus = [int(gpu) for gpu in status.get("gpus") or []]
    servers = status.get("servers") or []
    if not servers:
        return {"serve": gpus}
    per_server = len(gpus) // len(servers)
    roles: dict[str, list[int]] = defaultdict(list)
    for number, server in enumerate(servers):
        roles[server["role"]] += gpus[number * per_server : (number + 1) * per_server]
    return dict(roles)


def steady_seconds(cell: Path, start: float) -> list[int]:
    """Unix seconds of a cell in which running >= 90% of its peak (serving instance)."""
    path = cell / "metrics_timeseries.jsonl"
    if not path.exists():
        return []
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    peak = max((sample.get("running") or 0 for sample in samples), default=0)
    if peak <= 0:
        return []
    return sorted(
        {
            int(start + sample["t"])
            for sample in samples
            if (sample.get("running") or 0) >= STEADY_FRACTION * peak
        }
    )


def _mean(series: dict[int, dict[str, float]], seconds: list[int]) -> dict[str, Any]:
    present = [second for second in seconds if second in series]
    totals: dict[str, list[float]] = defaultdict(list)
    for second in present:
        for name, value in series[second].items():
            totals[name].append(value)
    return {
        "seconds": len(present),
        "metrics": {name: round(sum(v) / len(v), 3) for name, v in sorted(totals.items())},
    }


def cells(sweep: Path) -> list[dict[str, Any]]:
    data = load_per_second(sweep / "gpu-metrics" / "per-second.csv.gz")
    records = []
    for status_path in sorted(sweep.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        roles = arm_roles(status)
        for metadata_path in sorted(status_path.parent.glob("*/*/c*/metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            start = datetime.fromisoformat(metadata["started_at"]).timestamp()
            end = datetime.fromisoformat(metadata["ended_at"]).timestamp()
            window = list(range(int(start) + 1, int(end)))
            steady = [s for s in steady_seconds(metadata_path.parent, start) if s in window]
            record = {
                "arm": status["arm"],
                "cell": str(metadata_path.parent.relative_to(sweep)),
                "concurrency": int(metadata_path.parent.name[1:]),
                "window_s": round(end - start, 1),
                "gpus": {},
            }
            for role, gpus in roles.items():
                for gpu in gpus:
                    record["gpus"][f"{role}:{gpu}"] = {
                        "window": _mean(data.get(gpu, {}), window),
                        "steady": _mean(data.get(gpu, {}), steady),
                    }
            records.append(record)
    (sweep / "gpu-metrics" / "cells.json").write_text(json.dumps(records, indent=1) + "\n")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus_bench.gpu_metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "cells"):
        command = sub.add_parser(name)
        command.add_argument("sweep", type=Path)
    sub.choices["export"].add_argument("--nsys", default=DEFAULT_NSYS)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.sweep, args.nsys)
    else:
        cells(args.sweep)


if __name__ == "__main__":
    main()
