#!/usr/bin/env python3
"""CPU-only review of preserved layout outputs; never launches or changes a run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

EXPECTED_STAGES = [
    "preflight",
    "tp4-dspark-off-k7",
    "tp4-eagle-k7",
    "tp4-dspark-on-k7",
    "think-pd",
    "tp2dp2",
    "pd-off",
]
UNHEALTHY = {"tp4-b1-20261024T132121Z"}
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
KV = re.compile(r"GPU KV cache size:\s*([\d,]+)\s+tokens")
LIMITS = [
    (
        "Screening data only: independent deployment confirmation repeats "
        "and confidence intervals are not established."
    ),
    "Calibrated per-layout correctness controls are not established by these load tests.",
    (
        "Fixed-concurrency tests do not establish an open-loop SLO knee "
        "or sustainable traffic capacity."
    ),
    "Load-generator capacity needs its own evidence; client success alone does not prove it.",
    (
        "Natural-EOS outputs can differ across deployments; "
        "compare completion volumes alongside throughput."
    ),
    "P/D load/acceptance counters cover only the first decode instance, not the whole node.",
    "GPU resident memory includes preallocated KV pools; it is not the minimum memory required.",
]


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def mapped(value: str, mappings: list[tuple[str, str]]) -> Path:
    for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        try:
            relative = Path(value).relative_to(old)
        except ValueError:
            continue
        return Path(new) / relative
    return Path(value)


def raw_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    return compressed if compressed.is_file() else None


def text_open(path: Path):
    return (
        gzip.open(path, "rt", errors="replace")
        if path.suffix == ".gz"
        else path.open(errors="replace")
    )


def json_value(value):
    return json.loads(value) if isinstance(value, str) else value or {}


def cache_configs(snapshot: Path | None) -> list[dict[str, Any]]:
    """Retain each logical engine; never multiply a TP pool by GPU count."""
    if snapshot is None:
        return []
    records = {}
    with text_open(snapshot) as stream:
        for line in stream:
            if not line.startswith("vllm:cache_config_info{"):
                continue
            labels = {k: json.loads('"' + v + '"') for k, v in LABEL.findall(line)}
            engine = labels.get("engine", "single")
            records[engine] = {
                "engine": engine,
                "kv_tokens": int(labels["kv_cache_size_tokens"])
                if labels.get("kv_cache_size_tokens")
                else None,
                "gpu_memory_utilization": float(labels["gpu_memory_utilization"])
                if labels.get("gpu_memory_utilization")
                else None,
                "num_gpu_blocks": labels.get("num_gpu_blocks"),
                "block_size": labels.get("block_size"),
            }
    return list(records.values())


def used_kv_bounds(configs: list[dict], summed_usage: float | None) -> dict:
    capacities = [row["kv_tokens"] for row in configs]
    if (
        summed_usage is None
        or not capacities
        or any(v is None for v in capacities)
        or not 0 <= summed_usage <= len(capacities) + 1e-6
    ):
        return {"lower_bound": None, "upper_bound": None}

    def allocate(order):
        remaining = min(summed_usage, len(order))
        total = 0
        for capacity in order:
            fraction = min(1.0, remaining)
            total += fraction * capacity
            remaining -= fraction
        return total

    return {
        "lower_bound": math.floor(allocate(sorted(capacities))),
        "upper_bound": math.ceil(allocate(sorted(capacities, reverse=True))),
        "method": (
            "Bound observed peak summed usage by allocating fractions to "
            "smallest/largest logical pools; scope follows sampled metrics."
        ),
    }


@functools.cache
def run_provenance(run: Path) -> dict:
    result = {}
    for name in [
        "git-revision.txt",
        "image.sha256",
        "slurm.txt",
        "job.txt",
        "patches.txt",
        "environment.toml",
        "checkout.patch",
        "submit-command.txt",
    ]:
        p = run / name
        if p.exists():
            result[name] = {"path": str(p), "sha256": digest(p)}
            if name in {"git-revision.txt", "image.sha256", "slurm.txt", "job.txt"}:
                result[name]["text"] = p.read_text()[:4096].strip()
    return result


@functools.cache
def pd_startup(arm_dir: Path) -> dict[str, Any]:
    """Bounded prefixes per role; report ambiguity instead of extrapolating."""
    rows = []
    files = sorted(arm_dir.glob("replica-*.out")) + sorted(arm_dir.glob("replica-*.out.gz"))
    seen = set()
    for path in files:
        name = path.name.removesuffix(".gz")
        if name in seen:
            continue
        seen.add(name)
        with text_open(path) as stream:
            prefix = stream.read(1024 * 1024)
        values = sorted({int(value.replace(",", "")) for value in KV.findall(prefix)})
        rows.append(
            {
                "file": str(path),
                "prefix_characters": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "role": "prefill" if "-prefill-" in name else "decode",
                "startup_kv_token_values": values,
                "kv_tokens": values[0] if len(values) == 1 else None,
            }
        )
    return {"role_logs": rows, "prefix_limit_characters_per_log": 1024 * 1024}


@functools.cache
def gpu_samples(run: Path, offset_seconds: int) -> list[tuple[float, int, float]]:
    path = raw_path(run / "gpu-memory.csv")
    if path is None:
        return []
    result = []
    zone = dt.timezone(dt.timedelta(seconds=offset_seconds))
    with text_open(path) as stream:
        for fields in csv.reader(stream):
            try:
                timestamp = (
                    dt.datetime.strptime(fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=zone)
                    .timestamp()
                )
                result.append((timestamp, int(fields[1]), float(fields[2])))
            except (ValueError, IndexError):
                continue
    return result


def gpu_window(run: Path, status: dict, meta: dict) -> dict:
    if not all(meta.get(key) for key in ["started_at", "ended_at"]) or not status.get("started"):
        return {"available": False}
    start, end = (
        dt.datetime.fromisoformat(meta[key]).timestamp() for key in ["started_at", "ended_at"]
    )
    offset = dt.datetime.fromisoformat(status["started"]).utcoffset()
    if offset is None:
        return {"available": False}
    samples = gpu_samples(run, int(offset.total_seconds()))
    by_gpu = {}
    buckets = {}
    ids = set(status.get("gpus", []))
    for timestamp, gpu, memory in samples:
        if start <= timestamp <= end and gpu in ids:
            by_gpu.setdefault(gpu, []).append(memory)
            buckets.setdefault(math.floor(timestamp), {})[gpu] = memory
    totals = [sum(row.values()) for row in buckets.values() if set(row) == ids]
    return {
        "available": bool(by_gpu),
        "timezone_assumption": (
            "GPU CSV timestamps use the same UTC offset as arm status; sampled intervals only."
        ),
        "per_gpu": {
            str(g): {"mean_mib": statistics.mean(v), "peak_mib": max(v), "samples": len(v)}
            for g, v in by_gpu.items()
        },
        "node_resident_mean_mib": statistics.mean(totals) if totals else None,
        "node_resident_peak_mib": max(totals) if totals else None,
        "complete_node_sample_seconds": len(totals),
        "node_sample_coverage_fraction": min(1.0, len(totals) / max(1.0, end - start)),
    }


def usage_check(cell: Path) -> dict:
    path = raw_path(cell / "requests.jsonl")
    if path is None:
        return {"passed": False, "error": "Missing requests"}
    counts = {"attempted": 0, "successful": 0, "missing_usage": 0, "completion_tokens": 0}
    prompt_sizes = []
    with text_open(path) as stream:
        for line in stream:
            row = json.loads(line)
            counts["attempted"] += 1
            if row.get("success"):
                counts["successful"] += 1
                if row.get("prompt_tokens") is None or row.get("completion_tokens") is None:
                    counts["missing_usage"] += 1
                else:
                    prompt_sizes.append(row["prompt_tokens"])
                    counts["completion_tokens"] += row["completion_tokens"]
    ordered = sorted(prompt_sizes)
    counts["prompt_tokens_distribution"] = (
        {
            "min": min(ordered),
            "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": max(ordered),
        }
        if ordered
        else None
    )
    summary = read_json(cell / "summary.json")
    counts["passed"] = (
        counts["missing_usage"] == 0
        and counts["completion_tokens"] == summary["tokens"]["completion"]
        and counts["successful"] == summary["requests"]["successful"]
        and counts["attempted"] == summary["requests"]["attempted"]
    )
    counts["source_sha256"] = digest(path)
    return counts


def enrich(row: dict, mappings, verify_requests: bool) -> dict:
    result = dict(row)
    summary = mapped(row["summary"], mappings)
    if not summary.exists():
        result.update(
            review_eligible=False, review_errors=["Raw summary unavailable; collect its source run"]
        )
        return result
    cell = summary.parent
    arm_dir = cell.parents[2]
    run = arm_dir.parent
    raw = read_json(summary)
    meta_path = cell / "metadata.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    status_path = arm_dir / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    extra = meta.get("extra", {})
    engine = json_value(extra.get("engine"))
    pd = json_value(extra.get("disaggregated"))
    configs = cache_configs(raw_path(cell / "metrics_before.prom"))
    capacity = {
        "scope": "first_decode_instance" if pd else "all_observed_logical_engines",
        "engines": configs,
        "observed_logical_kv_tokens": sum(r["kv_tokens"] for r in configs)
        if configs and all(r["kv_tokens"] is not None for r in configs)
        else None,
        "summary_last_engine_kv_tokens": (raw.get("load") or {}).get("kv_cache_size_tokens"),
    }
    capacity["observed_used_kv_tokens_peak_bounds"] = used_kv_bounds(
        configs, (raw.get("load") or {}).get("kv_cache_usage", {}).get("max")
    )
    if pd:
        startup = pd_startup(arm_dir)
        logs = startup["role_logs"]
        expected = int(pd.get("prefill", 0)) + int(pd.get("decode", 0))
        complete = len(logs) == expected and all(r["kv_tokens"] is not None for r in logs)
        capacity.update(
            startup=startup,
            startup_complete_all_roles=complete,
            startup_node_logical_kv_tokens=sum(r["kv_tokens"] for r in logs) if complete else None,
            startup_decode_logical_kv_tokens=sum(
                r["kv_tokens"] for r in logs if r["role"] == "decode"
            )
            if complete
            else None,
        )
    errors = []
    if digest(summary) != row["summary_sha256"]:
        errors.append("Summary hash changed")
    expected_meta_hash = row.get("quality", {}).get("metadata_sha256")
    if expected_meta_hash and (not meta_path.exists() or digest(meta_path) != expected_meta_hash):
        errors.append("Metadata hash changed")
    if not row.get("quality", {}).get("passed"):
        errors.append("Original provenance/counter gate missing or failed")
    if raw["requests"]["success_rate"] < 0.99:
        errors.append("Below 99% request success")
    if status.get("status") != "ok":
        errors.append("Arm did not complete successfully")
    if run.name in UNHEALTHY:
        errors.append("Known unhealthy plain allocation; exclude whole attempt")
    if not row.get("allocation_baseline_health_passed", True):
        errors.append("Controller baseline-health gate failed for this allocation")
    if (
        not meta.get("workload_file_sha256")
        or "prompt_ids" not in meta
        or not meta.get("cell", {}).get("generation")
    ):
        errors.append("Missing corpus/prompt/generation pairing provenance")
    if row.get("stage") == "preflight":
        errors.append("Short infrastructure stress is not workload evidence")
    usage = usage_check(cell) if verify_requests else row.get("usage_validation")
    if usage is not None and not usage.get("passed"):
        errors.append("Independent raw usage validation failed")
    result.update(
        summary=str(summary),
        run=str(run),
        engine=engine,
        disaggregated=pd,
        deployment_id=extra.get("deployment_id"),
        arm_status=status.get("status"),
        raw_metadata_sha256=digest(meta_path) if meta_path.exists() else None,
        started_at=meta.get("started_at"),
        ended_at=meta.get("ended_at"),
        prompt_ids_sha256=hashlib.sha256(json.dumps(meta.get("prompt_ids")).encode()).hexdigest(),
        generation=meta.get("cell", {}).get("generation"),
        workload_digest=meta.get("workload_file_sha256"),
        prompt_tokens=raw["tokens"].get("prompt"),
        request_success_rate=raw["requests"]["success_rate"],
        cache_capacity=capacity,
        gpu_memory=gpu_window(run, status, meta),
        usage_validation=usage
        or {
            "passed": None,
            "status": "Not independently resummed; harness successful requests require exact usage",
        },
        review_eligible=not errors,
        review_errors=errors,
        counters_scope="first_decode_instance" if pd else "all_observed_metrics_engines",
        run_provenance=run_provenance(run),
        raw_metric_snapshots=[
            {"path": str(p), "sha256": digest(p)}
            for name in ["metrics_before.prom", "metrics_after.prom"]
            if (p := raw_path(cell / name))
        ],
    )
    return result


def baseline_name(arm: str) -> str:
    if arm.startswith("pd-"):
        return "-".join(arm.split("-")[:2]) + "-plain"
    for suffix in ["tp2dp2", "tp4", "dp4"]:
        if arm.endswith(suffix):
            return "plain-" + suffix
    return "plain"


def number(row: dict, *keys):
    value = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def compare(treatment: dict, baseline: dict, kind: str) -> dict:
    matched = all(
        treatment.get(key) == baseline.get(key)
        for key in ["generation", "workload_digest", "prompt_ids_sha256"]
    )
    eligible = bool(
        treatment.get("review_eligible") and baseline.get("review_eligible") and matched
    )
    tp, bp = treatment["throughput_tok_s"], baseline["throughput_tok_s"]
    result = {
        "run": treatment["run"],
        "stage": treatment["stage"],
        "arm": treatment["arm"],
        "baseline": baseline["arm"],
        "comparison": kind,
        "label": treatment["label"],
        "concurrency": treatment["concurrency"],
        "screening_pair_eligible": eligible,
        "treatment_review_errors": "; ".join(treatment.get("review_errors", [])),
        "baseline_review_errors": "; ".join(baseline.get("review_errors", [])),
        "matched_generation_corpus_prompts": matched,
        "baseline_tok_s": bp,
        "treatment_tok_s": tp,
        "delta_tok_s": tp - bp,
        "speedup": tp / bp,
        "baseline_summary": baseline["summary"],
        "treatment_summary": treatment["summary"],
        "baseline_wall_s": baseline["wall_seconds"],
        "treatment_wall_s": treatment["wall_seconds"],
        "baseline_completion_tokens": baseline["completion_tokens"],
        "treatment_completion_tokens": treatment["completion_tokens"],
        "baseline_success_rate": baseline.get("request_success_rate"),
        "treatment_success_rate": treatment.get("request_success_rate"),
        "engine_settings_equal": treatment.get("engine") == baseline.get("engine"),
        "counter_scope": treatment.get("counters_scope"),
        "effect_class": "substantial_screening_gain"
        if eligible and tp / bp >= 1.1
        else "substantial_regression"
        if eligible and tp / bp <= 0.9
        else "small_or_mixed_effect"
        if eligible
        else "excluded_or_incomplete",
        "implementation_recommendation": (
            "unestablished; no promotion without original scientific gates"
        ),
    }
    for side, row in [("baseline", baseline), ("treatment", treatment)]:
        for family in ["e2e", "ttft", "tpot"]:
            for stat in ["mean", "p50", "p95"]:
                result[f"{side}_{family}_{stat}_ms"] = number(row, "latency_ms", family, stat)
        for key in ["preemptions", "seconds_kv_above_90pct", "kv_cache_tokens_peak"]:
            result[f"{side}_{key}"] = number(row, "load", key)
        result[f"{side}_waiting_max"] = number(row, "load", "waiting", "max")
        result[f"{side}_kv_usage_max"] = number(row, "load", "kv_cache_usage", "max")
        for key in [
            "observed_logical_kv_tokens",
            "startup_node_logical_kv_tokens",
            "startup_decode_logical_kv_tokens",
            "scope",
        ]:
            result[f"{side}_{key}"] = number(row, "cache_capacity", key)
        for key in [
            "node_resident_mean_mib",
            "node_resident_peak_mib",
            "node_sample_coverage_fraction",
        ]:
            result[f"{side}_{key}"] = number(row, "gpu_memory", key)
        result[f"{side}_memory_utilization"] = number(row, "engine", "gpu_memory_utilization")
    for key in [
        "e2e_p50_ms",
        "e2e_p95_ms",
        "ttft_p95_ms",
        "tpot_p95_ms",
        "preemptions",
        "observed_logical_kv_tokens",
        "startup_node_logical_kv_tokens",
        "startup_decode_logical_kv_tokens",
        "node_resident_mean_mib",
    ]:
        a, b = result.get("treatment_" + key), result.get("baseline_" + key)
        result["delta_" + key] = a - b if a is not None and b is not None else None
    scopes_equal = treatment.get("cache_capacity", {}).get("scope") == baseline.get(
        "cache_capacity", {}
    ).get("scope")
    result["kv_capacity_delta_comparable_scope"] = scopes_equal
    if not scopes_equal:
        result["delta_observed_logical_kv_tokens"] = None
    if treatment.get("counters_scope") != baseline.get("counters_scope"):
        result["delta_preemptions"] = None
    result["gpu_memory_delta_coverage_adequate"] = all(
        number(r, "gpu_memory", "node_sample_coverage_fraction") is not None
        and number(r, "gpu_memory", "node_sample_coverage_fraction") >= 0.8
        for r in [treatment, baseline]
    )
    if not result["gpu_memory_delta_coverage_adequate"]:
        result["delta_node_resident_mean_mib"] = None
    return result


def comparisons(rows: list[dict]) -> list[dict]:
    index = {(r["run"], r["arm"], r["label"], r["concurrency"]): r for r in rows}
    result = []
    for row in rows:
        if row.get("review_errors") == ["Raw summary unavailable; collect its source run"]:
            continue
        wanted = []
        name = baseline_name(row["arm"])
        if name != row["arm"]:
            wanted.append((name, "same_layout_speculation"))
        if name != "plain-dp4" and row["arm"] != "plain-dp4":
            wanted.append(("plain-dp4", "layout_vs_plain_dp4"))
        for name, kind in wanted:
            base = index.get((row["run"], name, row["label"], row["concurrency"]))
            if base:
                result.append(compare(row, base, kind))
    return result


def partial_rows(record: dict, mappings) -> list[dict]:
    root = mapped(record["output"], mappings)
    rows = []
    qualities = {
        (r.get("arm"), r.get("label"), r.get("concurrency")): r
        for r in record.get("gate", {}).get("checks", [])
        if r.get("kind") == "provenance_counter_continuity"
    }
    for p in root.glob("*/**/c*/summary.json"):
        try:
            d = read_json(p)
        except (OSError, ValueError):
            continue
        rel = p.relative_to(root).parts
        key = (rel[0], rel[1], int(p.parent.name[1:]))
        rows.append(
            {
                "stage": record["stage"],
                "run": str(root),
                "arm": key[0],
                "label": key[1],
                "concurrency": key[2],
                "summary": str(p),
                "summary_sha256": digest(p),
                "quality": qualities.get(
                    key, {"passed": False, "errors": ["No completed original quality gate"]}
                ),
                "throughput_tok_s": d["tokens"]["output_tokens_per_second"],
                "successful": d["requests"]["successful"],
                "failed": d["requests"]["failed"],
                "acceptance_length": d.get("speculative_decoding", {}).get(
                    "mean_acceptance_length"
                ),
                "latency_ms": d.get("latency_ms"),
                "load": d.get("load"),
                "completion_tokens": d["tokens"]["completion"],
                "wall_seconds": d["wall_seconds"],
            }
        )
    return rows


def cell_table(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        flat = {
            key: row.get(key)
            for key in [
                "run",
                "stage",
                "arm",
                "label",
                "concurrency",
                "review_eligible",
                "throughput_tok_s",
                "successful",
                "failed",
                "request_success_rate",
                "wall_seconds",
                "completion_tokens",
                "prompt_tokens",
                "counters_scope",
                "summary",
                "summary_sha256",
                "deployment_id",
            ]
        }
        flat["review_errors"] = "; ".join(row.get("review_errors", []))
        for family in ["e2e", "ttft", "tpot"]:
            for stat in ["mean", "p50", "p95"]:
                flat[family + "_" + stat + "_ms"] = number(row, "latency_ms", family, stat)
        for key in ["preemptions", "kv_cache_tokens_peak", "seconds_kv_above_90pct"]:
            flat[key] = number(row, "load", key)
        for key in [
            "scope",
            "observed_logical_kv_tokens",
            "startup_node_logical_kv_tokens",
            "startup_decode_logical_kv_tokens",
        ]:
            flat[key] = number(row, "cache_capacity", key)
        for key in ["lower_bound", "upper_bound"]:
            flat["observed_used_kv_tokens_peak_" + key] = number(
                row, "cache_capacity", "observed_used_kv_tokens_peak_bounds", key
            )
        for key in [
            "node_resident_mean_mib",
            "node_resident_peak_mib",
            "node_sample_coverage_fraction",
        ]:
            flat[key] = number(row, "gpu_memory", key)
        flat["gpu_memory_utilization"] = number(row, "engine", "gpu_memory_utilization")
        result.append(flat)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(analyses: list[Path], controller_state: Path | None, mappings):
    sources = []
    rows = []
    missing = []
    state = None
    if controller_state:
        state = read_json(controller_state)
        sources.append({"path": str(controller_state), "sha256": digest(controller_state)})
        seen = set()
        for record in state["stages"]:
            seen.add(record["stage"])
            path = controller_state.parent / f"analysis-{record['stage']}-{record['job_id']}.json"
            if path.exists():
                data = read_json(path)
                stage_rows = data["rows"]
                sources.append({"path": str(path), "sha256": digest(path)})
            else:
                stage_rows = partial_rows(record, mappings)
                missing.append(
                    {
                        "stage": record["stage"],
                        "job_id": record["job_id"],
                        "stage_state": record["state"],
                        "missing_analysis": str(path),
                        "raw_output": record["output"],
                        "partial_rows_preserved": len(stage_rows),
                    }
                )
            health = all(
                check["passed"]
                for check in record.get("gate", {}).get("checks", [])
                if check.get("kind") == "baseline_health"
            )
            for row in stage_rows:
                row["allocation_baseline_health_passed"] = health
                row["controller_stage_state"] = record["state"]
            rows.extend(stage_rows)
        for stage in EXPECTED_STAGES:
            if stage not in seen:
                missing.append({"stage": stage, "job_id": None, "stage_state": "not_submitted"})
    for path in dict.fromkeys(analyses):
        data = read_json(path)
        sources.append({"path": str(path), "sha256": digest(path)})
        rows.extend(data["rows"])
    unique = {r["summary"]: r for r in rows}
    return sources, list(unique.values()), missing, state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, action="append", default=[])
    ap.add_argument("--controller-state", type=Path)
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument(
        "--verify-requests",
        action="store_true",
        help="CPU-intensive raw usage/prompt distribution check; run locally after collection",
    )
    ap.add_argument("--output", type=Path, required=True, help="Fresh immutable output directory")
    a = ap.parse_args()
    maps = [tuple(v.split("=", 1)) for v in a.path_map]
    if any(len(v) != 2 for v in maps):
        ap.error("Path maps need OLD=NEW")
    sources, rows, missing, state = load_inputs(a.analysis, a.controller_state, maps)
    if not rows:
        ap.error("No preserved analysis rows")
    a.output.mkdir(parents=True, exist_ok=False)
    enriched = [enrich(row, maps, a.verify_requests) for row in rows]
    pairs = comparisons(enriched)
    report = {
        "helper_sha256": digest(Path(__file__)),
        "input_analyses": sources,
        "controller_state_path": str(a.controller_state),
        "controller_state_sha256": digest(a.controller_state) if a.controller_state else None,
        "campaign_state": state.get("campaign_state") if state else "historical_screening_only",
        "scientific_state": "pending_scientific_review",
        "rows": enriched,
        "comparisons": pairs,
        "missing_stage_analyses": missing,
        "original_scientific_gate_limits": LIMITS,
        "capacity_note": (
            "Cache tokens are logical engine pools, not summed TP shards. "
            "P/D first-decode metrics never imply node totals; "
            "startup role totals are separately labeled and not fungible across roles."
        ),
    }
    (a.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(a.output / "comparisons.csv", pairs)
    write_csv(a.output / "cells.csv", cell_table(enriched))
    groups = {}
    for row in pairs:
        groups.setdefault((Path(row["run"]).name, row["arm"], row["comparison"]), []).append(row)
    lines = [
        "# Preserved layout comparison review",
        "",
        f"Campaign state: `{report['campaign_state']}`. Scientific verdict remains pending.",
        "",
        (
            "All absolute metrics and both P/D baselines are in `comparisons.csv`; "
            "raw scopes/engine capacities/hashes are in `review.json`."
        ),
        (
            "Effects of at least 10% are screening candidates only. "
            "Small gains never become implementation recommendations."
        ),
        "",
        "| Run | Arm | Comparison | Eligible/seen | Speedup range | Losing cells >10% |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (run, arm, kind), group in sorted(groups.items()):
        eligible = [r for r in group if r["screening_pair_eligible"]]
        effects = [r["speedup"] for r in eligible]
        span = f"{min(effects):.2f}–{max(effects):.2f}x" if effects else "unestablished"
        lines.append(
            f"|{run}|{arm}|{kind}|{len(eligible)}/{len(group)}|"
            f"{span}|{sum(v < 0.9 for v in effects)}|"
        )
    lines += ["", "Original scientific gates/claim limits:", ""] + ["- " + line for line in LIMITS]
    lines += [
        "",
        (
            f"Missing stage analyses: {len(missing)}. "
            "A terminal Slurm state alone is not campaign completion."
        ),
        "",
        (
            "GPU-memory windows assume the CSV clock shares the arm-status UTC offset. "
            "Verify the reported coverage; missing fields stay unavailable."
        ),
        (
            "These are observed resident allocations, including preallocated KV pools. "
            "Negative deltas are savings only within the declared metric scope."
        ),
    ]
    (a.output / "review.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "rows": len(enriched),
                "comparisons": len(pairs),
                "eligible_pairs": sum(r["screening_pair_eligible"] for r in pairs),
                "missing_stage_analyses": len(missing),
                "output": str(a.output),
            }
        )
    )


if __name__ == "__main__":
    main()
