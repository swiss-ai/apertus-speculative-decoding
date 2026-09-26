from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from apertus_bench.diagnostics import normalize_scheduled_tokens


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        value = value[key]
    return value


def _extra(metadata: dict[str, Any]) -> dict[str, Any]:
    extra = metadata.get("extra") or {}
    return extra if isinstance(extra, dict) else {}


def _generation(metadata: dict[str, Any]) -> dict[str, Any]:
    cell = metadata.get("cell") or {}
    generation = cell.get("generation") or {}
    return generation if isinstance(generation, dict) else {}


def _variant_field(variant: dict[str, Any], extra: dict[str, Any], key: str) -> Any:
    if variant.get(key) not in (None, ""):
        return variant[key]
    return extra.get(key)


# Target identity fields. A baseline from another target, revision, tokenizer,
# target TP or precision is never a denominator (8B and 70B never pool).
TARGET_MATCH_FIELDS = (
    "target_model",
    "target_revision",
    "tokenizer_sha256",
    "target_tensor_parallel_size",
    "precision",
)


def operational_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Fields that must match for an operational baseline comparison."""
    return tuple(row.get(field) for field in TARGET_MATCH_FIELDS) + (
        row.get("workload"),
        row.get("concurrency"),
        row.get("workload_file_sha256"),
        row.get("temperature"),
        row.get("top_p"),
        row.get("ignore_eos"),
        row.get("container_image"),
        row.get("vllm_revision"),
        row.get("max_model_len"),
        row.get("prefix_caching"),
    )


def matched_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Operational key plus the scheduler/budget knobs a matched baseline copies."""
    return operational_key(row) + (
        row.get("async_scheduling"),
        row.get("effective_scheduled_tokens"),
    )


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        metadata_path = summary_path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        variant = metadata["variant"]
        cell = metadata["cell"]
        extra = _extra(metadata)
        generation = _generation(metadata)
        spec = summary["speculative_decoding"]
        latency = summary.get("latency_ms") or {}
        rows.append(
            {
                "path": str(summary_path.parent),
                "variant": variant["name"],
                "method": variant["method"],
                "algorithm": _variant_field(variant, extra, "algorithm"),
                "num_speculative_tokens": variant.get("num_speculative_tokens"),
                "draft_tensor_parallel_size": variant.get("draft_tensor_parallel_size"),
                "prompt_lookup_max": variant.get("prompt_lookup_max"),
                "parallel_drafting": variant.get("parallel_drafting"),
                "workload": cell["workload"],
                "concurrency": cell["concurrency"],
                "repeat": cell["repeat"],
                "success_rate": _nested(summary, "requests", "success_rate"),
                "output_tokens_per_second": _nested(summary, "tokens", "output_tokens_per_second"),
                "ttft_p50_ms": _nested(summary, "latency_ms", "ttft", "p50"),
                "ttft_p95_ms": _nested(summary, "latency_ms", "ttft", "p95"),
                "tpot_p50_ms": _nested(summary, "latency_ms", "tpot", "p50"),
                "tpot_p95_ms": _nested(summary, "latency_ms", "tpot", "p95"),
                "e2e_p50_ms": (latency.get("e2e") or {}).get("p50"),
                "e2e_p95_ms": (latency.get("e2e") or {}).get("p95"),
                "acceptance_rate": spec.get("acceptance_rate"),
                "mean_acceptance_length": spec.get("mean_acceptance_length"),
                "acceptance_metrics_available": spec.get("enabled"),
                "completion_tokens": _nested(summary, "tokens", "completion"),
                "prompt_tokens": _nested(summary, "tokens", "prompt"),
                "drafts": spec.get("drafts"),
                "draft_tokens": spec.get("draft_tokens"),
                "accepted_tokens": spec.get("accepted_tokens"),
                "workload_file_sha256": metadata.get("workload_file_sha256"),
                "temperature": generation.get("temperature"),
                "top_p": generation.get("top_p"),
                "ignore_eos": generation.get("ignore_eos"),
                "container_image": extra.get("container_image"),
                "vllm_revision": extra.get("vllm_revision"),
                "serving_revision": extra.get("serving_revision") or extra.get("vllm_revision"),
                "training_revision": extra.get("training_revision"),
                "max_model_len": extra.get("max_model_len"),
                "prefix_caching": extra.get("prefix_caching"),
                "async_scheduling": extra.get("async_scheduling"),
                "effective_scheduled_tokens": normalize_scheduled_tokens(
                    extra.get("effective_scheduled_tokens")
                ),
                "max_num_batched_tokens": extra.get("max_num_batched_tokens"),
                "precision": extra.get("precision"),
                "target_model": extra.get("target_model"),
                "target_revision": extra.get("target_revision"),
                "tokenizer_sha256": extra.get("tokenizer_sha256"),
                "target_tensor_parallel_size": _as_int(extra.get("target_tensor_parallel_size")),
                "cache_policy": extra.get("cache_policy") or extra.get("prefix_caching"),
                "checkpoint_sha256": extra.get("checkpoint_sha256"),
                "deployment_id": extra.get("deployment_id"),
                "block_id": extra.get("block_id"),
                "baseline_role": extra.get("baseline_role"),
                "paired_baseline_id": extra.get("paired_baseline_id"),
                "diagnostic_id": extra.get("diagnostic_id") or variant["name"],
                "harness_git_revision": metadata.get("harness_git_revision"),
            }
        )
    return rows


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def check_treatment_factors(rows: list[dict[str, Any]]) -> None:
    """Draft TP is a treatment factor: every repeat of one candidate must agree."""
    seen: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for row in rows:
        if row["method"] == "none":
            continue
        key = (
            row.get("variant"),
            row.get("target_model"),
            row.get("workload"),
            row.get("concurrency"),
        )
        seen[key].add(row.get("draft_tensor_parallel_size"))
    mixed = {key: values for key, values in seen.items() if len(values) > 1}
    if mixed:
        raise ValueError(
            "speculative repeats disagree on draft_tensor_parallel_size: "
            + "; ".join(
                f"{key[0]} {key[2]} c{key[3]}: {sorted(map(str, v))}" for key, v in mixed.items()
            )
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _is_operational_baseline(row: dict[str, Any]) -> bool:
    role = row.get("baseline_role")
    return row["method"] == "none" and role in (None, "", "operational")


def _is_matched_baseline(row: dict[str, Any]) -> bool:
    return row["method"] == "none" and row.get("baseline_role") == "matched"


def add_speedups(rows: list[dict[str, Any]], *, require_baseline: bool = True) -> None:
    check_treatment_factors(rows)
    operational: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    matched: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    named: dict[tuple[Any, ...], list[float]] = defaultdict(list)

    for row in rows:
        rate = row["output_tokens_per_second"]
        if rate is None:
            continue
        if _is_operational_baseline(row):
            operational[operational_key(row)].append(float(rate))
        elif _is_matched_baseline(row):
            matched[matched_key(row)].append(float(rate))
        if row["method"] == "none" and row.get("deployment_id"):
            named[(operational_key(row), row["deployment_id"])].append(float(rate))

    operational_means = {key: _mean(values) for key, values in operational.items()}
    matched_means = {key: _mean(values) for key, values in matched.items()}
    missing: list[str] = []

    for row in rows:
        rate = row["output_tokens_per_second"]
        op_key = operational_key(row)
        operational_rate = operational_means.get(op_key)
        paired_id = row.get("paired_baseline_id")
        if paired_id:
            named_rate = named.get((op_key, paired_id))
            operational_rate = _mean(named_rate) if named_rate else None
        row["speedup_vs_baseline"] = (
            float(rate) / operational_rate if rate is not None and operational_rate else None
        )
        matched_rate = matched_means.get(matched_key(row))
        row["speedup_vs_matched_baseline"] = (
            float(rate) / matched_rate if rate is not None and matched_rate else None
        )
        if (
            require_baseline
            and row["method"] != "none"
            and operational_rate is None
        ):
            missing.append(
                f"{row.get('variant')} {row.get('workload')} c{row.get('concurrency')}"
            )

    if missing:
        raise ValueError(
            "no compatible operational baseline for: "
            + "; ".join(missing)
            + ". Refusing to pool unmatched method=none rows."
        )


def add_inferred_round_costs(rows: list[dict[str, Any]]) -> None:
    from apertus_bench.diagnostics import inferred_round_metrics

    for row in rows:
        row.update(
            inferred_round_metrics(
                row.get("completion_tokens"),
                row.get("accepted_tokens"),
                row.get("tpot_p50_ms"),
                row.get("drafts"),
            )
        )


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("no result cells found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
