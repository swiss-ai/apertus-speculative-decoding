"""Read-only paired analysis for the native-effort experiment.

Incomplete, invalid or unmatched cells never produce a proceed decision. Every
ratio uses a contemporaneous baseline in the same deployment block and effort.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

EFFORTS = ("low", "medium", "high")
IDENTITY = (
    "target_sha256",
    "target_revision",
    "target_tensor_parallel_size",
    "tokenizer_sha256",
    "template_sha256",
    "rendered_prompts_sha256",
    "container_sha256",
    "vllm_version",
    "slurm_node_id",
    "gpu_identity",
    "source_patch_sha256",
    "harmony_overlay_sha256",
    "harness_git_revision",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _decode_cost(records: list[dict[str, Any]]) -> float | None:
    if any(r.get("ttft_ms") is None or r.get("completion_tokens") is None for r in records):
        return None
    tokens = sum(max(r["completion_tokens"] - 1, 0) for r in records)
    return sum(r["e2e_ms"] - r["ttft_ms"] for r in records) / tokens if tokens else None


def _strata(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for workload in sorted({r["workload"] for r in records}):
        rows = [r for r in records if r["workload"] == workload]
        accepted = sum(
            (r.get("speculative_decoding") or {}).get("accepted_tokens", 0) for r in rows
        )
        rounds = sum((r.get("speculative_decoding") or {}).get("drafts", 0) for r in rows)
        groups[workload] = {
            "requests": len(rows),
            "e2e_p50_ms": median(r["e2e_ms"] for r in rows),
            "decode_ms_per_token": _decode_cost(rows),
            "accepted_draft_tokens_per_round": accepted / rounds if rounds else None,
        }
    return groups


def build_report(campaign: Path, *, phase: str = "phase1") -> dict[str, Any]:
    lock = _read(campaign / "inputs/lock.json")
    spec = lock["spec"]
    variants = (
        ("plain", "dflash")
        if phase == "phase1"
        else (
            "plain",
            "low",
            "medium",
            "high",
            "mixed",
        )
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "decision": "pending",
        "campaign": str(campaign),
        "rows": [],
        "problems": [],
        "thresholds": {
            k: spec[k]
            for k in (
                "acceptance_change_threshold",
                "max_cap_fraction",
                "specialist_latency_threshold",
            )
        },
    }
    cells = {}
    for block in range(spec["repeats"]):
        block_dir = campaign / phase / f"block-{block}"
        block_status = block_dir / "status.json"
        if not block_status.exists() or _read(block_status).get("status") != "complete":
            result["problems"].append(f"block {block} is incomplete or failed")
        for name in variants:
            for effort in EFFORTS:
                cell = block_dir / name / "cells" / effort
                status = cell / "status.json"
                if not status.exists() or _read(status).get("status") != "complete":
                    result["problems"].append(f"missing/invalid cell: {block}/{name}/{effort}")
                    continue
                metadata = _read(cell / "metadata.json")
                summary = _read(cell / "summary.json")
                records = [
                    json.loads(line)
                    for line in (cell / "requests.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                settings = metadata["cell"]
                generation = settings["generation"]
                if not summary.get("validity", {}).get("valid"):
                    result["problems"].append(f"cell failed scientific validity: {cell}")
                if (
                    settings["concurrency"] != 1
                    or settings["repeat"] != block
                    or generation["reasoning_effort"] != effort
                ):
                    result["problems"].append(f"mislabelled experiment cell: {cell}")
                if (
                    len(records) != spec["expected_prompts"]
                    or [r["prompt_id"] for r in records] != lock["workload"]["prompt_ids"]
                ):
                    result["problems"].append(f"prompt set/order differs: {cell}")
                for key in ("temperature", "top_p", "top_k", "min_p", "max_tokens"):
                    if generation[key] != spec[key]:
                        result["problems"].append(f"generation {key} differs: {cell}")
                if generation["seed"] != block or generation["ignore_eos"]:
                    result["problems"].append(f"seed or termination differs: {cell}")
                counters = summary["speculative_decoding"]
                rounds = counters.get("drafts")
                row = {
                    "block": block,
                    "variant": name,
                    "effort": effort,
                    "path": str(cell),
                    "e2e_p50_ms": summary["latency_ms"]["e2e"]["p50"],
                    "decode_ms_per_token": _decode_cost(records),
                    "output_tokens_per_second": summary["tokens"]["output_tokens_per_second"],
                    "acceptance_rate": counters.get("acceptance_rate"),
                    "accepted_draft_tokens_per_round": counters.get("accepted_tokens", 0) / rounds
                    if rounds
                    else None,
                    "channel_tokens": summary.get("channel_tokens"),
                    "cap_fraction": summary["validity"]["cap_fraction"],
                    "strata": _strata(records),
                }
                cells[block, name, effort] = metadata, row
                result["rows"].append(row)
    expected = spec["repeats"] * len(variants) * len(EFFORTS)
    if len(cells) != expected:
        return result
    # Node and GPU can differ between blocks. Model, renderer, code, container
    # and generation settings must remain fixed for the entire campaign.
    reference_meta = cells[0, "plain", "low"][0]
    for (block, name, effort), (metadata, _) in cells.items():
        for key in set(IDENTITY) - {"slurm_node_id", "gpu_identity"}:
            if metadata["extra"].get(key) != reference_meta["extra"].get(key):
                result["problems"].append(
                    f"campaign identity changed: {key}/{block}/{name}/{effort}"
                )
    for (block, name, effort), (metadata, row) in cells.items():
        baseline_meta, baseline = cells[block, "plain", effort]
        for key in IDENTITY:
            if not metadata["extra"].get(key) or metadata["extra"][key] != baseline_meta[
                "extra"
            ].get(key):
                result["problems"].append(f"unmatched {key}: {block}/{name}/{effort}")
        if (
            metadata["workload_file_sha256"] != baseline_meta["workload_file_sha256"]
            or metadata["prompt_ids"] != baseline_meta["prompt_ids"]
        ):
            result["problems"].append(f"unmatched corpus: {block}/{name}/{effort}")
        row["latency_speedup_vs_plain"] = baseline["e2e_p50_ms"] / row["e2e_p50_ms"]
        if row["decode_ms_per_token"] and baseline["decode_ms_per_token"]:
            row["decode_speedup_vs_plain"] = (
                baseline["decode_ms_per_token"] / row["decode_ms_per_token"]
            )
    if result["problems"]:
        result["decision"] = "invalid"
        return result
    if phase == "phase1":
        changes = {}
        for effort in ("low", "high"):
            values = []
            for block in range(3):
                reference = cells[block, "dflash", "medium"][1]["accepted_draft_tokens_per_round"]
                value = cells[block, "dflash", effort][1]["accepted_draft_tokens_per_round"]
                if not reference or value is None:
                    result["problems"].append("acceptance unavailable for effort comparison")
                    result["decision"] = "invalid"
                    return result
                values.append(value / reference - 1)
            changes[effort] = values
        threshold = spec["acceptance_change_threshold"]
        sensitive = [
            effort
            for effort, values in changes.items()
            if all(v >= threshold for v in values) or all(v <= -threshold for v in values)
        ]
        wins = [
            effort
            for effort in EFFORTS
            if all(cells[b, "dflash", effort][1]["latency_speedup_vs_plain"] > 1 for b in range(3))
        ]
        losses = [
            effort
            for effort in EFFORTS
            if all(cells[b, "dflash", effort][1]["latency_speedup_vs_plain"] < 1 for b in range(3))
        ]
        result.update(
            {
                "decision": "proceed" if sensitive or (wins and losses) else "stop",
                "relative_acceptance_changes_vs_medium": changes,
                "sensitive_efforts": sensitive,
                "consistent_wins": wins,
                "consistent_losses": losses,
                "interpretation": "Sensitivity motivates testing weights; it does not prove "
                "specialists are necessary. A stop decision is limited to this head and probe.",
            }
        )
    else:
        useful = []
        for effort in EFFORTS:
            gains = []
            normalized = []
            for block in range(3):
                specialist = cells[block, effort, effort][1]
                mixed = cells[block, "mixed", effort][1]
                gains.append(1 - specialist["e2e_p50_ms"] / mixed["e2e_p50_ms"])
                normalized.append(
                    bool(
                        specialist["decode_ms_per_token"]
                        and mixed["decode_ms_per_token"]
                        and specialist["decode_ms_per_token"] < mixed["decode_ms_per_token"]
                    )
                )
            if all(g >= spec["specialist_latency_threshold"] for g in gains) and all(normalized):
                useful.append(effort)
        result.update(
            {
                "decision": "specialists_help" if useful else "retain_shared",
                "useful_specialist_efforts": useful,
                "interpretation": "Report storage/training costs before recommending "
                "multiple heads; "
                "this does not evaluate adapters, offloading or production capacity.",
            }
        )
    return result


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Native reasoning effort {report['phase']} results",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Block | Drafter | Effort | Accepted draft tokens/round | Latency speedup | "
        "Decode speedup | Cap fraction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.3f}"

    for row in report["rows"]:
        lines.append(
            f"| {row['block']} | {row['variant']} | {row['effort']} | "
            f"{fmt(row['accepted_draft_tokens_per_round'])} | "
            f"{fmt(row.get('latency_speedup_vs_plain'))} | "
            f"{fmt(row.get('decode_speedup_vs_plain'))} | {fmt(row['cap_fraction'])} |"
        )
    lines += [
        "",
        report.get("interpretation", "Incomplete or invalid comparisons have no verdict."),
    ]
    if report["problems"]:
        lines += ["", *[f"- {problem}" for problem in report["problems"]]]
    return "\n".join(lines) + "\n"
