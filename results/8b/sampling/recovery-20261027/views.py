#!/usr/bin/env python3
"""Produce descriptive sampling effects and report tables from validated cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.analysis / "summary.json"
    data = json.loads(source.read_text())
    absolute, pairs = data["absolute"], data["matched"]
    cells = {
        (r["run"], r["arm"], r["workload"], r["sampling"], r["concurrency"]): r for r in absolute
    }
    effects = []
    for row in absolute:
        if row["sampling"] not in ["t1-k50", "t1-k20", "t1-k5"]:
            continue
        baseline = cells[row["run"], row["arm"], row["workload"], "t1-kall", row["concurrency"]]
        for key, expected in [("temperature", 1.0), ("top_p", 1.0), ("seed_base", 1)]:
            assert row[key] == baseline[key] == expected
        for key in ["requests", "min_output_cap", "max_output_cap", "mean_output_cap"]:
            assert row[key] == baseline[key]
        effect = {
            key: row[key] for key in ["run", "mode", "arm", "draft_k", "workload", "concurrency"]
        }
        effect["sampling_top_k"] = row["sampling_top_k"]
        for key in [
            "output_tok_s",
            "mean_output_tokens",
            "length_finish_rate",
            "gpu_s_per_million_output_tokens",
            "e2e_mean_ms",
            "tpot_p50_ms",
            "accepted_per_step",
        ]:
            effect["unfiltered_" + key] = baseline[key]
            effect["filtered_" + key] = row[key]
        effect["throughput_ratio"] = row["output_tok_s"] / baseline["output_tok_s"]
        effect["output_length_ratio"] = row["mean_output_tokens"] / baseline["mean_output_tokens"]
        effect["unfiltered_cell"] = baseline["cell_path"]
        effect["filtered_cell"] = row["cell_path"]
        effects.append(effect)
    policy = []
    for mode in ["off", "on"]:
        for workload in sorted({r["workload"] for r in absolute if r["mode"] == mode}):
            for arm in sorted(
                {p["arm"] for p in pairs if p["mode"] == mode and p["workload"] == workload}
            ):
                concurrencies = sorted(
                    {
                        p["concurrency"]
                        for p in pairs
                        if p["mode"] == mode and p["workload"] == workload
                    }
                )
                for concurrency in concurrencies:
                    subset = [
                        p
                        for p in pairs
                        if p["mode"] == mode
                        and p["workload"] == workload
                        and p["arm"] == arm
                        and p["concurrency"] == concurrency
                    ]
                    assert len(subset) == 6
                    policy.append(
                        {
                            "mode": mode,
                            "workload": workload,
                            "arm": arm,
                            "concurrency": concurrency,
                            "minimum_throughput_ratio_to_matched_plain": min(
                                p["throughput_ratio"] for p in subset
                            ),
                            "maximum_throughput_ratio_to_matched_plain": max(
                                p["throughput_ratio"] for p in subset
                            ),
                            "tested_sampling_settings": len(subset),
                        }
                    )
    selected_keys = [
        ("off", "dspark-off-k7", "opb-default", 1),
        ("off", "dspark-off-k3", "arena-hard-clean-default", 64),
        ("off", "dspark-off-k3", "arena-hard-clean-default", 256),
        ("off", "dspark-off-k7", "arena-hard-clean-t1-kall", 256),
        ("on", "dspark-on-k7", "opb-think-greedy", 1),
        ("on", "dspark-on-k3", "arena-hard-clean-think-default", 64),
        ("on", "dspark-on-k3", "opb-think-default", 256),
        ("on", "dspark-on-k3", "arena-hard-clean-think-default", 256),
        ("on", "dspark-on-k3", "arena-hard-clean-think-t1-kall", 256),
    ]
    selected = [
        next(p for p in pairs if (p["mode"], p["arm"], p["label"], p["concurrency"]) == key)
        for key in selected_keys
    ]
    table = [
        "| Mode / workload / sampling | C | DSpark draft k | Output tok/s, plain → spec | "
        "Ratio | GPU-s / M output tokens, plain → spec | Mean request s, plain → spec | "
        "Accepted / step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in selected:
        table.append(
            f"| {p['mode']} / {p['workload']} / {p['sampling']} | {p['concurrency']} | "
            f"{p['draft_k']} | {p['plain_output_tok_s']:,.0f} → "
            f"{p['candidate_output_tok_s']:,.0f} | {p['throughput_ratio']:.2f}× | "
            f"{p['plain_gpu_s_per_million_output_tokens']:,.1f} → "
            f"{p['candidate_gpu_s_per_million_output_tokens']:,.1f} | "
            f"{p['plain_e2e_mean_ms'] / 1000:.2f} → {p['candidate_e2e_mean_ms'] / 1000:.2f} | "
            f"{p['accepted_per_step']:.2f} |"
        )
    args.output.mkdir(parents=True)
    write_csv(args.output / "sampling-topk-effects.csv", effects)
    write_csv(args.output / "draft-load-ranges.csv", policy)
    write_csv(args.output / "selected-comparisons.csv", selected)
    (args.output / "selected-table.md").write_text("\n".join(table) + "\n")
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "summary_source": str(source),
                "summary_sha256": checksum(source),
                "views_source_sha256": checksum(Path(__file__)),
                "sampling_comparisons": len(effects),
                "load_ranges": len(policy),
                "claim_limit": "Descriptive comparisons of a single deployment per arm; "
                "sampling comparisons change the target distribution and have no quality evidence.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"sampling_comparisons": len(effects), "load_ranges": len(policy)}))


if __name__ == "__main__":
    main()
