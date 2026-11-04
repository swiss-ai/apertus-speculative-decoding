"""Summarize the immutable T4 replay and corrected cost projection.

Run the documented tree_sim --reproject command first. This script verifies
that acceptance aggregates did not change, then writes new summary files only.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from apertus_bench import tree_sim as ts


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best(rows: list[dict], concurrency: str) -> dict:
    return max(rows, key=lambda row: row["projected"][concurrency]["tok_s"])


def point(row: dict, concurrency: str) -> dict:
    return {
        "policy": row["policy"],
        "budget": row["budget"],
        "tokens_per_step": row["accepted_per_step"],
        **row["projected"][concurrency],
    }


def rate_ratio(a: dict, b: dict, concurrency: str) -> float:
    return a["projected"][concurrency]["tok_s"] / b["projected"][concurrency]["tok_s"]


def comparison(report: dict) -> list[dict]:
    comparisons = []
    for workload in ("opb", "chat"):
        rows = [row for row in report["rows"] if row["workload"] == workload]
        for concurrency in ("1", "8", "64", "128", "256"):
            chain = best([r for r in rows if r["policy"].startswith("chain-")], concurrency)
            single = best(
                [r for r in rows if r["policy"].startswith(("chain-", "tree-"))], concurrency
            )
            multi = best(
                [r for r in rows if not r["policy"].startswith(("chain-", "tree-"))], concurrency
            )
            neural_pair = best(
                [r for r in rows if r["policy"] in ("graft-all", "graft-dspark+eagle")], concurrency
            )
            comparisons.append(
                {
                    "workload": workload,
                    "concurrency": int(concurrency),
                    "best_chain": point(chain, concurrency),
                    "best_single": point(single, concurrency),
                    "best_multi": point(multi, concurrency),
                    "best_two_neural_drafters": point(neural_pair, concurrency),
                    "multi_vs_chain": rate_ratio(multi, chain, concurrency),
                    "multi_vs_single": rate_ratio(multi, single, concurrency),
                    "two_neural_vs_single": rate_ratio(neural_pair, single, concurrency),
                }
            )
    return comparisons


class TailSensitivity(ts.CostModel):
    tail_scale: float = 1.0

    def verify(self, tokens: float) -> float:
        end = self.verify_tokens[-1]
        if tokens <= end:
            return super().verify(tokens)
        return super().verify(end) + self.tail_scale * self.description()[
            "verify_tail_ms_per_token"
        ] * (tokens - end)


def main() -> None:
    directory = Path(__file__).resolve().parent
    root = directory.parent
    source = root / "raw/full-20261024T1000Z/tree-sim.json"
    corrected = directory / "tree-sim.json"
    table = root / "inputs/cost-table.json"
    if (directory / "summary.json").exists():
        raise SystemExit("summary already exists; use a fresh analysis version")
    original, report = json.loads(source.read_text()), json.loads(corrected.read_text())
    assert report["reanalysis"]["source_sha256"] == digest(source)
    assert len(original["rows"]) == len(report["rows"]) == 441
    for before, after in zip(original["rows"], report["rows"], strict=True):
        assert {k: v for k, v in before.items() if k != "projected"} == {
            k: v for k, v in after.items() if k != "projected"
        }
        assert all(p["step_ms"]["verify"] > 0 for p in after["projected"].values())
    comparisons = comparison(report)
    gate = []
    for workload in ("opb", "chat"):
        rows = [r for r in report["rows"] if r["workload"] == workload]
        for budget, threshold in ((4, 0.10), (16, 0.15)):
            eligible = [r for r in rows if r["budget"] <= budget]
            chain = max(
                (r for r in eligible if r["policy"].startswith("chain-")),
                key=lambda r: r["accepted_per_step"],
            )
            single = max(
                (r for r in eligible if r["policy"].startswith(("chain-", "tree-"))),
                key=lambda r: r["accepted_per_step"],
            )
            multi = max(
                (r for r in eligible if not r["policy"].startswith(("chain-", "tree-"))),
                key=lambda r: r["accepted_per_step"],
            )
            gain = multi["accepted_per_step"] / chain["accepted_per_step"] - 1
            gate.append(
                {
                    "workload": workload,
                    "budget_cap": budget,
                    "threshold": threshold,
                    "chain_tokens_per_step": chain["accepted_per_step"],
                    "single_tree_tokens_per_step": single["accepted_per_step"],
                    "multi_tokens_per_step": multi["accepted_per_step"],
                    "multi_vs_chain_gain": gain,
                    "multi_vs_single_gain": multi["accepted_per_step"] / single["accepted_per_step"]
                    - 1,
                    "passed": gain >= threshold,
                }
            )
    summaries = sorted((root / "raw/full-20261024T1000Z/candidates").glob("*.summary.json"))
    shards = [json.loads(p.read_text()) for p in summaries]
    sensitivity = {}
    for scale in (0.5, 1.5):
        model = TailSensitivity.from_table(table)
        model.tail_scale = scale
        sensitivity[str(scale)] = comparison(ts.reproject(original, model, [1, 8, 64, 128, 256]))
    for alternative in sensitivity.values():
        for base, variant in zip(comparisons, alternative, strict=True):
            for endpoint in ("best_chain", "best_single", "best_multi"):
                assert (base[endpoint]["policy"], base[endpoint]["budget"]) == (
                    variant[endpoint]["policy"],
                    variant[endpoint]["budget"],
                )
    result = {
        "claim_limit": (
            "offline candidate replay and modeled decode-only throughput; no tree-serving benchmark"
        ),
        "source_report_sha256": digest(source),
        "corrected_report_sha256": digest(corrected),
        "sequences": report["sequences"],
        "anchors": sum(s["anchors"] for s in shards),
        "hf_argmax_agreement_range_across_shards": [
            min(s["hf_argmax_agreement"] for s in shards),
            max(s["hf_argmax_agreement"] for s in shards),
        ],
        "comparisons": comparisons,
        "G4_acceptance_screen": gate,
        "gate_interpretation": (
            "low-load chain comparison passes, but does not isolate multi-drafter value"
        ),
        "implementation_verdict": (
            "do not implement multi-drafter serving; "
            "single-DSpark-tree low-load headroom is unvalidated"
        ),
        "invalidated_original_projection": {
            "reason": "last downward timing segment extrapolated to negative verification time",
            "minimum_verify_ms": min(
                p["step_ms"]["verify"] for r in original["rows"] for p in r["projected"].values()
            ),
            "negative_verify_points": sum(
                p["step_ms"]["verify"] < 0
                for r in original["rows"]
                for p in r["projected"].values()
            ),
        },
        "tail_sensitivity": sensitivity,
        "checks": {
            "acceptance_aggregates_unchanged": True,
            "all_corrected_verify_times_positive": True,
            "selected_best_chain_single_multi_within_measured_verify_range": all(
                not c[endpoint]["verify_extrapolated"]
                for c in comparisons
                for endpoint in ("best_chain", "best_single", "best_multi")
            ),
            "best_settings_unchanged_under_half_and_1_5x_tail_slope": True,
        },
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (directory / "comparisons.csv").open("x", newline="") as handle:
        fields = [
            "workload",
            "concurrency",
            "endpoint",
            "policy",
            "budget",
            "tokens_per_step",
            "tok_s",
            "speedup_vs_plain",
            "verify_extrapolated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            for endpoint in ("best_chain", "best_single", "best_multi", "best_two_neural_drafters"):
                writer.writerow(
                    {
                        "workload": row["workload"],
                        "concurrency": row["concurrency"],
                        "endpoint": endpoint,
                        **{k: v for k, v in row[endpoint].items() if k in fields},
                    }
                )
    provenance = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): digest(p) for p in [Path(__file__), Path(ts.__file__), source, table, corrected]
        },
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checks": result["checks"], "gate": gate, "anchors": result["anchors"]}))


if __name__ == "__main__":
    main()
