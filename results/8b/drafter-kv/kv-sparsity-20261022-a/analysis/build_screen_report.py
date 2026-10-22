"""Report the two completed screen replacements without modifying raw artifacts.

Run only after both repeats and the screen-v1 evidence/gate reports exist:
    python3 results/8b/drafter-kv/kv-sparsity-20261022-a/analysis/build_screen_report.py

Requires PyYAML. Outputs are exclusively created in analysis/screen-report-v1.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

ALLOCATIONS = ("screen-01-r2", "screen-02-r2")
NATIVE = "native2048"
METRICS = ("throughput_tok_s", "success_rate", "acceptance_length", "ttft_p95_ms", "tpot_p95_ms")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mapping() -> dict | None:
    source = os.environ.get("APERTUS_DRAFT_KV_SOURCE_ROOT")
    target = os.environ.get("APERTUS_DRAFT_KV_ARTIFACT_ROOT")
    if source is None and target is None:
        return None
    if not source or not target or not Path(source).is_absolute() or not Path(target).is_absolute():
        raise ValueError(
            "artifact mapping requires both nonempty absolute root environment variables")
    return {"source_root": source, "artifact_root": target,
            "method": "lexical_relative_to_prefix; canonical artifact contents unchanged"}


class Inputs:
    def __init__(self) -> None:
        self.paths: set[Path] = set()

    def read(self, path: Path, *, is_yaml: bool = False) -> dict:
        self.paths.add(path)
        return yaml.safe_load(path.read_text()) if is_yaml else json.loads(path.read_text())

    def manifest(self) -> list[dict]:
        return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(self.paths)]


def number(value: object, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"missing or nonfinite metric: {label}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"invalid metric: {label}")
    return float(value)


def geomean(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        raise ValueError("paired geometric means require positive ratios")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def collect(campaign: Path, inputs: Inputs) -> tuple[list[dict], dict]:
    rows, revisions = [], {}
    for allocation in ALLOCATIONS:
        parent = campaign / "allocations" / allocation
        results = parent / "results"
        spec = inputs.read(parent / "spec.yaml", is_yaml=True)
        status = inputs.read(results / "sweep-status.json")
        registration = inputs.read(parent / "job-registration.json")
        launch = inputs.read(parent / "launch-provenance.json")
        revisions[allocation] = launch["git_revision"]
        if (set(status) != {a["name"] for a in spec["arms"]}
                or any(s != "ok" for s in status.values())):
            raise ValueError(f"{allocation}: expected every screen arm to complete successfully")
        for name in ["job.sh", "draft-kv-overlay-manifest.json", "drafter-kv-applied.patch"]:
            path = results / name
            if path.is_file():
                inputs.paths.add(path)
        for arm in spec["arms"]:
            policy = arm["metadata"]["kv_policy"]
            arm_status = inputs.read(results / arm["name"] / "status.json")
            if arm_status["status"] != "ok":
                raise ValueError(f"{allocation}/{arm['name']}: incomplete arm")
            for test in spec["loadtests"]:
                workload, label = test["workload"], test.get("label", test["workload"])
                corpus = next(c for c in spec["drafter_kv"]["corpora"].values()
                              if c["path"] == test["workloads_file"])
                for concurrency in test["concurrencies"]:
                    cell = results / arm["name"] / label / workload / f"c{concurrency}"
                    summary = inputs.read(cell / "summary.json")
                    metadata = inputs.read(cell / "metadata.json")
                    if (metadata["cell"]["workload"] != workload
                            or metadata["cell"]["concurrency"] != concurrency
                            or metadata["extra"]["kv_policy"] != policy
                            or metadata["workload_file_sha256"] != corpus["sha256"]):
                        raise ValueError(f"cell identity or frozen corpus mismatch: {cell}")
                    row = {
                        "node_block": allocation, "job_id": registration["job_id"],
                        "node_id": metadata["extra"].get("slurm_node_id"),
                        "gpu_ids": ",".join(map(str, arm_status["gpus"])),
                        "deployment_id": metadata["extra"]["deployment_id"], "policy": policy,
                        "workload": workload, "concurrency": int(concurrency),
                        "throughput_tok_s": summary["tokens"]["output_tokens_per_second"],
                        "success_rate": summary["requests"]["success_rate"],
                        "acceptance_length": summary["speculative_decoding"].get(
                            "mean_acceptance_length"),
                        "ttft_p95_ms": summary["latency_ms"]["ttft"]["p95"],
                        "tpot_p95_ms": summary["latency_ms"]["tpot"]["p95"],
                        "preemptions": summary["load"]["preemptions"],
                        "requests": summary["requests"]["attempted"],
                        "successful_requests": summary["requests"]["successful"],
                        "failed_requests": summary["requests"]["failed"],
                        "warmup_requests": summary["warmup"]["requests"],
                        "warmup_failed": summary["warmup"]["failed"],
                        "corpus_sha256": metadata["workload_file_sha256"],
                        "generation": metadata["cell"]["generation"],
                        "raw_harness_revision": metadata["harness_git_revision"],
                    }
                    for metric in METRICS:
                        row[metric] = number(row[metric], f"{cell}/{metric}", positive=True)
                    row["preemptions"] = number(row["preemptions"], f"{cell}/preemptions")
                    rows.append(row)
    return rows, revisions


def paired(rows: list[dict]) -> tuple[list[dict], dict]:
    index = {(r["node_block"], r["policy"], r["workload"], r["concurrency"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("screen contains duplicate policy cells within one allocation")
    policies = sorted({r["policy"] for r in rows} - {NATIVE})
    cells = sorted({(r["workload"], r["concurrency"]) for r in rows})
    ratios = []
    for policy in policies:
        for workload, concurrency in cells:
            blocks = []
            for block in ALLOCATIONS:
                base = index[(block, NATIVE, workload, concurrency)]
                row = index[(block, policy, workload, concurrency)]
                for field in ["corpus_sha256", "generation", "requests"]:
                    if row[field] != base[field]:
                        raise ValueError(
                            f"unpaired {field}: {block}/{policy}/{workload}/c{concurrency}")
                blocks.append({"node_block": block,
                               **{m: row[m] / base[m] for m in METRICS}})
            ratios.append({"policy": policy, "workload": workload, "concurrency": concurrency,
                           "node_ratios": blocks,
                           **{f"{m}_ratio": geomean([b[m] for b in blocks]) for m in METRICS}})
    scores = {}
    for policy in policies:
        selected = [r for r in ratios if r["policy"] == policy and r["concurrency"] in {32, 256}]
        scores[policy] = {
            "geomean_throughput_ratio_c32_c256": geomean([
                b["throughput_tok_s"] for r in selected for b in r["node_ratios"]]),
            "per_node_scores": {block: geomean([
                b["throughput_tok_s"] for r in selected for b in r["node_ratios"]
                if b["node_block"] == block]) for block in ALLOCATIONS},
        }
    return ratios, scores


def write_json(path: Path, data: dict) -> None:
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    analysis = Path(__file__).resolve().parent
    campaign = analysis.parent
    output = analysis / "screen-report-v1"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output}")
    artifact_mapping = mapping()
    inputs = Inputs()
    # Missing final evidence is an error; do not manufacture pass/fail flags.
    gate = inputs.read(analysis / "screen-gate-v1.json")
    evidence = inputs.read(analysis / "screen-v1/evidence.json")
    memory = inputs.read(analysis / "screen-v1/memory.json")
    correctness = inputs.read(analysis / "screen-v1/correctness-controls.json")
    if any(report["phase"] != "screen" for report in [gate, evidence, memory, correctness]):
        raise ValueError("all final gate/evidence inputs must belong to the screen phase")
    if set(gate["selected_allocations"]) != set(ALLOCATIONS):
        raise ValueError("final gate belongs to different screen attempts")
    rows, revisions = collect(campaign, inputs)
    ratios, scores = paired(rows)
    final = {}
    for policy in sorted(scores):
        decision = gate["decisions"][policy]
        policy_evidence = evidence["policies"][policy]
        final[policy] = {
            "stable_and_storage_passed": decision["stable_and_storage_passed"],
            "failure_reasons": decision["failure_reasons"],
            "promising": decision["promising"],
            "memory": memory["policies"][policy]["memory"],
            "memory_passed": memory["policies"][policy]["passed"],
            "matched_memory_strata": len(memory["policies"][policy]["matches"]),
            "correctness_passed": decision["correctness_controls_passed"],
            "correctness_artifact_passed": correctness["policies"][policy]["passed"],
            "correctness_failure_reasons": correctness["policies"][policy]["failure_reasons"],
            "evidence_correctness_passed": policy_evidence["correctness_controls_passed"],
        }
        if len({final[policy][key] for key in ["correctness_passed", "correctness_artifact_passed",
                                              "evidence_correctness_passed"]}) != 1:
            raise ValueError(f"per-policy correctness inputs disagree: {policy}")
        if not math.isclose(scores[policy]["geomean_throughput_ratio_c32_c256"],
                            decision["score_c32_c256"], rel_tol=1e-10):
            raise ValueError(f"computed high-C score disagrees with source gate: {policy}")
    inputs.paths.add(Path(__file__))
    provenance = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(), "script_sha256": sha256(Path(__file__)),
        "source_revision": next(iter(set(revisions.values()))) if len(set(revisions.values())) == 1
        else None,
        "launch_source_revisions": revisions,
        "raw_harness_revisions": sorted({r["raw_harness_revision"] for r in rows}),
        "artifact_path_mapping": artifact_mapping,
        "input_artifact_path_mappings": {name: report.get("artifact_path_mapping")
                                        for name, report in [("gate", gate), ("evidence", evidence),
                                                             ("memory", memory),
                                                             ("correctness", correctness)]},
        "inputs": inputs.manifest(),
        "method": "geometric mean of within-allocation candidate/native ratios across two nodes",
        "high_C_method": (
            "geometric mean of paired throughput ratios over all workloads at C32/C256"),
        "uncertainty": "two independent allocations; no confidence interval is inferred",
        "physical_node_id": (
            "null when absent in raw cell metadata; allocation identifies node block"),
        "raw_files_unchanged": True,
    }
    summary = {"phase": "screen", "allocations": list(ALLOCATIONS), "absolute": rows,
               "paired_ratios": ratios, "high_C_scores": scores,
               "final_gate": {"passed": gate["passed"], "candidate": gate["candidate"],
                              "policies": final},
               "superseded_attempt_history": gate["superseded_attempt_history"]}
    lines = ["# Paired screen measurements", "",
             "Ratios are paired candidate/native geometric means across two allocations.",
             "Above 1 means more throughput/acceptance or higher latency. No CI is inferred.",
             "", "| Workload | C | Policy | Throughput | Acceptance | p95 TTFT | p95 TPOT |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(ratios, key=lambda r: (r["workload"], r["concurrency"], r["policy"])):
        lines.append(f"| {row['workload']} | {row['concurrency']} | {row['policy']} | "
                     f"{row['throughput_tok_s_ratio']:.4f} | "
                     f"{row['acceptance_length_ratio']:.4f} | "
                     f"{row['ttft_p95_ms_ratio']:.4f} | {row['tpot_p95_ms_ratio']:.4f} |")
    lines += ["", "| Policy | High-C throughput score | Draft storage ratio | Correctness | Gate |",
              "| --- | ---: | ---: | --- | --- |"]
    for policy in sorted(scores):
        item = final[policy]
        storage = item["memory"]["draft_storage_ratio"]
        storage_text = f"{storage:.4f}" if storage is not None else "unavailable"
        lines.append(f"| {policy} | {scores[policy]['geomean_throughput_ratio_c32_c256']:.4f} | "
                     f"{storage_text} | {item['correctness_passed']} | "
                     f"{item['stable_and_storage_passed']} |")
    lines += ["", f"Source gate: passed={gate['passed']}, candidate={gate['candidate']}.",
              "Absolutes, counts, preemptions and input hashes are in CSV/JSON.", ""]
    # All inputs and consistency checks have passed before claiming the output directory.
    output.mkdir(exist_ok=False)
    absolute_csv = [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                     for k, v in row.items()} for row in rows]
    ratio_csv = [{k: v for k, v in row.items() if k != "node_ratios"} for row in ratios]
    write_csv(output / "absolute.csv", absolute_csv)
    write_csv(output / "paired-ratios.csv", ratio_csv)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"output": str(output), "absolute_cells": len(rows),
                      "paired_cells": len(ratios), "source_gate_passed": gate["passed"]}))


if __name__ == "__main__":
    main()
