"""Rebuild this preflight diagnostic without modifying raw measurements.

From the isolated repository root:
  PYTHONPATH=src python3 <this script> --output NEW_REPORT.json

The output must not already exist. This is a derived preflight diagnostic,
not a screen/confirmation gate and not an independent deployment repeat.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path

import apertus_bench.drafter_kv_evidence as evidence
import apertus_bench.drafter_kv_experiment as controller
from apertus_bench.drafter_kv_evidence import Deployment, memory_evidence, trace_snapshots
from apertus_bench.drafter_kv_experiment import digest


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a derived report")
    allocation = Path(__file__).resolve().parent.parent
    results = allocation / "results"
    arms = ["native-a", "native-b", "full-custom", "importance512"]
    paths = {allocation / "spec.yaml", allocation / "launch-provenance.json",
             allocation / "submit-command.json", allocation / "job-registration.json",
             allocation / "calibrated-correctness.json", results / "sweep-status.json"}
    deployments = [Deployment("preflight04", name,
                              "importance512" if name == "importance512" else "native2048",
                              results / name, None)
                   for name in ["native-a", "native-b", "importance512"]]
    traces = {d.identity: trace_snapshots(d) for d in deployments}
    physical = memory_evidence(deployments, "importance512", "native2048", [], traces)
    # First completed prefills can temporarily own >window pages in native.
    # Keep a separately labelled, post-hoc bounded-cache diagnostic; retain the
    # unfiltered evidence above exactly as produced by the evidence module.
    bounded_traces = copy.deepcopy(traces)
    for deployment in deployments:
        if deployment.policy == "native2048":
            trace = bounded_traces[deployment.identity]
            trace["samples"] = [s for s in trace["samples"] if all(
                pages <= 130 for group in s["groups"] if group["drafter"]
                for pages in group["pages_by_request"])]
    bounded = memory_evidence(deployments, "importance512", "native2048", [], bounded_traces)
    runtime = {}
    for arm in arms:
        path = results / arm / "draft-kv.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        workers = [r for r in records if r["event"] == "step"]
        releases = [r for r in records if r["event"] == "bootstrap_release"]
        runtime[arm] = {
            "first_completed_worker_events": sum(bool(r.get("first_completed_prompt_request_ids"))
                                                 for r in workers),
            "release_events": len(releases),
            "positive_release_events": sum(r["released_pages"] > 0 for r in releases),
            "total_released_pages": sum(r["released_pages"] for r in releases),
            "graph_captured_max": max((r.get("drafter_graphs_captured", 0) for r in workers),
                                      default=0),
            "graph_replays_max": max((r.get("drafter_graph_replays", 0) for r in workers), default=0),
            "graph_metadata_peak_bytes": max((r.get("draft_graph_metadata_bytes", 0) for r in workers),
                                             default=0),
            "metadata_peak_bytes": max((r.get("draft_metadata_bytes", 0) for r in workers), default=0),
            "note": "worker-only peaks may include graph profiling; ownership evidence joins allocator",
        }
        paths.update([path, results / arm / "status.json"])
    performance = []
    git_fields = set()
    for workload in ["chat", "code", "summarization"]:
        for concurrency in [1, 8]:
            rows = {}
            for arm in arms:
                cell = results / arm / workload / workload / f"c{concurrency}"
                summary = json.loads((cell / "summary.json").read_text())
                metadata = json.loads((cell / "metadata.json").read_text())
                git_fields.add(metadata["harness_git_revision"])
                rows[arm] = {
                    "output_tokens_per_second": summary["tokens"]["output_tokens_per_second"],
                    "mean_acceptance_length": summary["speculative_decoding"]["mean_acceptance_length"],
                    "success_rate": summary["requests"]["success_rate"],
                    "requests": summary["requests"]["attempted"],
                    "warmup_failed": summary["warmup"]["failed"],
                    "ttft_p95_ms": summary["latency_ms"]["ttft"]["p95"],
                    "tpot_p95_ms": summary["latency_ms"]["tpot"]["p95"],
                    "cell_settings": metadata["cell"],
                    "workload_file_sha256": metadata["workload_file_sha256"],
                }
                paths.update(p for p in cell.iterdir() if p.is_file())
                paths.add(results / arm / workload / "loadtest-summary.json")
            denominator = {m: (rows["native-a"][m] + rows["native-b"][m]) / 2
                           for m in ["output_tokens_per_second", "mean_acceptance_length",
                                     "ttft_p95_ms", "tpot_p95_ms"]}
            ratios = {arm: {metric: rows[arm][metric] / base for metric, base in denominator.items()}
                      for arm in ["full-custom", "importance512"]}
            performance.append({
                "workload": workload, "concurrency": concurrency, "absolute": rows,
                "native_arithmetic_mean": denominator, "ratios_to_native_mean": ratios,
                "native_a_over_b_throughput": rows["native-a"]["output_tokens_per_second"]
                / rows["native-b"]["output_tokens_per_second"],
            })
    launch = json.loads((allocation / "launch-provenance.json").read_text())
    report = {
        "schema_version": 1, "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "method": "post_hoc_installed_runtime_preflight_diagnostic_v1",
        "allocation": "preflight-runtime-04", "slurm_job_id": "3723858",
        "raw_memory_evidence": physical, "trace_snapshots_and_coverage": traces,
        "bounded_native_cache_diagnostic": {
            "method": "post_hoc_native_owned_draft_pages_per_request_at_most130",
            "not_an_expansion_gate": True,
            "reason": "exclude transient initial native prefill ownership above its window",
            "evidence": bounded,
        },
        "runtime_trace_summary": runtime,
        "per_cell_performance": performance,
        "performance_method": "arithmetic mean of native-a/native-b absolute metrics per cell",
        "six_cell_geomean_throughput_ratio": {
            arm: geomean([r["ratios_to_native_mean"][arm]["output_tokens_per_second"]
                          for r in performance]) for arm in ["full-custom", "importance512"]},
        "provenance": {
            "launch_git_revision": launch.get("git_revision"),
            "raw_cell_harness_git_revisions": sorted(git_fields),
            "cwd_git_caveat": (
                "Cell harness_git_revision reflects the old environment workdir rather than the "
                "isolated PYTHONPATH source. Retain raw fields; launch-provenance/job command and "
                "overlay manifest identify actual code. A future launch could cd to the isolated "
                "repository before benchmarking; no launch change is made by this analysis."
            ),
            "analysis_sources": [source(Path(__file__)), source(Path(evidence.__file__)),
                                 source(Path(controller.__file__))],
            "raw_sources": [source(p) for p in sorted(paths)],
        },
        "limits": [
            "Two prompts per stratum; only2 or8 measured requests per cell, cap128; no performance conclusion.",
            "One node allocation; two native GPUs are technical controls, not independent node repeats.",
            "Full-coverage custom backend is materially slower despite similar acceptance; eviction alone does not explain the slowdown.",
            "First-completed-prefill snapshots can include transient native ownership; a separate bounded-cache diagnostic is labelled post hoc.",
            "Worst combined physical target/draft ratio in this probe narrowly misses5% reduction; no expansion is authorized here.",
            "This script performs local derived analysis only; it does not modify raw artifacts or submit experiments.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "memory": physical["memory"], "bounded_memory": bounded["memory"],
                      "six_cell_geomean_throughput_ratio": report["six_cell_geomean_throughput_ratio"]},
                     indent=2))


if __name__ == "__main__":
    main()
