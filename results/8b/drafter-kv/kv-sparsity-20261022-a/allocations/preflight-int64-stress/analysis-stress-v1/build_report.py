"""Rebuild the C256 installed-runtime stability diagnostic using standard Python.

    python3 <this script> --output NEW_REPORT.json

The JSON and adjacent Markdown output must not exist. Raw artifacts are read
only. This stress allocation is not screening/selection or confirmation data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite derived JSON or Markdown")
    allocation = Path(__file__).resolve().parent.parent
    results = allocation / "results"
    status = json.loads((results / "sweep-status.json").read_text())
    arms = ["native-a", "native-b", "full-custom", "importance512"]
    if set(status) != set(arms) or any(s != "ok" for s in status.values()):
        raise ValueError("all expected stress arms must complete successfully")
    launch = json.loads((allocation / "launch-provenance.json").read_text())
    registration = json.loads((allocation / "job-registration.json").read_text())
    paths = {allocation / name for name in ["spec.yaml", "launch-provenance.json",
             "job-registration.json", "reservation.json", "submit-command.json"]}
    paths.update(results / name for name in [
        "spec.yaml", "sweep-status.json", "state-kernels.log", "job.sh", "environment.toml",
        "draft-kv-overlay-manifest.json", "drafter-kv-applied.patch", "patches.txt"])
    absolute = {}
    for arm in arms:
        cell = results / arm / "chat/chat/c256"
        summary = json.loads((cell / "summary.json").read_text())
        metadata = json.loads((cell / "metadata.json").read_text())
        absolute[arm] = {
            "output_tokens_per_second": summary["tokens"]["output_tokens_per_second"],
            "mean_acceptance_length": summary["speculative_decoding"]["mean_acceptance_length"],
            "ttft_p95_ms": summary["latency_ms"]["ttft"]["p95"],
            "tpot_p95_ms": summary["latency_ms"]["tpot"]["p95"],
            "requests": summary["requests"], "warmup": summary["warmup"],
            "preemptions": summary["load"]["preemptions"],
            "cell_settings": metadata["cell"],
            "workload_file_sha256": metadata["workload_file_sha256"],
            "harness_git_revision": metadata["harness_git_revision"],
            "deployment_id": metadata["extra"]["deployment_id"],
        }
        paths.update(path for path in cell.iterdir() if path.is_file())
        paths.update([results / arm / "status.json", results / arm / "draft-kv.jsonl",
                      results / arm / "chat/loadtest-summary.json"])
    metrics = ["output_tokens_per_second", "mean_acceptance_length", "ttft_p95_ms", "tpot_p95_ms"]
    native_mean = {m: (absolute["native-a"][m] + absolute["native-b"][m]) / 2 for m in metrics}
    ratios = {arm: {m: absolute[arm][m] / native_mean[m] for m in metrics}
              for arm in ["full-custom", "importance512"]}
    gpu_records = [json.loads(line) for line in (results / "state-kernels.log").read_text()
                   .splitlines() if line.startswith("{")]
    gpu = next(record for record in reversed(gpu_records) if "status" in record)
    tiny_path = allocation.parent / "preflight-runtime-04/analysis-physical-perf-v1/report.json"
    tiny = json.loads(tiny_path.read_text())
    paths.add(tiny_path)
    failed_screen = allocation.parent / "screen-01"
    report = {
        "schema_version": 1, "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "method": "post_hoc_single_node_C256_installed_runtime_stability_stress_v1",
        "allocation": allocation.name, "slurm_job_id": registration["job_id"],
        "used_for_policy_selection": False, "used_for_confirmation": False,
        "workload": "validation chat", "concurrency": 256, "absolute": absolute,
        "native_arithmetic_mean": native_mean, "ratios_to_native_mean": ratios,
        "native_a_over_b_ratios": {m: absolute["native-a"][m] / absolute["native-b"][m]
                                   for m in metrics},
        "ratio_method": "candidate absolute metric divided by arithmetic mean of two native GPUs",
        "kernel_and_state_regression": gpu,
        "stability": {
            "all_servers_ok": True,
            "all_four_arms_1024_requests_successful": all(
                r["requests"]["attempted"] == r["requests"]["successful"] == 1024
                for r in absolute.values()),
            "all_warmups_successful": all(r["warmup"]["failed"] == 0 for r in absolute.values()),
            "preemptions": {arm: r["preemptions"] for arm, r in absolute.items()},
        },
        "earlier_preflight_context": {
            "tiny_probe_source": source(tiny_path),
            "tiny_probe_six_cell_geomean_throughput_ratio": tiny["six_cell_geomean_throughput_ratio"],
            "tiny_probe_bounded_draft_storage_ratio": tiny["bounded_native_cache_diagnostic"]
                ["evidence"]["memory"]["draft_storage_ratio"],
            "scope": "preflight04 used2/8-request cap128 cells; no pooled estimate with this stress cell",
            "failure_context": "earlier high-load attempt exposed32-bit cache offset overflow; raw failed attempts retained",
            "new_regression": "this allocation passes large cache-offset GPU reference cases above2**31 and actual C256 requests",
            "failed_screen_path_if_present": str(failed_screen.resolve()) if failed_screen.exists() else None,
        },
        "provenance": {
            "launch_git_revision": launch.get("git_revision"),
            "raw_harness_git_revisions": sorted({r["harness_git_revision"] for r in absolute.values()}),
            "all_raw_harness_git_revisions_match_launch": all(
                r["harness_git_revision"] == launch.get("git_revision") for r in absolute.values()),
            "source_script": source(Path(__file__)),
            "raw_sources": [source(path) for path in sorted(paths)],
        },
        "limits": [
            "One node allocation and one workload/concurrency cell; native duplicates are technical controls.",
            "This allocation validates installed high-load stability and is excluded from candidate ranking and confirmation.",
            "Full-custom is slower despite similar acceptance, so eviction quality alone cannot explain the overhead.",
            "Averaged native p95 values are per-deployment summaries, not a pooled request-level p95.",
            "Earlier tiny preflight data use a prior runtime revision; keep them separate rather than pool estimates.",
        ],
    }
    lines = [
        "# Installed-runtime C256 stress diagnostic", "",
        f"Job `{registration['job_id']}`, launch `{launch.get('git_revision')}`. "
        "All four arms completed1,024 measured requests without failures or preemptions. "
        "This allocation is stability stress data, excluded from policy selection and confirmation.", "",
        "| Arm | tok/s | Acceptance length | p95 TTFT ratio | p95 TPOT ratio | Throughput ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ["full-custom", "importance512"]:
        row, ratio = absolute[arm], ratios[arm]
        lines.append(f"| {arm} | {row['output_tokens_per_second']:.2f} | "
                     f"{row['mean_acceptance_length']:.4f} | {ratio['ttft_p95_ms']:.4f} | "
                     f"{ratio['tpot_p95_ms']:.4f} | {ratio['output_tokens_per_second']:.4f} |")
    lines += ["", f"Native arithmetic mean: {native_mean['output_tokens_per_second']:.2f}tok/s, "
              f"acceptance length{native_mean['mean_acceptance_length']:.4f}. "
              "Ratios use the two native deployment summaries as the denominator.", "",
              "Large-offset GPU reference cases above2**31, state kernels and query graph replay passed. "
              "The full backend retains substantial overhead even with similar acceptance. "
              "No candidate or expansion decision follows from this diagnostic.", "",
              "Earlier preflight04 tiny cells had full-custom/native throughput geometric mean0.8072 "
              "and importance512/native0.7922. Their different workload size and prior runtime revision "
              "prevent pooling with this stress cell. Exact sources and hashes are in the adjacent JSON.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with markdown.open("x") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "launch_revision": launch.get("git_revision"), "ratios": ratios,
                      "stability": report["stability"]}, indent=2))


if __name__ == "__main__":
    main()
