#!/usr/bin/env python3
"""Summarize recorded trainer timings and ten-second memory samples (stdlib).

Compatible with the Beverin login node's Python 3.6. Raw logs remain intact.
Timing summaries omit each torchrun attempt's first (compiling) step.
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def summarize(work, run_id, job_id):
    run = work / "runs" / run_id
    attempts = []
    for log in sorted((run / "logs").glob("train-{}*-node0.log".format(job_id))):
        content = log.read_text(errors="replace")
        steps = []
        for match in re.finditer(
            r"profile/fetch_ms=.*?global_step=\d+", content, re.DOTALL
        ):
            pairs = re.findall(
                r"(profile/[a-z_]+|global_step)=([0-9.eE+\-]+)", match.group()
            )
            steps.append({key: float(value) for key, value in pairs})
        timings = {}
        for phase in ("fetch", "fwd", "bwd", "opt", "step"):
            values = [s["profile/{}_ms".format(phase)] / 1000 for s in steps[1:]
                      if "profile/{}_ms".format(phase) in s]
            if values:
                timings[phase + "_s"] = {
                    "median": statistics.median(values),
                    "min": min(values), "max": max(values), "n": len(values),
                }
        attempts.append({
            "log": str(log), "steps": steps, "steady": timings,
            "checkpoint_saved": "Checkpoint saved to" in content,
            "resume": re.findall(
                r"Resuming mid-epoch from.*?global_step=\d+", content, re.DOTALL
            ),
        })
    memory = {}
    for log in sorted((work / "logs").glob("mem-{}-node*.log".format(job_id))):
        samples = re.findall(r"(\S+) used_gb=(\d+) avail_gb=(\d+)", log.read_text())
        if samples:
            memory[log.name] = {
                "peak_used_gib": max(int(s[1]) for s in samples),
                "min_available_gib": min(int(s[2]) for s in samples),
                "samples": len(samples),
            }
    return {"run_id": run_id, "job_id": job_id, "attempts": attempts, "memory": memory}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.work, args.run_id, args.job_id), indent=2))
