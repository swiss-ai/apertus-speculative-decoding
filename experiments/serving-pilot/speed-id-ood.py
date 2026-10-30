"""Paired same-GPU deployment repeats using the existing node-sweep benchmark.

Run inside the pilot's preserved container environment through srun --overlap.
No Slurm submission, gateway credentials, or changes to the existing server.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from apertus_bench.node_sweep import build_plan, run_arm
from apertus_bench.workloads import file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_corpus(source: Path, output: Path):
    """Deterministic sample without tuning on observed acceptance or speed."""
    definitions = [
        ("id-opb", "opb-heldout.jsonl", "opb"),
        ("ood-chat", "eagle-8b-test.jsonl", "chat"),
        ("ood-summary", "eagle-8b-test.jsonl", "summarization"),
    ]
    records, manifest = [], {}
    for label, filename, workload in definitions:
        path = source / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["workload"] == workload]
        rows.sort(key=lambda r: hashlib.sha256(("speed-20261030:" + r["id"]).encode()).hexdigest())
        chosen = rows[:32]
        if len(chosen) != 32:
            raise ValueError(f"need 32 prompts in {label}, found {len(chosen)}")
        manifest[label] = {
            "source": str(path),
            "source_sha256": file_sha256(path),
            "available": len(rows),
            "ids": [r["id"] for r in chosen],
        }
        for row in chosen:
            row = dict(
                row, workload=label, max_tokens=256, chat_template_kwargs={"enable_thinking": False}
            )
            records.append(row)
    output.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest["corpus_sha256"] = file_sha256(output)
    return manifest


def spec_for(pilot: Path, corpus: Path):
    profile = json.loads((pilot / "profile.json").read_text())
    loadtests = []
    for label in ("id-opb", "ood-chat", "ood-summary"):
        for fixed in (False, True):
            loadtests.append(
                {
                    "label": label + ("-fixed" if fixed else "-natural"),
                    "workload": label,
                    "workloads_file": str(corpus),
                    "concurrencies": [1, 128] if fixed else [1, 8, 128],
                    "min_requests": 32,
                    "requests_per_slot": 1,
                    "max_tokens": 256,
                    "generation": {"temperature": 0, "top_p": 1, "seed": 1, "ignore_eos": fixed},
                }
            )
    return {
        "name": "dspark-e10-id-ood",
        "stage": "8b",
        "probe": False,
        "target_model": profile["target"]["path"],
        "node_gpus": 1,
        "engine": {
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 256,
        },
        "loadtests": loadtests,
        "arms": [
            {
                "name": name,
                "method": method,
                **({"drafter": profile["drafter"]["path"], "k": 3} if method == "dspark" else {}),
                "vllm_args": [
                    "--dtype",
                    "bfloat16",
                    "--no-enable-log-requests",
                    "--chat-template",
                    str(pilot / "chat_template.jinja"),
                ],
            }
            for name, method in [("plain", "baseline"), ("dspark", "dspark")]
        ],
    }


async def campaign(args):
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = args.output / "corpus.jsonl"
    manifest = prepare_corpus(args.workloads, corpus)
    write_json(args.output / "corpus-manifest.json", manifest)
    spec = spec_for(args.pilot, corpus)
    write_json(args.output / "spec.json", spec)
    for name in (
        "preflight.json",
        "profile.json",
        "launch-manifest.json",
        "source-manifest.json",
        "environment.toml",
        "chat_template.jinja",
        "checkpoint-check.json",
    ):
        shutil.copy2(args.pilot / name, args.output / name)
    # Copy the exact overlay sources; do not modify the running pilot's files.
    shutil.copytree(args.pilot / "overlays", args.output / "overlays")
    provenance = {
        "branch": os.environ["SPEED_SOURCE_BRANCH"],
        "revision": os.environ["SPEED_SOURCE_REVISION"],
        "argv": sys.argv,
        "gpu_index": args.gpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "gpu_inventory": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,clocks.sm,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        "pilot": str(args.pilot),
        "design": "three paired deployment repeats on one GPU; AB, BA, AB",
        "unit": "deployment pair, all on one physical node; not independent nodes",
    }
    shutil.copy2(__file__, args.output / "speed-id-ood.py")
    write_json(args.output / "provenance.json", provenance)
    all_status = []
    for repeat, order in enumerate(((0, 1), (1, 0), (0, 1)), 1):
        plan = build_plan(spec)
        target = args.output / f"repeat-{repeat}"
        for index in order:
            arm = plan.arms[index]
            # This release requires the module API server for --model syntax.
            arm.serve_argv[:2] = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
            arm.metadata.update(
                {
                    "checkpoint_sha256": json.loads((args.pilot / "preflight.json").read_text())[
                        "drafter"
                    ]["checkpoint_sha256"],
                    "target_revision": spec_target_revision(args.pilot),
                    "target_model": spec["target_model"],
                    "target_tensor_parallel_size": "1",
                    "source_revision": provenance["revision"],
                    "source_branch": provenance["branch"],
                    "pair": str(repeat),
                    "physical_gpu": str(args.gpu),
                    "corpus_sha256": manifest["corpus_sha256"],
                }
            )
            print(f"starting pair={repeat} arm={arm.name}", flush=True)
            status = await run_arm(arm, [args.gpu], plan, target, f"speed-pair-{repeat}")
            all_status.append({"pair": repeat, **status})
            write_json(args.output / "campaign-status.json", all_status)
            if status.get("status") != "ok":
                raise RuntimeError(f"arm failed: {status}")
    print("campaign complete", flush=True)


def spec_target_revision(pilot):
    return json.loads((pilot / "profile.json").read_text())["target"]["revision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(campaign(args))


if __name__ == "__main__":
    main()
