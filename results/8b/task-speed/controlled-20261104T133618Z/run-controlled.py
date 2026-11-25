"""Controlled per-task speed: plain/DSpark pairs on the same GPU, one pair per GPU.

Runs inside the serving pilot's container (environment.toml of the pilot that
the live Deployment was frozen from), as the step of a Slurm job submitted by
submit-controlled.sh. Each GPU runs a fresh plain and a fresh DSpark deployment
one after the other; the order alternates across GPUs (AB, BA, AB, BA), so each
GPU is one deployment pair. Pairs run concurrently on the node's GPUs.
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
from datetime import UTC, datetime
from pathlib import Path

from apertus_bench import task_speed
from apertus_bench.node_sweep import build_plan, run_arm
from apertus_bench.workloads import file_sha256

PILOT_FILES = (
    "preflight.json",
    "profile.json",
    "launch-manifest.json",
    "source-manifest.json",
    "environment.toml",
    "chat_template.jinja",
    "checkpoint-check.json",
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recheck_runtime(root: Path) -> dict:
    """Hash image, target and drafter after all timing, against the pilot preflight."""
    profile = json.loads((root / "profile.json").read_text())
    expected = json.loads((root / "preflight.json").read_text())
    checks = [(Path(profile["runtime"]["image"]), expected["image_sha256"])]
    for group, prefix in [
        (expected["target_files"], profile["target"]["path"]),
        (expected["drafter"]["files"], profile["drafter"]["path"]),
    ]:
        checks.extend((Path(prefix) / name, digest) for name, digest in group.items())
    files = [{"path": str(p), "expected": d, "actual": sha256(p)} for p, d in checks]
    result = {
        "verified_at": datetime.now(UTC).isoformat(),
        "passed": all(f["expected"] == f["actual"] for f in files),
        "files": files,
    }
    write_json(root / "runtime-recheck.json", result)
    return result


def artifact_manifest(root: Path) -> None:
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != "corpus.jsonl" and "vllm-cache" not in p.parts
    )
    write_json(
        root / "artifact-manifest.json",
        {str(p.relative_to(root)): sha256(p) for p in files},
    )


def parse_select(path: Path | None) -> dict[str, list[int]] | None:
    """Cells to extend: a report's inconclusive rows ({"select": {slice: [C, ...]}})."""
    if path is None:
        return None
    return json.loads(path.read_text())["select"]


async def campaign(args) -> None:
    suite = task_speed.load_suite(args.suite)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.corpus:
        # Extension pairs reuse the first run's corpus byte for byte.
        manifest = json.loads((args.corpus.parent / "corpus-manifest.json").read_text())
        if file_sha256(args.corpus) != manifest["corpus_sha256"]:
            raise ValueError("corpus does not match its manifest")
        corpus = args.output / "corpus.jsonl"
        shutil.copy2(args.corpus, corpus)
    else:
        corpus = args.output / "corpus.jsonl"
        manifest = task_speed.build_corpus(suite, args.inputs, corpus)
    write_json(args.output / "corpus-manifest.json", manifest)
    for name in PILOT_FILES:
        shutil.copy2(args.pilot / name, args.output / name)
    shutil.copytree(args.pilot / "overlays", args.output / "overlays")
    shutil.copy2(args.suite, args.output / "suite.yaml")
    shutil.copy2(__file__, args.output / Path(__file__).name)
    profile = json.loads((args.pilot / "profile.json").read_text())
    preflight = json.loads((args.pilot / "preflight.json").read_text())
    select = parse_select(args.select)
    spec = task_speed.controlled_spec(
        suite,
        corpus,
        profile["target"]["path"],
        profile["drafter"]["path"],
        str(args.pilot / "chat_template.jinja"),
        select,
    )
    write_json(args.output / "spec.json", spec)
    gpus = [int(g) for g in args.gpus.split(",")]
    provenance = {
        "branch": os.environ["TASK_SPEED_BRANCH"],
        "revision": os.environ["TASK_SPEED_REVISION"],
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "gpus": gpus,
        "gpu_inventory": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,clocks.max.sm,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        "pilot": str(args.pilot),
        "first_pair": args.first_pair,
        "select": select,
        "design": "one plain/DSpark deployment pair per GPU, run concurrently; "
        "order AB on even pairs, BA on odd pairs",
    }
    write_json(args.output / "provenance.json", provenance)
    statuses = []

    async def one_pair(slot: int, gpu: int) -> None:
        pair = args.first_pair + slot
        plan = build_plan(spec)
        order = (0, 1) if pair % 2 else (1, 0)  # pair 1 AB, pair 2 BA, ...
        target = args.output / f"pair-{pair}"
        for index in order:
            arm = plan.arms[index]
            arm.index += 2 * slot  # distinct ports for the concurrent pairs
            # This release requires the module API server for --model syntax.
            arm.serve_argv[:2] = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
            arm.metadata.update(
                {
                    "checkpoint_sha256": preflight["drafter"]["checkpoint_sha256"],
                    "target_revision": profile["target"]["revision"],
                    "target_model": profile["target"]["path"],
                    "target_tensor_parallel_size": "1",
                    "source_revision": provenance["revision"],
                    "source_branch": provenance["branch"],
                    "pair": str(pair),
                    "physical_gpu": str(gpu),
                    "node": provenance["node"] or "",
                    "corpus_sha256": manifest["corpus_sha256"],
                    "suite": suite["name"],
                }
            )
            print(f"start pair={pair} gpu={gpu} arm={arm.name}", flush=True)
            status = await run_arm(arm, [gpu], plan, target, f"task-speed-pair-{pair}")
            statuses.append({"pair": pair, "gpu": gpu, **status})
            write_json(args.output / "campaign-status.json", statuses)
            print(f"done pair={pair} arm={arm.name} status={status.get('status')}", flush=True)
            if status.get("status") != "ok":
                return  # keep the other GPUs' pairs going; the report rejects this pair

    await asyncio.gather(*(one_pair(slot, gpu) for slot, gpu in enumerate(gpus)))
    recheck = recheck_runtime(args.output)
    artifact_manifest(args.output)
    failed = [s for s in statuses if s.get("status") != "ok"]
    print(json.dumps({"arms": len(statuses), "failed": len(failed), "runtime": recheck["passed"]}))
    if failed or not recheck["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="directory with the suite's source files")
    parser.add_argument("--corpus", type=Path, help="reuse an earlier run's corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--first-pair", type=int, default=1)
    parser.add_argument("--select", type=Path, help="JSON with the cells to extend")
    args = parser.parse_args()
    if (args.inputs is None) == (args.corpus is None):
        parser.error("give exactly one of --inputs and --corpus")
    asyncio.run(campaign(args))


if __name__ == "__main__":
    main()
