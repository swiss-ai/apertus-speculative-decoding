"""Durable Slurm controller for the bounded four-head effort experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

from apertus_bench.effort import REPO, read_spec, source_identity, write_json
from apertus_bench.effort_submit import submit, submit_job, write_immutable
from apertus_bench.effort_training import TRAINING_CONFIG, load_config

TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}


def job_state(job: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job, "--noheader", "--format=JobIDRaw,State%30", "-P"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == job:
            return fields[1].split()[0]
    return "UNKNOWN"


def wait_success(job: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        state = job_state(job)
        if state == "COMPLETED":
            return
        if state in TERMINAL_FAILURES:
            raise RuntimeError(
                f"Slurm job {job} ended as {state}; retained artifacts must be inspected"
            )
        time.sleep(30)
    raise TimeoutError(f"controller wait expired while job {job} was {job_state(job)}")


def stage_job(args, campaign: Path, name: str) -> dict:
    directory = campaign / "submissions" / name
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "python3",
        "-m",
        "apertus_bench.effort_training",
        name,
        "--campaign",
        str(campaign),
        "--config",
        str(campaign / "phase2.yaml"),
        "--upstream",
        str(args.upstream.resolve()),
    ]
    if name == "data":
        command += ["--phase1-reference", str(args.phase1_reference.resolve())]
    pythonpath = ":".join(
        str(p)
        for p in (
            REPO / "src",
            args.upstream / "src",
            args.upstream / "hs_connectors/src",
        )
    )
    bootstrap = shlex.join(
        [
            "python3",
            str(REPO / "experiments/thinking-effort/bootstrap-phase2.py"),
            "--upstream",
            str(args.upstream.resolve()),
            "--output",
            str(campaign / "training-environment"),
        ]
    )
    inside = (
        f"export PYTHONPATH={shlex.quote(pythonpath)}; "
        "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; "
        f"cd {shlex.quote(str(REPO))}; "
        f'EFFORT_PYTHON="$({bootstrap})"; exec "$EFFORT_PYTHON" ' + shlex.join(command[1:])
    )
    script = directory / "job.sh"
    write_immutable(
        script,
        "#!/bin/bash\nset -euo pipefail\n"
        + shlex.join(
            [
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--environment={campaign / 'environment.toml'}",
                "bash",
                "-c",
                inside,
            ]
        )
        + "\n",
    )
    argv = [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--partition={args.partition}",
        f"--time={args.time}",
        "--nodes=1",
        "--ntasks=1",
        "--exclusive",
        "--cpus-per-task=32",
        f"--job-name=effort-phase2-{name}",
        f"--output={directory / 'job.out'}",
        f"--error={directory / 'job.err'}",
        str(script),
    ]
    write_immutable(
        directory / "submission.json",
        json.dumps({"argv": argv, "container_argv": command, "source": source_identity()}, indent=2)
        + "\n",
    )
    return submit_job(
        argv, directory, {"campaign": str(campaign), "directory": str(directory)}, args
    )


def initialize(args, campaign: Path) -> None:
    campaign.mkdir(parents=True, exist_ok=True)
    parent = args.phase1_reference.resolve()
    reports = sorted(parent.glob("reports/phase1-*/comparison.json"))
    if not reports or json.loads(reports[-1].read_text())["decision"] != "proceed":
        raise ValueError("the parent campaign has no completed proceed report")
    spec = read_spec(parent / "experiment.yaml")
    source = Path(spec["source_workload"])
    if not source.is_absolute():
        spec["source_workload"] = str(parent.parents[3] / source)
    # Parent is <repo>/results/gpt-oss-20b/thinking-effort/<campaign>.
    write_immutable(campaign / "experiment.yaml", yaml.safe_dump(spec, sort_keys=False))
    load_config(args.config)
    write_immutable(campaign / "phase2.yaml", args.config.read_text())
    environment = args.environment.read_text()
    old_workdir = "/users/faruk_zahiragic/apertus-spec-decoding"
    environment = environment.replace(old_workdir, str(REPO))
    write_immutable(campaign / "environment.toml", environment)
    write_immutable(
        campaign / "controller-config.json",
        json.dumps(
            {
                "argv": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "source": source_identity(),
                "phase1_report": str(reports[-1]),
            },
            indent=2,
        )
        + "\n",
    )


def run(args) -> None:
    campaign = args.output.resolve()
    initialize(args, campaign)
    deadline = time.monotonic() + args.controller_timeout
    status = campaign / "controller-status.json"
    try:
        write_json(
            status,
            {
                "status": "waiting_for_training_export_qualification",
                "job_id": args.qualification_job,
            },
        )
        wait_success(args.qualification_job, deadline)
        report = json.loads((args.qualification / "training-smoke/report.json").read_text())
        serving = json.loads(
            (args.qualification / "serving-preflight/phase1/block-0/status.json").read_text()
        )
        if (
            report["status"] != "training_export_passed"
            or serving["status"] != "preflight_complete"
        ):
            raise ValueError("GPT-OSS DFlash training/export/serving qualification failed")
        write_immutable(
            campaign / "qualification.json",
            json.dumps(
                {"source": str(args.qualification), "report": report, "serving": serving}, indent=2
            )
            + "\n",
        )
        for stage in ("data", "train"):
            finished = campaign / (
                "data-status.json" if stage == "data" else "training-status.json"
            )
            record = stage_job(args, campaign, stage)
            write_json(status, {"status": stage, "job_id": record["job_id"]})
            wait_success(record["job_id"], deadline)
            if not finished.exists() or json.loads(finished.read_text())["status"] != "complete":
                raise ValueError(f"{stage} job completed without a valid artifact gate")
        evaluation = SimpleNamespace(
            spec=campaign / "experiment.yaml",
            output=campaign,
            environment=campaign / "environment.toml",
            blocks=[0, 1, 2],
            phase="phase2",
            heads=campaign / "heads.json",
            phase1_reference=args.phase1_reference.resolve(),
            prepare=False,
            preflight_only=False,
            dependency=None,
            partition=args.partition,
            time=args.time,
            account=args.account,
            validate_only=False,
            supervise=True,
            retry_seconds=args.retry_seconds,
            submission_timeout=args.submission_timeout,
        )
        write_json(status, {"status": "submitting_evaluation_blocks"})
        jobs = submit(evaluation)
        write_json(status, {"status": "evaluation", "jobs": jobs})
        for record in jobs:
            wait_success(record["job_id"], deadline)
        # Report generation reads raw records, so keep it in a Slurm allocation.
        reports = sorted(campaign.glob("reports/phase2-*/comparison.json"))
        if not reports:
            raise ValueError("evaluation completed without a Phase 2 comparison report")
        result = json.loads(reports[-1].read_text())
        if result["decision"] in {"pending", "invalid"} or result["problems"]:
            raise ValueError(f"Phase 2 comparison is not valid: {result['problems']}")
        write_json(
            status,
            {
                "status": "complete",
                "decision": result["decision"],
                "report": str(reports[-1]),
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
    except BaseException as error:
        write_json(
            status,
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "time": datetime.now(UTC).isoformat(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase1-reference", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-job", required=True)
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument(
        "--environment", type=Path, default=REPO / "experiments/thinking-effort/env.toml"
    )
    parser.add_argument("--partition", default="debug")
    parser.add_argument("--time", default="01:30:00")
    parser.add_argument("--account", default="infra01")
    parser.add_argument("--retry-seconds", type=float, default=30)
    parser.add_argument("--submission-timeout", type=float, default=86400)
    parser.add_argument("--controller-timeout", type=float, default=172800)
    args = parser.parse_args()
    args.dependency = None
    args.supervise = True
    if not 1 <= args.retry_seconds <= 60:
        parser.error("retry interval must be between 1 and 60 seconds")
    run(args)


if __name__ == "__main__":
    main()
