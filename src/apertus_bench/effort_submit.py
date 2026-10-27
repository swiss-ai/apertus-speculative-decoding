"""Submit native-effort blocks through the repository's Slurm/EDF workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from apertus_bench.effort import REPO, SPEC_PATH, read_spec, write_json


def container_command(argv: list[str], environment: Path) -> str:
    inside = (
        f'export PYTHONPATH={shlex.quote(str(REPO / "src"))}:"${{PYTHONPATH:-}}"; '
        + "exec "
        + shlex.join(argv)
    )
    return shlex.join(
        [
            "srun",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            f"--environment={environment}",
            "bash",
            "-c",
            inside,
        ]
    )


def submit(args: argparse.Namespace) -> list[dict[str, str]]:
    spec = read_spec(args.spec)
    output = (
        args.output.resolve()
        if args.output
        else REPO
        / (
            "results/gpt-oss-20b/thinking-effort/"
            + datetime.now(UTC).strftime("campaign-%Y%m%dT%H%M%S%fZ")
        )
    )
    phase = args.phase
    if args.prepare and phase != "phase1":
        raise ValueError("asset preparation and preflight belong to Phase 1")
    if phase == "phase2" and args.heads is None:
        raise ValueError("Phase 2 requires --heads")
    if not args.prepare and not args.dependency and not args.preflight_only:
        status = output / "preflight/phase1/block-0/status.json"
        if not status.exists() or json.loads(status.read_text())["status"] != "preflight_complete":
            raise ValueError("run --prepare first, or supply its afterok --dependency job ID")
    environment_source = args.environment.resolve()
    if not environment_source.is_file():
        raise ValueError(f"missing EDF environment: {environment_source}")
    environment_data = tomllib.loads(environment_source.read_text())
    if environment_data["image"] != spec["image"]:
        raise ValueError("EDF image differs from the image recorded in the experiment spec")
    # Immutable copies, not a reference to a mutable YAML/image tag at launch.
    output.mkdir(parents=True, exist_ok=True)
    config = output / "experiment.yaml"
    import yaml

    rendered = yaml.safe_dump(spec, sort_keys=False)
    if config.exists():
        if config.read_text() != rendered:
            raise ValueError("campaign spec is already locked to different settings")
    else:
        config.write_text(rendered)
    blocks = (
        [None] if args.prepare else (args.blocks or ([0] if args.preflight_only else [0, 1, 2]))
    )
    if len(blocks) != len(set(blocks)):
        raise ValueError("a block can only be submitted once per campaign")
    submissions = []
    for block in blocks:
        name = "prepare" if block is None else f"{phase}-block-{block}"
        if args.preflight_only and block is not None:
            name += "-preflight"
        directory = output / "submissions" / name
        directory.mkdir(parents=True, exist_ok=False)
        environment = directory / "environment.toml"
        environment.write_text(environment_source.read_text())
        commands = []
        if args.prepare:
            commands.append(
                [
                    "python3",
                    str(REPO / "experiments/thinking-effort/prepare-assets.py"),
                    str(config),
                    "--output",
                    str(output / "asset-preparation"),
                ]
            )
            commands.append(
                [
                    "python3",
                    "-m",
                    "apertus_bench.effort",
                    "--spec",
                    str(config),
                    "run",
                    "--output",
                    str(output / "preflight"),
                    "--block",
                    "0",
                    "--preflight-only",
                ]
            )
        else:
            command = [
                "python3",
                "-m",
                "apertus_bench.effort",
                "--spec",
                str(config),
                "run",
                "--output",
                str(output),
                "--block",
                str(block),
                "--phase",
                phase,
            ]
            if args.preflight_only:
                command.append("--preflight-only")
            if args.heads:
                command += ["--heads", str(args.heads.resolve())]
            commands.append(command)
            if not args.preflight_only:
                commands.append(
                    [
                        "python3",
                        "-m",
                        "apertus_bench.effort",
                        "--spec",
                        str(config),
                        "report",
                        str(output),
                        "--phase",
                        phase,
                    ]
                )
        script = directory / "job.sh"
        script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"cd {shlex.quote(str(REPO))}",
                    *[container_command(command, environment) for command in commands],
                    "",
                ]
            )
        )
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=effort-{name}",
            f"--account={args.account}",
            f"--partition={args.partition}",
            f"--time={args.time}",
            "--nodes=1",
            "--exclusive",
            f"--output={directory / 'job.out'}",
            f"--error={directory / 'job.err'}",
        ]
        if args.dependency:
            if not args.dependency.isdigit():
                raise ValueError("dependency must be a Slurm job ID")
            command.append(f"--dependency=afterok:{args.dependency}")
        command.append(str(script))
        write_json(
            directory / "submission.json",
            {
                "argv": command,
                "container_commands": commands,
                "environment_source": str(environment_source),
                "spec": spec,
            },
        )
        record = {"directory": str(directory), "campaign": str(output)}
        if not args.validate_only:
            job = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
            record["job_id"] = job.split(";")[0]
            write_json(directory / "job.json", record)
        else:
            record["status"] = "validated_not_submitted"
        submissions.append(record)
    return submissions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment", type=Path, default=REPO / "experiments/thinking-effort/env.toml"
    )
    parser.add_argument("--blocks", nargs="+", type=int, choices=range(3))
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    parser.add_argument("--heads", type=Path)
    parser.add_argument(
        "--prepare", action="store_true", help="download head and preflight both arms"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dependency", help="afterok dependency, normally the prepare job ID")
    parser.add_argument("--partition", default="normal")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--account", default="infra01")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit(args), indent=2))


if __name__ == "__main__":
    main()
