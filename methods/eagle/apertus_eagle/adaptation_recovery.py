"""Durable T5 stage controller; run on the login host in a tmux session.

Only lightweight control/data slicing runs here. All generation and training
uses the existing Slurm launchers. One owned job is queued at a time. The
original 19 finite adaptation curves are the entire training scope; serving
confirmation is selected after their metrics have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

SOURCE_DATA = "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates"
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
}


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def curve_report(run: Path, head: Path, expected_steps: int = 384) -> dict[str, Any]:
    needed = [
        run / name
        for name in ("train-summary.json", "provenance.json", "evals.json", "history.json")
    ]
    needed += [head / "config.json", head / "export-report.json"]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return {"state": "incomplete", "missing": missing, "run_dir": str(run)}
    summary = json.loads(needed[0].read_text())
    export = json.loads((head / "export-report.json").read_text())
    initial, final = summary["eval_initial"], summary["eval_final"]
    steps_done = summary["steps"] == expected_steps
    infrastructure = (
        steps_done
        and summary["gates"].get("finite_losses") is True
        and summary["gates"].get("real_updates") is True
        and export.get("reload", {}).get("pass") is True
    )
    return {
        "state": "complete" if infrastructure else "infrastructure_failed",
        "run_dir": str(run),
        "head_dir": str(head),
        "steps": summary["steps"],
        "wall_seconds": summary["wall_seconds"],
        "scientific_gates": summary["gates"],
        "scientific_gates_passed": all(summary["gates"].values()),
        "domain_acceptance_initial": initial["simulated_acc_len"],
        "domain_acceptance_final": final["simulated_acc_len"],
        "opb_acceptance_initial": initial.get("extra", {}).get("opb", {}).get("simulated_acc_len"),
        "opb_acceptance_final": final.get("extra", {}).get("opb", {}).get("simulated_acc_len"),
        "artifacts": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in needed},
    }


def main(argv: list[str] | None = None) -> None:
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--trace-job", required=True)
    parser.add_argument("--run-id", default="recovery-20261027")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    root, repo = args.campaign_root, args.repo
    receipts = root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    lock = (receipts / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = receipts / "campaign.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "workstream": "adaptation",
            "job_ids": [args.trace_job],
            "batches": {},
            "created_at": stamp(),
        }
    )
    if state.get("state") == "awaiting_confirmation":
        print(json.dumps(state), flush=True)
        return
    state["pid"] = os.getpid()

    def save(**updates: Any) -> None:
        state.update(updates, updated_at=stamp())
        write_state(state_path, state)

    def command(command_argv: list[str], env: dict[str, str] | None = None) -> str:
        print(json.dumps({"at": stamp(), "argv": command_argv}), flush=True)
        return subprocess.check_output(command_argv, cwd=repo, env=env, text=True)

    def wait(job: str) -> str:
        while True:
            try:
                rows = command(
                    ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
                )
                fields = next(
                    line.split("|") for line in rows.splitlines() if line.startswith(job + "|")
                )
                status = fields[1].split()[0].rstrip("+")
                save(state="waiting", current_job=job, slurm_state=status)
                if status in TERMINAL:
                    return status
            except (subprocess.CalledProcessError, StopIteration) as error:
                save(state="waiting", current_job=job, status_error=str(error))
            time.sleep(max(args.poll_seconds, 30))

    save(
        state="waiting", next_action="Wait for trace generation, then the original bounded curves."
    )
    if wait(args.trace_job) != "COMPLETED":
        save(
            state="needs_repair",
            next_action="Inspect the trace job before any training submission.",
        )
        return
    env = os.environ.copy()
    env["DRAFTER_DATA_ROOT"] = str(root / "data")
    if not (root / "data/adapt-data/manifest.json").exists():
        output = command(["bash", "methods/eagle/launch/drafter-adapt-data.sh"], env)
        (receipts / "prepare-data.log").write_text(output)

    templates = sorted((repo / "methods/eagle/configs/8b/adapt").glob("adapt-*.yaml"))
    templates = [path for path in templates if path.name != "adapt-smoke.yaml"]
    if len(templates) != 19:
        raise RuntimeError(f"Expected the original 19 configurations, found {len(templates)}")
    cfg_dir = root / "configs"
    cfg_dir.mkdir(exist_ok=True)
    manifest = {}
    groups: dict[str, list[Path]] = {"chat": [], "summ": [], "mult": []}
    for source in templates:
        content = source.read_text()
        for old, new in [
            ("adapt-data", "data/adapt-data"),
            ("traces", "data/traces"),
            ("runs", "runs"),
            ("heads", "heads"),
        ]:
            content = content.replace(f"{SOURCE_DATA}/{old}", str(root / new))
        target = cfg_dir / source.name
        if target.exists():
            if target.read_text() != content:
                raise RuntimeError(f"Refusing to change a frozen configuration: {target}")
        else:
            target.write_text(content)
        domain = source.stem.split("-")[1]
        groups[domain].append(target)
        manifest[source.name] = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "resolved": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    frozen = root / "configuration-manifest.json"
    if not frozen.exists():
        frozen.write_text(json.dumps(manifest, indent=2) + "\n")

    # Cases are independent. The existing on-node driver continues after a
    # scientific gate failure, while every full gate result remains recorded.
    all_reports = {}
    for domain, configs in groups.items():
        batch = state["batches"].setdefault(domain, {"job_ids": []})
        while True:
            reports = {}
            remaining = []
            for config in configs:
                name = re.search(r"^run_name:\s*(\S+)", config.read_text(), re.MULTILINE)[1]
                full_name = f"{name}-{args.run_id}"
                report = curve_report(
                    root / "runs" / full_name, root / "heads" / (full_name + "-se")
                )
                reports[name] = report
                if report["state"] == "incomplete":
                    remaining.append(config)
                elif report["state"] != "complete":
                    save(
                        state="needs_repair",
                        next_action=f"Inspect infrastructure failure: {name}",
                        curves=reports,
                    )
                    return
            if not remaining:
                all_reports.update(reports)
                break
            if batch.get("active_job"):
                job = batch["active_job"]
            else:
                if len(batch["job_ids"]) >= 3:
                    save(
                        state="needs_repair",
                        next_action=f"Three bounded attempts exhausted for {domain}.",
                        curves=reports,
                    )
                    return
                training_env = env | {
                    "EAGLE_TRAIN_CONFIG": " ".join(str(path) for path in remaining),
                    "EAGLE_CUDA_DEVICES": "0,1,2,3",
                    "EAGLE_TRAIN_PARTITION": "normal",
                    "EAGLE_TRAIN_TIME": "08:00:00",
                    "EAGLE_STEPS": "train,export",
                    "EAGLE_RUN_ID": args.run_id,
                    "EAGLE_VLLM_PATCH_DIR": str(
                        root / "overlays" / f"{domain}-{len(batch['job_ids'])}"
                    ),
                }
                output = command(
                    ["bash", "methods/eagle/launch/submit-eagle-train.sh"], training_env
                )
                job = re.findall(r"Submitted batch job (\d+)", output)[-1]
                (receipts / f"submit-{domain}-{job}.txt").write_text(output)
                batch["job_ids"].append(job)
                batch["active_job"] = job
                state["job_ids"].append(job)
                save(
                    state="waiting",
                    next_action=f"Wait for {domain} curves; inspect artifacts and full gates.",
                )
            wait(job)
            batch.pop("active_job", None)
            save(state="collecting")
    matrix = root / "adaptation-matrix-v1.json"
    with matrix.open("x") as handle:
        handle.write(json.dumps(all_reports, indent=2) + "\n")
    save(
        state="awaiting_confirmation",
        curves=all_reports,
        artifact_paths=[str(matrix)],
        next_action=(
            "Review scientific gates/forgetting; select 2-3 heads for the original "
            "paired serving confirmation."
        ),
    )


if __name__ == "__main__":
    main()
