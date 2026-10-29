"""Durable T5 controller; run detached on the login host.

Only lightweight control/data slicing runs here. All generation and training
uses the existing Slurm launchers. One owned job is queued at a time. The
original 19 finite adaptation curves and preregistered serving confirmation
are the entire scope. It closes positive or negative results automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from apertus_eagle.adaptation_decision import (
    digest,
    final_verdict,
    immutable_json,
    read,
    select_heads,
    workload_entries,
)

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
        and summary.get("non_finite_grad_steps") == 0
        and export.get("reload", {}).get("pass") is True
        and export.get("reload", {}).get("exported_tensors_bitwise_equal") is True
        and all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in (
                initial["simulated_acc_len"],
                final["simulated_acc_len"],
                initial.get("extra", {}).get("opb", {}).get("simulated_acc_len"),
                final.get("extra", {}).get("opb", {}).get("simulated_acc_len"),
            )
        )
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
    parser.add_argument("--overlay-root", type=Path)
    args = parser.parse_args(argv)
    root, repo = args.campaign_root, args.repo
    short_root = args.overlay_root or Path(
        f"/iopsstor/scratch/cscs/{os.environ['USER']}/t5-"
        + hashlib.sha256(str(root).encode()).hexdigest()[:8]
    )
    short_root.mkdir(parents=True, exist_ok=True)
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
    if state.get("state") in {"complete", "negative_complete"}:
        print(json.dumps(state), flush=True)
        return
    state["pid"] = os.getpid()

    def save(**updates: Any) -> None:
        state.update(updates, updated_at=stamp())
        write_state(state_path, state)

    def command(command_argv: list[str], env: dict[str, str] | None = None) -> str:
        print(json.dumps({"at": stamp(), "argv": command_argv}), flush=True)
        return subprocess.check_output(command_argv, cwd=repo, env=env, text=True)

    policy = read(repo / "methods/eagle/configs/8b/adapt-recovery-policy.json")
    immutable_json(root / "preregistration.json", policy)
    source = read(repo / "source-manifest.json")
    if command(["git", "rev-parse", "HEAD"]).strip() != source["revision"]:
        raise RuntimeError("Frozen controller Git revision changed")
    for name, sha in source["files"].items():
        if digest(repo / name) != sha:
            raise RuntimeError(f"Frozen source changed: {name}")
    state["source_manifest"] = str(repo / "source-manifest.json")
    state["policy_sha256"] = digest(root / "preregistration.json")
    state["short_overlay_root"] = str(short_root)
    immutable_json(
        root / "overlay-locations.json",
        {
            "root": str(short_root),
            "training": "t-<chat|summ|mult>-<attempt_index>",
            "serving": "s-<chat|summ|mult>; distinct from every training deployment",
            "reason": "Short private mount-source paths avoid the Pyxis mount-list length limit.",
        },
    )

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
            if batch.get("last_terminal_state") not in {None, "COMPLETED"}:
                save(
                    state="needs_repair",
                    next_action=f"Inspect {domain} failure before resubmitting incomplete curves.",
                    curves=reports,
                )
                return
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
                    "EAGLE_CHAIN": "1",
                    "EAGLE_NODES": "1",
                    "EAGLE_ENV_SOURCE": str(repo / "methods/eagle/configs/train-env.toml"),
                    "SML_ENV_DIR": str(root / "training-environments"),
                    "EXTRA_MOUNTS": "",
                    "EAGLE_DEPLOYMENT_ROOT": str(root / "training-deployments"),
                    "EAGLE_VLLM_PATCH_DIR": str(short_root / f"t-{domain}-{len(batch['job_ids'])}"),
                }
                output = command(
                    ["bash", "methods/eagle/launch/submit-eagle-train.sh"], training_env
                )
                job = re.findall(r"Submitted batch job (\d+)", output)[-1]
                (receipts / f"submit-{domain}-{job}.txt").write_text(output)
                deployment = root / "training-deployments" / job
                deployment.mkdir(parents=True, exist_ok=False)
                launch = Path(re.search(r"^sbatch_file=(.+)$", output, re.MULTILINE)[1])
                shutil.copy2(launch, deployment / "launch.sbatch")
                environment = Path(re.search(r"^environment=(.+)$", output, re.MULTILINE)[1])
                shutil.copy2(environment, deployment / "environment.toml")
                (deployment / "submission-environment.json").write_text(
                    json.dumps(
                        {
                            key: value
                            for key, value in training_env.items()
                            if key
                            in {
                                "EAGLE_TRAIN_CONFIG",
                                "EAGLE_CUDA_DEVICES",
                                "EAGLE_STEPS",
                                "EAGLE_RUN_ID",
                                "EAGLE_VLLM_PATCH_DIR",
                                "EAGLE_DEPLOYMENT_ROOT",
                            }
                        },
                        indent=2,
                    )
                    + "\n"
                )
                batch["job_ids"].append(job)
                batch["active_job"] = job
                state["job_ids"].append(job)
                save(
                    state="waiting",
                    next_action=f"Wait for {domain} curves; inspect artifacts and full gates.",
                )
            batch["last_terminal_state"] = wait(job)
            batch.pop("active_job", None)
            save(state="collecting")
    matrix = root / "adaptation-matrix-v1.json"
    immutable_json(matrix, all_reports)
    selection = select_heads(all_reports, policy)
    immutable_json(root / "selection.json", selection)
    confirmations = []
    state.setdefault("confirmations", {})
    for selected in selection["selected"]:
        name = selected["name"]
        spec_dir = root / "confirmation-configs"
        spec_dir.mkdir(exist_ok=True)
        selected = selected | {"workloads": workload_entries(repo, root, selected["domain"])}
        selected["corpus_sha256"] = {
            w["workloads_file"]: digest(Path(w["workloads_file"])) for w in selected["workloads"]
        }
        selected_path = spec_dir / (name + "-selection.json")
        immutable_json(selected_path, selected)
        out = root / "confirmations" / name
        receipt = state["confirmations"].setdefault(name, {})
        report_path = out / "paired" / "paired" / "confirmation-summary.json"
        if not receipt.get("job_id"):
            if out.exists():
                raise RuntimeError(
                    f"Unreceipted confirmation output; inspect before submission: {out}"
                )
            spec = {
                "name": name + "-confirmation",
                "stage": "8b",
                "loadtests": [{"workload": "opb", "concurrencies": [1]}],
                "arms": [
                    {
                        "name": "paired",
                        "method": "command",
                        "gpus": 4,
                        "pythonpath": [str(repo / "methods/eagle")],
                        "command": [
                            "python3",
                            "-m",
                            "apertus_bench.adaptation_confirmation",
                            "--selection",
                            str(selected_path),
                            "--policy",
                            str(root / "preregistration.json"),
                            "--environment",
                            str(out / "environment.toml"),
                            "--output",
                            "{arm_dir}/paired",
                        ],
                    }
                ],
            }
            spec_path = spec_dir / (name + ".yaml")
            immutable_json(spec_path, spec)  # JSON is a strict subset of YAML.
            confirmation_env = env | {
                "SWEEP_PARTITION": policy["confirmation"]["partition"],
                "SWEEP_TIME": policy["confirmation"]["wall_time_per_head"],
                "SWEEP_OUTPUT": str(out),
                "VLLM_PATCH_DIR": str(short_root / ("s-" + selected["domain"])),
                "SML_ENV_DIR": str(root / "serving-environments" / name),
                "EXTRA_MOUNTS": "",
                "SWEEP_PATCHES": "",
                "ENV_SOURCE": str(repo / "methods/eagle/configs/train-env.toml"),
            }
            submitted = command(["bash", "serving/node-sweep.sh", str(spec_path)], confirmation_env)
            (receipts / f"submit-confirm-{name}.txt").write_text(submitted)
            job = re.findall(r"job=(\d+) output=", submitted)[-1]
            receipt.update(job_id=job, output=str(out), report=str(report_path))
            state["job_ids"].append(job)
            save(state="waiting", next_action=f"Wait for paired serving confirmation: {name}")
        terminal = wait(receipt["job_id"])
        if terminal != "COMPLETED" or not report_path.is_file():
            save(
                state="needs_repair",
                next_action=f"Inspect confirmation {name}; no automatic resubmission.",
            )
            return
        report = read(report_path)
        if not report["correctness_passed"]:
            save(
                state="needs_repair", next_action=f"Inspect calibrated correctness failure: {name}"
            )
            return
        receipt["state"] = "complete"
        confirmations.append(report)
        save(state="collecting")
    jobs = ",".join(state["job_ids"])
    accounting = command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            jobs,
            "--format=JobIDRaw,ElapsedRaw,AllocNodes,State,ExitCode",
        ]
    )
    (receipts / "allocation-accounting.txt").write_text(accounting)
    allocations = {}
    for line in accounting.splitlines():
        fields = line.split("|")
        if fields[0] in state["job_ids"]:
            allocations[fields[0]] = {
                "elapsed_seconds": int(fields[1]),
                "nodes": int(fields[2]),
                "gpu_seconds": 4 * int(fields[1]) * int(fields[2]),
                "state": fields[3],
                "exit_code": fields[4],
            }
    if set(allocations) != set(state["job_ids"]):
        raise RuntimeError("Incomplete allocation accounting; refusing an unsupported cost claim")
    compute = {
        "trace_allocated_gpu_seconds": sum(
            allocations[j]["gpu_seconds"] for j in state.get("trace_job_ids", [args.trace_job])
        ),
        "whole_campaign_allocated_gpu_seconds": sum(v["gpu_seconds"] for v in allocations.values()),
        "domain_training_allocated_gpu_seconds": {
            domain: sum(allocations[j]["gpu_seconds"] for j in batch["job_ids"])
            for domain, batch in state["batches"].items()
        },
        "basis": (
            "4 physical GPUs allocated per exclusive Clariden node, including startup"
            " and failed attempts; original historical runs excluded."
        ),
        "jobs": allocations,
    }
    verdict = final_verdict(selection, confirmations, policy, compute)
    verdict_path = root / "verdict.json"
    immutable_json(verdict_path, verdict)
    lines = [
        "# T5 domain adaptation verdict",
        "",
        verdict["verdict"],
        "",
        (
            "Simulated acceptance was used only for development selection. Serving "
            "cells use paired GPU-swapped repeats."
        ),
        "",
    ]
    for head in verdict["heads"]:
        lines.append(
            f"- {head['name']}: {head['recommendation']}; "
            f"forgetting regression: {head['forgetting_regression']}."
        )
        benefits = [
            cell
            for cell in head["cells"]
            if cell["role"] != "forgetting" and cell["substantial_repeatable_benefit"]
        ]
        benefits.sort(
            key=lambda cell: -min(p["ratios"]["output_tokens_per_second"] for p in cell["pairs"])
        )
        for cell in benefits[:1]:  # Compact synopsis; every cell remains in verdict.json.
            if cell["substantial_repeatable_benefit"]:
                p = cell["pairs"][0]
                lines.append(
                    f"  - {cell['label']} k={cell['k']} C={cell['concurrency']}: "
                    f"{cell['verdict']}; first-repeat throughput "
                    f"{p['baseline']['output_tokens_per_second']:.1f} to "
                    f"{p['adapted']['output_tokens_per_second']:.1f} tok/s, median TPOT "
                    f"{p['baseline']['tpot_p50_ms']:.2f} to {p['adapted']['tpot_p50_ms']:.2f} ms. "
                    "Both repeats and absolute cost/latency tradeoffs are in verdict.json."
                )
    if not selection["selected"]:
        lines.append(
            "No final head passed the original scientific gates and the preregistered"
            " development gain threshold; no serving promotion was attempted."
        )
    lines += [
        "",
        (
            "Calibrated correctness is consistent with losslessness, not proof. No "
            "production deployment or routing change was made."
        ),
        f"Whole-recovery allocated compute: "
        f"{compute['whole_campaign_allocated_gpu_seconds'] / 3600:.2f} GPU-hours. "
        "No monetary tariff assumed.",
    ]
    summary_path = root / "verdict.md"
    if not summary_path.exists():
        summary_path.write_text("\n".join(lines) + "\n")
    save(
        state=verdict["state"],
        verdict=verdict["verdict"],
        conditional_dspark_followup=verdict["conditional_dspark_followup"],
        artifact_paths=[
            str(matrix),
            str(root / "selection.json"),
            str(verdict_path),
            str(summary_path),
        ],
        next_action=(
            "Original bounded workflow finished; collect compact verdict/provenance "
            "for the experiment canvas. No deployment."
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        raise  # Another live controller owns the lock; never alter its receipt.
    except Exception as error:
        import sys

        root = Path(sys.argv[sys.argv.index("--campaign-root") + 1])
        path = root / "receipts/campaign.json"
        state = read(path) if path.exists() else {"workstream": "adaptation", "job_ids": []}
        state.update(
            state="needs_repair",
            error=f"{type(error).__name__}: {error}",
            next_action="Repair the recorded failure before resuming owned work.",
            updated_at=stamp(),
        )
        write_state(path, state)
        raise
