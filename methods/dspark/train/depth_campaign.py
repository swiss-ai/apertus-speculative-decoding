"""Fail-closed, bounded post-training controller for the two-run depth experiment.

The submitter freezes only executable sources, never weights or earlier results.
Each stage runs in its own Slurm allocation, afterok-dependent on its predecessor.
Operational campaign state is replaceable; scientific artifacts are write-once.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import depth_recovery as recovery

ROOT = Path(__file__).resolve().parents[3]
TRAIN_FILES = (
    "methods/dspark/train/train-node.sh",
    "methods/dspark/train/draft-init-args.sh",
    "methods/dspark/train/common.sh",
    "methods/dspark/train/rank-wrapper.sh",
    "methods/dspark/train/submit-train.sh",
    "methods/dspark/train/rebind_verifier.py",
    "methods/dspark/train/depth_recovery.py",
)


class CorrectnessGateFailed(ValueError):
    """Completed captures exceed the unchanged calibrated correctness limits."""


def state(campaign: Path, **changes: object) -> None:
    path = campaign / "campaign.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(changes, updated_at=dt.datetime.now(dt.UTC).isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    with (campaign / "events.jsonl").open("a") as out:
        out.write(json.dumps({"at": previous["updated_at"], **changes}) + "\n")


def verify_sources(campaign: Path) -> dict:
    manifest = json.loads((campaign / "source-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if recovery.sha256(campaign / "source" / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    # The submitted training script refers to this dedicated checkout. Preserve
    # before/after observations rather than mislabelling the printed SHA as an
    # assertion that every runtime file came from that revision.
    for relative, expected in manifest["training_files"].items():
        if recovery.sha256(Path(manifest["training_repo"]) / relative) != expected:
            raise ValueError(f"training launch source changed: {relative}")
    return manifest


def submitted_training(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    exports = {}
    for line in text.splitlines():
        if not line.startswith("export "):
            continue
        tokens = shlex.split(line)
        if len(tokens) == 2 and "=" in tokens[1]:
            key, value = tokens[1].split("=", 1)
            exports[key] = value
    for key in ("DSPARK_DIR", "RUN_ID", "DATA", "FROM_PRETRAINED"):
        if not exports.get(key):
            raise ValueError(f"training submission missing {key}")
    match = re.search(r"git=([0-9a-f]{7,40})", text)
    if match is None:
        raise ValueError("training submission does not record its source revision")
    return exports, match[1]


def completed_training_prerequisite(campaign: Path, accounting_path: Path) -> None:
    source = verify_sources(campaign)
    rows = [line.split("|") for line in accounting_path.read_text().splitlines() if line.strip()]
    matching = [row for row in rows if row[0].strip() == source["training_job"]]
    if len(matching) != 1 or matching[0][1:3] != ["COMPLETED", "0:0"]:
        raise ValueError("training prerequisite is not an accounting-confirmed successful job")
    checkpoint = Path(source["candidate"])
    state_path, config_path = checkpoint / "training_state.json", checkpoint / "config.json"
    state_value = json.loads(state_path.read_text())
    config = json.loads(config_path.read_text())
    if (
        state_value["global_step"] != 6200
        or config["transformer_layer_config"]["num_hidden_layers"] != 1
        or config["aux_hidden_state_layer_ids"] != [1, 8, 15, 22, 29]
    ):
        raise ValueError("completed training checkpoint has wrong step count or model definition")
    validation_path = checkpoint / "val_metrics.json"
    eal = json.loads(validation_path.read_text())["eal_epoch"]
    if not math.isfinite(eal) or not 1 <= eal <= 8:
        raise ValueError("completed training has no valid held-out validation")
    identity_path = Path(source["warm_start_preflight_report"])
    identity = json.loads(identity_path.read_text())
    if identity.get("passed") is not True or identity["initializer"] != source["warm_start"]:
        raise ValueError("completed training identity gate did not pass")
    recovery.write_new(
        campaign / "completed-training-prerequisite.json",
        {
            "training_job": source["training_job"],
            "passed": True,
            "state": state_value,
            "validation_eal": eal,
            "metadata_sha256": {
                str(path): recovery.sha256(path)
                for path in (
                    state_path,
                    config_path,
                    validation_path,
                    identity_path,
                    accounting_path,
                )
            },
            "scheduler_note": "completed job aged out of live Slurm dependency records; "
            "accounting/checkpoint verified; full hash gates rerun in allocation",
        },
    )


def snapshot(
    campaign: Path, training_repo: Path, revision: str, train_job: str, training_submit: Path
) -> None:
    import shutil

    data = Path("/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter")
    speculators = data / "speculators"
    source = campaign / "source"
    exports, launch_revision = submitted_training(training_submit)
    if exports["DSPARK_DIR"] != str(data) or exports["DATA"] != str(data / "data"):
        raise ValueError("recovery submission changed the original data roots")
    run_root = data / "runs" / exports["RUN_ID"]
    identity_path = run_root / "provenance" / f"warm-start-{train_job}.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    training_files = {}
    for relative in TRAIN_FILES:
        original = subprocess.check_output(
            ["git", "-C", str(training_repo), "show", f"{launch_revision}:{relative}"]
        )
        expected = hashlib.sha256(original).hexdigest()
        if recovery.sha256(training_repo / relative) != expected:
            raise ValueError(f"launch identity mismatch: {relative}")
        training_files[relative] = expected
    revision_spec = subprocess.check_output(
        ["git", "-C", str(speculators), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(speculators), "diff", "HEAD"])
    patch = f"# repo: {speculators} ({revision_spec})\n".encode() + diff.strip()
    (campaign / "speculators-at-queue.patch").write_bytes(patch)
    original_submit = training_submit
    shutil.copyfile(original_submit, campaign / "training-submit.sbatch")
    environment = re.search(r"--environment=([^ ]+)", original_submit.read_text())[1]
    shutil.copyfile(environment, campaign / "training-environment.toml")
    recovery.write_new(
        campaign / "source-manifest.json",
        {
            "revision": revision,
            "branch": subprocess.check_output(
                ["git", "-C", str(training_repo), "branch", "--show-current"], text=True
            ).strip(),
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "files": {
                str(p.relative_to(source)): recovery.sha256(p)
                for p in sorted(source.rglob("*"))
                if p.is_file()
            },
            "training_files": training_files,
            "training_repo": str(training_repo),
            "training_job": train_job,
            "printed_training_launch_revision": launch_revision,
            "training_source_observation_revision": subprocess.check_output(
                ["git", "-C", str(training_repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "speculators_revision": revision_spec,
            "speculators_patch_sha256": hashlib.sha256(patch).hexdigest(),
            "candidate": str(run_root / "checkpoints/epoch0_end"),
            "control": str(data / "runs/depth-l5-scratch/checkpoints/epoch0_end"),
            "data": str(data / "data"),
            "warm_start": exports["FROM_PRETRAINED"],
            "warm_start_preflight_report": str(
                run_root / "provenance" / f"warm-start-{train_job}.json"
            ),
            "workload": recovery.DEFAULT_WORKLOAD,
            "e10": recovery.DEFAULT_E10,
            "eagle": recovery.DEFAULT_EAGLE,
            "image": recovery.DEFAULT_IMAGE,
            "expected_serving_image_sha256": identity.get("container", {}).get("sha256"),
        },
    )
    state(
        campaign,
        workstream="shallow_drafter",
        state="submitting",
        job_ids=[train_job],
        training_job=train_job,
        source_revision=revision,
        next_action="afterok validation and calibrated correctness; then counterbalanced a/b",
    )


def check_training_recipe(candidate: Path, manifest: dict) -> dict:
    import yaml

    recipe = yaml.safe_load((candidate.parent / "run.yaml").read_text())["train"]
    expected = {
        ("speculator_type",): "dspark",
        ("data", "data_path"): manifest["data"],
        ("data", "total_seq_len"): 16384,
        ("data", "max_anchors"): 1024,
        ("data", "num_workers"): 8,
        ("data", "prefetch_factor"): 4,
        ("optimizer", "optimizer"): "muon",
        ("optimizer", "lr"): 0.0003,
        ("loss", "loss_implementation"): "eager",
        ("trainer", "max_steps"): 6200,
        ("trainer", "epochs"): 1,
        ("trainer", "checkpoint_freq"): 0.0625,
        ("dflash", "block_size"): 8,
        ("dspark", "markov_rank"): 256,
        ("dspark", "enable_confidence_head"): True,
        ("dspark", "confidence_head_with_markov"): True,
    }
    for keys, value in expected.items():
        observed = recipe
        for key in keys:
            observed = observed[key]
        if observed != value:
            raise ValueError(f"training recipe changed: {keys}: {observed!r} != {value!r}")
    if json.loads(recipe["loss"]["loss_fn"]) != {"ce": 0.1, "tv": 0.9}:
        raise ValueError("training loss mix changed")
    command_path = candidate.parent / "train_command.txt"
    command = command_path.read_text()
    argv = shlex.split(next(line for line in command.splitlines() if not line.startswith("#")))
    if "--num-layers" in argv or "--fsdp-shard" in argv:
        raise ValueError("conflicting model definition or changed FSDP configuration")
    if argv[argv.index("--from-pretrained") + 1] != manifest["warm_start"]:
        raise ValueError("wrong warm start")
    for required in ("# World size: 4", f"# Git SHA: {manifest['speculators_revision']}"):
        if required not in command:
            raise ValueError(f"training provenance does not contain {required}")
    patch_path = candidate.parent / "speculators.patch"
    if recovery.sha256(patch_path) != manifest["speculators_patch_sha256"]:
        raise ValueError("Speculators training source diff changed")
    warm_preflight = None
    if "warm_start_preflight_report" in manifest:
        warm_preflight = json.loads(Path(manifest["warm_start_preflight_report"]).read_text())
        if (
            warm_preflight.get("passed") is not True
            or warm_preflight["initializer"] != manifest["warm_start"]
        ):
            raise ValueError("allocated initializer/target hash gate did not pass")
    return {
        "recipe": recipe,
        "command_sha256": recovery.sha256(command_path),
        "patch_sha256": recovery.sha256(patch_path),
        "training_source_observation": manifest["training_files"],
        "warm_start_preflight": warm_preflight,
        "observation_limit": "matching training sources before queue and after training; argv and "
        "Speculators patch recorded by the training process itself",
    }


def capture_paths(root: Path, arm: str) -> list[Path]:
    return [root / arm / "correctness" / f"capture-{repeat}.json" for repeat in (1, 2)]


def calibrated_checks(a: Path, b: Path, arms: list[str], output: Path) -> list[dict]:
    from apertus_bench.correctness import compare_captures, run_correctness_gate

    for arm in ["plain", *arms]:
        aa, bb = capture_paths(a, arm), capture_paths(b, arm)
        for path in [*aa, *bb]:
            capture = json.loads(path.read_text())
            if len(capture["cases"]) != 6 or not all(
                case["measurement"]["success"] for case in capture["cases"]
            ):
                raise ValueError(f"{path}: incomplete calibration capture")
        for left, right in [aa, bb, (aa[0], bb[0]), (aa[1], bb[1])]:
            comparison = compare_captures(
                json.loads(left.read_text()), json.loads(right.read_text())
            )
            if comparison["comparable_cases"] != 6:
                raise ValueError("incomplete calibration control")
        # Each treatment uses controls of its own configuration and plain;
        # controls from other k/depth values cannot loosen its envelope.
    reports = []
    for arm in arms:
        pairs = []
        for name in ("plain", arm):
            aa, bb = capture_paths(a, name), capture_paths(b, name)
            pairs.extend([tuple(aa), tuple(bb), (aa[0], bb[0]), (aa[1], bb[1])])
        for number, root in enumerate((a, b), 1):
            for repeat in (1, 2):
                report = run_correctness_gate(
                    tuple(
                        [
                            capture_paths(root, "plain")[repeat - 1],
                            capture_paths(root, arm)[repeat - 1],
                        ]
                    ),
                    pairs,
                    output / f"{arm}-deployment{number}-capture{repeat}.json",
                )
                reports.append({"arm": arm, "deployment": number, "capture": repeat, **report})
    if not all(report["passed"] for report in reports):
        raise CorrectnessGateFailed("calibrated correctness exceeded the existing protocol margins")
    return reports


def make_smoke(workload: Path, output: Path) -> None:
    # Fixed, declared six-prompt smoke subset of the original held-out OPB set.
    # Full throughput passes still use every prompt and its original cap.
    records = [json.loads(line) for line in workload.read_text().splitlines() if line.strip()]
    if len(records) < 6:
        raise ValueError("insufficient smoke prompts")
    with output.open("x") as out:
        for record in records[:6]:
            out.write(json.dumps(record) + "\n")


def require_same_image(source: dict, preflight: dict) -> None:
    expected = source.get("expected_serving_image_sha256")
    if expected is not None and preflight["image"]["sha256"] != expected:
        raise ValueError("serving image differs from the completed training identity gate")


def collect_training_evidence(candidate: Path, output: Path, job: str) -> None:
    import gzip
    import shutil

    output.mkdir(exist_ok=False)
    files = [
        *(candidate / name for name in ("config.json", "training_state.json", "val_metrics.json")),
        *(
            candidate.parent / name
            for name in ("run.yaml", "train_command.txt", "speculators.patch")
        ),
    ]
    preflight = candidate.parent.parent / "provenance" / f"warm-start-{job}.json"
    if preflight.is_file():
        files.append(preflight)
    log_root = candidate.parent.parent / "logs"
    files.extend(
        path
        for path in sorted(log_root.rglob("*"))
        if path.is_file() and (job in path.name or job in str(path.parent))
    )
    entries = []
    for source in files:
        relative = source.relative_to(candidate.parent.parent)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        compressed = source.suffix in {".log", ".out", ".csv"}
        if compressed:
            destination = destination.with_suffix(destination.suffix + ".gz")
            with source.open("rb") as src, gzip.open(destination, "xb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(source, destination)
        entries.append(
            {
                "source": str(source),
                "source_sha256": recovery.sha256(source),
                "artifact": str(destination.relative_to(output)),
                "gzip": compressed,
            }
        )
    recovery.write_new(output / "collection-manifest.json", {"entries": entries})


async def run_spec(spec: dict, output: Path, *, capture_only: bool = False) -> None:
    from apertus_bench import node_sweep as sweep

    output.mkdir(parents=True, exist_ok=False)
    recovery.write_new(output / "spec.yaml", spec)
    plan = sweep.build_plan(spec)
    if "DEPTH_SOURCE_MANIFEST" in os.environ:
        source_manifest = Path(os.environ["DEPTH_SOURCE_MANIFEST"])
        revision = json.loads(source_manifest.read_text())["revision"]
        for arm in plan.arms:
            arm.metadata.update(
                source_revision=revision,
                source_manifest_sha256=recovery.sha256(source_manifest),
                source_manifest_path=str(source_manifest),
                preflight_manifest_path=str(output / "preflight/manifest.json"),
            )
    if capture_only:
        plan = dataclasses.replace(plan, loadtests=[], probe=None)
    # The generic sweep lets other arms continue after a failed command. This
    # campaign executes its all-GPU preflight separately and aborts on failure.
    preflight = [arm for arm in plan.arms if arm.name == "preflight"]
    statuses = {}
    if preflight:
        result = await sweep.run_arm(preflight[0], list(range(4)), plan, output, output.name)
        statuses["preflight"] = result["status"]
        if result["status"] != "ok":
            recovery.write_new(output / "sweep-status.json", statuses)
            raise ValueError("compute preflight failed; no serving arms started")
        if "DEPTH_SOURCE_MANIFEST" in os.environ:
            require_same_image(
                json.loads(Path(os.environ["DEPTH_SOURCE_MANIFEST"]).read_text()),
                json.loads((output / "preflight/manifest.json").read_text()),
            )
    remaining = dataclasses.replace(
        plan, arms=[arm for arm in plan.arms if arm.name != "preflight"]
    )
    results = await sweep.run_plan(remaining, output)
    statuses.update({result["arm"]: result["status"] for result in results})
    (output / "sweep-status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if any(value != "ok" for value in statuses.values()):
        raise ValueError(f"{output}: serving or measurement failure")


def runtime_provenance(campaign: Path, stage: str, manifest: dict) -> dict:
    overlay = Path(os.environ["VLLM_PATCH_DIR"])
    environment = Path(os.environ["DEPTH_RESOLVED_ENV"])
    return {
        "source_revision": manifest["revision"],
        "source_manifest_sha256": recovery.sha256(campaign / "source-manifest.json"),
        "job_id": os.environ["SLURM_JOB_ID"],
        "nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "stage": stage,
        "overlay_files": {
            str(path.relative_to(overlay)): recovery.sha256(path)
            for path in sorted(overlay.rglob("*"))
            if path.is_file()
        },
        "resolved_environment": str(environment),
        "environment_sha256": recovery.sha256(environment),
    }


def execute(campaign: Path, stage: str) -> None:
    manifest = verify_sources(campaign)
    os.environ["DEPTH_SOURCE_MANIFEST"] = str(campaign / "source-manifest.json")
    run = campaign / stage
    run.mkdir(exist_ok=False)
    state(campaign, state="running", stage=stage, active_job=os.environ["SLURM_JOB_ID"])
    recovery.write_new(
        run / "runtime-provenance.json", runtime_provenance(campaign, stage, manifest)
    )
    candidate, control = manifest["candidate"], manifest["control"]
    smoke = campaign / "smoke.jsonl"
    if stage == "validate":
        recipe = check_training_recipe(Path(candidate), manifest)
        recovery.write_new(run / "training-recipe-checked.json", recipe)
        collect_training_evidence(
            Path(candidate), run / "training-evidence", manifest["training_job"]
        )
        make_smoke(Path(manifest["workload"]), smoke)
    else:
        if not json.loads((campaign / "validate" / "gate.json").read_text())["passed"]:
            raise ValueError("validation/correctness prerequisite absent")
        if stage == "b":
            recovery.read_sweep(campaign / "a" / "sweep")
    a, b = recovery.specs(
        candidate, control, manifest["e10"], manifest["eagle"], manifest["workload"]
    )
    for spec in (a, b):
        spec["captures"] = {"workloads_file": str(smoke), "repeats": 2, "max_tokens": 384}
    if stage == "validate":
        # k7 is the most speculative original configuration, checked before
        # benchmark allocations. All measured k values also receive their own
        # captures in a/b and must pass individually before final acceptance.
        base = {
            **a,
            "name": "depth-validation-a",
            "gpu_order": [0, 1, 2, 3],
            "arms": [
                arm
                for arm in a["arms"]
                if arm["name"]
                in {"preflight", "plain", "dspark-e10-k7", "depth-l1-k7", "depth-l5-k7"}
            ],
        }
        repeat = {
            **base,
            "name": "depth-validation-b",
            "gpu_order": [3, 2, 1, 0],
            "arms": [arm for arm in base["arms"] if arm["name"] != "preflight"],
        }
        asyncio.run(run_spec(base, run / "repeat-a", capture_only=True))
        asyncio.run(run_spec(repeat, run / "repeat-b", capture_only=True))
        reports = calibrated_checks(
            run / "repeat-a",
            run / "repeat-b",
            ["dspark-e10-k7", "depth-l1-k7", "depth-l5-k7"],
            run / "correctness",
        )
        recovery.write_new(
            run / "gate.json",
            {"passed": True, "reports": reports, "scope": "six OPB prompts, k7, 384-token cap"},
        )
        state(campaign, state="waiting", stage="validate_passed", next_action="dependent sweep a")
    else:
        asyncio.run(run_spec(a if stage == "a" else b, run / "sweep"))
        recovery.read_sweep(run / "sweep")
        if stage == "a":
            state(campaign, state="waiting", stage="a_passed", next_action="dependent sweep b")
        else:
            # Refuse even a negative *valid comparison* if correctness or
            # infrastructure failed. That is unresolved, not evidence of no gain.
            arms = [arm["name"] for arm in a["arms"] if arm["name"] not in {"preflight", "plain"}]
            reports = calibrated_checks(
                campaign / "a" / "sweep", run / "sweep", arms, run / "correctness"
            )
            result = recovery.decision(campaign / "a" / "sweep", run / "sweep")
            result.update(
                correctness_confirmation_required_before_recommendation=False,
                calibrated_correctness_passed=True,
                correctness_reports=reports,
            )
            positive = result["gate_g3_repeatable_throughput_passed"]
            result["verdict"] = (
                "original_G3_pass_implementation_review_pending" if positive else "negative"
            )
            recovery.write_new(run / "decision.json", result)
            strict_path = campaign / "strict-review" / "final-review.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "methods/dspark/train/depth_review.py"),
                    str(campaign / "a/sweep"),
                    str(run / "sweep"),
                    "--raw-decision",
                    str(run / "decision.json"),
                    "--output",
                    str(strict_path),
                ],
                check=True,
            )
            strict = json.loads(strict_path.read_text())
            state(
                campaign,
                state="complete"
                if strict["implementation_evidence_passed"]
                else "negative_complete",
                stage="concluded",
                decision_path=str(run / "decision.json"),
                strict_review_path=str(strict_path),
                original_G3_passed=positive,
                implementation_verdict=strict["implementation_verdict"],
                next_action="collect compact evidence and update verdict",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "verify", "run", "fail", "resume"])
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--stage", choices=["validate", "a", "b"])
    parser.add_argument("--training-repo", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--train-job")
    parser.add_argument("--training-submit", type=Path)
    parser.add_argument("--accounting", type=Path)
    args = parser.parse_args()
    if args.action == "resume":
        completed_training_prerequisite(args.campaign, args.accounting)
        return
    if args.action == "fail":
        previous = json.loads((args.campaign / "campaign.json").read_text())
        if previous.get("state") == "failed" and previous.get("error"):
            state(args.campaign, host_launcher_also_failed=True)
            return
        state(
            args.campaign,
            state="failed",
            stage=args.stage,
            error="host launcher failed; inspect stage job.out and preserved submit script",
            next_action="stop; diagnose infrastructure failure; no scientific verdict",
        )
        return
    if args.action == "snapshot":
        snapshot(
            args.campaign, args.training_repo, args.revision, args.train_job, args.training_submit
        )
        return
    if args.action == "verify":
        verify_sources(args.campaign)
        return
    try:
        execute(args.campaign, args.stage)
    except BaseException as error:
        state(
            args.campaign,
            state="failed",
            stage=args.stage,
            error=f"{type(error).__name__}: {error}",
            failure_kind="correctness"
            if isinstance(error, CorrectnessGateFailed)
            else "infrastructure",
            next_action=(
                "stop; diagnose preserved infrastructure/correctness artifacts; "
                "no scientific verdict"
            ),
        )
        raise


if __name__ == "__main__":
    main()
