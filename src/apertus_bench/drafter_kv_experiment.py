"""Bounded, provenance-preserving controller for the drafter-KV experiment.

This module never starts another phase automatically. The submitter is an explicit
entrypoint; reservations survive ambiguous submission failures. Allocation hours
count all four GPUs, including pending reservations and startup/idle time.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

PHASE_LIMITS = {"preflight": 6.0, "screen": 12.0, "confirm": 30.0, "on": 12.0, "eagle": 12.0}
TOTAL_GPU_HOURS = 72.0
GPUS = 4
MAX_SECONDS = 5400
PHASE_REPEATS = {"screen": 2, "confirm": 5, "on": 2, "eagle": 2}
SCREEN_POLICIES = ("native2048", "recent512", "importance1024", "importance512")
CANDIDATES = ("importance1024", "importance512")
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
HEAD_PATHS = {
    "off": "/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/"
    "dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final",
    "on": "/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/"
    "dspark_apertus15-8b_open-perfectblend_thinking-on_2026-10-10_epoch10_final",
    "eagle": "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/heads/"
    "e31-opb-thinkoff-stage1-se",
}


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, data: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".ledger-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def ledger(campaign: Path):
    with (campaign / ".ledger.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = campaign / "ledger.json"
        data = json.loads(path.read_text())
        yield data
        _atomic_json(path, data)


def _name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", value):
        raise ValueError(f"unsafe name: {value!r}")
    return value


def init_campaign(campaign: Path, campaign_id: str) -> dict[str, Any]:
    campaign_id = _name(campaign_id)
    campaign.mkdir(parents=True, exist_ok=False)
    data = {
        "schema_version": 1, "campaign_id": campaign_id, "created_at": now(),
        "phase_limits_gpu_hours": PHASE_LIMITS, "total_limit_gpu_hours": TOTAL_GPU_HOURS,
        "allocated_gpus_per_node": GPUS, "allocations": {}, "phase_authorizations": {},
        "halted": False,
        "predeclared_speed_cell": {"workload": "summarization", "concurrency": 256},
    }
    exclusive_json(campaign / "campaign.json", data)
    exclusive_json(campaign / "ledger.json", data)
    (campaign / "allocations").mkdir()
    return data


def budget(data: dict[str, Any]) -> dict[str, Any]:
    phases = dict.fromkeys(PHASE_LIMITS, 0.0)
    actual = 0.0
    for allocation in data["allocations"].values():
        if allocation["state"] == "released_before_submit":
            continue
        used = allocation.get("elapsed_seconds", 0) * GPUS / 3600
        actual += used
        charge = used if allocation["state"] == "terminal" else max(
            used, allocation["reserved_seconds"] * GPUS / 3600
        )
        phases[allocation["phase"]] += charge
    return {
        "actual_gpu_hours": actual, "committed_gpu_hours": sum(phases.values()),
        "remaining_gpu_hours": TOTAL_GPU_HOURS - sum(phases.values()),
        "phase_committed_gpu_hours": phases,
        "phase_remaining_gpu_hours": {p: PHASE_LIMITS[p] - h for p, h in phases.items()},
    }


def reserve(
    campaign: Path, phase: str, allocation: str, seconds: int,
    command: list[str] | None = None,
) -> dict[str, Any]:
    _name(allocation)
    if phase not in PHASE_LIMITS or not 0 < seconds <= MAX_SECONDS:
        raise ValueError("invalid phase or walltime (must be 1..5400 seconds)")
    with ledger(campaign) as data:
        if data["halted"]:
            raise ValueError("campaign halted; no more allocations may be reserved")
        if phase in {"confirm", "on", "eagle"} and phase not in data["phase_authorizations"]:
            raise ValueError(f"{phase} requires an explicit recorded parent gate")
        if allocation in data["allocations"]:
            raise FileExistsError(f"allocation already recorded: {allocation}")
        if phase in PHASE_REPEATS:
            count = sum(a["phase"] == phase and a["state"] != "released_before_submit"
                        for a in data["allocations"].values())
            if count >= PHASE_REPEATS[phase]:
                raise ValueError(f"{phase} already has its declared independent allocations")
        status = budget(data)
        hours = seconds * GPUS / 3600
        if hours > status["remaining_gpu_hours"] + 1e-9:
            raise ValueError("total allocation budget exhausted, including pending reservations")
        if hours > status["phase_remaining_gpu_hours"][phase] + 1e-9:
            raise ValueError(f"{phase} phase allocation budget exhausted")
        output = campaign.resolve() / "allocations" / allocation
        output.mkdir(exist_ok=False)
        item = {
            "phase": phase, "allocation": allocation, "reserved_at": now(),
            "reserved_seconds": seconds, "allocated_gpus": GPUS, "state": "reserved",
            "output": str(output / "results"), "command": command,
            "job_comment": (f"apertus-drafter-kv:{data['campaign_id']}:"
                            f"{allocation}:{uuid.uuid4().hex}"),
            "submission_attempted": False,
        }
        exclusive_json(output / "reservation.json", item)
        data["allocations"][allocation] = item
    return item


def mark_submission(campaign: Path, allocation: str) -> dict[str, Any]:
    with ledger(campaign) as data:
        item = data["allocations"][allocation]
        if item["state"] != "reserved" or item["submission_attempted"]:
            raise ValueError("allocation has already been submitted or released")
        if data["halted"]:
            raise ValueError("campaign halted")
        item["submission_attempted"] = True
        item["state"] = "submission_unknown"
        item["submission_started_at"] = now()
    return item


def register(campaign: Path, allocation: str, job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("only one numeric Slurm job ID may be registered")
    with ledger(campaign) as data:
        item = data["allocations"][allocation]
        if item["state"] not in {"reserved", "submission_unknown"}:
            raise ValueError("allocation cannot be registered again")
        if any(a.get("job_id") == job_id for a in data["allocations"].values()):
            raise ValueError("job already belongs to an allocation")
        item.update({"job_id": job_id, "state": "submitted", "registered_at": now(),
                     "submission_attempted": True})
        exclusive_json(campaign / "allocations" / allocation / "job-registration.json", item)
    return item


def release_unsubmitted(campaign: Path, allocation: str) -> dict[str, Any]:
    with ledger(campaign) as data:
        item = data["allocations"][allocation]
        if item["submission_attempted"] or item.get("job_id") or item["state"] != "reserved":
            raise ValueError("cannot prove no sbatch submission; keep reservation and reconcile")
        item["state"] = "released_before_submit"
        item["released_at"] = now()
    return item


def authorize_phase(campaign: Path, phase: str, evidence: Path) -> dict[str, Any]:
    if phase not in {"confirm", "on", "eagle"}:
        raise ValueError("only confirm/on/eagle are explicitly gated")
    report = json.loads(evidence.read_text())
    if not report.get("passed"):
        raise ValueError("gate report must record passed=true")
    if report.get("phase") != ({"confirm": "screen", "on": "confirm", "eagle": "confirm"}[phase]):
        raise ValueError("wrong source phase in gate report")
    if phase in {"on", "eagle"} and not report.get("promising"):
        raise ValueError("extensions require the confirmation report to be promising")
    if not report.get("correctness_controls_passed"):
        raise ValueError("native/candidate correctness controls must pass")
    if report.get("candidate") not in CANDIDATES:
        raise ValueError("extension candidate must be a screened importance policy")
    with ledger(campaign) as data:
        if phase in data["phase_authorizations"]:
            raise FileExistsError("phase already authorized")
        entry = {"authorized_at": now(), "evidence": str(evidence.resolve()),
                 "evidence_sha256": digest(evidence), "candidate": report["candidate"],
                 "speed_cell": report.get("selected_speed_cell", data["predeclared_speed_cell"])}
        data["phase_authorizations"][phase] = entry
        exclusive_json(campaign / f"{phase}-authorization.json", entry)
    return entry


def reconcile(campaign: Path) -> dict[str, Any]:
    with ledger(campaign) as data:
        jobs = {a["job_id"]: a for a in data["allocations"].values() if a.get("job_id")}
        if not jobs:
            return budget(data)
        result = subprocess.run(
            ["sacct", "-j", ",".join(jobs), "--noheader", "--parsable2", "--format",
             "JobIDRaw,State,ElapsedRaw,AllocNodes,AllocTRES"],
            check=True, text=True, capture_output=True,
        )
        for line in result.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 5 or fields[0] not in jobs:
                continue  # job steps do not count as additional allocations
            job_id, state, elapsed, nodes, tres = fields[:5]
            item = jobs[job_id]
            state = state.split()[0].rstrip("+")
            if int(nodes or 0) > 1:
                raise ValueError(f"unexpected multi-node allocation {job_id}; halt campaign")
            gpu = re.search(r"(?:^|,)gres/gpu(?:[:=][^,=]+)?=([0-9]+)(?:,|$)", tres)
            if gpu and int(gpu.group(1)) != GPUS:
                raise ValueError(f"unexpected GPU count for {job_id}; halt campaign")
            item["elapsed_seconds"] = max(item.get("elapsed_seconds", 0), int(elapsed or 0))
            item["slurm_state"] = state
            item["accounting_seen_at"] = now()
            if state in TERMINAL:
                item["state"] = "terminal"
        return budget(data)


def stop_owned(campaign: Path, reason: str) -> dict[str, Any]:
    """Halt first, then cancel only recorded IDs with the matching Slurm comment."""
    with ledger(campaign) as data:
        data["halted"] = True
        data["halt_reason"] = reason
        data["halted_at"] = now()
        items = copy.deepcopy(list(data["allocations"].values()))
    cancelled, uncertain = [], []
    for item in items:
        if item["state"] == "terminal" or not item.get("job_id"):
            if item["state"] == "submission_unknown" and not item.get("job_id"):
                uncertain.append(item["allocation"])
            continue
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", item["job_id"]],
            text=True, capture_output=True, check=False,
        )
        if result.returncode or f"Comment={item['job_comment']}" not in result.stdout.split():
            uncertain.append(item["job_id"])
            continue
        subprocess.run(["scancel", item["job_id"]], check=True)
        cancelled.append(item["job_id"])
    report = {"halted": True, "reason": reason, "cancelled_owned_jobs": cancelled,
              "unresolved_jobs_or_submissions": uncertain, "recorded_at": now()}
    exclusive_json(campaign / f"halt-{uuid.uuid4().hex}.json", report)
    return report


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *argv], text=True, check=True,
                          capture_output=True).stdout.strip()


def freeze_inputs(
    campaign: Path, repo: Path, head: str, output: Path, *, hash_weights: bool,
    image: Path | None = None,
) -> dict[str, Any]:
    """Capture exact input identities. Full weight hashing is compute-node only."""
    if hash_weights and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("weight hashing must run in a recorded Slurm allocation")
    if head not in HEAD_PATHS:
        raise ValueError("head must be off/on/eagle")
    path = Path(HEAD_PATHS[head])
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("trained head has no safetensors weights")
    config = path / "config.json"
    contract = repo / "targets/8b/contract.json"
    files = {}
    for name in ["eagle-8b-validation.jsonl", "eagle-8b-test.jsonl", "opb-heldout.jsonl"]:
        p = repo / "workloads/8b" / name
        files[name] = {"path": str(p.resolve()), "sha256": digest(p)}
    smoke = output.with_name(output.stem + "-correctness.jsonl")
    selected = dict.fromkeys(["chat", "code", "summarization"], 0)
    with Path(files["eagle-8b-validation.jsonl"]["path"]).open() as source, smoke.open("x") as sink:
        for line in source:
            record = json.loads(line)
            workload = record["workload"]
            if workload not in selected or selected[workload] >= 2:
                continue
            selected[workload] += 1
            record["max_tokens"] = min(record["max_tokens"], 128)
            record["chat_template_kwargs"] = {"enable_thinking": head == "on"}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not all(n == 2 for n in selected.values()):
        raise ValueError("correctness smoke requires two validation prompts per stratum")
    files["correctness.jsonl"] = {"path": str(smoke.resolve()), "sha256": digest(smoke),
                                 "selection": "first two per validation stratum, cap128"}
    report = {
        "schema_version": 1, "created_at": now(), "head": head, "head_path": str(path),
        "head_config": json.loads(config.read_text()), "head_config_sha256": digest(config),
        "weights": [{"name": p.name, "bytes": p.stat().st_size,
                     "sha256": digest(p) if hash_weights else None} for p in weights],
        "weights_hashed": hash_weights, "corpora": files,
        "target_contract": json.loads(contract.read_text()),
        "target_contract_sha256": digest(contract), "repo": str(repo.resolve()),
        "git_revision": _git(repo, "rev-parse", "HEAD"),
        "git_tracked_diff_sha256": hashlib.sha256(
            _git(repo, "diff", "HEAD", "--binary").encode()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_id": os.environ.get("SLURMD_NODENAME"),
    }
    artifact_root = output.with_name(output.stem + "-artifacts")
    artifact_root.mkdir(exist_ok=False)
    artifacts = [config, path / "run.yaml", path / "train_command.txt", path / "speculators.patch",
                 path / "training_state.json", path / "val_metrics.json",
                 path / "export-report.json"]
    run_dir = report["head_config"].get("apertus_training", {}).get("run_dir")
    if run_dir:
        report["training_run_dir"] = run_dir
        artifacts += [Path(run_dir) / name for name in [
            "train-summary.json", "provenance.json", "evals.json", "history.json"]]
    report["training_and_export_artifacts"] = []
    for source in artifacts:
        if not source.is_file():
            continue
        record = {"source": str(source.resolve()), "bytes": source.stat().st_size,
                  "sha256": digest(source)}
        if source.stat().st_size <= 16 * 1024 * 1024:
            destination = artifact_root / source.name
            if destination.exists():
                raise FileExistsError(f"duplicate provenance artifact name: {source.name}")
            shutil.copyfile(source, destination)
            record["copy"] = str(destination.resolve())
        else:
            record["preserved_at_original_path"] = True
        report["training_and_export_artifacts"].append(record)
    if hash_weights:
        payload = {"config_sha256": report["head_config_sha256"], "files": report["weights"]}
        report["checkpoint_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if image is not None:
        if not hash_weights:
            raise ValueError("image hashing also requires a recorded compute allocation")
        report["container"] = {"path": str(image.resolve()), "bytes": image.stat().st_size,
                               "sha256": digest(image)}
    exclusive_json(output, report)
    return report


def _policy_env(policy: str) -> dict[str, str]:
    if policy in {"native2048", "nativefull"}:
        return {}
    match = re.fullmatch(r"(recent|importance)(512|1024|2048)", policy)
    if not match:
        raise ValueError(f"unsupported cache policy: {policy}")
    kind, size = match.groups()
    return {"VLLM_APERTUS_DRAFT_KV_POLICY": kind, "VLLM_APERTUS_DRAFT_KV_BUDGET": size,
            "VLLM_APERTUS_DRAFT_KV_RECENT": "128", "VLLM_APERTUS_DRAFT_KV_GRAPHS": "1"}


def generate_specs(
    campaign: Path, phase: str, inventory: Path, *, candidate: str | None = None,
) -> list[Path]:
    if phase not in PHASE_REPEATS:
        raise ValueError("generate supports screen/confirm/on/eagle")
    inputs = json.loads(inventory.read_text())
    if not inputs.get("weights_hashed") or not inputs.get("checkpoint_sha256"):
        raise ValueError("a compute-node frozen inventory with full weight hashes is required")
    expected_head = {"screen": "off", "confirm": "off", "on": "on", "eagle": "eagle"}[phase]
    if inputs["head"] != expected_head:
        raise ValueError(f"{phase} needs the trained {expected_head} head")
    with ledger(campaign) as data:
        if phase != "screen":
            authorization = data["phase_authorizations"].get(phase)
            if not authorization:
                raise ValueError(f"{phase} has no recorded parent gate")
            candidate = candidate or authorization["candidate"]
            if candidate != authorization["candidate"]:
                raise ValueError("candidate differs from the recorded parent decision")
    if phase != "screen" and candidate not in CANDIDATES:
        raise ValueError("candidate must be one screened importance policy")
    output = campaign / "specs" / phase
    output.mkdir(parents=True, exist_ok=False)
    corpora = inputs["corpora"]
    source = corpora["eagle-8b-validation.jsonl" if phase == "screen" else "eagle-8b-test.jsonl"]
    concurrencies = [1, 32, 256] if phase == "screen" else [1, 8, 32, 128, 256]
    tests = [{"workload": w, "workloads_file": source["path"],
              "concurrencies": concurrencies,
              "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 1}}
             for w in ["chat", "code", "summarization"]]
    if phase != "screen":
        tests.append({"workload": "opb", "workloads_file": corpora["opb-heldout.jsonl"]["path"],
                      "concurrencies": concurrencies,
                      "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 1}})
    # Thinking-on uses the same frozen messages with explicit kwargs and a useful cap.
    if phase == "on":
        derived = output / "workloads"
        derived.mkdir()
        from apertus_bench.workloads import derive_workload
        for source_path in {t["workloads_file"] for t in tests}:
            target = derived / Path(source_path).name
            derive_workload(Path(source_path), target,
                            chat_template_kwargs={"enable_thinking": True},
                            max_tokens=8192, id_suffix="-think")
            for test in tests:
                if test["workloads_file"] == source_path:
                    test["workloads_file"] = str(target.resolve())
            corpora[target.name + "-thinking"] = {
                "path": str(target.resolve()), "sha256": digest(target),
                "derived_from": source_path, "enable_thinking": True, "max_tokens": 8192,
            }
    paths = []
    for repeat in range(PHASE_REPEATS[phase]):
        native = "nativefull" if phase == "eagle" else "native2048"
        policies = list(SCREEN_POLICIES) if phase == "screen" else [
            native, candidate, native, candidate]
        if repeat % 2:
            policies = (policies[2:] + policies[:2] if phase == "screen"
                        else list(reversed(policies)))
        arms = []
        counts: dict[str, int] = {}
        for policy in policies:
            counts[policy] = counts.get(policy, 0) + 1
            arm_name = f"{policy}-{counts[policy]}"
            metadata = {
                "kv_policy": policy, "kv_phase": phase, "kv_node_block": str(repeat + 1),
                "head_config_sha256": inputs["head_config_sha256"],
                "exact_head_checkpoint_sha256": inputs["checkpoint_sha256"],
                "inventory_sha256": digest(inventory), "target_tensor_parallel_size": "1",
                "target_model": inputs["target_contract"]["target"]["identity"]["model_id"],
                "target_revision": inputs["target_contract"]["target"]["identity"]["revision"],
            }
            if phase != "eagle":
                metadata["checkpoint_sha256"] = inputs["checkpoint_sha256"]
            if inputs.get("container"):
                metadata.update({"container_sha256": inputs["container"]["sha256"],
                                 "container_image": inputs["container"]["path"]})
            arms.append({"name": arm_name, "method": "eagle" if phase == "eagle" else "dspark",
                         "drafter": inputs["head_path"], "k": 7,
                         "env": {**_policy_env(policy),
                                 "VLLM_DRAFT_KV_TRACE": "{arm_dir}/draft-kv.jsonl",
                                 "VLLM_DRAFT_KV_CORRECTNESS_WORKLOADS":
                                     corpora["correctness.jsonl"]["path"]},
                         "metadata": metadata})
        spec = {"name": f"drafter-kv-{phase}-{repeat + 1:02d}", "stage": "8b",
                "node_gpus": 4, "probe": False,
                "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256,
                           "async_scheduling": False},
                "loadtests": tests, "arms": arms,
                "drafter_kv": {"phase": phase, "repeat": repeat + 1,
                               "inventory": str(inventory.resolve()), "corpora": corpora,
                               "inventory_sha256": digest(inventory)}}
        from apertus_bench.node_sweep import build_plan
        build_plan(spec, check_heads=False)
        path = output / f"{phase}-{repeat + 1:02d}.yaml"
        with path.open("x") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)
        paths.append(path)
    exclusive_json(output / "generation.json", {
        "phase": phase, "generated_at": now(), "candidate": candidate,
        "inventory": str(inventory.resolve()), "inventory_sha256": digest(inventory),
        "specs": [{"path": str(p.resolve()), "sha256": digest(p)} for p in paths],
        "thinking_on_max_tokens": 8192 if phase == "on" else None,
    })
    return paths


def validate_frozen_spec(spec: dict[str, Any]) -> None:
    frozen = spec["drafter_kv"]
    if digest(Path(frozen["inventory"])) != frozen["inventory_sha256"]:
        raise ValueError("frozen head inventory changed")
    inventory = json.loads(Path(frozen["inventory"]).read_text())
    if digest(Path(inventory["head_path"]) / "config.json") != inventory["head_config_sha256"]:
        raise ValueError("exact head config changed")
    for weight in inventory.get("weights") or []:
        path = Path(inventory["head_path"]) / weight["name"]
        if not path.is_file() or path.stat().st_size != weight["bytes"]:
            raise ValueError("frozen checkpoint file changed or disappeared")
    if inventory.get("container"):
        container = inventory["container"]
        path = Path(container["path"])
        if not path.is_file() or path.stat().st_size != container["bytes"]:
            raise ValueError("frozen container image changed or disappeared")
    for entry in frozen["corpora"].values():
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"immutable corpus changed: {entry['path']}")


def prepare_submission(campaign: Path, allocation: str, spec_path: Path, repo: Path) -> Path:
    spec = yaml.safe_load(spec_path.read_text())
    validate_frozen_spec(spec)
    with ledger(campaign) as data:
        item = copy.deepcopy(data["allocations"][allocation])
        if item["phase"] != spec["drafter_kv"]["phase"]:
            raise ValueError("spec phase differs from budget reservation")
        if item["state"] != "reserved" or item["submission_attempted"]:
            raise ValueError("allocation is not unsubmitted")
    spec_root = (campaign / "specs" / item["phase"]).resolve()
    if not spec_path.resolve().is_relative_to(spec_root):
        raise ValueError("spec must belong to the reserved campaign phase")
    generation_path = spec_path.resolve().parent / "generation.json"
    generation = json.loads(generation_path.read_text())
    expected_spec = next((p for p in generation["specs"]
                          if Path(p["path"]).resolve() == spec_path.resolve()), None)
    if not expected_spec or expected_spec["sha256"] != digest(spec_path):
        raise ValueError("only unchanged campaign-generated specs may be submitted")
    from apertus_bench.node_sweep import build_plan
    plan = build_plan(spec, check_heads=False)
    provenance = {
        "created_at": now(), "reservation": item, "spec": str(spec_path.resolve()),
        "spec_sha256": digest(spec_path), "git_revision": _git(repo, "rev-parse", "HEAD"),
        "git_diff": _git(repo, "diff", "HEAD", "--binary"),
        "expanded_arms": [{"name": a.name, "serve_argv": a.serve_argv,
                           "env": a.env, "metadata": a.metadata} for a in plan.arms],
        "frozen_inputs": spec["drafter_kv"],
        "SWEEP_DRAFT_KV": "1",
    }
    path = campaign / "allocations" / allocation / "launch-provenance.json"
    exclusive_json(path, provenance)
    return path


def submit(campaign: Path, allocation: str, spec_path: Path, repo: Path) -> dict[str, Any]:
    """Explicit submission only. Unknown outcomes retain their full reservation."""
    prepare_submission(campaign, allocation, spec_path, repo)
    with ledger(campaign) as data:
        item = copy.deepcopy(data["allocations"][allocation])
    output = Path(item["output"])
    # Claim the output before calling the existing launcher, which permits mkdir -p.
    output.mkdir(exist_ok=False)
    frozen_spec = output.parent / "spec.yaml"
    with frozen_spec.open("x") as handle:
        handle.write(spec_path.read_text())
    seconds = item["reserved_seconds"]
    walltime = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    command = [str(repo / "serving/node-sweep.sh"), str(frozen_spec.resolve())]
    environment = {
        **os.environ, "SWEEP_OUTPUT": str(output), "SWEEP_TIME": walltime,
        "SBATCH_COMMENT": item["job_comment"], "SBATCH_NO_REQUEUE": "1",
        "SWEEP_DRAFT_KV": "1",
        "VLLM_PATCH_DIR": str(output.parent / "vllm-overlay"),
        "SWEEP_DRAFT_KV_PREFLIGHT": "1" if item["phase"] == "preflight" else "0",
        "SWEEP_JOB_COMMENT": item["job_comment"], "SWEEP_NO_REQUEUE": "1",
    }
    exclusive_json(output.parent / "submit-command.json", {
        "argv": command, "SWEEP_OUTPUT": str(output), "SWEEP_TIME": walltime,
        "SBATCH_COMMENT": item["job_comment"],
        "SWEEP_DRAFT_KV": "1", "SBATCH_NO_REQUEUE": "1",
        "VLLM_PATCH_DIR": environment["VLLM_PATCH_DIR"],
        "SWEEP_DRAFT_KV_PREFLIGHT": environment["SWEEP_DRAFT_KV_PREFLIGHT"],
        "SWEEP_JOB_COMMENT": environment["SWEEP_JOB_COMMENT"], "SWEEP_NO_REQUEUE": "1",
        "SWEEP_PATCHES": environment.get("SWEEP_PATCHES", ""),
        "IMAGE": environment.get("IMAGE"), "ENV_SOURCE": environment.get("ENV_SOURCE"),
    })
    mark_submission(campaign, allocation)
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    with (output.parent / "submit.stdout").open("x") as handle:
        handle.write(result.stdout)
    with (output.parent / "submit.stderr").open("x") as handle:
        handle.write(result.stderr)
    match = re.search(r"(?:^|\n)job=([0-9]+) output=", result.stdout)
    if match:
        registered = register(campaign, allocation, match.group(1))
        if result.returncode:
            raise RuntimeError(
                f"launcher failed after reporting job {registered['job_id']}; inspect logs")
        return registered
    raise RuntimeError(
        "submission outcome unknown; reservation retained. Inspect submit logs and recover the "
        "job by its exact job_comment before another allocation is submitted"
    )


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return (ordered[n // 2] + ordered[(n - 1) // 2]) / 2


def node_bootstrap_ci(values: list[float], *, samples: int = 20000) -> list[float] | None:
    """Paired log-ratio bootstrap: one independent draw per node allocation."""
    if len(values) < 5:
        return None
    rng = random.Random(20261022)
    logs = [math.log(v) for v in values]
    draws = sorted(math.exp(sum(rng.choice(logs) for _ in logs) / len(logs))
                   for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]]


def trace_summary(path: Path) -> dict[str, Any]:
    count = filled = allocations = 0
    peaks = dict.fromkeys([
        "draft_allocated_bytes", "target_allocated_bytes", "draft_live_bytes", "target_live_bytes",
        "draft_context_live_bytes", "draft_metadata_bytes", "draft_query_scratch_bytes",
        "draft_context_projection_scratch_bytes", "bootstrap_peak_pages",
        "shared_pool_reserved_bytes",
    ], 0)
    if not path.is_file():
        return {"available": False}
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") not in {"step", "allocation"}:
                continue
            if event["event"] == "step":
                count += 1
                lengths = event.get("native_sequence_lengths") or []
                filled += bool(lengths and min(lengths) >= 2048
                               and event.get("bootstrap_active_requests") == 0)
            else:
                allocations += 1
            for field in peaks:
                value = event.get(field)
                if isinstance(value, (int, float)):
                    peaks[field] = max(peaks[field], value)
    return {"available": True, "step_samples": count, "steady_filled_samples": filled,
            "allocation_samples": allocations, "peaks": peaks,
            "path": str(path), "sha256": digest(path)}


def _allocation_cells(allocation: dict[str, Any]) -> tuple[dict, list[str], dict]:
    output = Path(allocation["output"])
    spec_path = output.parent / "spec.yaml"
    errors: list[str] = []
    cells: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    traces = {}
    if not spec_path.is_file() or not (output / "sweep-status.json").is_file():
        return {}, [f"{allocation['allocation']}: missing spec or final sweep status"], traces
    spec = yaml.safe_load(spec_path.read_text())
    status = json.loads((output / "sweep-status.json").read_text())
    expected = {a["name"] for a in spec["arms"]}
    if set(status) != expected or any(s != "ok" for s in status.values()):
        errors.append(f"{allocation['allocation']}: not every expected arm completed successfully")
    for arm in spec["arms"]:
        policy = arm["metadata"]["kv_policy"]
        traces[arm["name"]] = trace_summary(output / arm["name"] / "draft-kv.jsonl")
        for test in spec["loadtests"]:
            label = test.get("label", test["workload"])
            for concurrency in test["concurrencies"]:
                path = output / arm["name"] / label / test["workload"] / f"c{concurrency}"
                required = ["metadata.json", "summary.json", "requests.jsonl",
                            "metrics_before.prom", "metrics_after.prom"]
                if any(not (path / name).is_file() for name in required):
                    errors.append(f"missing raw cell artifacts: {path}")
                    continue
                metadata = json.loads((path / "metadata.json").read_text())
                summary = json.loads((path / "summary.json").read_text())
                expected_digest = next(
                    (x["sha256"] for x in spec["drafter_kv"]["corpora"].values()
                     if x["path"] == test["workloads_file"]), None)
                if metadata.get("workload_file_sha256") != expected_digest:
                    errors.append(f"frozen corpus mismatch: {path}")
                extra = metadata.get("extra") or {}
                if extra.get("kv_policy") != policy:
                    errors.append(f"cache policy provenance mismatch: {path}")
                metrics = {
                    "throughput": summary["tokens"]["output_tokens_per_second"],
                    "success": summary["requests"]["success_rate"],
                    "acceptance": summary["speculative_decoding"].get("mean_acceptance_length"),
                    "ttft_p95": summary["latency_ms"]["ttft"].get("p95"),
                    "tpot_p95": summary["latency_ms"]["tpot"].get("p95"),
                }
                if any(v is None or not math.isfinite(v) or v <= 0 for v in metrics.values()):
                    errors.append(f"missing/nonpositive decision metrics: {path}")
                    continue
                key = (policy, test["workload"], int(concurrency))
                row = cells.setdefault(key, {m: [] for m in metrics})
                for metric, value in metrics.items():
                    row[metric].append(value)
    # Duplicate GPUs are technical replicates; reduce within node before uncertainty estimation.
    reduced = {key: {m: sum(values) / len(values) for m, values in metrics.items()}
               for key, metrics in cells.items()}
    return reduced, errors, traces


def analyze(campaign: Path, phase: str, evidence_path: Path, output: Path) -> dict[str, Any]:
    if phase not in PHASE_REPEATS:
        raise ValueError("analyze supports screen/confirm/on/eagle")
    evidence = json.loads(evidence_path.read_text())
    with ledger(campaign) as data:
        allocations = [copy.deepcopy(a) for a in data["allocations"].values()
                       if a["phase"] == phase and a["state"] != "released_before_submit"]
        authorization = copy.deepcopy(data["phase_authorizations"].get(phase))
    errors, blocks, traces = [], [], {}
    if len(allocations) != PHASE_REPEATS[phase]:
        errors.append(f"expected {PHASE_REPEATS[phase]} independent node allocations")
    for allocation in allocations:
        if allocation["state"] != "terminal" or allocation.get("slurm_state") != "COMPLETED":
            errors.append(f"{allocation['allocation']}: no completed Slurm accounting")
        cells, failures, block_traces = _allocation_cells(allocation)
        blocks.append(cells)
        errors.extend(failures)
        traces[allocation["allocation"]] = block_traces
    policies = SCREEN_POLICIES[1:] if phase == "screen" else [
        authorization["candidate"] if authorization else evidence.get("candidate")]
    decisions = {}
    for policy in policies:
        failures = list(errors)
        source = (evidence.get("policies") or {}).get(policy, {})
        memory = source.get("memory") or {}
        artifacts = source.get("artifact_paths") or []
        if not artifacts or any(not Path(p).is_file() for p in artifacts):
            failures.append("missing explicit correctness/storage evidence artifacts")
        correctness = source.get("correctness_controls_passed") is True
        if not correctness:
            failures.append("native and candidate correctness controls have not passed")
        storage_ok = all(memory.get(k) is True for k in [
            "includes_selection_metadata", "native_filled_window", "physical_eviction_verified"])
        storage_ratio = memory.get("draft_storage_ratio")
        if (not storage_ok or not isinstance(storage_ratio, (float, int))
                or not math.isfinite(storage_ratio) or not 0 <= storage_ratio <= 0.55):
            failures.append("45% lower physical draft storage including metadata not established")
        ratios: dict[tuple[str, int], dict[str, list[float]]] = {}
        for block in blocks:
            native = "nativefull" if phase == "eagle" else "native2048"
            keys = [(w, c) for p, w, c in block if p == native]
            for workload, concurrency in keys:
                baseline = block[(native, workload, concurrency)]
                candidate = block.get((policy, workload, concurrency))
                if candidate is None:
                    failures.append(f"candidate cell missing: {workload}/c{concurrency}")
                    continue
                pair = ratios.setdefault((workload, concurrency), {
                    m: [] for m in ["throughput", "acceptance", "ttft_p95", "tpot_p95"]})
                if min(candidate["success"], baseline["success"]) < 0.99:
                    failures.append(f"success below99%: {workload}/c{concurrency}")
                for metric in pair:
                    pair[metric].append(candidate[metric] / baseline[metric])
        if not ratios:
            failures.append("no complete paired cells")
        cells_report = []
        for (workload, concurrency), metrics in sorted(ratios.items()):
            if len(metrics["throughput"]) != PHASE_REPEATS[phase]:
                failures.append(f"incomplete node blocks: {workload}/c{concurrency}")
            mean = {k: _geomean(v) for k, v in metrics.items()}
            if mean["acceptance"] < 0.95:
                failures.append(f"acceptance below95%: {workload}/c{concurrency}")
            if mean["throughput"] < 0.97:
                failures.append(f"throughput regression over3%: {workload}/c{concurrency}")
            if max(mean["ttft_p95"], mean["tpot_p95"]) > 1.05:
                failures.append(f"p95 latency regression over5%: {workload}/c{concurrency}")
            cells_report.append({"workload": workload, "concurrency": concurrency,
                                 "ratios_geomean": mean, "node_ratios": metrics,
                                 "throughput_ratio_ci95": node_bootstrap_ci(metrics["throughput"])})
        score_values = [r for (w, c), m in ratios.items() if c in {32, 256}
                        for r in m["throughput"]]
        score = _geomean(score_values) if score_values else None
        if phase == "screen" and ratios:
            speed_key = min(ratios, key=lambda k: (-_median(ratios[k]["throughput"]), k[0], k[1]))
            speed_cell = {"workload": speed_key[0], "concurrency": speed_key[1]}
        else:
            speed_cell = (authorization or {}).get("speed_cell", data["predeclared_speed_cell"])
        selected = next((c for c in cells_report if c["workload"] == speed_cell["workload"]
                         and c["concurrency"] == speed_cell["concurrency"]), None)
        speed_pass = bool(selected and selected["ratios_geomean"]["throughput"] >= 1.05
                          and selected["throughput_ratio_ci95"]
                          and selected["throughput_ratio_ci95"][0] > 1.0)
        capacity_pass = (memory.get("matched_lengths") is True
                         and isinstance(memory.get("combined_live_kv_bytes_ratio"), (int, float))
                         and math.isfinite(memory["combined_live_kv_bytes_ratio"])
                         and 0 <= memory["combined_live_kv_bytes_ratio"] <= 0.95)
        decisions[policy] = {
            "stable_and_storage_passed": not failures, "failure_reasons": sorted(set(failures)),
            "score_c32_c256": score, "selected_speed_cell": speed_cell,
            "speed_improvement_passed": speed_pass, "capacity_improvement_passed": capacity_pass,
            "promising": not failures and (speed_pass or capacity_pass),
            "correctness_controls_passed": correctness, "memory": memory, "cells": cells_report,
            "artifacts": [{"path": str(Path(p).resolve()), "sha256": digest(Path(p))}
                          for p in artifacts if Path(p).is_file()],
        }
    qualified = [p for p, d in decisions.items()
                 if p in CANDIDATES and d["stable_and_storage_passed"]]
    if phase == "screen" and qualified:
        best = max(decisions[p]["score_c32_c256"] for p in qualified)
        tied = [p for p in qualified if decisions[p]["score_c32_c256"] >= best * 0.98]
        candidate = min(tied, key=lambda p: (int(re.search(r"[0-9]+$", p).group()), p))
    else:
        candidate = policies[0] if policies else None
    chosen = decisions.get(candidate) or {}
    passed = bool(chosen.get("stable_and_storage_passed") if phase == "screen"
                  else chosen.get("promising"))
    report = {"schema_version": 1, "phase": phase, "created_at": now(), "candidate": candidate,
              "passed": passed, "promising": chosen.get("promising", False),
              "correctness_controls_passed": chosen.get("correctness_controls_passed", False),
              "selected_speed_cell": chosen.get("selected_speed_cell"), "decisions": decisions,
              "trace_summaries": traces, "evidence": str(evidence_path.resolve()),
              "evidence_sha256": digest(evidence_path),
              "uncertainty_unit": (
                  "independent node allocation; GPU replicates averaged within node"),
              "bootstrap": {"samples": 20000, "seed": 20261022, "scale": "paired log ratio"}}
    exclusive_json(output, report)
    return report


def monitor(campaign: Path, *, once: bool = False) -> dict[str, Any]:
    """Enforce only this campaign's budget/failure stops; no launches or phase changes."""
    while True:
        current = reconcile(campaign)
        with ledger(campaign) as data:
            allocations = copy.deepcopy(list(data["allocations"].values()))
        over_budget = current["committed_gpu_hours"] > TOTAL_GPU_HOURS + 1e-9 or any(
            h > PHASE_LIMITS[p] + 1e-9 for p, h in current["phase_committed_gpu_hours"].items())
        failed = any(a.get("slurm_state") in TERMINAL - {"COMPLETED"} for a in allocations)
        for allocation in allocations:
            status_path = Path(allocation["output"]) / "sweep-status.json"
            if status_path.is_file():
                statuses = json.loads(status_path.read_text())
                failed = failed or any(value != "ok" for value in statuses.values())
        if over_budget or failed:
            stopped = stop_owned(campaign, "budget" if over_budget else "failure")
            return {"budget": current, **stopped}
        live = [a for a in allocations if a["state"] not in {"terminal", "released_before_submit"}]
        if once or not live:
            return {"budget": current, "active_allocations": [a["allocation"] for a in live]}
        time.sleep(30)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("campaign", type=Path)
    init.add_argument("--campaign-id", required=True)
    for action in ["status", "reconcile", "stop", "reserve", "register", "mark-submit", "release",
                   "authorize", "freeze", "generate", "prepare", "submit", "analyze", "monitor"]:
        command = sub.add_parser(action)
        command.add_argument("campaign", type=Path)
        if action in {"reserve", "register", "mark-submit", "release", "prepare", "submit"}:
            command.add_argument("--allocation", required=True)
        if action in {"reserve", "authorize", "generate", "analyze"}:
            command.add_argument("--phase", choices=list(PHASE_LIMITS), required=True)
        if action == "reserve":
            command.add_argument("--seconds", type=int, default=MAX_SECONDS)
            command.add_argument("--command-json")
        if action == "register":
            command.add_argument("--job-id", required=True)
        if action == "stop":
            command.add_argument("--reason", choices=["budget", "failure", "usage-limit", "user"],
                                 required=True)
        if action == "authorize":
            command.add_argument("--evidence", type=Path, required=True)
        if action in {"freeze", "prepare", "submit"}:
            command.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
        if action == "freeze":
            command.add_argument("--head", choices=list(HEAD_PATHS), required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--hash-weights", action="store_true")
            command.add_argument("--image", type=Path)
        if action == "generate":
            command.add_argument("--inventory", type=Path, required=True)
            command.add_argument("--candidate", choices=CANDIDATES)
        if action in {"prepare", "submit"}:
            command.add_argument("--spec", type=Path, required=True)
        if action == "analyze":
            command.add_argument("--evidence", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if action == "monitor":
            command.add_argument("--once", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    action = args.action
    if action == "init":
        result = init_campaign(args.campaign, args.campaign_id)
    elif action == "status":
        with ledger(args.campaign) as data:
            result = {**data, "budget": budget(data)}
    elif action == "reserve":
        result = reserve(args.campaign, args.phase, args.allocation, args.seconds,
                         json.loads(args.command_json) if args.command_json else None)
    elif action == "register":
        result = register(args.campaign, args.allocation, args.job_id)
    elif action == "mark-submit":
        result = mark_submission(args.campaign, args.allocation)
    elif action == "release":
        result = release_unsubmitted(args.campaign, args.allocation)
    elif action == "authorize":
        result = authorize_phase(args.campaign, args.phase, args.evidence)
    elif action == "reconcile":
        result = reconcile(args.campaign)
    elif action == "stop":
        result = stop_owned(args.campaign, args.reason)
    elif action == "freeze":
        result = freeze_inputs(args.campaign, args.repo, args.head, args.output,
                               hash_weights=args.hash_weights, image=args.image)
    elif action == "generate":
        result = [str(p) for p in generate_specs(args.campaign, args.phase, args.inventory,
                                               candidate=args.candidate)]
    elif action == "prepare":
        result = str(prepare_submission(args.campaign, args.allocation, args.spec, args.repo))
    elif action == "submit":
        result = submit(args.campaign, args.allocation, args.spec, args.repo)
    elif action == "analyze":
        result = analyze(args.campaign, args.phase, args.evidence, args.output)
    elif action == "monitor":
        result = monitor(args.campaign, once=args.once)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
