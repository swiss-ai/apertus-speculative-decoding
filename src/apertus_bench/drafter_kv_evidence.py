"""Derive immutable correctness and storage evidence from drafter-KV raw artifacts.

    python -m apertus_bench.drafter_kv_evidence CAMPAIGN --phase screen --output NEW_DIR

No jobs are submitted and no phase is authorized. Missing controls, physical
ownership traces, or conservative native/sparse matches produce a failed gate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from apertus_bench.correctness import DEFAULT_TOLERANCE, calibrate_gate, compare_captures
from apertus_bench.drafter_kv_experiment import (
    PHASE_REPEATS,
    SCREEN_POLICIES,
    artifact_path,
    artifact_path_mapping,
    digest,
    exclusive_json,
    now,
    phase_attempts,
)

PAGE = 16
NATIVE_WINDOW = 2048
# Frozen experiment: k=7 and anchor-first block8. An unaligned interval
# covering the2048 retained context tokens plus8 native query/lookahead tokens
# intersects at most ceil((2048+8+15)/16)=130 physical pages. Native completed
# prefills may temporarily own the entire prompt; those are bootstrap, not the
# bounded steady-state denominator for an eviction storage claim.
SPECULATIVE_TOKENS = 7
NATIVE_QUERY_TOKENS = SPECULATIVE_TOKENS + 1
NATIVE_STEADY_MAX_PAGES = math.ceil((NATIVE_WINDOW + NATIVE_QUERY_TOKENS + PAGE - 1) / PAGE)
MAX_JOIN_AGE_NS = 1_000_000_000


@dataclass(frozen=True)
class Deployment:
    allocation: str
    arm: str
    policy: str
    directory: Path
    correctness_workload: Path | None
    correctness_sha256: str | None = None

    @property
    def identity(self) -> str:
        return f"{self.allocation}/{self.arm}"


def discover(campaign: Path, phase: str) -> tuple[list[Deployment], list[str]]:
    state = json.loads((campaign / "ledger.json").read_text())
    allocations, _ = phase_attempts(state, phase)
    errors = []
    if len(allocations) != PHASE_REPEATS[phase]:
        errors.append(f"expected {PHASE_REPEATS[phase]} independent {phase} allocations")
    deployments = []
    identities = set()
    for allocation in allocations:
        output = artifact_path(allocation["output"])
        if allocation["state"] != "terminal" or allocation.get("slurm_state") != "COMPLETED":
            errors.append(f"{allocation['allocation']}: incomplete Slurm accounting")
        spec_path = output.parent / "spec.yaml"
        if not spec_path.is_file():
            errors.append(f"{allocation['allocation']}: missing frozen spec")
            continue
        spec = yaml.safe_load(spec_path.read_text())
        for arm in spec["arms"]:
            metadata = arm.get("metadata") or {}
            policy = metadata.get("kv_policy")
            identity = tuple(metadata.get(name) for name in [
                "head_config_sha256", "exact_head_checkpoint_sha256", "target_model",
                "target_revision", "target_tensor_parallel_size",
            ])
            if any(not field for field in identity):
                errors.append(f"{allocation['allocation']}/{arm['name']}: incomplete head identity")
            identities.add(identity)
            workload = (arm.get("env") or {}).get("VLLM_DRAFT_KV_CORRECTNESS_WORKLOADS")
            if not policy:
                errors.append(f"{allocation['allocation']}/{arm['name']}: missing policy identity")
                continue
            deployments.append(Deployment(
                allocation["allocation"], arm["name"], policy, output / arm["name"],
                artifact_path(workload) if workload else None,
                next((c["sha256"] for c in spec.get("drafter_kv", {}).get("corpora", {}).values()
                      if c["path"] == workload), None),
            ))
    if len(identities) != 1:
        errors.append("native/candidate controls do not share exactly one head/target identity")
    return deployments, errors


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def _comparison(left: tuple[Deployment, Path, dict], right: tuple[Deployment, Path, dict]) -> dict:
    report = compare_captures(left[2], right[2])
    # Identical localhost URLs/model names on separate nodes are separate deployments.
    report["same_deployment"] = left[0].identity == right[0].identity
    report["baseline_deployment"] = left[0].identity
    report["candidate_deployment"] = right[0].identity
    report["baseline_capture"] = str(left[1])
    report["candidate_capture"] = str(right[1])
    return report


def correctness_evidence(
    deployments: list[Deployment], policy: str, native: str, initial_errors: list[str],
) -> dict[str, Any]:
    errors = list(initial_errors)
    selected = [d for d in deployments if d.policy in {native, policy}]
    captures: dict[str, list[tuple[Deployment, Path, dict]]] = {}
    sources = []
    declared_digests = {d.correctness_sha256 for d in selected}
    if None in declared_digests or len(declared_digests) != 1:
        errors.append("native/candidate deployments lack one identical frozen correctness digest")
    for deployment in selected:
        captures[deployment.identity] = []
        expected_ids = None
        try:
            if deployment.correctness_workload is None:
                raise ValueError("missing declared correctness workload")
            if digest(deployment.correctness_workload) != deployment.correctness_sha256:
                raise ValueError("frozen correctness workload changed")
            records = [json.loads(line) for line in deployment.correctness_workload.read_text()
                       .splitlines() if line.strip()]
            expected_ids = {r["id"] for r in records}
            if not expected_ids or len(expected_ids) != len(records):
                raise ValueError("empty or duplicate correctness prompt IDs")
            sources.append(_source(deployment.correctness_workload))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{deployment.identity}: {error}")
        for number in (1, 2):
            path = deployment.directory / "correctness" / f"capture-{number}.json"
            try:
                capture = json.loads(path.read_text())
                cases = capture["cases"]
                ids = [c["prompt_id"] for c in cases]
                if (expected_ids is None or set(ids) != expected_ids
                        or len(ids) != len(expected_ids)):
                    raise ValueError("capture does not cover the frozen correctness prompt IDs")
                sampling = capture["sampling"]
                if sampling.get("temperature") != 0 or sampling.get("top_p") != 1:
                    raise ValueError("correctness capture was not greedy")
                if not all(c["measurement"].get("success") is True for c in cases):
                    raise ValueError("unsuccessful correctness request")
                if any(not isinstance(c["measurement"].get("completion_tokens"), int)
                       for c in cases):
                    raise ValueError("capture lacks completion token counts")
                captures[deployment.identity].append((deployment, path, capture))
                sources.append(_source(path))
            except (OSError, ValueError, KeyError) as error:
                errors.append(f"{path}: {error}")
    controls, treatments = [], []
    try:
        for pair in captures.values():
            if len(pair) == 2:
                controls.append(_comparison(*pair))
        for configuration in [native, policy]:
            configs = [d for d in selected if d.policy == configuration]
            cross = 0
            for left, right in itertools.combinations(configs, 2):
                if left.allocation == right.allocation:
                    continue
                left_caps, right_caps = captures[left.identity], captures[right.identity]
                for pair in itertools.product(left_caps, right_caps):
                    controls.append(_comparison(*pair))
                    cross += 1
            if not cross:
                errors.append(f"missing cross-allocation {configuration}/{configuration} controls")
        for left in [d for d in selected if d.policy == native]:
            for right in [d for d in selected if d.policy == policy
                          and d.allocation == left.allocation]:
                for pair in itertools.product(captures[left.identity], captures[right.identity]):
                    comparison = _comparison(*pair)
                    gate = calibrate_gate(comparison, controls, DEFAULT_TOLERANCE)
                    gate["treatment_comparison"] = comparison
                    treatments.append(gate)
        if not treatments:
            errors.append("no native/candidate treatment comparisons")
        if any(c["comparable_cases"] != c["total_cases"] for c in controls):
            errors.append("a calibration control has failed requests")
    except (ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    passed = not errors and bool(treatments) and all(t["passed"] for t in treatments)
    return {
        "passed": passed, "policy": policy, "native": native, "failure_reasons": errors,
        "controls": controls, "treatments": treatments,
        "within_control_pairs": sum(c["same_deployment"] for c in controls),
        "cross_control_pairs": sum(not c["same_deployment"] for c in controls),
        "sources": sources,
        "claim_limit": "calibrated greedy divergence evidence; not a proof of losslessness",
    }


def _nonnegative(record: dict, name: str) -> float:
    value = record.get(name)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"missing/nonfinite/negative {name}")
    return value


def _snapshot(allocation: dict, worker: dict, released: set[str], sparse: bool) -> dict:
    owners = allocation["request_ids"]
    scheduled = allocation["scheduled_request_ids"]
    step_ids = worker["request_ids"]
    if len(owners) != len(set(owners)) or not owners:
        raise ValueError("empty or duplicate allocator owners")
    if set(owners) != set(scheduled) or set(owners) != set(step_ids):
        raise ValueError("worker lengths do not cover every physical cache owner")
    lengths = dict(zip(step_ids, worker["native_sequence_lengths"], strict=True))
    if any(not isinstance(n, int) or n < NATIVE_WINDOW for n in lengths.values()):
        raise ValueError("native window is not filled for every cache owner")
    if worker.get("bootstrap_active_requests") != 0:
        raise ValueError("bootstrap is still active")
    if sparse and not set(owners).issubset(released):
        raise ValueError("no physical release proof for every sparse cache owner")
    metadata = _nonnegative(worker, "draft_metadata_bytes")
    query = _nonnegative(worker, "draft_query_scratch_bytes")
    projection = worker.get("draft_context_projection_scratch_bytes")
    if projection is not None:
        projection = _nonnegative(worker, "draft_context_projection_scratch_bytes")
    physical_draft = _nonnegative(allocation, "draft_allocated_bytes")
    physical_target = _nonnegative(allocation, "target_allocated_bytes")
    logical_draft = _nonnegative(allocation, "draft_live_bytes")
    logical_target = _nonnegative(allocation, "target_live_bytes")
    block_bytes = _nonnegative(allocation, "physical_bytes_per_pool_block")
    if not block_bytes or not physical_draft or not physical_target:
        raise ValueError("empty cache ownership cannot establish storage reduction")
    target_groups, draft_groups = [], []
    for group in allocation["groups"]:
        pages = dict(zip(owners, group["pages_by_request"], strict=True))
        if any(not isinstance(n, int) or n < 0 for n in pages.values()):
            raise ValueError("invalid page ownership counts")
        (draft_groups if group["drafter"] else target_groups).append((group, pages))
    if not target_groups or not draft_groups:
        raise ValueError("missing target or draft ownership groups")
    maximum_draft_pages = (int(worker["budget_tokens"]) // PAGE + 1
                           if sparse else NATIVE_STEADY_MAX_PAGES)
    if any(count > maximum_draft_pages for _, pages in draft_groups for count in pages.values()):
        kind = "sparse budget" if sparse else "native steady-state"
        raise ValueError(f"drafter ownership exceeds {kind} bound{maximum_draft_pages}pages")
    if sum(sum(p.values()) for _, p in draft_groups) * block_bytes != physical_draft:
        raise ValueError("draft physical bytes disagree with owned pages")
    if sum(sum(p.values()) for _, p in target_groups) * block_bytes != physical_target:
        raise ValueError("target physical bytes disagree with owned pages")
    target_shapes = [tuple(p[r] for _, p in target_groups) for r in owners]
    # Match the full multiset of per-request length pages AND target ownership.
    strata = tuple(sorted(((lengths[r] + PAGE - 1) // PAGE, target_shapes[i])
                          for i, r in enumerate(owners)))
    return {
        "stratum": (block_bytes, physical_target, strata),
        "timestamp_ns": allocation["timestamp_ns"], "worker_timestamp_ns": worker["timestamp_ns"],
        "request_ids": owners, "native_sequence_lengths": [lengths[r] for r in owners],
        "draft_physical_bytes": physical_draft, "target_physical_bytes": physical_target,
        "draft_logical_live_bytes": logical_draft, "target_logical_live_bytes": logical_target,
        "draft_metadata_bytes": metadata, "draft_query_scratch_bytes": query,
        "draft_context_projection_scratch_bytes": projection,
        "draft_storage_bytes_including_metadata": physical_draft + metadata,
        "combined_physical_bytes_including_metadata": physical_draft + physical_target + metadata,
        "groups": allocation["groups"],
    }


def trace_snapshots(deployment: Deployment) -> dict[str, Any]:
    path = deployment.directory / "draft-kv.jsonl"
    rejected: dict[str, int] = defaultdict(int)
    errors, samples, releases = [], [], []
    released = set()
    workers: dict[frozenset[str], dict] = {}
    max_metadata = max_query = max_bootstrap_pages = 0
    max_projection = None
    allocation_events = worker_events = 0
    try:
        # Processes append atomically, but timestamps can precede file order.
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        records.sort(key=lambda event: event["timestamp_ns"])
        for record in records:
            event = record.get("event")
            expected_policy = "off" if deployment.policy.startswith("native") else (
                "importance" if deployment.policy.startswith("importance") else "recent")
            expected_budget = (NATIVE_WINDOW if deployment.policy.startswith("native") else
                               int("".join(c for c in deployment.policy if c.isdigit())))
            if event in {"step", "allocation", "bootstrap_release"} and (
                record.get("policy") != expected_policy
                or record.get("budget_tokens") != expected_budget
            ):
                errors.append("runtime policy/budget differs from frozen deployment identity")
                continue
            if event == "step":
                worker_events += 1
                workers[frozenset(record["request_ids"])] = record
                max_metadata = max(max_metadata, _nonnegative(record, "draft_metadata_bytes"))
                max_query = max(max_query, _nonnegative(record, "draft_query_scratch_bytes"))
                if record.get("draft_context_projection_scratch_bytes") is not None:
                    max_projection = max(max_projection or 0, _nonnegative(
                        record, "draft_context_projection_scratch_bytes"))
                max_bootstrap_pages = max(max_bootstrap_pages,
                                         record.get("bootstrap_peak_pages_per_request", 0))
            elif event == "bootstrap_release":
                before = _nonnegative(record, "allocated_pages_before")
                after = _nonnegative(record, "allocated_pages_after")
                freed = _nonnegative(record, "released_pages")
                if freed != before - after or record.get("page_size") != PAGE:
                    errors.append("inconsistent physical bootstrap release event")
                elif freed > 0:
                    released.add(record["request_id"])
                    releases.append(record)
            elif event == "allocation":
                allocation_events += 1
                key = frozenset(record["scheduled_request_ids"])
                # New traces carry the sampled worker payload through the same
                # ModelRunnerOutput. Historical traces use the strict time/ID join.
                worker = record.get("worker_snapshot") or workers.get(key)
                if worker is None:
                    rejected["no preceding workerstep with identical scheduled request IDs"] += 1
                    continue
                if (worker.get("policy") != expected_policy
                        or worker.get("budget_tokens") != expected_budget):
                    errors.append("joined worker runtime policy/budget differs from deployment")
                    continue
                age = record["timestamp_ns"] - worker["timestamp_ns"]
                if not 0 <= age <= MAX_JOIN_AGE_NS:
                    rejected["worker/allocation timestamp gap exceeds1second"] += 1
                    continue
                try:
                    max_metadata = max(max_metadata, _nonnegative(worker, "draft_metadata_bytes"))
                    max_query = max(max_query, _nonnegative(worker, "draft_query_scratch_bytes"))
                    if worker.get("draft_context_projection_scratch_bytes") is not None:
                        max_projection = max(max_projection or 0, _nonnegative(
                            worker, "draft_context_projection_scratch_bytes"))
                    samples.append(_snapshot(record, worker, released,
                                             sparse=not deployment.policy.startswith("native")))
                except (ValueError, KeyError, TypeError) as error:
                    rejected[str(error)] += 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{path}: {error}")
    return {
        "deployment": deployment.identity, "samples": samples, "releases": releases,
        "worker_events": worker_events, "allocation_events": allocation_events,
        "rejected_allocation_samples": dict(rejected), "failure_reasons": errors,
        "max_request_selection_metadata_bytes": max_metadata, "query_scratch_peak_bytes": max_query,
        "context_projection_scratch_peak_bytes": max_projection,
        "bootstrap_peak_pages_per_request": max_bootstrap_pages,
        "source": _source(path) if path.is_file() else None,
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return (ordered[len(ordered) // 2] + ordered[(len(ordered) - 1) // 2]) / 2


def memory_evidence(
    deployments: list[Deployment], policy: str, native: str, initial_errors: list[str],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors = list(initial_errors)
    matches = []
    blocks = sorted({d.allocation for d in deployments})
    for block in blocks:
        native_samples: dict[tuple, list[dict]] = defaultdict(list)
        candidate_samples: dict[tuple, list[dict]] = defaultdict(list)
        for deployment in [d for d in deployments if d.allocation == block
                           and d.policy in {native, policy}]:
            trace = traces[deployment.identity]
            errors.extend(trace["failure_reasons"])
            table = native_samples if deployment.policy == native else candidate_samples
            for sample in trace["samples"]:
                table[sample["stratum"]].append(sample)
        matched = native_samples.keys() & candidate_samples.keys()
        if not matched:
            errors.append(f"{block}: no filled-window, target-page/length-matched physical samples")
        for key in sorted(matched):
            left, right = native_samples[key], candidate_samples[key]
            # Use the highest measured global selection tables/metadata in this stratum.
            native_metadata = max(s["draft_metadata_bytes"] for s in left)
            sparse_metadata = max(s["draft_metadata_bytes"] for s in right)
            native_draft = _median([s["draft_physical_bytes"] for s in left]) + native_metadata
            sparse_draft = _median([s["draft_physical_bytes"] for s in right]) + sparse_metadata
            target = key[1]
            matches.append({
                "allocation": block, "stratum": key,
                "native_samples": len(left), "candidate_samples": len(right),
                "native_draft_physical_bytes": native_draft - native_metadata,
                "candidate_draft_physical_bytes": sparse_draft - sparse_metadata,
                "native_selection_metadata_bytes": native_metadata,
                "candidate_selection_metadata_bytes": sparse_metadata,
                "native_target_physical_bytes": target, "candidate_target_physical_bytes": target,
                "draft_storage_ratio": sparse_draft / native_draft,
                "combined_live_kv_bytes_ratio": (sparse_draft + target) / (native_draft + target),
                "native_draft_logical_live_bytes": _median(
                    [s["draft_logical_live_bytes"] for s in left]),
                "candidate_draft_logical_live_bytes": _median(
                    [s["draft_logical_live_bytes"] for s in right]),
                "native_target_logical_live_bytes": _median(
                    [s["target_logical_live_bytes"] for s in left]),
                "candidate_target_logical_live_bytes": _median(
                    [s["target_logical_live_bytes"] for s in right]),
            })
    complete = bool(matches) and not errors
    candidate_traces = [traces[d.identity] for d in deployments if d.policy == policy]
    releases_proven = bool(candidate_traces) and all(t["releases"] for t in candidate_traces)
    memory = {
        "draft_storage_ratio": max((m["draft_storage_ratio"] for m in matches), default=None),
        "combined_live_kv_bytes_ratio": max(
            (m["combined_live_kv_bytes_ratio"] for m in matches), default=None),
        "includes_selection_metadata": complete, "native_filled_window": complete,
        "physical_eviction_verified": complete and releases_proven,
        "matched_lengths": complete,
    }
    passed = (complete and releases_proven and memory["draft_storage_ratio"] <= 0.55)
    return {
        "passed": passed, "policy": policy, "native": native, "memory": memory,
        "failure_reasons": errors, "matches": matches,
        "measurement": "owned physical pool bytes plus incremental selection metadata",
        "logical_live_bytes_are_separate": True,
        "shared_pool_reservation_is_not_a_savings_measure": True,
        "scratch_and_bootstrap_reported_separately": True,
        "worker_allocator_maximum_join_age_ns": MAX_JOIN_AGE_NS,
        "steady_state_ownership_bounds": {
            "page_tokens": PAGE, "native_window_tokens": NATIVE_WINDOW,
            "frozen_speculative_tokens": SPECULATIVE_TOKENS,
            "native_query_tokens_including_anchor": NATIVE_QUERY_TOKENS,
            "native_maximum_owned_pages_per_request": NATIVE_STEADY_MAX_PAGES,
            "native_derivation": "ceil((window_tokens+query_tokens+page_tokens-1)/page_tokens)",
            "sparse_maximum_owned_pages_per_request": "budget_tokens/page_tokens+1stagingpage",
        },
        "join": "same-output embedded worker snapshot, otherwise strict timestamp/request-ID join",
        "length_stratum_tokens": PAGE,
        "coverage": {d.identity: {
            "eligible_snapshots": len(traces[d.identity]["samples"]),
            "allocation_events": traces[d.identity]["allocation_events"],
            "rejected_samples": traces[d.identity]["rejected_allocation_samples"],
        } for d in deployments if d.policy in {native, policy}},
    }


def build_evidence(campaign: Path, phase: str, output: Path) -> dict[str, Any]:
    if phase not in PHASE_REPEATS:
        raise ValueError("phase must be screen/confirm/on/eagle")
    output.mkdir(parents=True, exist_ok=False)
    deployments, errors = discover(campaign, phase)
    _, failed_history = phase_attempts(json.loads((campaign / "ledger.json").read_text()), phase)
    native = "nativefull" if phase == "eagle" else "native2048"
    policies = list(SCREEN_POLICIES[1:]) if phase == "screen" else sorted(
        {d.policy for d in deployments if d.policy != native})
    traces = {d.identity: trace_snapshots(d) for d in deployments}
    controls = {p: correctness_evidence(deployments, p, native, errors) for p in policies}
    memory = {p: memory_evidence(deployments, p, native, errors, traces) for p in policies}
    control_path, memory_path = output / "correctness-controls.json", output / "memory.json"
    exclusive_json(control_path, {"created_at": now(), "phase": phase, "policies": controls,
                                  "artifact_path_mapping": artifact_path_mapping(),
                                  "superseded_attempt_history": failed_history})
    exclusive_json(memory_path, {"created_at": now(), "phase": phase, "policies": memory,
                                 "traces": traces, "artifact_path_mapping": artifact_path_mapping(),
                                 "superseded_attempt_history": failed_history})
    report = {"schema_version": 1, "phase": phase, "created_at": now(), "policies": {
        p: {"correctness_controls_passed": controls[p]["passed"], "memory": memory[p]["memory"],
            "artifact_paths": [str(control_path.resolve()), str(memory_path.resolve())]}
        for p in policies}, "source_campaign": str(campaign.resolve()),
        "artifact_path_mapping": artifact_path_mapping(),
        "selected_allocations": sorted({d.allocation for d in deployments}),
        "superseded_attempt_history": failed_history,
        "source_ledger_sha256": digest(campaign / "ledger.json")}
    exclusive_json(output / "evidence.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--phase", choices=list(PHASE_REPEATS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_evidence(args.campaign, args.phase, args.output), indent=2))


if __name__ == "__main__":
    main()
