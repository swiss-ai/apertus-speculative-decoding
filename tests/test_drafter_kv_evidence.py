import json
from pathlib import Path

import pytest

from apertus_bench import drafter_kv_evidence as evidence_module
from apertus_bench.drafter_kv_evidence import (
    NATIVE_STEADY_MAX_PAGES,
    Deployment,
    correctness_evidence,
    discover,
    memory_evidence,
    trace_snapshots,
)
from apertus_bench.drafter_kv_experiment import digest


def deployment(tmp_path, block, policy):
    directory = tmp_path / block / policy
    directory.mkdir(parents=True)
    workload = tmp_path / "correctness.jsonl"
    if not workload.exists():
        workload.write_text(json.dumps({"id": "prompt", "messages": [{"role": "user",
                                                                      "content": "question"}]})
                            + "\n")
    return Deployment(block, policy, policy, directory, workload, digest(workload))


def capture(deployment, text="abcdefghij", success=True):
    directory = deployment.directory / "correctness"
    directory.mkdir()
    report = {"model": deployment.policy, "base_url": "http://127.0.0.1:8100",
              "sampling": {"temperature": 0, "top_p": 1, "seed": 1},
              "cases": [{"prompt_id": "prompt", "workload": "chat",
                         "measurement": {"success": success, "output": text,
                                         "completion_tokens": 10}}]}
    for number in (1, 2):
        (directory / f"capture-{number}.json").write_text(json.dumps(report))


def test_correctness_uses_real_cross_allocation_controls_despite_same_localhost_urls(tmp_path):
    deployments = [deployment(tmp_path, block, policy) for block in ["one", "two"]
                   for policy in ["native2048", "importance512"]]
    for item in deployments:
        capture(item)
    evidence = correctness_evidence(deployments, "importance512", "native2048", [])
    assert evidence["passed"]
    assert evidence["within_control_pairs"] == 4
    assert evidence["cross_control_pairs"] == 8
    assert len(evidence["treatments"]) == 8
    assert all(t["control_envelope"]["cross_deployment_pairs"] == 8
               for t in evidence["treatments"])


def test_stable_but_different_candidate_outputs_fail_calibrated_gate(tmp_path):
    deployments = [deployment(tmp_path, block, policy) for block in ["one", "two"]
                   for policy in ["native2048", "importance512"]]
    for item in deployments:
        capture(item, "abcdefghij" if item.policy == "native2048" else "ZYXWVUTSRQ")
    evidence = correctness_evidence(deployments, "importance512", "native2048", [])
    assert not evidence["passed"]
    assert all(not t["passed"] for t in evidence["treatments"])


def test_missing_repeated_capture_or_failed_control_never_passes(tmp_path):
    deployments = [deployment(tmp_path, block, policy) for block in ["one", "two"]
                   for policy in ["native2048", "importance512"]]
    for item in deployments:
        capture(item)
    (deployments[0].directory / "correctness/capture-2.json").unlink()
    evidence = correctness_evidence(deployments, "importance512", "native2048", [])
    assert not evidence["passed"]
    assert any("capture-2.json" in reason for reason in evidence["failure_reasons"])


def trace(item, *, metadata=500, target_pages=256, release=True,
          worker_owner="request", allocation_owners=None, gap_ns=100, length=4096,
          owned_pages=None):
    sparse = item.policy == "importance512"
    budget = 512 if sparse else 2048
    policy = "importance" if sparse else "off"
    owned = owned_pages if owned_pages is not None else (33 if sparse else 129)
    worker = {"event": "step", "timestamp_ns": 100, "request_ids": [worker_owner],
              "policy": policy, "budget_tokens": budget, "native_sequence_lengths": [length],
              "draft_metadata_bytes": metadata if sparse else 0, "draft_query_scratch_bytes": 100,
              "draft_context_projection_scratch_bytes": 200, "bootstrap_active_requests": 0,
              "bootstrap_peak_pages_per_request": 129}
    owners = allocation_owners or [worker_owner]
    allocation = {"event": "allocation", "timestamp_ns": 100 + gap_ns, "request_ids": owners,
                  "scheduled_request_ids": [worker_owner], "policy": policy,
                  "budget_tokens": budget, "draft_allocated_bytes": owned * 1000,
                  "target_allocated_bytes": target_pages * 1000,
                  "draft_live_bytes": owned * 200, "target_live_bytes": target_pages * 400,
                  "physical_bytes_per_pool_block": 1000,
                  "groups": [{"group_id": 0, "drafter": True, "pages_by_request": [owned],
                              "bytes_per_logical_page": 200},
                             {"group_id": 1, "drafter": False, "pages_by_request": [target_pages],
                              "bytes_per_logical_page": 400}]}
    records = [worker, allocation]
    if sparse and release:
        records.insert(1, {"event": "bootstrap_release", "timestamp_ns": 101,
                           "request_id": worker_owner, "policy": policy, "budget_tokens": budget,
                           "allocated_pages_before": 129, "allocated_pages_after": 33,
                           "released_pages": 96, "page_size": 16})
    (item.directory / "draft-kv.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))


def matched_memory(tmp_path, *, metadata=500, sparse_target_pages=256, release=True):
    deployments = [deployment(tmp_path, "one", p) for p in ["native2048", "importance512"]]
    trace(deployments[0], worker_owner="native-request")
    trace(deployments[1], metadata=metadata, target_pages=sparse_target_pages,
          worker_owner="sparse-request", release=release)
    snapshots = {d.identity: trace_snapshots(d) for d in deployments}
    return memory_evidence(deployments, "importance512", "native2048", [], snapshots), snapshots


def test_physical_memory_is_matched_by_lengths_and_target_ownership_not_request_uuid(tmp_path):
    evidence, snapshots = matched_memory(tmp_path)
    assert evidence["passed"]
    assert evidence["memory"]["physical_eviction_verified"]
    assert evidence["memory"]["draft_storage_ratio"] == pytest.approx(33500 / 129000)
    match = evidence["matches"][0]
    assert match["candidate_draft_logical_live_bytes"] == 33 * 200
    assert match["candidate_draft_physical_bytes"] == 33 * 1000
    assert match["candidate_selection_metadata_bytes"] == 500
    assert all(t["query_scratch_peak_bytes"] == 100 for t in snapshots.values())


def test_global_selection_metadata_can_erase_the_nominal_page_savings(tmp_path):
    evidence, _ = matched_memory(tmp_path, metadata=100000)
    assert not evidence["passed"]
    assert evidence["memory"]["draft_storage_ratio"] > 1


def test_target_physical_page_mismatch_refuses_capacity_claim(tmp_path):
    evidence, _ = matched_memory(tmp_path, sparse_target_pages=257)
    assert not evidence["passed"]
    assert not evidence["memory"]["matched_lengths"]
    assert evidence["memory"]["draft_storage_ratio"] is None


def test_native_full_prefill_transient_cannot_inflate_eviction_savings(tmp_path):
    assert NATIVE_STEADY_MAX_PAGES == 130
    deployments = [deployment(tmp_path, "one", p) for p in ["native2048", "importance512"]]
    # The sparse metadata makes this policy miss the45% storage gate against
    # bounded native KV. A full-prefill denominator would incorrectly pass it.
    trace(deployments[0], owned_pages=200)
    trace(deployments[1], metadata=45000)
    traces = {d.identity: trace_snapshots(d) for d in deployments}
    native = traces[deployments[0].identity]
    assert not native["samples"]
    assert any("native steady-state bound130pages" in reason
               for reason in native["rejected_allocation_samples"])
    evidence = memory_evidence(deployments, "importance512", "native2048", [], traces)
    assert not evidence["passed"]
    assert evidence["memory"]["draft_storage_ratio"] is None
    trace(deployments[0], owned_pages=129)
    traces[deployments[0].identity] = trace_snapshots(deployments[0])
    bounded = memory_evidence(deployments, "importance512", "native2048", [], traces)
    assert bounded["memory"]["draft_storage_ratio"] == pytest.approx(78000 / 129000)
    assert not bounded["passed"]


def test_sparse_allocator_ownership_must_fit_budget_plus_one_staging_page(tmp_path):
    item = deployment(tmp_path, "one", "importance512")
    trace(item, owned_pages=34)
    report = trace_snapshots(item)
    assert not report["samples"]
    assert any("sparse budget bound33pages" in reason
               for reason in report["rejected_allocation_samples"])
    trace(item, owned_pages=33)
    assert len(trace_snapshots(item)["samples"]) == 1


def test_attention_mask_without_observed_physical_page_release_is_not_sparsification(tmp_path):
    evidence, _ = matched_memory(tmp_path, release=False)
    assert not evidence["passed"]
    assert not evidence["memory"]["physical_eviction_verified"]


@pytest.mark.parametrize(
    "kwargs, reason",
    [({"allocation_owners": ["request", "unscheduled"]}, "every physical cache owner"),
     ({"gap_ns": 1_000_000_001}, "gap exceeds1second"),
     ({"length": 1024}, "window is not filled")],
)
def test_stale_incomplete_or_unfilled_snapshot_joins_fail_closed(tmp_path, kwargs, reason):
    item = deployment(tmp_path, "one", "importance512")
    trace(item, **kwargs)
    evidence = trace_snapshots(item)
    assert not evidence["samples"]
    assert any(reason in key for key in evidence["rejected_allocation_samples"])


def test_budget_identity_is_checked_against_actual_runtime_trace(tmp_path):
    item = deployment(tmp_path, "one", "importance512")
    trace(item)
    path = item.directory / "draft-kv.jsonl"
    path.write_text(path.read_text().replace('"budget_tokens": 512', '"budget_tokens": 1024'))
    evidence = trace_snapshots(item)
    assert not evidence["samples"]
    assert any("runtime policy/budget" in reason for reason in evidence["failure_reasons"])


def test_embedded_same_output_worker_snapshot_handles_independent_trace_cadence(tmp_path):
    item = deployment(tmp_path, "one", "importance512")
    trace(item)
    path = item.directory / "draft-kv.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    worker = next(r for r in records if r["event"] == "step")
    allocation = next(r for r in records if r["event"] == "allocation")
    allocation["worker_snapshot"] = worker
    records = [r for r in records if r["event"] != "step"]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    evidence = trace_snapshots(item)
    assert len(evidence["samples"]) == 1
    assert evidence["samples"][0]["worker_timestamp_ns"] == worker["timestamp_ns"]
    assert evidence["query_scratch_peak_bytes"] == 100


def test_builder_writes_immutable_analyzer_inputs_from_captures_and_traces(tmp_path, monkeypatch):
    deployments = [deployment(tmp_path, block, policy) for block in ["one", "two"]
                   for policy in ["native2048", "importance512"]]
    for item in deployments:
        capture(item)
        trace(item)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "ledger.json").write_text('{"allocations":{}}')
    monkeypatch.setattr(evidence_module, "discover", lambda *args: (deployments, []))
    output = tmp_path / "evidence"
    report = evidence_module.build_evidence(campaign, "screen", output)
    policy = report["policies"]["importance512"]
    assert policy["correctness_controls_passed"]
    assert policy["memory"]["physical_eviction_verified"]
    assert policy["memory"]["draft_storage_ratio"] < 0.55
    assert all(Path(p).is_file() for p in policy["artifact_paths"])
    assert not report["policies"]["importance1024"]["correctness_controls_passed"]
    with pytest.raises(FileExistsError):
        evidence_module.build_evidence(campaign, "screen", output)


def test_discovery_selects_only_unsuperseded_attempts_for_two_logical_repeats(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    allocations = {}
    for name in ["failed", "replacement", "second"]:
        parent = campaign / "allocations" / name
        parent.mkdir(parents=True)
        allocations[name] = {"allocation": name, "phase": "screen", "state": "terminal",
                             "slurm_state": "COMPLETED", "output": str(parent / "results")}
        if name == "failed":
            allocations[name].update({"slurm_state": "FAILED", "superseded_by": "replacement"})
            continue  # Its failed/missing outputs must not enter replacement evidence.
        metadata = {key: "same-exact-identity" for key in [
            "head_config_sha256", "exact_head_checkpoint_sha256", "target_model",
            "target_revision", "target_tensor_parallel_size"]}
        metadata["kv_policy"] = "importance512"
        spec = {"arms": [{"name": "importance512", "metadata": metadata}]}
        (parent / "spec.yaml").write_text(json.dumps(spec))
    (campaign / "ledger.json").write_text(json.dumps({"allocations": allocations}))
    deployments, errors = discover(campaign, "screen")
    assert not errors
    assert {d.allocation for d in deployments} == {"replacement", "second"}
