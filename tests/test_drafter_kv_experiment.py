import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apertus_bench import drafter_kv_experiment as kv
from apertus_bench.node_sweep import build_plan


@pytest.fixture
def campaign(tmp_path):
    path = tmp_path / "campaign"
    kv.init_campaign(path, "test-kv")
    return path


def status(campaign):
    return json.loads((campaign / "ledger.json").read_text())


def test_pending_reservations_charge_whole_four_gpu_allocation(campaign):
    kv.reserve(campaign, "preflight", "hash", 1200)
    charge = kv.budget(status(campaign))
    assert charge["committed_gpu_hours"] == pytest.approx(4 / 3)
    assert charge["actual_gpu_hours"] == 0
    kv.reserve(campaign, "preflight", "runtime", 4200)
    with pytest.raises(ValueError, match="phase allocation budget"):
        kv.reserve(campaign, "preflight", "would-overspend", 1)


def test_independent_run_names_and_phase_counts_cannot_be_reused(campaign):
    kv.reserve(campaign, "screen", "screen-01", 3600)
    with pytest.raises(FileExistsError):
        kv.reserve(campaign, "screen", "screen-01", 3600)
    kv.reserve(campaign, "screen", "screen-02", 3600)
    with pytest.raises(ValueError, match="independent allocations"):
        kv.reserve(campaign, "screen", "screen-03", 1)
    with pytest.raises(FileExistsError):
        kv.init_campaign(campaign, "test-kv")


def test_ambiguous_submission_never_releases_its_reservation(campaign):
    kv.reserve(campaign, "preflight", "audit", 1200)
    kv.mark_submission(campaign, "audit")
    with pytest.raises(ValueError, match="cannot prove no sbatch"):
        kv.release_unsubmitted(campaign, "audit")
    assert kv.budget(status(campaign))["committed_gpu_hours"] == pytest.approx(4 / 3)
    kv.register(campaign, "audit", "123")
    with pytest.raises(ValueError):
        kv.register(campaign, "audit", "124")


def test_only_provably_unsubmitted_reservation_may_be_released(campaign):
    kv.reserve(campaign, "preflight", "audit", 1200)
    kv.release_unsubmitted(campaign, "audit")
    assert kv.budget(status(campaign))["committed_gpu_hours"] == 0


def test_sacct_steps_are_not_additional_allocations_and_pending_not_free(campaign, monkeypatch):
    kv.reserve(campaign, "preflight", "finished", 1200)
    kv.reserve(campaign, "preflight", "pending", 1200)
    kv.register(campaign, "finished", "123")
    kv.register(campaign, "pending", "124")
    monkeypatch.setattr(kv.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="123|COMPLETED|600|1|cpu=288,gres/gpu=4|\n"
               "123.batch|COMPLETED|600|1|cpu=288,gres/gpu=4|\n"
               "124|PENDING|0|0||\n", returncode=0))
    charge = kv.reconcile(campaign)
    assert charge["actual_gpu_hours"] == pytest.approx(2 / 3)
    assert charge["committed_gpu_hours"] == pytest.approx(2)


def test_stop_cancels_only_registered_jobs_with_exact_ownership_comment(campaign, monkeypatch):
    owned = kv.reserve(campaign, "preflight", "owned", 1200)
    kv.reserve(campaign, "preflight", "wrong-comment", 1200)
    kv.register(campaign, "owned", "123")
    kv.register(campaign, "wrong-comment", "124")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["scontrol", "show"]:
            comment = owned["job_comment"] if command[-1] == "123" else "unrelated"
            return SimpleNamespace(stdout=f"JobId={command[-1]} Comment={comment}", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(kv.subprocess, "run", run)
    result = kv.stop_owned(campaign, "usage-limit")
    assert [c for c in calls if c[0] == "scancel"] == [["scancel", "123"]]
    assert result["unresolved_jobs_or_submissions"] == ["124"]
    with pytest.raises(ValueError, match="halted"):
        kv.reserve(campaign, "screen", "new-work", 1200)


def test_extensions_require_real_parent_gate_and_correctness_controls(campaign, tmp_path):
    with pytest.raises(ValueError, match="parent gate"):
        kv.reserve(campaign, "on", "on-01", 1200)
    report = tmp_path / "gate.json"
    report.write_text(json.dumps({"phase": "confirm", "passed": True, "promising": True,
                                  "candidate": "importance512"}))
    with pytest.raises(ValueError, match="correctness"):
        kv.authorize_phase(campaign, "on", report)
    payload = json.loads(report.read_text())
    payload["correctness_controls_passed"] = True
    report.write_text(json.dumps(payload))
    kv.authorize_phase(campaign, "on", report)
    kv.reserve(campaign, "on", "on-01", 1200)


def inventory(tmp_path, head="off"):
    corpora = {}
    for name in ["eagle-8b-validation.jsonl", "eagle-8b-test.jsonl", "opb-heldout.jsonl",
                 "correctness.jsonl"]:
        path = tmp_path / name
        path.write_text("".join(json.dumps({"id": f"{w}-{i}", "workload": w,
                                          "messages": [{"role": "user", "content": "Hello"}],
                                          "max_tokens": 128}) + "\n"
                                for w in ["chat", "code", "summarization", "opb"]
                                for i in range(2)))
        corpora[name] = {"path": str(path), "sha256": kv.digest(path)}
    head_path = tmp_path / "head"
    head_path.mkdir(exist_ok=True)
    config = head_path / "config.json"
    config.write_text('{"sliding_window":2048}\n')
    payload = {"head": head, "head_path": str(head_path), "weights_hashed": True,
               "checkpoint_sha256": "a" * 64, "head_config_sha256": kv.digest(config),
               "corpora": corpora,
               "target_contract": {"target": {"identity": {"model_id": "swiss-ai/Apertus-v1.5-8B",
                                                          "revision": "target-revision"}}}}
    path = tmp_path / f"{head}-inventory.json"
    path.write_text(json.dumps(payload))
    return path


def test_screen_counterbalances_gpu_slots_without_changing_policies(campaign, tmp_path):
    paths = kv.generate_specs(campaign, "screen", inventory(tmp_path))
    first, second = [yaml.safe_load(path.read_text()) for path in paths]
    p1 = [a["metadata"]["kv_policy"] for a in first["arms"]]
    p2 = [a["metadata"]["kv_policy"] for a in second["arms"]]
    assert p1 == list(kv.SCREEN_POLICIES)
    assert set(p1) == set(p2)
    assert all(p1.index(policy) != p2.index(policy) for policy in p1)
    assert first["arms"][0]["env"].get("VLLM_APERTUS_DRAFT_KV_POLICY") is None
    assert all(a["k"] == 7 for a in first["arms"])
    assert all(t["concurrencies"] == [1, 32, 256] for t in first["loadtests"])
    kv.validate_frozen_spec(first)
    source = Path(first["loadtests"][0]["workloads_file"])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="immutable corpus"):
        kv.validate_frozen_spec(first)


def test_metadata_cannot_overwrite_generated_provenance(campaign, tmp_path):
    (path, _) = kv.generate_specs(campaign, "screen", inventory(tmp_path))
    spec = yaml.safe_load(path.read_text())
    spec["arms"][0]["metadata"]["sweep_arm"] = "another-deployment"
    with pytest.raises(ValueError, match="cannot replace"):
        build_plan(spec, check_heads=False)


def test_duplicate_gpus_do_not_manufacture_five_independent_nodes():
    assert kv.node_bootstrap_ci([1.10] * 4) is None
    assert kv.node_bootstrap_ci([1.10] * 5) == pytest.approx([1.10, 1.10])


def test_missing_results_or_memory_controls_cannot_pass(campaign, tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"policies":{}}')
    output = tmp_path / "report.json"
    report = kv.analyze(campaign, "screen", evidence, output)
    assert not report["passed"]
    assert not report["correctness_controls_passed"]
    assert all(d["failure_reasons"] for d in report["decisions"].values())
    with pytest.raises(FileExistsError):
        kv.analyze(campaign, "screen", evidence, output)


def test_freeze_refuses_full_weight_hashing_on_login(campaign, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm allocation"):
        kv.freeze_inputs(campaign, tmp_path, "off", tmp_path / "inputs.json", hash_weights=True)


def test_memory_reduction_is_not_inferred_from_trace_masks_alone(tmp_path):
    trace = tmp_path / "draft-kv.jsonl"
    trace.write_text(json.dumps({"event": "step", "native_sequence_lengths": [4096],
                                 "bootstrap_active_requests": 0, "draft_allocated_bytes": 100,
                                 "draft_metadata_bytes": 20}) + "\n")
    summary = kv.trace_summary(trace)
    assert summary["steady_filled_samples"] == 1
    assert "draft_storage_ratio" not in summary
    assert summary["peaks"]["draft_metadata_bytes"] == 20


def test_unknown_submit_outcome_retains_pending_budget(campaign, tmp_path, monkeypatch):
    kv.reserve(campaign, "screen", "screen-01", 5400)
    spec = tmp_path / "spec.yaml"
    spec.write_text("stage: 8b\n")
    monkeypatch.setattr(kv, "prepare_submission", lambda *args: None)
    monkeypatch.setattr(kv.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="connection lost while submitting\n", stderr="", returncode=1))
    with pytest.raises(RuntimeError, match="outcome unknown"):
        kv.submit(campaign, "screen-01", spec, tmp_path)
    state = status(campaign)
    assert state["allocations"]["screen-01"]["state"] == "submission_unknown"
    assert kv.budget(state)["committed_gpu_hours"] == 6


def test_freeze_preserves_exact_provenance_and_compute_hashes(campaign, tmp_path, monkeypatch):
    head = tmp_path / "head"
    head.mkdir()
    (head / "config.json").write_text('{"architecture":"DSparkDraftModel"}')
    (head / "model.safetensors").write_bytes(b"trained-weights")
    (head / "train_command.txt").write_text("trainer --epochs 10\n")
    repo = tmp_path / "repo"
    (repo / "targets/8b").mkdir(parents=True)
    (repo / "targets/8b/contract.json").write_text('{"target":"Apertus-v1.5-8B"}')
    workloads = repo / "workloads/8b"
    workloads.mkdir(parents=True)
    for name in ["eagle-8b-validation.jsonl", "eagle-8b-test.jsonl", "opb-heldout.jsonl"]:
        rows = [{"id": f"{w}-{i}", "workload": w, "messages": [{"role": "user", "content": "x"}],
                 "max_tokens": 512} for w in ["chat", "code", "summarization"] for i in range(2)]
        (workloads / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setitem(kv.HEAD_PATHS, "off", str(head))
    monkeypatch.setattr(kv, "_git", lambda *args: "revision")
    output = campaign / "off-inputs.json"
    report = kv.freeze_inputs(campaign, repo, "off", output, hash_weights=True, image=image)
    assert report["weights"][0]["sha256"] == kv.digest(head / "model.safetensors")
    assert report["container"]["sha256"] == kv.digest(image)
    preserved = report["training_and_export_artifacts"]
    assert any(Path(a["copy"]).read_text() == "trainer --epochs 10\n"
               for a in preserved if a["source"].endswith("train_command.txt"))
    smoke = Path(report["corpora"]["correctness.jsonl"]["path"])
    assert len(smoke.read_text().splitlines()) == 6
    assert all(json.loads(line)["max_tokens"] == 128 for line in smoke.read_text().splitlines())


def test_screen_ranks_importance_only_and_freezes_validation_speed_cell(
    campaign, tmp_path, monkeypatch,
):
    for repeat in [1, 2]:
        name = f"screen-{repeat}"
        kv.reserve(campaign, "screen", name, 1200)
        with kv.ledger(campaign) as state:
            state["allocations"][name].update({"state": "terminal", "slurm_state": "COMPLETED"})
    cells = {}
    for policy in kv.SCREEN_POLICIES:
        multiplier = {"native2048": 1, "recent512": 1.3,
                      "importance1024": 1.085, "importance512": 1.07}[policy]
        for workload in ["chat", "code", "summarization"]:
            for concurrency in [1, 32, 256]:
                throughput = multiplier
                if policy == "importance512" and workload == "code" and concurrency == 32:
                    throughput = 1.1
                cells[(policy, workload, concurrency)] = {
                    "throughput": 100 * throughput, "success": 1, "acceptance": 4,
                    "ttft_p95": 10, "tpot_p95": 5}
    monkeypatch.setattr(kv, "_allocation_cells", lambda allocation: (cells, [], {}))
    artifact = tmp_path / "real-control-report.json"
    artifact.write_text('{"test_fixture": true}')
    evidence = tmp_path / "evidence.json"
    policy_evidence = {"correctness_controls_passed": True, "artifact_paths": [str(artifact)],
                       "memory": {"includes_selection_metadata": True, "native_filled_window": True,
                                  "physical_eviction_verified": True, "draft_storage_ratio": 0.5}}
    evidence.write_text(json.dumps({"policies": {p: policy_evidence
                                                for p in kv.SCREEN_POLICIES[1:]}}))
    report_path = tmp_path / "screen-gate.json"
    report = kv.analyze(campaign, "screen", evidence, report_path)
    assert report["passed"]
    assert report["candidate"] == "importance512"
    assert report["selected_speed_cell"] == {"workload": "code", "concurrency": 32}
    kv.authorize_phase(campaign, "confirm", report_path)
    assert status(campaign)["phase_authorizations"]["confirm"]["speed_cell"] == {
        "workload": "code", "concurrency": 32}
