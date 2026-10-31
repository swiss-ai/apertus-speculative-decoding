import copy
import json
import shutil
from pathlib import Path

import pytest
from apertus_eagle.adaptation_decision import (
    classify_cell,
    final_verdict,
    paired_metrics,
    select_heads,
    workload_entries,
)

from apertus_bench.adaptation_confirmation import collect_report, round_spec
from apertus_bench.node_sweep import build_plan

REPO = Path(__file__).resolve().parents[1]
POLICY = json.loads((REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json").read_text())


def curve(gain=0.5, opb=3.0, passed=True):
    return {
        "state": "complete",
        "head_dir": "/head",
        "run_dir": "/run",
        "wall_seconds": 100,
        "scientific_gates": {"finite_losses": True, "loss_decreased": passed},
        "domain_acceptance_initial": 2.0,
        "domain_acceptance_final": 2.0 + gain,
        "opb_acceptance_initial": 4.0,
        "opb_acceptance_final": opb,
    }


def level(speed=100.0, tpot=20.0):
    return {
        "concurrency": 1,
        "success_rate": 1.0,
        "warmup_failed": 0,
        "output_tokens_per_second": speed,
        "tpot_p50_ms": tpot,
        "tpot_p95_ms": 30.0,
        "ttft_p50_ms": 40.0,
        "ttft_p95_ms": 50.0,
        "e2e_p95_ms": 1000.0,
        "mean_acceptance_length": 2.0,
    }


def test_development_selection_preserves_specialist_and_all_original_gates():
    matrix = {
        "adapt-chat-n8k-r0": curve(),
        "adapt-chat-n500-r20": curve(),
        "adapt-summ-n500-r0": curve(gain=1.0, passed=False),
        "adapt-mult-n500-r0": curve(gain=0.19),
    }
    report = select_heads(matrix, POLICY)
    assert [r["name"] for r in report["selected"]] == ["adapt-chat-n500-r20"]
    assert report["selected"][0]["opb_ratio"] == 0.75  # forgetting is not a selection filter
    assert report["excluded"]["adapt-summ-n500-r0"] == ["original_scientific_gate_failed"]
    assert not matrix["adapt-summ-n500-r0"]["scientific_gates"]["loss_decreased"]
    matrix["adapt-mult-n500-r0"]["domain_acceptance_final"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        select_heads(matrix, POLICY)


def test_repeatable_threshold_rejects_minute_or_unreplicated_gain():
    strong = paired_metrics(level(), level(120.0, 16.0))
    minute = paired_metrics(level(), level(103.0, 19.8))
    assert classify_cell([strong, strong], POLICY)["substantial_repeatable_benefit"]
    assert not classify_cell([strong, minute], POLICY)["substantial_repeatable_benefit"]
    assert not classify_cell([minute, minute], POLICY)["substantial_repeatable_benefit"]
    with pytest.raises(ValueError, match="warmups"):
        paired_metrics(level(), level() | {"warmup_failed": 1})


def test_negative_closure_and_conditional_forgetting_verdict():
    compute = {
        "trace_allocated_gpu_seconds": 400.0,
        "whole_campaign_allocated_gpu_seconds": 2000.0,
        "domain_training_allocated_gpu_seconds": {"chat": 500.0},
    }
    empty = {"selected": [], "excluded": {"arm": ["scientific_gate_failed"]}}
    assert final_verdict(empty, [], POLICY, compute)["state"] == "negative_complete"
    selection = select_heads({"adapt-chat-n500-r0": curve()}, POLICY)
    strong = paired_metrics(level(), level(120.0, 16.0))
    report = {
        "name": "adapt-chat-n500-r0",
        "correctness_passed": True,
        "cells": [
            {"label": "chat", "role": "domain", "k": 3, "concurrency": 1, "pairs": [strong, strong]}
        ],
    }
    result = final_verdict(selection, [report], POLICY, compute)
    assert result["state"] == "complete"
    head = result["heads"][0]
    assert head["recommendation"] == "conditional_specialist_tradeoff"
    assert 540000 <= head["cells"][0]["conservative_break_even_output_tokens"] <= 540001
    assert strong["gpu_seconds_per_million_tokens"]["baseline"] == 10000
    report["correctness_passed"] = False
    with pytest.raises(ValueError, match="correctness"):
        final_verdict(selection, [report], POLICY, compute)


def test_original_multilingual_workloads_and_no_added_variants(tmp_path):
    entries = workload_entries(REPO, tmp_path, "mult")
    assert {w["label"] for w in entries} == {
        "opb",
        "multilingual",
        "lang-deu",
        "lang-fra",
        "lang-ita",
        "lang-spa",
        "translate-from-en",
        "multi-turn",
        "code-multi",
        "math-gsm8k",
    }
    assert entries[1]["workloads_file"].endswith("/data/pools/multilingual/test.jsonl")
    selection = {"name": "adapt-mult-n500-r0", "head_dir": "/adapt", "workloads": entries}
    for repeat in ("a", "b"):
        spec = round_spec(selection, POLICY, repeat)
        plan = build_plan(spec, check_heads=False)
        assert [a.name for a in plan.arms] == list(POLICY["confirmation"][f"gpus_round_{repeat}"])
        assert all(t.concurrencies == [1, 64, 256] for t in plan.loadtests)
        assert len(plan.captures) == 10
        assert all(c.limit == 32 and c.max_tokens == 256 for c in plan.captures)
        assert all(a.gpus == 1 for a in plan.arms)


def make_confirmation(tmp_path):
    policy = copy.deepcopy(POLICY)
    policy["confirmation"]["k"] = [3]
    policy["confirmation"]["concurrencies"] = [1]
    selection = {
        "name": "adapt-chat-n500-r0",
        "workloads": [{"label": "chat", "workload": "chat", "role": "domain"}],
    }
    for repeat in ("a", "b"):
        round_dir = tmp_path / repeat
        round_dir.mkdir()
        status = {}
        for arm, gpu in policy["confirmation"][f"gpus_round_{repeat}"].items():
            folder = round_dir / arm
            folder.mkdir()
            status[arm] = "ok"
            (folder / "status.json").write_text(json.dumps({"gpus": [gpu]}))
            capture = {
                "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 1},
                "model": arm,
                "base_url": f"http://gpu{gpu}",
                "cases": [
                    {
                        "prompt_id": "p",
                        "workload": "chat",
                        "measurement": {"success": True, "output": "same", "completion_tokens": 1},
                    }
                ],
            }
            (folder / "capture-chat.json").write_text(json.dumps(capture))
            label_dir = folder / "chat"
            label_dir.mkdir()
            (label_dir / "loadtest-summary.json").write_text(json.dumps({"levels": [level()]}))
            cell = label_dir / "chat/c1"
            cell.mkdir(parents=True)
            for name in (
                "metadata.json",
                "requests.jsonl",
                "summary.json",
                "metrics_before.prom",
                "metrics_after.prom",
            ):
                (cell / name).touch()
        (round_dir / "sweep-status.json").write_text(json.dumps(status))
    return selection, policy


def test_collect_requires_complete_raw_cells_and_actual_gpu_swap(tmp_path):
    selection, policy = make_confirmation(tmp_path)
    report = collect_report(tmp_path, selection, policy)
    assert report["correctness_passed"]
    assert len(report["cells"][0]["pairs"]) == 2
    status = tmp_path / "b/base-k3/status.json"
    status.write_text(json.dumps({"gpus": [0]}))
    with pytest.raises(ValueError, match="GPU assignment"):
        collect_report(tmp_path, selection, policy)
    status.write_text(json.dumps({"gpus": [1]}))
    (tmp_path / "a/base-k3/chat/chat/c1/requests.jsonl").unlink()
    with pytest.raises(ValueError, match="raw measurement"):
        collect_report(tmp_path, selection, policy)


def test_controller_automatically_closes_negative_without_submitting_serving(tmp_path, monkeypatch):
    from apertus_eagle import adaptation_recovery as recovery

    repo = tmp_path / "repo"
    root = tmp_path / "campaign"
    configs = repo / "methods/eagle/configs/8b"
    configs.mkdir(parents=True)
    shutil.copytree(REPO / "methods/eagle/configs/8b/adapt", configs / "adapt")
    shutil.copy2(REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json", configs)
    (repo / "source-manifest.json").write_text(json.dumps({"revision": "frozen", "files": {}}))
    (root / "data/adapt-data").mkdir(parents=True)
    (root / "data/adapt-data/manifest.json").write_text("{}")
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return "frozen\n"
        assert argv[0] == "sacct", f"Unexpected submission: {argv}"
        if argv[-1].endswith("AllocNodes,State,ExitCode"):
            return "123|100|1|COMPLETED|0:0|\n"
        return "123|COMPLETED|0:0|\n"

    monkeypatch.setattr(recovery.subprocess, "check_output", command)
    monkeypatch.setattr(recovery, "curve_report", lambda *_: curve(gain=0.5, passed=False))
    recovery.main(
        [
            "--repo",
            str(repo),
            "--campaign-root",
            str(root),
            "--trace-job",
            "123",
            "--overlay-root",
            str(tmp_path / "overlays"),
        ]
    )
    state = json.loads((root / "receipts/campaign.json").read_text())
    assert state["state"] == "negative_complete"
    assert state["job_ids"] == ["123"]
    assert state["confirmations"] == {}
    assert len(json.loads((root / "selection.json").read_text())["excluded"]) == 19
    assert json.loads((root / "verdict.json").read_text())["state"] == "negative_complete"
    assert not any("node-sweep.sh" in str(call) for call in calls)


@pytest.mark.parametrize("correctness_passed", [True, False])
def test_controller_runs_confirmation_and_records_terminal_verdict(
    tmp_path, monkeypatch, correctness_passed
):
    from apertus_eagle import adaptation_recovery as recovery

    repo, root = tmp_path / "repo", tmp_path / "campaign"
    configs = repo / "methods/eagle/configs/8b"
    configs.mkdir(parents=True)
    shutil.copytree(REPO / "methods/eagle/configs/8b/adapt", configs / "adapt")
    shutil.copy2(REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json", configs)
    (repo / "source-manifest.json").write_text(json.dumps({"revision": "frozen", "files": {}}))
    (root / "data/adapt-data").mkdir(parents=True)
    (root / "data/adapt-data/manifest.json").write_text("{}")
    workloads = repo / "workloads/8b"
    workloads.mkdir(parents=True)
    for name in ("eagle-8b-test.jsonl", "opb-heldout.jsonl"):
        (workloads / name).write_text("{}\n")
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return "frozen\n"
        if argv[:2] == ["bash", "serving/node-sweep.sh"]:
            report_path = (
                root / "confirmations/adapt-chat-n500-r0/paired/paired/confirmation-summary.json"
            )
            report_path.parent.mkdir(parents=True)
            pair = paired_metrics(level(), level(120.0, 16.0))
            report_path.write_text(
                json.dumps(
                    {
                        "name": "adapt-chat-n500-r0",
                        "correctness_passed": correctness_passed,
                        "cells": [
                            {
                                "label": "chat",
                                "role": "domain",
                                "k": 3,
                                "concurrency": 1,
                                "pairs": [pair, pair],
                            }
                        ],
                    }
                )
            )
            return "job=42 output=confirmation\n"
        assert argv[0] == "sacct", argv
        if argv[-1].endswith("AllocNodes,State,ExitCode"):
            return "123|100|1|COMPLETED|0:0|\n42|100|1|COMPLETED|0:0|\n"
        return argv[5] + "|COMPLETED|0:0|\n"

    def curve_for_path(run, head):
        return curve(passed=run.name.startswith("adapt-chat-n500-r0-"))

    monkeypatch.setattr(recovery.subprocess, "check_output", command)
    monkeypatch.setattr(recovery, "curve_report", curve_for_path)
    argv = [
        "--repo",
        str(repo),
        "--campaign-root",
        str(root),
        "--trace-job",
        "123",
        "--overlay-root",
        str(tmp_path / "overlays"),
    ]
    recovery.main(argv)
    state = json.loads((root / "receipts/campaign.json").read_text())
    assert state["state"] == ("complete" if correctness_passed else "needs_repair")
    assert state["job_ids"] == ["123", "42"]
    assert state["confirmations"]["adapt-chat-n500-r0"]["job_id"] == "42"
    if correctness_passed:
        verdict = json.loads((root / "verdict.json").read_text())
        assert verdict["scope_completed"].startswith("Original bounded EAGLE")
        assert verdict["conditional_dspark_followup"]["promising_eagle_result"]
        recovery.main(argv)
        assert sum(call[:2] == ["bash", "serving/node-sweep.sh"] for call in calls) == 1
    else:
        assert not (root / "verdict.json").exists()


def test_between_arm_validation_distinguishes_deadline_from_infrastructure(tmp_path, monkeypatch):
    from apertus_eagle import adaptation_validate as validation

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        json.dumps(
            {
                "run_name": "adapt",
                "output_root": str(tmp_path / "runs"),
                "head_root": str(tmp_path / "heads"),
                "training": {"num_train_steps": 384},
            }
        )
    )
    monkeypatch.setattr(validation, "curve_report", lambda *_: curve(passed=False))
    assert validation.inspect(cfg, "id")[0] == 0
    monkeypatch.setattr(validation, "curve_report", lambda *_: {"state": "infrastructure_failed"})
    assert validation.inspect(cfg, "id")[0] == 2
    monkeypatch.setattr(validation, "curve_report", lambda *_: {"state": "incomplete"})
    run = tmp_path / "runs/adapt-id"
    (run / "checkpoints/resume").mkdir(parents=True)
    (run / "training-incomplete.json").write_text("{}")
    assert validation.inspect(cfg, "id")[0] == 4


def test_confirmation_only_uses_frozen_results_and_fresh_attempt_without_training(
    tmp_path, monkeypatch
):
    from apertus_eagle import adaptation_recovery as recovery
    from apertus_eagle.adaptation_decision import immutable_json

    repo, root = tmp_path / "repo", tmp_path / "campaign"
    (repo / "methods/eagle/configs/8b").mkdir(parents=True)
    shutil.copy2(
        REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json",
        repo / "methods/eagle/configs/8b",
    )
    immutable_json(repo / "source-manifest.json", {"revision": "frozen", "files": {}})
    (root / "data/adapt-data").mkdir(parents=True)
    immutable_json(root / "data/adapt-data/manifest.json", {})
    workload = repo / "workloads/8b"
    workload.mkdir(parents=True)
    for name in ("opb-heldout.jsonl", "eagle-8b-test.jsonl"):
        (workload / name).write_text("{}\n")
    matrix = {}
    templates = sorted((REPO / "methods/eagle/configs/8b/adapt").glob("adapt-*.yaml"))
    for domain in ("chat", "summ", "mult"):
        for path in templates:
            if path.stem.startswith("adapt-" + domain + "-"):
                matrix[path.stem] = curve(passed=path.stem == "adapt-chat-n500-r0")
    immutable_json(root / "adaptation-matrix-v1.json", matrix)
    immutable_json(root / "selection.json", select_heads(matrix, POLICY))
    originals = {
        name: (root / name).read_bytes() for name in ("adaptation-matrix-v1.json", "selection.json")
    }
    old = root / "confirmations/adapt-chat-n500-r0"
    old.mkdir(parents=True)
    (old / "environment.toml").write_text("old failed setup\n")
    immutable_json(
        root / "receipts/campaign.json",
        {
            "workstream": "adaptation",
            "state": "needs_repair",
            "job_ids": ["123"],
            "batches": {domain: {"job_ids": []} for domain in ("chat", "summ", "mult")},
        },
    )
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "rev-parse"]:
            return "frozen\n"
        if argv[:2] == ["bash", "serving/node-sweep.sh"]:
            env = kwargs["env"]
            assert Path(env["TMPDIR"]).is_relative_to(root / "submission-tmp")
            assert env["TMPDIR"] != "/tmp"
            assert env["SWEEP_TIME"] == "12:00:00"
            output = Path(env["SWEEP_OUTPUT"])
            assert output == old / "repair-test"
            report = output / "paired/paired/confirmation-summary.json"
            report.parent.mkdir(parents=True)
            strong = paired_metrics(level(), level(120.0, 16.0))
            immutable_json(
                report,
                {
                    "name": "adapt-chat-n500-r0",
                    "correctness_passed": True,
                    "cells": [
                        {
                            "label": "chat",
                            "role": "domain",
                            "k": 3,
                            "concurrency": 1,
                            "pairs": [strong, strong],
                        }
                    ],
                },
            )
            return "job=42 output=confirmation\n"
        assert argv[0] == "sacct", f"Training or unexpected command: {argv}"
        if argv[-1].endswith("AllocNodes,State,ExitCode"):
            return "123|100|1|COMPLETED|0:0|\n42|100|1|COMPLETED|0:0|\n"
        return "42|COMPLETED|0:0|\n"

    def no_training_reads(*_args):
        raise AssertionError("Confirmation repair must not re-read completed arms")

    monkeypatch.setattr(recovery.subprocess, "check_output", command)
    monkeypatch.setattr(recovery, "curve_report", no_training_reads)
    recovery.main(
        [
            "--repo",
            str(repo),
            "--campaign-root",
            str(root),
            "--trace-job",
            "123",
            "--overlay-root",
            str(tmp_path / "overlays"),
            "--confirmation-only",
            "--confirmation-attempt",
            "repair-test",
            "--confirmation-wall-time",
            "12:00:00",
        ]
    )
    assert (old / "environment.toml").read_text() == "old failed setup\n"
    assert all((root / name).read_bytes() == content for name, content in originals.items())
    state = json.loads((root / "receipts/campaign.json").read_text())
    assert state["state"] == "complete"
    assert state["confirmations"]["adapt-chat-n500-r0"]["attempt"] == "repair-test"
    assert state["confirmation_schedule"]["frozen_wall_budget"] == "24:00:00"
    assert state["confirmation_schedule"]["requested_wall_time"] == "12:00:00"
    assert not any("submit-eagle-train" in str(call) for call in calls)


@pytest.mark.parametrize("wall_time", ["25:00:00", "00:00:00", "12:99:00", "12h"])
def test_confirmation_scheduling_cannot_expand_frozen_budget(tmp_path, wall_time):
    from apertus_eagle import adaptation_recovery as recovery

    repo = tmp_path / "repo"
    config_dir = repo / "methods/eagle/configs/8b"
    config_dir.mkdir(parents=True)
    shutil.copy2(REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json", config_dir)
    with pytest.raises(ValueError, match="wall time|scheduling cap"):
        recovery.main(
            [
                "--repo", str(repo), "--campaign-root", str(tmp_path / "campaign"),
                "--trace-job", "123", "--overlay-root", str(tmp_path / "overlays"),
                "--confirmation-wall-time", wall_time,
            ]
        )


def test_confirmation_only_rejects_incomplete_frozen_matrix(tmp_path, monkeypatch):
    from apertus_eagle import adaptation_recovery as recovery
    from apertus_eagle.adaptation_decision import immutable_json

    repo, root = tmp_path / "repo", tmp_path / "campaign"
    (repo / "methods/eagle/configs/8b").mkdir(parents=True)
    shutil.copy2(
        REPO / "methods/eagle/configs/8b/adapt-recovery-policy.json",
        repo / "methods/eagle/configs/8b",
    )
    immutable_json(repo / "source-manifest.json", {"revision": "frozen", "files": {}})
    immutable_json(root / "adaptation-matrix-v1.json", {"one": {"state": "incomplete"}})
    immutable_json(root / "selection.json", {"selected": []})
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *_args, **_kwargs: "frozen\n")
    with pytest.raises(RuntimeError, match="all original 19 completed curves"):
        recovery.main(
            [
                "--repo",
                str(repo),
                "--campaign-root",
                str(root),
                "--trace-job",
                "123",
                "--overlay-root",
                str(tmp_path / "overlays"),
                "--confirmation-only",
                "--confirmation-attempt",
                "repair-test",
            ]
        )
