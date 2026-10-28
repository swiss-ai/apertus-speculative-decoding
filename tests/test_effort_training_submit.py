import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from apertus_bench import effort_training_submit as controller


def test_failed_qualification_never_submits_data_or_training(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "initialize", lambda *args: None)

    def fail(*args):
        raise RuntimeError("Slurm job ended as FAILED")

    monkeypatch.setattr(controller, "wait_success", fail)
    submitted = []
    monkeypatch.setattr(controller, "stage_job", lambda *args: submitted.append(args))
    args = SimpleNamespace(output=tmp_path, controller_timeout=60, qualification_job="123")
    with pytest.raises(RuntimeError, match="FAILED"):
        controller.run(args)
    assert not submitted
    assert json.loads((tmp_path / "controller-status.json").read_text())["status"] == "failed"


def test_job_state_keeps_full_terminal_names(monkeypatch):
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="123|OUT_OF_MEMORY\n123.batch|FAILED\n"),
    )
    assert controller.job_state("123") == "OUT_OF_MEMORY"


def test_stage_allocation_and_container_step_explicitly_use_one_task(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(controller, "source_identity", lambda: {"revision": "frozen"})

    def capture(argv, directory, record, args):
        captured.append(argv)
        return record

    monkeypatch.setattr(controller, "submit_job", capture)
    args = SimpleNamespace(
        upstream=tmp_path / "upstream",
        phase1_reference=tmp_path / "phase1",
        account="infra01",
        partition="debug",
        time="01:30:00",
    )
    controller.stage_job(args, tmp_path, "data")
    assert "--ntasks=1" in captured[0]
    script = tmp_path / "submissions/data/job.sh"
    step = shlex.split(script.read_text().splitlines()[2])
    assert step[0] == "srun" and "--ntasks=1" in step and "--nodes=1" in step
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_parent_paths_and_frozen_source_are_recorded_without_reading_raw_requests(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    parent = repo / "results/gpt-oss-20b/thinking-effort/phase1"
    report = parent / "reports/phase1-final/comparison.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"decision":"proceed"}')
    (parent / "experiment.yaml").write_text("schema_version: 1")
    monkeypatch.setattr(
        controller, "read_spec", lambda _: {"source_workload": "workloads/probe.jsonl"}
    )
    monkeypatch.setattr(controller, "load_config", lambda _: {})
    monkeypatch.setattr(controller, "source_identity", lambda: {"revision": "new"})
    config = tmp_path / "phase2.yaml"
    config.write_text("schema_version: 1\n")
    env = tmp_path / "environment.toml"
    env.write_text('image="pinned"\n')
    campaign = tmp_path / "phase2"
    args = SimpleNamespace(phase1_reference=parent, config=config, environment=env)
    controller.initialize(args, campaign)
    import yaml

    spec = yaml.safe_load((campaign / "experiment.yaml").read_text())
    assert spec["source_workload"] == str(repo / "workloads/probe.jsonl")
    assert (
        json.loads((campaign / "controller-config.json").read_text())["source"]["revision"] == "new"
    )
