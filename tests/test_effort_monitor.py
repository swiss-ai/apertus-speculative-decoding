import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "effort_monitor", Path(__file__).resolve().parents[1] / "tools/monitor-effort-jobs.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def snapshot(state="RUNNING", alive=True, control="data"):
    return {
        "controller": {"status": control},
        "controller_alive": alive,
        "jobs": {"data": {"job_id": "123", "state": state}},
    }


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETED"])
def test_healthy_jobs_do_not_wake_agent(state):
    assert monitor.incident(snapshot(state)) is None


@pytest.mark.parametrize("state", sorted(monitor.FAILURES))
def test_terminal_errors_wake_repair(state):
    result = monitor.incident(snapshot(state))
    assert result["kind"] == "failure"
    assert result["jobs"]["data"]["job_id"] == "123"


def test_controller_failure_or_disappearance_wakes_repair():
    assert monitor.incident(snapshot(control="failed"))["kind"] == "failure"
    assert monitor.incident(snapshot(alive=False))["kind"] == "controller_stopped"


def test_completed_campaign_wakes_final_collection_before_disabling():
    assert monitor.incident(snapshot(alive=False, control="complete"))["kind"] == "complete"


def test_remote_probe_is_valid_python():
    compile(monitor.REMOTE_PROBE, "remote-probe", "exec")


def test_existing_repair_message_is_not_duplicated(monkeypatch, tmp_path):
    class Server:
        def __init__(self, _):
            pass

        def request(self, *args):
            return {"data": [{"id": "queued", "input": [{"text": monitor.PREFIX}]}]}

        def close(self):
            pass

    monkeypatch.setattr(monitor, "AppServer", Server)
    result = monitor.notify("codex", "thread", tmp_path, {"kind": "failure"})
    assert result == {"status": "already_queued", "id": "queued"}
