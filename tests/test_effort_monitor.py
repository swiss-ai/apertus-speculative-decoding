import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

        def request(self, method, *args):
            if method == "thread/read":
                return {"thread": {"status": {"type": "active"}}}
            return {"data": [{"id": "queued", "input": [{"text": monitor.PREFIX}]}]}

        def close(self):
            pass

    monkeypatch.setattr(monitor, "AppServer", Server)
    result = monitor.notify("codex", "thread", tmp_path, {"kind": "failure"})
    assert result == {"status": "already_queued", "id": "queued"}


def test_idle_thread_starts_queued_repair():
    calls = []

    class Server:
        def request(self, method, params):
            calls.append((method, params))
            return {"thread": {"status": {"type": "idle"}}}

    monitor.start_if_idle(Server(), "thread", "queued")
    assert calls[-1] == (
        "thread/queue/start",
        {"threadId": "thread", "queuedSubmissionId": "queued"},
    )


def monitor_args(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["monitor", "--state-dir", str(tmp_path), "--thread", "thread"],
    )
    monkeypatch.setattr(monitor, "notify", lambda *args: {"status": "test"})


def test_unreadable_certificate_blocks_without_attempting_ssh(monkeypatch, tmp_path):
    monitor_args(monkeypatch, tmp_path)
    certificate = {"error": "PermissionError: public certificate is unreadable"}
    monkeypatch.setattr(monitor, "certificate_status", lambda: certificate)

    def forbidden(*args, **kwargs):
        raise AssertionError("SSH must not run with unreadable credentials")

    monkeypatch.setattr(monitor.subprocess, "run", forbidden)
    monitor.main()
    state = json.loads((tmp_path / "monitor-state.json").read_text())
    assert state["access_blocked"]
    assert state["access_blocked_certificate"] == certificate
    assert json.loads((tmp_path / "latest.json").read_text())["incident"]["kind"] == (
        "authentication_required"
    )
    monitor.main()


@pytest.mark.parametrize("changed", [False, True])
def test_only_changed_readable_certificate_resumes_access(monkeypatch, tmp_path, changed):
    monitor_args(monkeypatch, tmp_path)
    path = tmp_path / "monitor-state.json"
    path.write_text(
        json.dumps({"access_blocked": True, "access_blocked_certificate": {"sha256": "old"}})
    )
    monkeypatch.setattr(
        monitor, "certificate_status", lambda: {"sha256": "new" if changed else "old"}
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps(snapshot()))

    monkeypatch.setattr(monitor.subprocess, "run", run)
    monitor.main()
    assert len(calls) == int(changed)
    assert json.loads(path.read_text())["access_blocked"] is not changed


def test_connection_error_preserves_actual_stderr(monkeypatch, tmp_path):
    monitor_args(monkeypatch, tmp_path)
    monkeypatch.setattr(monitor, "certificate_status", lambda: {"sha256": "public"})

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, ["ssh"], stderr="jump-host connection closed")

    monkeypatch.setattr(monitor.subprocess, "run", fail)
    monitor.main()
    result = json.loads((tmp_path / "latest.json").read_text())
    assert result["snapshot"]["connection_error"] == "jump-host connection closed"
