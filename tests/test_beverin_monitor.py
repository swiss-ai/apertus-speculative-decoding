"""No-network checks for scoped monitoring and durable notification decisions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
import monitor_one_epoch as monitor


def snapshot(step=200):
    return {"utc": "2026-10-27T18:00:00Z", "run": monitor.RUN,
            "jobs": [{"JobIDRaw": "698072", "State": "RUNNING"},
                     {"JobIDRaw": "698073", "State": "PENDING"}],
            "active_job": "698072", "phase": "training", "attempt": "train-698072-node0.log",
            "global_step": step, "checkpoint": {"global_step": 192}, "ready_nodes": 32,
            "step_seconds": {"n": 50, "mean": 19, "median": 19},
            "completion": None, "completion_error": None, "error": None}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_regular_progress_and_saves_are_quiet(self):
        old, new = snapshot(), snapshot(264)
        new["checkpoint"] = {"global_step": 256}
        self.assertEqual(monitor.changes(old, new), [])

    def test_state_resume_validation_and_errors_are_material(self):
        for field, value in [("phase", "validation"), ("attempt", "train-698072.a2-node0.log"),
                             ("error", "RuntimeError: failed")]:
            new = snapshot(); new[field] = value
            self.assertTrue(monitor.changes(snapshot(), new))
        new = snapshot(); new["jobs"][0]["State"] = "FAILED"
        self.assertTrue(monitor.changes(snapshot(), new))

    def test_three_hour_heartbeat_without_save_spam(self):
        messages = []
        def reply(*args, **kwargs):
            step = [200, 264, 1000][reply.calls]; reply.calls += 1
            return subprocess.CompletedProcess(args[0], 0, json.dumps(snapshot(step)), "")
        reply.calls = 0
        with patch.object(monitor.subprocess, "run", side_effect=reply), \
             patch.object(monitor, "deliver", side_effect=lambda text, thread: messages.append(text) or "receipt"), \
             patch.object(monitor.time, "time", side_effect=[1000, 1300, 11801]):
            monitor.poll(self.directory, monitor.THREAD, prime=True)
            monitor.poll(self.directory, monitor.THREAD)
            self.assertEqual(messages, [])
            monitor.poll(self.directory, monitor.THREAD)
        self.assertEqual(len(messages), 1)
        self.assertIn("three-hour heartbeat", messages[0])

    def test_authentication_notifies_once_and_suspends_ssh(self):
        result = subprocess.CompletedProcess([], 255, "", "Permission denied (publickey)")
        with patch.object(monitor.subprocess, "run", return_value=result) as ssh, \
             patch.object(monitor, "deliver", return_value="receipt") as send:
            monitor.poll(self.directory, monitor.THREAD)
            monitor.poll(self.directory, monitor.THREAD)
        self.assertEqual(ssh.call_count, 1)
        self.assertEqual(send.call_count, 1)
        self.assertTrue(json.loads((self.directory / "state.json").read_text())["auth_suspended"])

    def test_transport_failures_retry_and_do_not_claim_job_failure(self):
        (self.directory / "state.json").write_text(json.dumps({"last_notification_at": 1000,
                                                              "snapshot": snapshot()}))
        result = subprocess.CompletedProcess([], 255, "", "Connection timed out")
        with patch.object(monitor.subprocess, "run", return_value=result) as ssh, \
             patch.object(monitor, "deliver", return_value="receipt") as send, \
             patch.object(monitor.time, "time", return_value=1300):
            for _ in range(3): monitor.poll(self.directory, monitor.THREAD)
        self.assertEqual(ssh.call_count, 3)
        self.assertEqual(send.call_count, 1)
        self.assertIn("does not establish training failure", send.call_args.args[0])

    def test_final_completion_and_exhaustion_stop_remote_polling(self):
        for authenticated in (False, True):
            folder = self.directory / str(authenticated); folder.mkdir()
            current = snapshot(4077)
            if authenticated:
                current.update(phase="complete", completion={"global_step": 4077})
            else:
                for job in current["jobs"]: job["State"] = "FAILED"
                current["active_job"] = None
            result = subprocess.CompletedProcess([], 0, json.dumps(current), "")
            with patch.object(monitor.subprocess, "run", return_value=result) as ssh, \
                 patch.object(monitor, "deliver", return_value="receipt") as send:
                monitor.poll(folder, monitor.THREAD)
                monitor.poll(folder, monitor.THREAD)
            self.assertEqual(ssh.call_count, 1)
            self.assertEqual(send.call_count, 1)

    def test_sender_uses_exact_thread_and_drops_inherited_thread(self):
        with patch.dict(os.environ, {"CODEX_THREAD_ID": "unrelated"}), \
             patch.object(monitor.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "queued", "")) as run:
            monitor.deliver("observations only", monitor.THREAD)
        args = run.call_args.args[0]
        self.assertEqual(args[:4], [monitor.CODEX, "queue", "--thread", monitor.THREAD])
        self.assertNotIn("CODEX_THREAD_ID", run.call_args.kwargs["env"])

    def test_major_slowdown_and_recovery_notify_once_each(self):
        current = snapshot()
        with patch.object(monitor.subprocess, "run") as ssh, \
             patch.object(monitor, "deliver", return_value="receipt") as send:
            ssh.return_value = subprocess.CompletedProcess([], 0, json.dumps(current), "")
            monitor.poll(self.directory, monitor.THREAD, prime=True)
            current["step_seconds"]["mean"] = 45
            ssh.return_value = subprocess.CompletedProcess([], 0, json.dumps(current), "")
            monitor.poll(self.directory, monitor.THREAD)
            monitor.poll(self.directory, monitor.THREAD)
            current["step_seconds"]["mean"] = 19
            ssh.return_value = subprocess.CompletedProcess([], 0, json.dumps(current), "")
            monitor.poll(self.directory, monitor.THREAD)
        self.assertEqual(send.call_count, 2)


if __name__ == "__main__":
    unittest.main()
