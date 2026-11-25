"""CPU checks for the monitor, ALPS submitter, quotas and the epoch decision."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "methods/dspark/beverin"))
sys.path.insert(0, str(REPO / "methods/dspark/beverin/data_expansion"))
from blas_decision import choose
from dedupe import BoundedIndex, Index
from epoch_decision import decide, doubling_gain, parse_curve
from monitor_campaign import assess, format_heartbeat, heartbeat_due, parse_log, remote_script
from quotas import allocate, classify


class CampaignTests(unittest.TestCase):
    def test_monitor_sees_nan_spikes_and_stale_checkpoints(self):
        parsed = parse_log("train/loss=nan global_step=12\nBEVERIN_GRAD_NORM {\"global_step\": 12, \"grad_norm\": \"nan\"}\n")
        self.assertTrue(parsed["nonfinite"])
        self.assertTrue(assess({"nonfinite": True, "phase": "training"}))
        losses = [0.5] * 25 + [1.0]
        self.assertTrue(any("loss" in item for item in assess({"losses": losses, "phase": "training"})))
        quiet = assess({"losses": [0.5] * 30, "grad_norms": [1.0] * 30, "phase": "training",
                         "verified_age_s": 10, "step_age_s": 10, "min_avail_gib": 100,
                         "global_step": 128, "step_seconds": 20, "saving": False})
        self.assertEqual(quiet, [])
        stale = assess({"phase": "training", "saving": False, "verified_age_s": 64 * 20 * 3,
                         "global_step": 200, "step_seconds": 20, "step_age_s": 1000})
        self.assertTrue(stale)
        self.assertTrue(heartbeat_due(None, 10))
        self.assertFalse(heartbeat_due(0, 10))
        text = format_heartbeat({"global_step": 100, "step_seconds": 20, "utc": "t", "phase": "training"})
        self.assertIn("100/4077", text["progress"])
        compile(remote_script("opb-thinkoff-32n-alps-1epoch-test", ["123"]), "<remote>", "exec")

    def test_quotas_keep_all_tool_rows_and_cap_math(self):
        counts = {
            "Toucan-1.5M": 10, "glaiveai/glaive-function-calling-v2": 4,
            "wikipedia": 1000, "romansh-sft": 3, "smoltalk2": 500,
            "nvidia/OpenMathReasoning": 1000, "dolci-if-eng_Latn-200k": 7,
        }
        plan = allocate(counts)
        self.assertEqual(sum(plan["tool"].values()), 14)
        self.assertEqual(plan["math"]["nvidia/OpenMathReasoning"], 50000 if False else min(1000, 50000))
        self.assertLessEqual(sum(plan["math"].values()), 50000)
        self.assertEqual(plan["multilingual"]["romansh-sft"], 3)
        self.assertLessEqual(sum(plan["multilingual"].values()), 300000)
        self.assertEqual(classify("finepdfs-edu"), "summarization")
        index = Index()
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        index.add(text, "a")
        self.assertEqual(index.duplicate(text)[0], "exact")
        bounded = BoundedIndex()
        bounded.add(text, "a")
        self.assertEqual(bounded.duplicate(text)[0], "exact")
        self.assertIsNone(bounded.duplicate("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"))

    def test_decision_does_not_launch_epoch_two(self):
        rows = [(step, 1.0, 2.0 + (0.1 if step > 3000 else 0.0)) for step in range(1900, 4100, 10)]
        gain = doubling_gain(rows)
        self.assertGreater(gain, 0.03)
        report = decide(gain, {"chat": [1.0, 1.0, 1.0, 1.0]}, 4.0)
        self.assertFalse(report["launch_epoch_2"])
        self.assertIn("chat", report["coverage_gaps"])
        flat = decide(0.0, {"chat": [4.0, 4.0]}, 4.0)
        self.assertEqual(flat["action"], "stop")
        self.assertEqual(len(parse_curve("train/loss=0.5 train/eal=2.0 global_step=10\n")), 1)

    def test_default_blas_must_be_within_five_percent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "default").mkdir()
            (root / "cublas").mkdir()
            (root / "default/rank0.json").write_text(json.dumps({"status": "passed", "timing": {"median_s": 1.04}}))
            (root / "cublas/rank0.json").write_text(json.dumps({"status": "passed", "timing": {"median_s": 1.0}}))
            self.assertEqual(choose(root / "default", root / "cublas")["backend"], "default")
            (root / "default/rank0.json").write_text(json.dumps({"status": "passed", "timing": {"median_s": 1.2}}))
            self.assertEqual(choose(root / "default", root / "cublas")["backend"], "cublas")

    def test_rendered_sbatch_pins_alps_and_keeps_the_historical_default(self):
        with tempfile.TemporaryDirectory() as temp:
            stub_dir = Path(temp) / "bin"
            stub_dir.mkdir()
            stub = stub_dir / "sbatch"
            stub.write_text("#!/bin/bash\necho 4242\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            work = Path(temp) / "work"
            env = os.environ.copy()
            env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
            env["WORK"] = str(work)
            env["BEVERIN_ALPS"] = "1"
            env["EXTRACT_HEALTH_SECONDS"] = "7200"
            env["RUN_ID"] = "render-alps"
            env["NODES"] = "1"
            env["CHAIN"] = "1"
            env["MAX_ATTEMPTS"] = "1"
            env["EXTRACT_START_ATTEMPTS"] = "1"
            env["SBATCH_EXTRA"] = "--no-requeue"
            env.pop("CKPT_FREQ", None)
            subprocess.check_call(["bash", "methods/dspark/beverin/submit-train.sh"], cwd=str(REPO), env=env)
            script = next((work / "logs").glob("*.sbatch"))
            text = script.read_text()
            self.assertIn("--network=disable_rdzv_get", text)
            self.assertIn("env-vllm-alps.toml", text)
            self.assertIn("CKPT_FREQ=0.015625", text)
            self.assertIn("MAX_ATTEMPTS=1", text)
            self.assertIn("416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928", text)
            self.assertIn("BEVERIN_CHECKPOINT_RING=1", text)
            self.assertIn("EXTRACT_HEALTH_SECONDS=7200", text)
            self.assertIn("BEVERIN_ROLLBACK=1", text)
            env["BEVERIN_ALPS"] = "0"
            env["RUN_ID"] = "render-old"
            env.pop("CKPT_FREQ", None)
            subprocess.check_call(["bash", "methods/dspark/beverin/submit-train.sh"], cwd=str(REPO), env=env)
            old = next(path for path in (work / "logs").glob("*.sbatch") if "render-old" in path.name)
            old_text = old.read_text()
            self.assertNotIn("--network=disable_rdzv_get", old_text)
            self.assertIn("CKPT_FREQ=0.0625", old_text)
            self.assertIn("env-vllm031.toml", old_text)


if __name__ == "__main__":
    unittest.main()
