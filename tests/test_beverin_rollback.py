"""CPU checks for rollback decisions, continuity and the fault-injection gate."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
from fault_campaign import demonstrate
from fault_inject import injection_allowed, install
from numerical_guard import Guard
from rollback import decide, load_ledger


class RollbackTests(unittest.TestCase):
    def test_fault_campaign_rolls_back_then_stops_and_keeps_zero(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
                "RUN_ID": "faultinject-unit", "NODES": "1"}):
            os.environ.pop("SLURM_JOB_NUM_NODES", None)
            os.environ.pop("SLURM_NNODES", None)
            report = demonstrate(temp, str(Path(temp) / "train.log"))
        self.assertEqual(report["first"]["resume_step"], 10)
        self.assertEqual(report["second"]["resume_step"], 5)
        self.assertEqual(report["third"]["action"], "terminal")
        self.assertTrue(report["zero_intact_after_corrupt"])
        self.assertEqual(report["alerts"], 1)

    def test_fault_hook_refuses_multinode_and_plain_run_ids(self):
        class Trainer:
            def _optimizers_step(self):
                return None

        with patch.dict(os.environ, {"BEVERIN_FAULT_INJECT": "1", "RUN_ID": "faultinject-x",
                                      "SLURM_JOB_NUM_NODES": "32", "NODES": "1"}):
            with self.assertRaises(RuntimeError):
                install(Trainer, torch)
        with patch.dict(os.environ, {"BEVERIN_FAULT_INJECT": "1", "RUN_ID": "opb-real", "NODES": "1"}, clear=False):
            os.environ.pop("SLURM_JOB_NUM_NODES", None)
            with self.assertRaises(RuntimeError):
                install(Trainer, torch)
        with patch.dict(os.environ, {"RUN_ID": "faultinject-ok", "NODES": "1"}):
            os.environ.pop("SLURM_JOB_NUM_NODES", None)
            self.assertTrue(injection_allowed())

    def test_deadline_and_crash_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            from checkpoint_layout import promote
            from fault_campaign import _write_verified
            _write_verified(str(root), 10)
            log = root / "train.log"
            log.write_text("train/loss=0.4\n")
            soon = str(int(__import__("time").time()) + 60)
            stopped = decide(str(root), 1, str(log), "job", soon, "deadline")
            self.assertEqual(stopped["action"], "terminal")
            again = decide(str(root), 1, str(log), "job", 0, "crash-1")
            self.assertEqual(again["action"], "rollback")
            second = decide(str(root), 1, str(log), "job", 0, "crash-2")
            self.assertEqual(second["action"], "terminal")
            self.assertIn("allocation", second["reason"])
            promote  # imported so a missing helper fails this test loudly

    def test_optimizer_cpu_step_and_continuity_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight = torch.nn.Parameter(torch.ones(2))
            optimizer = torch.optim.AdamW([weight], lr=1e-3)
            weight.grad = torch.ones_like(weight)
            optimizer.step()
            optimizer.state[weight]["step"] = torch.tensor(4.0)
            trainer = type("T", (), {})()
            trainer.model = torch.nn.Linear(2, 1)
            trainer.optimizers = [optimizer]
            trainer.rank = 0
            trainer.is_distributed = False
            trainer.global_step = 10
            trainer.checkpointer = type("C", (), {"path": root})()
            guard = Guard(trainer, torch)
            names = [name for name, _value in guard.optimizer_state()]
            self.assertTrue(any(name.endswith("/step") for name in names))
            self.assertTrue(any("exp_avg" in name for name in names))
            (root / "rollback-ledger.json").write_text(json.dumps({
                "pending_continuity": {
                    "expected_global_step": 10,
                    "learning_rates": [1e-3],
                    "pre_median_loss": 0.2,
                    "window": 2,
                    "loss_ratio": 1.3,
                }
            }))
            guard.load_continuity()
            guard.observe_continuity((None, torch.tensor(0.5), {}))
            from numerical_guard import NumericalFailure
            with self.assertRaises(NumericalFailure):
                guard.observe_continuity((None, torch.tensor(0.5), {}))
            self.assertTrue((root / "numerical-failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
