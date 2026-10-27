"""CPU checks for one-epoch completion, validation recovery and continuation audits."""
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
from audit_train import audit
from single_epoch import install, read_completion, validated_artifacts, write_completion


def checkpoint(root, step=4, local=0):
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "optimizer_state_dict.pt", "scheduler_state_dict.pt", "config.json"):
        (folder / name).write_text("preserved-" + name)
    (folder / "training_state.json").write_text(json.dumps(
        {"epoch": 0, "local_step": local, "global_step": step}))


def trainer_type():
    class StubTrainer:
        def __init__(self, root, fail_val=False, interrupted=False):
            self.checkpointer = SimpleNamespace(path=root,
                previous_epoch=0 if (root / "0/training_state.json").exists() else -1)
            self.config = SimpleNamespace(num_epochs=1, max_steps=None)
            self.train_loader, self.val_loader = list(range(4)), [1, 2]
            self.rank, self.is_distributed = 0, False
            self.calls, self.trained_steps = [], 0
            self.fail_val, self.interrupted = fail_val, interrupted
            state = self._load_training_state()
            self.global_step = state.get("global_step", 0)
            self.current_epoch = 0 if not state or state["local_step"] else 1

        def _load_training_state(self):
            p = self.checkpointer.path / "0/training_state.json"
            return json.loads(p.read_text()) if p.exists() else {}

        def train_epoch(self, epoch):
            self.calls.append("train")
            self.trained_steps += 4 - self.global_step
            self.global_step = 4

        def maybe_save_checkpoint(self, epoch, local_step=0):
            self.calls.append("save")
            checkpoint(self.checkpointer.path, self.global_step, local_step)

        def val_epoch(self, epoch):
            self.calls.append("val")
            if self.fail_val:
                raise RuntimeError("validation failed")
            return {"loss_epoch": .5, "eal_epoch": 2.5}

        def run_training(self):
            if self.interrupted:
                return  # graceful_shutdown can swallow TrainingInterruptedError
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.train_epoch(epoch)
                self.maybe_save_checkpoint(epoch)
                result = self.val_epoch(epoch)
                (self.checkpointer.path / "0/val_metrics.json").write_text(json.dumps(result))

    install(StubTrainer, lambda self: None, logging.getLogger("single-epoch-test"))
    return StubTrainer


class SingleEpochTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "checkpoints"
        self.root.mkdir()
        self.environment = patch.dict(os.environ, {
            "BEVERIN_SINGLE_EPOCH": "1", "BEVERIN_EXPECTED_EPOCH_STEPS": "4",
            "BEVERIN_SEGMENT_STEPS": "0", "BEVERIN_PILOT_RESUME": "0",
            "BEVERIN_PROFILE_STEPS": "0",
        })
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_fresh_one_epoch(self):
        trainer = trainer_type()(self.root)
        trainer.run_training()
        self.assertEqual(trainer.calls, ["train", "save", "val"])
        self.assertEqual(trainer.trained_steps, 4)
        self.assertEqual(read_completion(self.root, 4)["global_step"], 4)

    def test_mid_epoch_resume_trains_only_remaining_steps(self):
        checkpoint(self.root, 2, 2)
        trainer = trainer_type()(self.root)
        trainer.run_training()
        self.assertEqual(trainer.trained_steps, 2)
        self.assertIsNotNone(read_completion(self.root, 4))

    def test_final_checkpoint_resumes_only_validation(self):
        checkpoint(self.root)
        before = (self.root / "0/model.safetensors").read_bytes()
        trainer = trainer_type()(self.root)
        trainer.run_training()
        self.assertEqual(trainer.calls, ["val"])
        self.assertEqual(trainer.trained_steps, 0)
        self.assertEqual((self.root / "0/model.safetensors").read_bytes(), before)
        self.assertIsNotNone(read_completion(self.root, 4))

    def test_validated_continuation_is_noop_and_does_not_overwrite(self):
        trainer_type()(self.root).run_training()
        marker = (self.root / "single-epoch-complete.json").read_bytes()
        trainer = trainer_type()(self.root)
        trainer.run_training()
        self.assertEqual(trainer.calls, [])
        self.assertEqual((self.root / "single-epoch-complete.json").read_bytes(), marker)

    def test_validation_failure_stays_incomplete_and_can_retry(self):
        trainer = trainer_type()(self.root, fail_val=True)
        with self.assertRaises(RuntimeError):
            trainer.run_training()
        self.assertIsNone(read_completion(self.root, 4))
        retry = trainer_type()(self.root)
        retry.run_training()
        self.assertEqual(retry.calls, ["val"])
        self.assertIsNotNone(read_completion(self.root, 4))

    def test_swallowed_interruption_does_not_mark_complete(self):
        trainer_type()(self.root, interrupted=True).run_training()
        self.assertIsNone(read_completion(self.root, 4))

    def test_torn_candidate_metrics_revalidate_but_marker_fails_closed(self):
        checkpoint(self.root)
        (self.root / "0/val_metrics.json").write_text("{")
        self.assertIsNone(validated_artifacts(self.root, 4))
        retry = trainer_type()(self.root)
        retry.run_training()
        self.assertEqual(retry.calls, ["val"])
        (self.root / "0/val_metrics.json").write_text("{")
        with self.assertRaises(ValueError):
            read_completion(self.root, 4)

    def test_invalid_epoch_or_pilot_contract_rejected(self):
        trainer = trainer_type()(self.root)
        trainer.config.num_epochs = 10
        with self.assertRaises(ValueError):
            trainer.run_training()
        trainer.config.num_epochs = 1
        with patch.dict(os.environ, {"BEVERIN_SEGMENT_STEPS": "100"}):
            with self.assertRaises(ValueError):
                trainer.run_training()
        self.assertEqual(trainer.calls, [])

    def test_audit_allows_only_authenticated_no_progress_paths(self):
        work = Path(self.temp.name) / "work"
        run = work / "runs/run"
        (run / "deployments/1").mkdir(parents=True)
        (run / "logs").mkdir()
        checkpoints = run / "checkpoints"
        checkpoint(checkpoints)
        self.assertEqual(audit(work, "run", "1", "start"), 0)
        self.assertEqual(audit(work, "run", "1", "finish", completion_skip=True), 1)
        trainer_type()(checkpoints).run_training()
        self.assertEqual(audit(work, "run", "1", "finish", completion_skip=True), 0)
        self.assertEqual(audit(work, "run", "1", "finish"), 1)
        (run / "logs/train-1-node0.log").write_text(
            "Resuming training on epoch 1.\nBEVERIN_VALIDATION_ONLY: resumed validation\n")
        self.assertEqual(audit(work, "run", "1", "finish"), 0)


if __name__ == "__main__":
    unittest.main()
