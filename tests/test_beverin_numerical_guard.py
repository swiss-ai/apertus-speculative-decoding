"""Numerical failures must stop every rank and preserve the previous save."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
from numerical_guard import install, NumericalFailure


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("nan")


class Toy(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.mode = mode

    def forward(self, x):
        loss = (self.weight * x).sum()
        if self.mode == "loss":
            loss = loss / 0
        if self.mode == "gradient":
            loss = BadBackward.apply(loss)
        return None, loss, {"loss": loss.detach()}


def trainer_type():
    class Trainer:
        def __init__(self, root, mode="finite", rank=0, distributed=False):
            self.rank, self.is_distributed = rank, distributed
            self.checkpointer = SimpleNamespace(path=root)
            self.model, self.global_step = Toy(mode), 518
            self.steps, self.mode = 0, mode
            if mode == "loaded":
                self.model.weight.data.fill_(float("nan"))
            self.setup_model()

        def setup_model(self):
            pass

        def _optimizers_step(self):
            self.steps += 1
            with torch.no_grad():
                self.model.weight.sub_(.01 * self.model.weight.grad)
                if self.mode == "update":
                    self.model.weight.fill_(float("nan"))

        def train_epoch(self, x):
            _, loss, _ = self.model(x=x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self._optimizers_step()
            self.global_step += 1
            (self.checkpointer.path / "last-save").write_text("updated")

    install(Trainer, torch)
    return Trainer


def distributed_failure(rank, root, store):
    dist.init_process_group("gloo", init_method="file://" + store,
                            rank=rank, world_size=2)
    try:
        trainer = trainer_type()(Path(root), rank=rank, distributed=True)
        x = torch.ones(2)
        if rank == 1:
            x[0] = float("nan")
        try:
            trainer.train_epoch(x)
            raise AssertionError("bad rank did not stop the group")
        except NumericalFailure as exc:
            assert "rank=1" in str(exc), str(exc)
            assert trainer.steps == 0
            assert torch.equal(trainer.model.weight, torch.ones(2))
    finally:
        dist.destroy_process_group()


class NumericalGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "last-save").write_text("preserved")
        self.environment = patch.dict(os.environ, {"BEVERIN_CAPTURE_BAD_BATCH": "1"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def failure(self, mode, expected, bad_input=False):
        trainer = trainer_type()(self.root, mode)
        clip_before = torch.nn.utils.clip_grad_norm_
        x = torch.tensor([float("nan"), 1.]) if bad_input else torch.ones(2)
        with self.assertRaises(NumericalFailure):
            trainer.train_epoch(x)
        self.assertIs(torch.nn.utils.clip_grad_norm_, clip_before)
        self.assertEqual((self.root / "last-save").read_text(), "preserved")
        report = json.loads((self.root / "numerical-failure.json").read_text())
        self.assertEqual(report["phase"], expected)
        self.assertEqual(report["global_step"], 518)
        self.assertTrue((self.root / "numerical-failure-batches/rank0.pt").is_file())
        return trainer

    def test_finite_training_proceeds(self):
        trainer = trainer_type()(self.root)
        trainer.train_epoch(torch.ones(2))
        self.assertEqual(trainer.steps, 1)
        self.assertEqual((self.root / "last-save").read_text(), "updated")
        self.assertFalse((self.root / "numerical-failure.json").exists())

    def test_input_failure_before_forward_or_optimizer(self):
        self.assertEqual(self.failure("finite", "inputs", True).steps, 0)

    def test_loss_failure_before_backward_or_optimizer(self):
        trainer = self.failure("loss", "forward-loss-and-metrics")
        self.assertEqual(trainer.steps, 0)
        self.assertIsNone(trainer.model.weight.grad)

    def test_finite_loss_with_invalid_backward(self):
        self.assertEqual(self.failure("gradient", "backward-gradients").steps, 0)

    def test_overflowed_norm_with_finite_gradients_stops_before_update(self):
        trainer = trainer_type()(self.root)
        with self.assertRaises(NumericalFailure):
            trainer.train_epoch(torch.full((2,), 1e20))
        self.assertEqual(trainer.steps, 0)
        report = json.loads((self.root / "numerical-failure.json").read_text())
        self.assertEqual(report["phase"], "gradient-norm")
        self.assertEqual((self.root / "last-save").read_text(), "preserved")

    def test_bad_optimizer_update_cannot_be_saved(self):
        self.assertEqual(self.failure("update", "updated-parameters").steps, 1)

    def test_corrupt_loaded_weights_rejected(self):
        with self.assertRaises(NumericalFailure):
            trainer_type()(self.root, "loaded")
        report = json.loads((self.root / "numerical-failure.json").read_text())
        self.assertEqual(report["phase"], "loaded-parameters")

    def test_terminal_marker_prevents_restart(self):
        self.failure("loss", "forward-loss-and-metrics")
        with self.assertRaisesRegex(NumericalFailure, "existing terminal marker"):
            trainer_type()(self.root)

    def test_one_bad_rank_stops_both_before_updates(self):
        mp.spawn(distributed_failure, args=(str(self.root), str(self.root / "store")),
                 nprocs=2, join=True)
        report = json.loads((self.root / "numerical-failure.json").read_text())
        self.assertEqual(report["rank"], 1)
        for rank in (0, 1):
            self.assertTrue((self.root / "numerical-failure-batches" /
                             ("rank%s.pt" % rank)).is_file())
        self.assertEqual((self.root / "last-save").read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
