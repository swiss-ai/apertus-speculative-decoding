"""CPU checks for multi-epoch resume and per-epoch archives on the checkpoint ring."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
from multi_epoch import archive_epoch, install, resume_point

STEPS = 5  # optimizer steps per epoch


def ring_save(root, epoch, local_step, global_step, weights):
    """What the ring leaves behind: the newest save in root/0, the old one renamed away."""
    zero = root / "0"
    if zero.exists():
        zero.rename(root / ("old-%d" % global_step))
    zero.mkdir(parents=True)
    (zero / "model.safetensors").write_text(weights)
    (zero / "config.json").write_text("{}")
    (zero / "training_state.json").write_text(json.dumps(
        {"epoch": epoch, "local_step": local_step, "global_step": global_step}))
    (zero / "verified-finite.json").write_text(json.dumps({"finite": True, "global_step": global_step}))


class Checkpointer:
    def __init__(self, path):
        self.path = path

    def val_metrics_path(self, epoch):
        return self.path / str(epoch) / "val_metrics.json"

    def update_best_symlink(self, epoch):
        link = self.path / "checkpoint_best"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path(str(epoch)), target_is_directory=True)


class Trainer:
    """Speculators' resume and epoch loop, reduced to what the ring changes."""

    def __init__(self, root, num_epochs, stop_at=None):
        self.checkpointer = Checkpointer(root)
        self.resume_from_checkpoint = True
        self.is_distributed = False
        self.num_epochs, self.stop_at = num_epochs, stop_at
        self.trained = []  # (epoch, local_step)

    def setup_trainer(self):
        # Speculators: the epoch is the highest integer directory, which is always 0.
        if (self.checkpointer.path / "0").exists():
            state = json.loads((self.checkpointer.path / "0/training_state.json").read_text())
            if state["epoch"] == 0 and state["local_step"] > 0:
                self.current_epoch, self._resume_local_step = 0, state["local_step"]
            else:
                self.current_epoch, self._resume_local_step = 1, 0
            self.global_step = state["global_step"]
        else:
            self.current_epoch, self._resume_local_step, self.global_step = 0, 0, 0

    def train_epoch(self, epoch):
        skip = self._resume_local_step if epoch == self.current_epoch else 0
        self._resume_local_step = 0
        for local in range(skip + 1, STEPS + 1):
            self.global_step += 1
            self.trained.append((epoch, local))
            if self.stop_at is not None and self.global_step == self.stop_at:
                self.maybe_save_checkpoint(epoch, local_step=local)
                raise SystemExit(0)

    def maybe_save_checkpoint(self, epoch, local_step=0):
        ring_save(self.checkpointer.path, epoch, local_step, self.global_step,
                  "weights-at-%d" % self.global_step)

    def val_epoch(self, epoch):
        return {"loss_epoch": 1.0 / (epoch + 1), "eal_epoch": 4.0 + epoch}

    def run(self):
        self.setup_trainer()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.train_epoch(epoch)
            self.maybe_save_checkpoint(epoch)
            metrics = self.val_epoch(epoch)
            path = self.checkpointer.val_metrics_path(epoch)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics))
            self.checkpointer.update_best_symlink(epoch)


class MultiEpochTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        class T(Trainer):
            pass

        class C(Checkpointer):
            pass

        install(T, C, log=lambda line: None)
        self.T, self.C = T, C

    def tearDown(self):
        self.tmp.cleanup()

    def trainer(self, **kwargs):
        t = self.T(self.root, **kwargs)
        t.checkpointer = self.C(self.root)
        return t

    def test_resume_point(self):
        self.assertEqual(resume_point({"epoch": 2, "local_step": 3, "global_step": 13}), (2, 3, 13))
        self.assertEqual(resume_point({"epoch": 2, "local_step": 0, "global_step": 15}), (3, 0, 15))

    def test_interrupted_mid_epoch_resumes_in_the_right_epoch(self):
        first = self.trainer(num_epochs=3, stop_at=STEPS + 3)  # stops at epoch 1, step 3
        with self.assertRaises(SystemExit):
            first.run()
        second = self.trainer(num_epochs=3)
        second.run()
        steps = first.trained + second.trained
        expected = [(e, s) for e in range(3) for s in range(1, STEPS + 1)]
        self.assertEqual(steps, expected)  # every step of every epoch exactly once
        self.assertEqual(second.global_step, 3 * STEPS)

    def test_epoch_archives_and_no_stray_epoch_directories(self):
        self.trainer(num_epochs=3).run()
        for epoch in range(3):
            folder = self.root / "epochs" / ("epoch-%d" % epoch)
            self.assertEqual((folder / "model.safetensors").read_text(),
                             "weights-at-%d" % ((epoch + 1) * STEPS))
            self.assertEqual(json.loads((folder / "val_metrics.json").read_text())["eal_epoch"],
                             4.0 + epoch)
        integer_dirs = sorted(p.name for p in self.root.iterdir() if p.name.isdigit())
        self.assertEqual(integer_dirs, ["0"])
        self.assertEqual(Path(str((self.root / "checkpoint_best").readlink())), Path("0"))

    def test_archive_refuses_a_mid_epoch_save(self):
        ring_save(self.root, 1, 3, 8, "w")
        with self.assertRaises(RuntimeError):
            archive_epoch(self.root, 1)


if __name__ == "__main__":
    unittest.main()
