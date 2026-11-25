"""Multi-epoch support for the Beverin checkpoint ring (BEVERIN_CHECKPOINT_RING=1).

The ring keeps the newest verified save in ``checkpoints/0`` whatever its
epoch, so speculators' own bookkeeping, which reads the epoch from the
directory name, is wrong from the second epoch on:

- resume: ``previous_epoch`` is always 0, so a save from epoch N (mid-epoch or
  end) would restart at epoch 1. Here the epoch and step come from
  ``checkpoints/0/training_state.json`` instead.
- validation metrics and ``checkpoint_best`` would create ``checkpoints/<N>``
  holding only ``val_metrics.json``, which the next resume would then try to
  load as weights. They are redirected into ``checkpoints/0``.
- each epoch's end-of-epoch weights would be rotated away. They are kept as
  hard links (no copy, survives the ring's renames and pruning) under
  ``checkpoints/epochs/epoch-<N>/``, with that epoch's validation metrics
  written there every epoch, not only when the loss improves.
"""
import json
import os
from pathlib import Path

ARCHIVE_FILES = (
    "model.safetensors", "config.json", "training_state.json",
    "verified-finite.json", "train_command.txt",
)


def _rank0(trainer):
    try:
        import torch.distributed as dist
        if getattr(trainer, "is_distributed", False) and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True


def resume_point(state):
    """(epoch to run next, local steps to skip, global step) from a training state."""
    epoch = int(state["epoch"])
    local_step = int(state.get("local_step", 0))
    global_step = int(state.get("global_step", 0))
    if local_step > 0:
        return epoch, local_step, global_step
    return epoch + 1, 0, global_step


def archive_epoch(root, epoch):
    """Hard-link the end-of-epoch save in ``root/0`` to ``root/epochs/epoch-<N>``."""
    root = Path(root)
    zero = root / "0"
    state = json.loads((zero / "training_state.json").read_text())
    if int(state["epoch"]) != epoch or int(state.get("local_step", 0)) != 0:
        raise RuntimeError("checkpoints/0 is not the end of epoch %d: %s" % (epoch, state))
    dest = root / "epochs" / ("epoch-%d" % epoch)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        source = zero / name
        target = dest / name
        if source.is_file() and not target.exists():
            os.link(str(source), str(target))
    for source in zero.glob("*.safetensors"):
        target = dest / source.name
        if not target.exists():
            os.link(str(source), str(target))
    return dest


def install(Trainer, Checkpointer, log=print):
    original_setup = Trainer.setup_trainer

    def setup_trainer(self):
        original_setup(self)
        state_path = Path(self.checkpointer.path) / "0" / "training_state.json"
        if not (self.resume_from_checkpoint and state_path.is_file()):
            return
        state = json.loads(state_path.read_text())
        epoch, skip, global_step = resume_point(state)
        self.current_epoch = epoch
        self._resume_local_step = skip
        self._resume_global_step = global_step
        self.global_step = global_step
        if _rank0(self):
            log("BEVERIN_MULTI_EPOCH_RESUME " + json.dumps({
                "saved": state, "epoch": epoch, "skip_local_steps": skip,
                "global_step": global_step,
            }))

    Trainer.setup_trainer = setup_trainer

    original_train_epoch = Trainer.train_epoch

    def train_epoch(self, epoch):
        # The ring's SIGTERM save reads the running epoch and its first global
        # step (to turn the global step into a step within the epoch).
        resumed = getattr(self, "_resume_local_step", 0) if epoch == self.current_epoch else 0
        self._beverin_epoch = epoch
        self._beverin_epoch_start_global = self.global_step - resumed
        return original_train_epoch(self, epoch)

    Trainer.train_epoch = train_epoch

    original_save = Trainer.maybe_save_checkpoint

    def maybe_save_checkpoint(self, epoch, local_step=0):
        if epoch == "interrupted":
            self.current_epoch = getattr(self, "_beverin_epoch", self.current_epoch)
        result = original_save(self, epoch, local_step=local_step)
        if isinstance(epoch, int) and int(local_step) == 0 and _rank0(self):
            dest = archive_epoch(self.checkpointer.path, epoch)
            log("BEVERIN_EPOCH_ARCHIVED " + json.dumps({"epoch": epoch, "path": str(dest)}))
        return result

    Trainer.maybe_save_checkpoint = maybe_save_checkpoint

    original_val = Trainer.val_epoch

    def val_epoch(self, epoch):
        metrics = original_val(self, epoch)
        if metrics is not None and _rank0(self):
            dest = Path(self.checkpointer.path) / "epochs" / ("epoch-%d" % epoch)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "val_metrics.json").write_text(json.dumps(metrics))
            log("BEVERIN_EPOCH_VALIDATION " + json.dumps({"epoch": epoch, "metrics": metrics}))
        return metrics

    Trainer.val_epoch = val_epoch

    # Per-epoch val metrics and checkpoint_best point at the ring head.
    Checkpointer.val_metrics_path = lambda self, epoch: Path(self.path) / "0" / "val_metrics.json"
    original_best_link = Checkpointer.update_best_symlink

    def update_best_symlink(self, epoch):
        return original_best_link(self, 0)

    Checkpointer.update_best_symlink = update_best_symlink
