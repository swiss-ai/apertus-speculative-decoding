"""Opt-in final-validation recovery and completion guard for a one-epoch run.

The stdlib helpers also run in the batch shell's Python 3.6. Training hooks
are installed only by the Beverin prelude when BEVERIN_SINGLE_EPOCH=1.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


MARKER = "single-epoch-complete.json"
ARTIFACTS = ("model.safetensors", "optimizer_state_dict.pt",
             "scheduler_state_dict.pt", "config.json", "training_state.json",
             "val_metrics.json")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def valid_metrics(metrics):
    return (isinstance(metrics, dict) and "loss_epoch" in metrics
            and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in metrics.values()))


def validated_artifacts(checkpoints, expected_steps):
    checkpoint = checkpoints / "0"
    if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size
               for name in ARTIFACTS):
        return None
    state = json.loads((checkpoint / "training_state.json").read_text())
    if state != {"epoch": 0, "local_step": 0, "global_step": expected_steps}:
        return None
    try:
        metrics = json.loads((checkpoint / "val_metrics.json").read_text())
    except (ValueError, OSError):
        # save_val_metrics itself is not atomic. A torn candidate means that
        # validation must run again; an existing marker still fails closed.
        return None
    if not valid_metrics(metrics):
        return None
    return {
        "num_epochs": 1, "global_step": expected_steps, "checkpoint": "0",
        "validation": metrics,
        "files": {name: {
            "size": (checkpoint / name).stat().st_size,
            "sha256": hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            if name not in ("model.safetensors", "optimizer_state_dict.pt") else None,
        } for name in ARTIFACTS},
    }


def read_completion(checkpoints, expected_steps):
    marker = checkpoints / MARKER
    if not marker.exists():
        return None
    recorded = json.loads(marker.read_text())
    current = validated_artifacts(checkpoints, expected_steps)
    if current is None or any(recorded.get(key) != current[key]
                              for key in ("num_epochs", "global_step", "checkpoint",
                                          "validation", "files")):
        raise ValueError("Completion marker does not match final validated artifacts")
    return recorded


def write_completion(checkpoints, expected_steps):
    existing = read_completion(checkpoints, expected_steps)
    if existing:
        return existing
    record = validated_artifacts(checkpoints, expected_steps)
    if record is None:
        raise ValueError("Cannot complete before final checkpoint and finite validation")
    best = checkpoints / "checkpoint_best"
    if not best.exists() and not best.is_symlink():
        best.symlink_to("0", target_is_directory=True)
    if not best.is_symlink() or best.resolve() != (checkpoints / "0").resolve():
        raise ValueError("Final checkpoint_best does not point to epoch 0")
    record.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "job_id": os.environ.get("SLURM_JOB_ID"),
                   "trainer_attempt": os.environ.get("TORCHELASTIC_RUN_ID")})
    atomic_json(checkpoints / MARKER, record)
    return record


def install(Trainer, barrier, log):
    original_run = Trainer.run_training
    original_train = Trainer.train_epoch
    original_save = Trainer.maybe_save_checkpoint
    original_val = Trainer.val_epoch
    expected = int(os.environ["BEVERIN_EXPECTED_EPOCH_STEPS"])

    def train(self, epoch):
        if getattr(self, "_beverin_validation_only", False):
            log.info("BEVERIN_VALIDATION_ONLY: preserving completed training at step %s",
                     self.global_step)
            return
        return original_train(self, epoch)

    def save(self, epoch, local_step=0):
        if getattr(self, "_beverin_validation_only", False) and epoch == 0:
            return
        return original_save(self, epoch, local_step=local_step)

    def val(self, epoch):
        result = original_val(self, epoch)
        if not valid_metrics(result):
            raise ValueError("Final validation did not return finite loss and metrics")
        self._beverin_validation_completed = True
        return result

    def run(self):
        if (self.config.num_epochs != 1 or self.config.max_steps is not None
                or self.val_loader is None or len(self.train_loader) != expected
                or int(os.environ.get("BEVERIN_SEGMENT_STEPS", "0"))
                or os.environ.get("BEVERIN_PILOT_RESUME", "0") == "1"
                or int(os.environ.get("BEVERIN_PROFILE_STEPS", "0"))):
            raise ValueError("One-epoch guard requires EPOCHS=1, full loader, validation and no MAX_STEPS")
        checkpoints = self.checkpointer.path
        self._beverin_validation_completed = False
        if validated_artifacts(checkpoints, expected) is not None:
            # A process may exit after writing validation metrics but before the
            # marker. Recover those completed results without running them again.
            barrier(self)
            if self.rank == 0:
                write_completion(checkpoints, expected)
            barrier(self)
            log.info("BEVERIN_VALIDATION_ONLY: already validated; completion confirmed")
            return
        state = self._load_training_state() if self.checkpointer.previous_epoch == 0 else {}
        if state == {"epoch": 0, "local_step": 0, "global_step": expected}:
            self._beverin_validation_only = True
            self.current_epoch = 0
            log.info("BEVERIN_VALIDATION_ONLY: resuming final validation at step %s", expected)
        result = original_run(self)
        # The graceful-shutdown decorator can swallow TrainingInterruptedError.
        # A normal return alone therefore does not prove validation completed.
        if not self._beverin_validation_completed:
            log.info("BEVERIN_SINGLE_EPOCH_INCOMPLETE: final validation has not completed")
            return result
        barrier(self)
        if self.rank == 0:
            write_completion(checkpoints, expected)
        barrier(self)
        log.info("BEVERIN_SINGLE_EPOCH_COMPLETE: trained and validated %s steps", expected)
        return result

    Trainer.train_epoch = train
    Trainer.maybe_save_checkpoint = save
    Trainer.val_epoch = val
    Trainer.run_training = run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()
    try:
        completed = read_completion(args.checkpoints, args.expected_steps)
        if completed is None and validated_artifacts(args.checkpoints, args.expected_steps):
            completed = write_completion(args.checkpoints, args.expected_steps)
        if completed:
            print("BEVERIN_COMPLETED_GUARD " + json.dumps(completed))
        raise SystemExit(0 if completed else 1)
    except (ValueError, OSError) as exc:
        print("BEVERIN_COMPLETION_ERROR " + str(exc))
        raise SystemExit(2)
