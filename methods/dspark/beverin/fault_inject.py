"""Test-only numerical fault injection.

The hook refuses to run unless RUN_ID starts with ``faultinject-`` and the
Slurm job, when present, is a single node. A 32-node campaign cannot enable it
by exporting NODES=1.
"""
import json
import os


def node_count():
    slurm = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
    if slurm:
        return str(slurm)
    return str(os.environ.get("NODES", ""))


def injection_allowed():
    run_id = os.environ.get("RUN_ID", "")
    return run_id.startswith("faultinject-") and node_count() == "1"


def corrupt_checkpoint(directory):
    """Flip one payload byte so read-back must reject the staging directory."""
    path = os.path.join(str(directory), "model.safetensors")
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size < 1:
            raise RuntimeError("checkpoint file is too small to corrupt")
        handle.seek(max(0, size // 2))
        original = handle.read(1)
        handle.seek(size // 2)
        handle.write(bytes([original[0] ^ 0xFF]))


def install(Trainer, torch):
    if os.environ.get("BEVERIN_FAULT_INJECT", "0") != "1":
        return
    if not injection_allowed():
        raise RuntimeError(
            "BEVERIN_FAULT_INJECT refused: RUN_ID must start with faultinject- "
            "and the allocation must be exactly 1 node")
    step = int(os.environ.get("BEVERIN_FAULT_STEP", "12"))
    original = Trainer._optimizers_step

    def poisoned(self):
        if int(getattr(self, "global_step", -1)) == step:
            parameter = next(self.model.parameters())
            parameter.data.fill_(float("nan"))
            print("BEVERIN_FAULT_INJECTED " + json.dumps({
                "global_step": int(self.global_step), "run_id": os.environ.get("RUN_ID"),
            }), flush=True)
            return None
        return original(self)

    Trainer._optimizers_step = poisoned
