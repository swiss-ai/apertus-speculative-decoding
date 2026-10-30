"""Tiny two-rank ROCm/FSDP preflight; launch with torchrun, no target model."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from numerical_guard import install, NumericalFailure


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * float("nan")


class Model(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)
        self.mode = mode

    def forward(self, x):
        loss = self.fc(x).float().square().mean()
        if self.mode == "gradient" and dist.get_rank() == 1:
            loss = BadBackward.apply(loss)
        return None, loss, {"loss": loss.detach()}


class Trainer:
    def __init__(self, root, mode):
        self.rank, self.local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        self.is_distributed, self.global_step, self.steps = True, 0, 0
        self.mode, self.checkpointer = mode, SimpleNamespace(path=root)
        self.model = Model(mode).to(self.local_rank, dtype=torch.bfloat16)
        self.setup_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)

    def setup_model(self):
        for parameter in self.model.parameters():
            dist.broadcast(parameter.data, src=0)
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(self.model.fc, mp_policy=policy)
        fully_shard(self.model, mp_policy=policy)

    def _optimizers_step(self):
        self.optimizer.step()
        self.steps += 1

    def train_epoch(self, x):
        _, loss, _ = self.model(x=x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self._optimizers_step()


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.preferred_blas_library("cublas")
    dist.init_process_group("nccl")
    install(Trainer, torch)
    root = Path(os.environ["BEVERIN_PROBE_DIR"])
    for mode in ("finite", "input", "gradient"):
        folder = root / mode
        folder.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(folder, mode)
        x = torch.ones(4, 64, dtype=torch.bfloat16, device=trainer.local_rank)
        if mode == "input" and trainer.rank == 1:
            x[0, 0] = float("nan")
        failed = False
        try:
            trainer.train_epoch(x)
        except NumericalFailure:
            failed = True
        assert failed == (mode != "finite"), (mode, failed)
        assert trainer.steps == (1 if mode == "finite" else 0)
        dist.barrier()
        if trainer.rank == 0:
            print("BEVERIN_GUARD_PROBE_PASS " + json.dumps(
                {"mode": mode, "torch": torch.__version__, "hip": torch.version.hip}), flush=True)
        del trainer
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
