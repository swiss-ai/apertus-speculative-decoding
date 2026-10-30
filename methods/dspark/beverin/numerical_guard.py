"""Fail together on non-finite values, before they can overwrite a checkpoint.

Installed only by the Beverin prelude. Full activation checks are optional;
inputs, loss, gradients, update results and loaded parameters are always checked.
No invalid batch is skipped and no NaN is replaced with an arbitrary value.
"""
import json
import os
from pathlib import Path
import time


MARKER = "numerical-failure.json"


class NumericalFailure(RuntimeError):
    pass


def tensors(value, prefix=""):
    """Flatten model inputs/results without importing torch in shell preflights."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for key, item in enumerate(value):
            yield from tensors(item, prefix + "/" + str(key))
    elif hasattr(value, "is_floating_point") and value.is_floating_point():
        yield prefix, value


class Guard:
    def __init__(self, trainer, torch):
        self.trainer, self.torch = trainer, torch
        self.root = Path(trainer.checkpointer.path)
        self.batch = {}
        self.rng = {}
        self.device = next(trainer.model.parameters()).device

    def check(self, values, phase):
        torch, trainer = self.torch, self.trainer
        values = [(name, value.detach().to_local() if hasattr(value, "to_local")
                   else value.detach()) for name, value in values]
        flags = [torch.isfinite(value).all() for _, value in values]
        ok = torch.stack(flags).all() if flags else torch.ones(
            (), dtype=torch.bool, device=self.device)
        rank = trainer.rank
        first_bad = torch.where(ok, torch.tensor(2**30, device=self.device),
                                torch.tensor(rank, device=self.device))
        if trainer.is_distributed:
            torch.distributed.all_reduce(first_bad, op=torch.distributed.ReduceOp.MIN)
        failing_rank = int(first_bad.item())
        if failing_rank == 2**30:
            return
        if rank == failing_rank:
            bad = []
            for name, value in values:
                if not torch.isfinite(value).all().item():
                    bad.append({"name": name, "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "nonfinite": int((~torch.isfinite(value)).sum().item())})
            record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "global_step": trainer.global_step, "phase": phase,
                      "rank": rank, "job_id": os.environ.get("SLURM_JOB_ID"),
                      "attempt": os.environ.get("TORCHELASTIC_RUN_ID"),
                      "bad_tensors": bad}
            self.root.mkdir(parents=True, exist_ok=True)
            # Publish the terminal marker before any potentially large batch dump.
            temporary = self.root / (MARKER + ".tmp-" + str(os.getpid()))
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(self.root / MARKER)
            print("BEVERIN_NUMERICAL_FAILURE " + json.dumps(record), flush=True)
            if self.batch and os.environ.get("BEVERIN_CAPTURE_BAD_BATCH", "1") == "1":
                try:
                    torch.save({"batch": {k: v.detach().cpu() if isinstance(v, torch.Tensor)
                                           else v for k, v in self.batch.items()},
                                "rng_before_forward": self.rng,
                                "failure": record}, self.root / "numerical-failure-batch.pt")
                except Exception as exc:
                    print("BEVERIN_NUMERICAL_FAILURE batch capture failed: " + str(exc), flush=True)
        raise NumericalFailure("BEVERIN_NUMERICAL_FAILURE phase=%s rank=%s step=%s; "
                               "automatic retry prohibited" %
                               (phase, failing_rank, trainer.global_step))

    def parameters(self):
        return [(name, param) for name, param in self.trainer.model.named_parameters()
                if param.requires_grad]


def install(Trainer, torch):
    original_setup = Trainer.setup_model
    original_step = Trainer._optimizers_step
    original_train = Trainer.train_epoch

    def setup(self):
        if (Path(self.checkpointer.path) / MARKER).exists():
            raise NumericalFailure("BEVERIN_NUMERICAL_FAILURE existing terminal marker; "
                                   "use a new run after diagnosing the failure")
        result = original_setup(self)
        guard = self._beverin_numerical_guard = Guard(self, torch)
        guard.check(guard.parameters(), "loaded-parameters")

        @torch.compiler.disable
        def before(_module, _args, kwargs):
            guard.batch = kwargs
            guard.rng = {"cpu": torch.get_rng_state(),
                         "accelerator": torch.cuda.get_rng_state()
                         if torch.cuda.is_available() else None}
            guard.check(list(tensors(kwargs)), "inputs")

        @torch.compiler.disable
        def after(_module, _args, output):
            guard.check(list(tensors(output)), "forward-loss-and-metrics")

        self.model.register_forward_pre_hook(before, with_kwargs=True)
        self.model.register_forward_hook(after)
        if os.environ.get("BEVERIN_CHECK_ACTIVATIONS", "0") == "1":
            stages = {"fc", "hidden_norm", "norm", "lm_head", "verifier_norm",
                      "verifier_lm_head", "markov_head.markov_w2", "confidence_head"}
            for name, module in self.model.named_modules():
                if name in stages or (name.startswith("layers.") and name.count(".") == 1):
                    def stage_hook(_module, _args, output, name=name):
                        guard.check(list(tensors(output)), "activation:" + name)
                    module.register_forward_hook(torch.compiler.disable(stage_hook))
        return result

    def train(self, *args, **kwargs):
        # The upstream trainer clips between backward and _optimizers_step.
        # Check before clipping as well as its returned norm: a non-finite norm
        # must never silently multiply every finite gradient by zero/NaN.
        original_clip = torch.nn.utils.clip_grad_norm_
        guard = self._beverin_numerical_guard

        def clip(parameters, *clip_args, **clip_kwargs):
            parameters = list(parameters)
            guard.check([(name, param.grad) for name, param in self.model.named_parameters()
                         if param.grad is not None], "backward-gradients")
            result = original_clip(parameters, *clip_args, **clip_kwargs)
            guard.check([("total_grad_norm", result)], "gradient-norm")
            return result

        torch.nn.utils.clip_grad_norm_ = clip
        try:
            return original_train(self, *args, **kwargs)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            guard.batch = {}

    def step(self):
        result = original_step(self)
        self._beverin_numerical_guard.check(
            self._beverin_numerical_guard.parameters(), "updated-parameters")
        return result

    Trainer.setup_model = setup
    Trainer.train_epoch = train
    Trainer._optimizers_step = step
