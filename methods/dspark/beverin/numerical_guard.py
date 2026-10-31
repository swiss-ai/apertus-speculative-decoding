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

    def optimizer_state(self):
        """Floating optimizer state, including CPU float ``step`` counters.

        Integer step counters cannot be NaN and are skipped. Float step counters
        stay on whatever device the optimizer stored them on.
        """
        named = []
        for index, optimizer in enumerate(getattr(self.trainer, "optimizers", None) or []):
            label = "%s%d" % (type(optimizer).__name__, index)
            state = getattr(optimizer, "state", {}) or {}
            for param_index, slot in enumerate(state.values()):
                if not isinstance(slot, dict):
                    continue
                for key, value in slot.items():
                    if not self.torch.is_tensor(value) or not value.is_floating_point():
                        continue
                    named.append(("%s/%d/%s" % (label, param_index, key), value))
        return named

    def load_continuity(self):
        from rollback import load_ledger
        pending = load_ledger(str(self.root)).get("pending_continuity")
        self._pending = pending if isinstance(pending, dict) else None
        self._losses = []
        self._continuity_started = False

    def observe_continuity(self, output):
        pending = getattr(self, "_pending", None)
        if not pending:
            return
        if not self._continuity_started:
            self._continuity_started = True
            self._check_resume_position(pending)
        value = self._loss_scalar(output)
        if value is None:
            return
        self._losses.append(value)
        if len(self._losses) >= int(pending.get("window") or 20):
            self._check_loss_window(pending)
            self._clear_continuity()

    def _check_resume_position(self, pending):
        expected = int(pending["expected_global_step"])
        if int(self.trainer.global_step) != expected:
            self._fail_continuity("continuity-step")
        expected_rates = [float(value) for value in pending.get("learning_rates") or []]
        if not expected_rates:
            return
        actual = []
        for optimizer in getattr(self.trainer, "optimizers", None) or []:
            for group in getattr(optimizer, "param_groups", None) or []:
                if "lr" in group:
                    actual.append(float(group["lr"]))
        if len(actual) != len(expected_rates) or any(
                abs(got - want) > 1e-5 + 1e-4 * abs(want)
                for got, want in zip(actual, expected_rates)):
            self._fail_continuity("continuity-lr")

    def _check_loss_window(self, pending):
        pre = pending.get("pre_median_loss")
        if pre is None:
            return
        ordered = sorted(self._losses)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        if median > float(pending.get("loss_ratio") or 1.3) * float(pre):
            self._fail_continuity("continuity-loss")

    def _fail_continuity(self, phase):
        bad = self.torch.tensor(float("nan"), device=self.device)
        self.check([(phase, bad)], phase)

    def _clear_continuity(self):
        from rollback import load_ledger, save_ledger
        if self.trainer.rank == 0:
            ledger = load_ledger(str(self.root))
            ledger["pending_continuity"] = None
            save_ledger(str(self.root), ledger)
        self._pending = None

    def _loss_scalar(self, output):
        torch = self.torch
        if not (isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[1])):
            return None
        local = float(output[1].detach().float().item())
        value = torch.tensor([local], dtype=torch.float32, device=self.device)
        if self.trainer.is_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value = value / torch.distributed.get_world_size()
        return float(value.item())

    def check(self, values, phase):
        torch, trainer = self.torch, self.trainer
        local_values = []
        flags = []
        for name, value in values:
            if not hasattr(value, "is_floating_point") or not value.is_floating_point():
                continue
            local = value.detach()
            if hasattr(local, "to_local"):
                local = local.to_local()
            local_values.append((name, local))
            flags.append(torch.isfinite(local).all().to(device=self.device))
        values = local_values
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
        record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "global_step": trainer.global_step, "phase": phase,
                  "rank": failing_rank, "job_id": os.environ.get("SLURM_JOB_ID"),
                  "attempt": os.environ.get("TORCHELASTIC_RUN_ID")}
        if rank == failing_rank:
            bad = []
            for name, value in values:
                if not torch.isfinite(value).all().item():
                    bad.append({"name": name, "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "nonfinite": int((~torch.isfinite(value)).sum().item())})
            record["bad_tensors"] = bad
            self.root.mkdir(parents=True, exist_ok=True)
            # Publish the terminal marker before any potentially large batch dump.
            temporary = self.root / (MARKER + ".tmp-" + str(os.getpid()))
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(self.root / MARKER)
            print("BEVERIN_NUMERICAL_FAILURE " + json.dumps(record), flush=True)
        # FSDP can spread one rank's bad gradient before this check observes it.
        # Retain every rank's batch rather than assuming the first observed rank
        # originated the failure. Keep peers alive until all dumps finish.
        if self.batch and os.environ.get("BEVERIN_CAPTURE_BAD_BATCH", "1") == "1":
            try:
                folder = self.root / "numerical-failure-batches"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save({"batch": {k: v.detach().cpu() if isinstance(v, torch.Tensor)
                                       else v for k, v in self.batch.items()},
                            "rng_before_forward": self.rng, "capture_rank": rank,
                            "failure": record}, folder / ("rank%s.pt" % rank))
            except Exception as exc:
                print("BEVERIN_NUMERICAL_FAILURE batch capture failed: " + str(exc), flush=True)
        if trainer.is_distributed:
            torch.distributed.barrier()
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
        guard.load_continuity()

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
            guard.observe_continuity(output)

        self.model.register_forward_pre_hook(before, with_kwargs=True)
        self.model.register_forward_hook(after)
        if os.environ.get("BEVERIN_CHECK_ACTIVATIONS", "0") == "1":
            stages = {"fc", "hidden_norm", "norm", "lm_head", "verifier_norm",
                      "verifier_lm_head", "markov_head.markov_w1",
                      "markov_head.markov_w2", "confidence_head"}
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
            norm = result.detach() if torch.is_tensor(result) else torch.tensor(float(result))
            try:
                shown = float(norm.detach().float().cpu()) if torch.is_tensor(norm) else float(norm)
            except (TypeError, ValueError):
                shown = float("nan")
            if not (shown == shown) or shown in (float("inf"), float("-inf")):
                shown_json = "nan" if shown != shown else ("-inf" if shown < 0 else "inf")
            else:
                shown_json = shown
            if self.rank == 0:
                print("BEVERIN_GRAD_NORM " + json.dumps({
                    "global_step": int(self.global_step), "grad_norm": shown_json,
                }), flush=True)
            guard.check([("total_grad_norm", norm)], "gradient-norm")
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

    if hasattr(Trainer, "setup_optimizer"):
        original_setup_optimizer = Trainer.setup_optimizer

        def setup_optimizer(self):
            result = original_setup_optimizer(self)
            guard = getattr(self, "_beverin_numerical_guard", None)
            if guard is not None:
                state = guard.optimizer_state()
                if state:
                    guard.check(state, "loaded-optimizer")
            if os.environ.get("BEVERIN_CHECK_ACTIVATIONS", "0") == "1":
                for optimizer in self.optimizers:
                    def optimizer_hook(opt, _args, _kwargs):
                        self._beverin_numerical_guard.check(
                            self._beverin_numerical_guard.parameters(),
                            "optimizer:" + type(opt).__name__)
                    optimizer.register_step_post_hook(optimizer_hook)
            return result

        Trainer.setup_optimizer = setup_optimizer
