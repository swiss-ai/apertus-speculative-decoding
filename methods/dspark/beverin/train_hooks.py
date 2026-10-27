"""Opt-in Beverin save/resume and CPU collective profiling hooks.

Loaded only by the Beverin prelude for speculators.train. The step limit
uses the existing deadline save path, keeping the ten-epoch LR schedule.
"""
import gc
import json
import os
import time
from pathlib import Path


def install():
    import torch
    import speculators.train.distributed as distributed
    from speculators.train.trainer import Trainer

    blas_backend = os.environ.get("BEVERIN_BLAS_BACKEND")
    if blas_backend:
        # PyTorch keeps the CUDA API names on ROCm: cublas selects rocBLAS,
        # whereas cublaslt selects hipBLASLt. The latter is very slow for
        # gathered DTensor inputs in the v0.31 image on Beverin.
        torch.backends.cuda.preferred_blas_library(blas_backend)

    reduce_dtype = os.environ.get("BEVERIN_REDUCE_DTYPE", "fp32")
    if reduce_dtype not in ("fp32", "bf16"):
        raise ValueError(f"Invalid BEVERIN_REDUCE_DTYPE: {reduce_dtype}")
    if reduce_dtype == "bf16":
        original_policy = distributed.MixedPrecisionPolicy

        def policy(*args, **kwargs):
            kwargs["reduce_dtype"] = torch.bfloat16
            return original_policy(*args, **kwargs)

        distributed.MixedPrecisionPolicy = policy

    segment_steps = int(os.environ.get("BEVERIN_SEGMENT_STEPS", "0"))
    if segment_steps:
        original_deadline = Trainer._deadline_reached

        def deadline(self):
            if not hasattr(self, "_beverin_stop_step"):
                # First call follows the first completed step.
                self._beverin_stop_step = self.global_step - 1 + segment_steps
            return self.global_step >= self._beverin_stop_step or original_deadline(self)

        Trainer._deadline_reached = deadline

    if os.environ.get("BEVERIN_EMPTY_CACHE_ON_SAVE", "0") == "1":
        original_save = Trainer.maybe_save_checkpoint

        def save(self, *args, **kwargs):
            print("BEVERIN_SAVE_START " + json.dumps({
                "unix_s": time.time(), "global_step": self.global_step,
            }), flush=True)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            result = original_save(self, *args, **kwargs)
            print("BEVERIN_SAVE_END " + json.dumps({
                "unix_s": time.time(), "global_step": self.global_step,
            }), flush=True)
            return result

        Trainer.maybe_save_checkpoint = save

    profile_steps = int(os.environ.get("BEVERIN_PROFILE_STEPS", "0"))
    if profile_steps:
        original_step = Trainer._optimizers_step
        profile_dir = Path(os.environ["BEVERIN_PROFILE_DIR"])
        profile_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        profiler = None
        completed = 0

        def step(self):
            nonlocal profiler, completed
            result = original_step(self)
            if completed == 0:
                # Skip the first step's compilation. CPU traces capture c10d
                # launches and waits without retaining GPU tensors or stacks.
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                )
                profiler.start()
            elif completed <= profile_steps:
                profiler.step()
                if completed == profile_steps:
                    profiler.stop()
                    profiler.export_chrome_trace(str(profile_dir / f"rank{rank}.json"))
                    (profile_dir / f"rank{rank}.txt").write_text(
                        profiler.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=80
                        )
                    )
                    profiler = None
            completed += 1
            return result

        Trainer._optimizers_step = step

    if os.environ.get("BEVERIN_SINGLE_EPOCH", "0") == "1":
        import torch.distributed as dist
        from speculators.train.trainer import root_logger
        from single_epoch import install as install_single_epoch

        def barrier(trainer):
            if trainer.is_distributed:
                dist.barrier()

        install_single_epoch(Trainer, barrier, root_logger)

    print("BEVERIN_HOOKS " + json.dumps({
        "reduce_dtype": reduce_dtype,
        "segment_steps": segment_steps,
        "empty_cache_on_save": os.environ.get("BEVERIN_EMPTY_CACHE_ON_SAVE", "0"),
        "profile_steps": profile_steps,
        "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
        "single_epoch": os.environ.get("BEVERIN_SINGLE_EPOCH", "0"),
    }), flush=True)
