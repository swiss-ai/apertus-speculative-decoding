"""NVTX ranges on the vLLM engine step, for Nsight Systems profiles of a sweep arm.

Loaded inside the serving container by the opt-in overlay
serving/patches/vllm-apertus-nvtx.patch, which calls ``install()`` at the end
of ``vllm/v1/worker/gpu_worker.py`` when ``VLLM_APERTUS_NVTX=1``. With one GPU
per arm the worker runs in the EngineCore process (UniProcExecutor), so the
scheduler, the model runner (v2) and the speculators are all in reach of one
install. Methods are wrapped, not edited, so the overlay applies on top of the
scheduler patches that rewrite the same files.

Ranges (CPU side; ``nsys stats --report nvtx_gpu_proj_sum`` projects them onto
the GPU work they launched):

    engine:step                one EngineCore step (step / step_with_batch_queue)
    engine:schedule            Scheduler.schedule
    engine:update_from_output  Scheduler.update_from_output (output processing)
    target:execute[mixed]      target forward of a step with prefill tokens
    target:execute[decode]     target forward of a decode/verify-only step
    target:graph               the target's CUDA-graph replay (full or piecewise)
    target:sample              sampling, including rejection sampling
    target:rejection_sample    rejection sampling of the drafted tokens
    target:postprocess         postprocess_sampled
    draft:propose              the whole draft step
    draft:fc                   DSpark/DFlash combine of the target aux hidden states
    draft:context_kv           DSpark/DFlash context K/V precompute (eager, every layer)
    draft:graph                DSpark/DFlash query-block CUDA-graph replay
    draft:backbone             DSpark/DFlash query-block forward (eager draft only)
    draft:lm_head              DSpark draft-vocab LM head over all slots (eager only)
    draft:markov               DSpark Markov embed/bias, once per slot (eager only)
    draft:markov_head          DSpark LM head plus the sequential Markov loop (eager only)
    draft:pass0                EAGLE first pass (draft prefill), graph or eager
    draft:pass                 EAGLE later passes, one range per pass
    draft:sample               EAGLE per-pass draft sampling (eager only)

The whole DSpark draft step is one full CUDA graph: ranges pushed while a graph
is captured are not replayed, so the per-component ranges inside it appear only
with ``VLLM_APERTUS_NVTX_EAGER_DRAFT=1``, which runs the draft step eagerly
(the target keeps its graphs). Use that only for component ratios, never for
timing headlines.

The worker's own CUDA-profiler annotation (``execute_context_N(T)_generation_M(T)``,
from ``--profiler-config.profiler cuda``) adds the per-step batch composition.
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Callable
from typing import Any

_state = threading.local()
_installed = False

Namer = str | Callable[..., str]


def _depth() -> dict[str, int]:
    depth = getattr(_state, "depth", None)
    if depth is None:
        depth = _state.depth = {}
    return depth


def _nvtx() -> tuple[Callable[[str], Any], Callable[[], Any]]:
    import torch

    return torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop


def ranged(fn: Callable[..., Any], name: Namer, push=None, pop=None) -> Callable[..., Any]:
    """``fn`` inside an NVTX range. ``name`` is a string or a function of the call's
    arguments. A call nested in a range of the same name pushes no second range, so
    a subclass method calling its wrapped parent is counted once."""
    if push is None or pop is None:
        push, pop = _nvtx()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        label = name(*args, **kwargs) if callable(name) else name
        depth = _depth()
        if depth.get(label):
            return fn(*args, **kwargs)
        depth[label] = 1
        push(label)
        try:
            return fn(*args, **kwargs)
        finally:
            pop()
            depth[label] = 0

    wrapper.__apertus_nvtx__ = label_of(name)  # type: ignore[attr-defined]
    return wrapper


def label_of(name: Namer) -> str:
    return name if isinstance(name, str) else getattr(name, "__name__", "dynamic")


def wrap_method(owner: Any, attr: str, name: Namer, push=None, pop=None) -> bool:
    """Wrap ``owner.attr`` (a class or an instance) once; False if it is missing.

    On a class only an attribute defined by that class is wrapped: an inherited
    one is wrapped on the class that defines it."""
    fn = owner.__dict__.get(attr) if isinstance(owner, type) else getattr(owner, attr, None)
    if fn is None or getattr(fn, "__apertus_nvtx__", None) is not None:
        return False
    setattr(owner, attr, ranged(fn, name, push, pop))
    return True


def step_kind(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> str:
    """``target:execute[mixed]`` when a step schedules prefill tokens, else ``[decode]``.

    A decode request schedules 1 + its draft tokens; anything else is prefill."""
    if getattr(scheduler_output, "scheduled_new_reqs", None):
        return "target:execute[mixed]"
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    for req_id, count in (getattr(scheduler_output, "num_scheduled_tokens", None) or {}).items():
        if count > 1 + len(spec.get(req_id, ())):
            return "target:execute[mixed]"
    return "target:execute[decode]"


def _wrap_managers(runner: Any, attrs: dict[str, str]) -> None:
    for attr, name in attrs.items():
        manager = getattr(runner, attr, None)
        if manager is None:
            continue
        for method in ("run_fullgraph", "run_pw_graph"):
            wrap_method(manager, method, name)


def install() -> list[str]:
    """Wrap the engine step; return the wrapped ``Class.method`` names."""
    global _installed
    if _installed:
        return []
    _installed = True
    from vllm.config import CUDAGraphMode
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.engine import core as core_mod
    from vllm.v1.worker.gpu import model_runner as runner_mod
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_mod
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar_mod
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_mod
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_mod

    wrapped: list[str] = []

    def wrap(cls: type, attr: str, name: Namer) -> None:
        if wrap_method(cls, attr, name):
            wrapped.append(f"{cls.__name__}.{attr}")

    engine = core_mod.EngineCore
    wrap(engine, "step", "engine:step")
    wrap(engine, "step_with_batch_queue", "engine:step")
    schedulers = [sched_mod.Scheduler]
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        schedulers.append(AsyncScheduler)
    except ImportError:
        pass
    for cls in schedulers:
        wrap(cls, "schedule", "engine:schedule")
        wrap(cls, "update_from_output", "engine:update_from_output")

    runner = runner_mod.GPUModelRunner
    wrap(runner, "execute_model", step_kind)
    wrap(runner, "sample", "target:sample")
    wrap(runner, "postprocess_sampled", "target:postprocess")
    wrap(rejection_mod.RejectionSampler, "__call__", "target:rejection_sample")

    original_init_kv = runner.__dict__.get("initialize_kv_cache")
    if original_init_kv is not None:

        @functools.wraps(original_init_kv)
        def initialize_kv_cache(self: Any, *args: Any, **kwargs: Any) -> Any:
            # The CUDA-graph managers exist once the KV cache is initialised.
            result = original_init_kv(self, *args, **kwargs)
            _wrap_managers(self, {"cudagraph_manager": "target:graph"})
            speculator = getattr(self, "speculator", None)
            if speculator is not None:
                _wrap_managers(
                    speculator,
                    {
                        "query_cudagraph_manager": "draft:graph",
                        "prefill_cudagraph_manager": "draft:pass0",
                        "decode_cudagraph_manager": "draft:pass",
                    },
                )
            return result

        runner.initialize_kv_cache = initialize_kv_cache
        wrapped.append("GPUModelRunner.initialize_kv_cache")

    dflash = dflash_mod.DFlashSpeculator
    dspark = dspark_mod.DSparkSpeculator
    autoregressive = ar_mod.AutoRegressiveSpeculator
    for cls in (dflash, dspark, autoregressive):
        wrap(cls, "propose", "draft:propose")
    wrap(dflash, "_run_model", "draft:backbone")
    wrap(dspark, "_sample_sequential", "draft:markov_head")
    wrap(autoregressive, "_prefill", "draft:pass0")
    wrap(autoregressive, "_generate_draft", "draft:pass")
    wrap(autoregressive, "sample_draft", "draft:sample")

    def model_ranges(cls: type) -> None:
        original = cls.__dict__.get("load_draft_model")
        if original is None:
            return

        @functools.wraps(original)
        def load_draft_model(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = original(self, *args, **kwargs)
            for attr, name in (
                ("combine_hidden_states", "draft:fc"),
                ("precompute_and_store_context_kv", "draft:context_kv"),
                ("compute_draft_logits", "draft:lm_head"),
                ("markov_embed", "draft:markov"),
                ("markov_bias", "draft:markov"),
            ):
                wrap_method(model, attr, name)
            return model

        cls.load_draft_model = load_draft_model
        wrapped.append(f"{cls.__name__}.load_draft_model")

    model_ranges(dflash)
    model_ranges(dspark)

    if os.environ.get("VLLM_APERTUS_NVTX_EAGER_DRAFT") == "1":
        for cls in (dflash, autoregressive):
            original_init = cls.__dict__["init_cudagraph_manager"]

            def init_cudagraph_manager(self: Any, mode: Any, _original=original_init) -> Any:
                return _original(self, CUDAGraphMode.NONE)

            cls.init_cudagraph_manager = init_cudagraph_manager
            wrapped.append(f"{cls.__name__}.init_cudagraph_manager[eager]")

    import logging

    logging.getLogger("vllm").warning("apertus NVTX ranges installed on: %s", ", ".join(wrapped))
    return wrapped
