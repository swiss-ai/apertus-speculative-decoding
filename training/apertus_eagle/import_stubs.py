"""Import stand-ins for packages the serving image does not ship.

TorchSpec's draft-class import reaches Accelerate, which calls
``importlib.util.find_spec``. A module sitting in ``sys.modules`` with
``__spec__ is None`` makes that call raise ``ValueError`` instead of
returning a spec. Job 3491232 died that way on ``wandb`` before the 70B
load. Every synthetic module installed here carries a ``ModuleSpec``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any


def bind_module_spec(module: types.ModuleType, *, is_package: bool) -> None:
    """Give ``module`` a spec so ``importlib.util.find_spec`` does not raise."""
    if getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec(
            module.__name__, loader=None, is_package=is_package
        )
    if is_package and getattr(module, "__path__", None) is None:
        module.__path__ = []


def _bind_tree(prefix: str) -> None:
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == prefix or name.startswith(prefix + "."):
            bind_module_spec(module, is_package=True)


def _ensure_wandb_stub() -> None:
    """Keep a spec-bearing wandb. Do not replace the on-disk pydeps stub.

    ``train-eagle-on-node.sh`` copies ``wandb_offline_stub.py`` onto
    ``$EAGLE_PYDEPS/wandb/__init__.py`` and prepends that directory to
    ``PYTHONPATH``. Importing that package yields a real spec. Replacing it
    with a bare ``ModuleType`` is what made Accelerate raise.
    """
    existing = sys.modules.get("wandb")
    if existing is not None and getattr(existing, "__spec__", None) is not None:
        _bind_tree("wandb")
        return
    try:
        import wandb
    except ModuleNotFoundError:
        wandb = None
    if wandb is not None and getattr(wandb, "__spec__", None) is not None:
        _bind_tree("wandb")
        return

    stub = types.ModuleType("wandb")
    stub.__version__ = "0.0.0-stub"
    stub.login = lambda **kwargs: None
    stub.init = lambda **kwargs: None
    stub.log = lambda *args, **kwargs: None
    stub.finish = lambda *args, **kwargs: None
    stub.run = None

    class _Util:
        @staticmethod
        def generate_id() -> str:
            return "offline"

    stub.util = _Util()
    bind_module_spec(stub, is_package=True)
    sys.modules["wandb"] = stub


def _ensure_ray_stub() -> None:
    """One-node overfit never starts a Ray actor.

    Job 3446638 died after the 70B load when Mooncake ``utils.py`` imported
    Ray. The Mooncake package import is lazy; this stub covers a later
    ``import ray`` and keeps ``find_spec('ray')`` from raising.
    """
    try:
        import ray  # noqa: F401
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        _bind_tree("ray")
        return

    ray = types.ModuleType("ray")
    util = types.ModuleType("ray.util")
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    private = types.ModuleType("ray._private")
    services = types.ModuleType("ray._private.services")

    class NodeAffinitySchedulingStrategy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    scheduling.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
    services.get_node_ip_address = lambda: "127.0.0.1"

    def remote(*args: Any, **kwargs: Any):
        del kwargs

        def deco(fn):
            return fn

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco

    ray.remote = remote
    ray.nodes = lambda: []
    ray.get_gpu_ids = lambda: []
    ray.util = util
    ray._private = private
    util.scheduling_strategies = scheduling
    private.services = services
    for module, is_package in (
        (ray, True),
        (util, True),
        (scheduling, False),
        (private, True),
        (services, False),
    ):
        bind_module_spec(module, is_package=is_package)
        sys.modules[module.__name__] = module


def _ensure_datasets_stub() -> None:
    """``train_config`` imports ``data.utils``, which imports datasets."""
    try:
        from datasets import Dataset, IterableDataset, load_dataset  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        bind_module_spec(sys.modules["datasets"], is_package=True)
        return

    datasets = types.ModuleType("datasets")

    class IterableDataset:
        pass

    class Dataset:
        pass

    def load_dataset(*args: Any, **kwargs: Any):
        del args, kwargs
        raise RuntimeError("datasets stub is import-only; overfit_direct does not load HF datasets")

    datasets.IterableDataset = IterableDataset
    datasets.Dataset = Dataset
    datasets.load_dataset = load_dataset
    bind_module_spec(datasets, is_package=True)
    sys.modules["datasets"] = datasets


def _ensure_pydantic_stub() -> None:
    """``data.template`` subclasses ``pydantic.BaseModel`` at import."""
    try:
        from pydantic import BaseModel  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        bind_module_spec(sys.modules["pydantic"], is_package=True)
        return

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    pydantic.BaseModel = BaseModel
    bind_module_spec(pydantic, is_package=True)
    sys.modules["pydantic"] = pydantic


def _ensure_numba_stub() -> None:
    """``loss_mask.py`` decorates with ``@numba.njit`` at import.

    The draft-class import reaches that module after Mooncake is lazy.
    A missing numba would fail the job in the same import, before training.
    """
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        bind_module_spec(sys.modules["numba"], is_package=True)
        return

    numba = types.ModuleType("numba")

    def njit(*args: Any, **kwargs: Any):
        del kwargs

        def deco(fn):
            return fn

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco

    numba.njit = njit
    bind_module_spec(numba, is_package=True)
    sys.modules["numba"] = numba


def ensure_import_stubs() -> None:
    _ensure_wandb_stub()
    _ensure_ray_stub()
    _ensure_datasets_stub()
    _ensure_pydantic_stub()
    _ensure_numba_stub()
