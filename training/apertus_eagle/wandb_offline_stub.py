"""Minimal wandb stand-in so TorchSpec can import without wandb deps.

Copied to scratch/pydeps/wandb/__init__.py on the allocated node. Not a
logging backend; WANDB_MODE=offline / WANDB_DISABLED=true for overfit.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

__version__ = "0.0.0-stub"


class _Run:
    id = "offline-stub"
    dir = "."

    def log(self, *args, **kwargs) -> None:
        del args, kwargs

    def finish(self, *args, **kwargs) -> None:
        del args, kwargs

    def __getattr__(self, name: str):
        del name
        return lambda *args, **kwargs: None


run = _Run()


def login(*args, **kwargs) -> bool:
    del args, kwargs
    return True


def init(*args, **kwargs):
    del args, kwargs
    global run
    run = _Run()
    return run


def log(*args, **kwargs) -> None:
    del args, kwargs


def finish(*args, **kwargs) -> None:
    del args, kwargs


def setup(*args, **kwargs) -> None:
    del args, kwargs


class Settings:
    def __init__(self, *args, **kwargs) -> None:
        del args
        self.__dict__.update(kwargs)


def _submodule(name: str) -> types.ModuleType:
    """Register a child module with a spec.

    Accelerate calls ``importlib.util.find_spec``. A child left in
    ``sys.modules`` with ``__spec__ is None`` raises ValueError. Job 3491232
    hit that on the top-level wandb module; the same trap applies here.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__path__ = []
    sys.modules[name] = module
    return module


util = _submodule("wandb.util")
util.generate_id = lambda: "000000"
env = _submodule("wandb.env")
sdk = _submodule("wandb.sdk")
sdk.lib = _submodule("wandb.sdk.lib")
