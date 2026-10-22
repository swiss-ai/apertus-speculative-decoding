"""Swap the drafter's weights inside a running vLLM server (T5, drafter updates).

The fork's ``update_weights`` / ``reload_weights`` reload the TARGET. This is a
vLLM worker extension that reloads only the draft model, in place:

    vllm serve ... --speculative-config '{"method": "eagle3", "model": HEAD_A, ...}' \
      --worker-extension-cls apertus_bench.drafter_reload.DrafterReloadExtension
    # with VLLM_SERVER_DEV_MODE=1, from any client:
    curl -X POST localhost:PORT/collective_rpc \
      -d '{"method": "reload_drafter", "args": ["HEAD_B"]}'

Every worker reads ``HEAD_B``'s safetensors and hands them to the draft model's
own ``load_weights`` (the same name mapping and tensor-parallel sharding as at
startup), which copies into the existing parameters. Storage addresses do not
change, so captured CUDA graphs and compiled code stay valid, and no KV cache
or request state is touched. The engine core runs the RPC between two steps:
requests in flight pause for the copy and continue with the new drafter.

Tensors the drafter shares with the target (embedding, lm_head when the head
carries a frozen copy or none) are never written: a shared tensor in the new
checkpoint must equal the target's, or the reload is refused before anything
is copied. The new head must have the old head's architecture
(``config.json`` keys in ``ARCHITECTURE_KEYS``); anything else needs a restart.

``drafter_info`` returns digests of the drafter's own and the target's
parameters, to show that a swap changed the former and not the latter.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

# A head with a different value for any of these cannot be loaded in place.
ARCHITECTURE_KEYS = (
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "draft_vocab_size",
    "eagle_config",
    "fc_norm",
    "norm_output",
    "norm_before_fc",
    "target_hidden_size",
)
SHARED_NAMES = ("embed_tokens", "lm_head")


def architecture_mismatch(old: dict[str, Any], new: dict[str, Any]) -> dict[str, list[Any]]:
    """Config keys whose values differ between the served head and the new one."""
    return {
        key: [old.get(key), new.get(key)]
        for key in ARCHITECTURE_KEYS
        if old.get(key) != new.get(key)
    }


def is_shared_name(name: str) -> bool:
    return any(part in name for part in SHARED_NAMES)


def checkpoint_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {path}")
    return files


class DrafterReloadExtension:
    """Mixed into the GPU worker by ``--worker-extension-cls``; methods run on every worker."""

    _drafter_path: str | None = None

    def _drafter_model(self):
        runner = self.model_runner  # type: ignore[attr-defined]
        speculator = getattr(runner, "speculator", None)  # model runner v2
        model = getattr(speculator, "model", None)
        if model is None:
            model = getattr(getattr(runner, "drafter", None), "model", None)  # v1
        if model is None:
            raise RuntimeError("this worker has no draft model")
        return model

    def _target_model(self):
        return self.model_runner.get_model()  # type: ignore[attr-defined]

    @staticmethod
    def _digest(named) -> str:
        import torch

        digest = hashlib.sha256()
        for name, tensor in sorted(named, key=lambda item: item[0]):
            digest.update(name.encode())
            data = tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy()
            digest.update(data.tobytes())
        return digest.hexdigest()

    def _own_and_shared(self) -> tuple[dict[str, Any], dict[str, Any]]:
        target_ptrs = {p.data_ptr() for p in self._target_model().parameters()}
        own, shared = {}, {}
        for name, param in self._drafter_model().named_parameters():
            (shared if param.data_ptr() in target_ptrs else own)[name] = param
        return own, shared

    def drafter_info(self) -> dict[str, Any]:
        own, shared = self._own_and_shared()
        target = self._target_model()
        # The full target digest reads 16 GB; its lm_head and one decoder layer suffice
        # to show the target was not written (shared tensors live there).
        sample = [
            (name, p)
            for name, p in target.named_parameters()
            if "lm_head" in name or "embed_tokens" in name or ".layers.0." in name
        ]
        return {
            "path": self._drafter_path or self.vllm_config.speculative_config.model,  # type: ignore[attr-defined]
            "own_parameters": sorted(own),
            "shared_with_target": sorted(shared),
            "own_sha256": self._digest(own.items()),
            "target_sample_sha256": self._digest(sample),
            "target_sample": len(sample),
        }

    def reload_drafter(self, path: str, config_check: str = "1") -> dict[str, Any]:
        import torch
        from safetensors import safe_open

        started = time.perf_counter()
        head = Path(path)
        model = self._drafter_model()
        if config_check == "1":
            # Compare config files, as both heads were exported alike; vLLM rewrites
            # parts of the config it serves (architectures, draft_vocab_size).
            served = Path(self._drafter_path or self.vllm_config.speculative_config.model)  # type: ignore[attr-defined]
            mismatch = architecture_mismatch(
                json.loads((served / "config.json").read_text()),
                json.loads((head / "config.json").read_text()),
            )
            if mismatch:
                raise ValueError(f"{head}: architecture differs from {served}: {mismatch}")
        own, shared = self._own_and_shared()
        shared_by_suffix = {name.rsplit(".", 2)[-2]: p for name, p in shared.items()}
        weights: list[tuple[str, Any]] = []
        checked_shared = []
        for file in checkpoint_files(head):
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118 - safe_open has no __iter__
                    if is_shared_name(name):
                        key = next((k for k in SHARED_NAMES if k in name), None)
                        param = shared_by_suffix.get(key)
                        if param is not None:
                            tensor = handle.get_tensor(name)
                            # vLLM may pad the vocabulary rows; compare the real rows.
                            rows = tensor.shape[0]
                            same = tuple(param.shape[1:]) == tuple(tensor.shape[1:]) and bool(
                                torch.equal(param[:rows].cpu(), tensor.to(param.dtype))
                            )
                            if not same:
                                raise ValueError(
                                    f"{name} is shared with the target and differs from it; "
                                    "this head needs its own copy: restart the server"
                                )
                            checked_shared.append(name)
                            continue
                    weights.append((name, handle.get_tensor(name)))
        read_s = time.perf_counter() - started
        before = {name: p.data_ptr() for name, p in own.items()}
        torch.cuda.synchronize()
        copy_started = time.perf_counter()
        loaded = model.load_weights(iter(weights))
        torch.cuda.synchronize()
        copy_s = time.perf_counter() - copy_started
        moved = [
            name
            for name, p in self._own_and_shared()[0].items()
            if before.get(name) != p.data_ptr()
        ]
        if moved:
            raise RuntimeError(f"reload moved parameters {moved[:4]}; CUDA graphs would be stale")
        self._drafter_path = str(head)
        return {
            "path": str(head),
            "tensors_loaded": len(weights),
            "loaded_names": len(loaded) if loaded is not None else None,
            "shared_checked_equal": checked_shared,
            "read_seconds": round(read_s, 4),
            "copy_seconds": round(copy_s, 4),
            "total_seconds": round(time.perf_counter() - started, 4),
        }
