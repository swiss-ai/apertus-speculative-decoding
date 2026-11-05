"""Verified checkpoint rotation installed around Trainer.maybe_save_checkpoint.

Speculators is not modified. Saves land in a staging directory, are read back
across ranks, and are promoted by rename only after ``verified-finite.json``
is written. A SIGTERM "interrupted" save uses this same path. Set
``BEVERIN_DISABLE_INTERRUPT_SAVE=1`` to skip that save instead.
"""
import io
import json
import os
import pickle
import struct
import time
import zipfile
from pathlib import Path

from checkpoint_layout import MARKER, atomic_json, promote, read_marker

_FLOAT_DTYPES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
}


class CheckpointRejected(RuntimeError):
    pass


def _rank_world(trainer, torch):
    distributed = getattr(trainer, "is_distributed", False)
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(getattr(trainer, "rank", 0)), 1


def _barrier(trainer, torch):
    if getattr(trainer, "is_distributed", False) and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _all_bad(trainer, torch, bad):
    _rank, world = _rank_world(trainer, torch)
    if world == 1 or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return bool(bad)
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    flag = torch.tensor([int(bool(bad))], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(int(flag.item()))


def assign_by_size(items, world, rank):
    """Deterministically assign ``(key, nbytes)`` items so ranks share the bytes."""
    loads = [0] * max(1, world)
    mine = []
    for key, nbytes in sorted(items, key=lambda item: (-int(item[1]), str(item[0]))):
        owner = min(range(len(loads)), key=lambda index: (loads[index], index))
        loads[owner] += int(nbytes)
        if owner == rank:
            mine.append(key)
    return mine


def _torch_dtype(torch, name):
    return {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
    }[name]


def safetensor_records(path):
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise RuntimeError("truncated safetensors header: %s" % path)
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
    data_start = 8 + header_len
    records = []
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        dtype = info.get("dtype")
        start, end = info.get("data_offsets", [0, 0])
        records.append((key, dtype, data_start + int(start), int(end) - int(start), info.get("shape")))
    return records


def _finite_buffer(torch, dtype_name, blob):
    if dtype_name not in _FLOAT_DTYPES:
        return True
    width = _FLOAT_DTYPES[dtype_name]
    if width and len(blob) % width:
        return False
    tensor = torch.frombuffer(bytearray(blob), dtype=_torch_dtype(torch, dtype_name))
    return bool(torch.isfinite(tensor).all().item())


def verify_safetensors(path, torch, rank, world):
    records = safetensor_records(path)
    items = [(key, length) for key, _dtype, _offset, length, _shape in records]
    mine = set(assign_by_size(items, world, rank))
    by_key = {key: (dtype, offset, length) for key, dtype, offset, length, _shape in records}
    with open(path, "rb") as handle:
        for key in mine:
            dtype, offset, length = by_key[key]
            handle.seek(offset)
            if not _finite_buffer(torch, dtype, handle.read(length)):
                return False
    return True


def _storage_is_float(storage_type):
    name = getattr(storage_type, "__name__", "")
    return any(token in name for token in ("Float", "Half", "Double", "Complex", "BFloat16"))


def _storage_width(storage_type):
    name = getattr(storage_type, "__name__", "")
    if "Complex" in name or "Double" in name or "Long" in name:
        return 8
    if "BFloat16" in name or "Half" in name or "Short" in name:
        return 2
    if "Byte" in name or "Char" in name or "Bool" in name:
        return 1
    return 4


def _dtype_for_storage(torch, storage_type):
    name = getattr(storage_type, "__name__", "")
    if "BFloat16" in name:
        return torch.bfloat16
    if "Half" in name or "Float16" in name:
        return torch.float16
    if "Double" in name:
        return torch.float64
    if "Complex" in name:
        return torch.complex64
    return torch.float32


def collect_zip_storages(torch, blob):
    """Record storage keys without allocating their bytes."""
    records = []
    rebuild = torch._utils._rebuild_tensor_v2

    def empty_rebuild(*_args, **_kwargs):
        return torch.empty(0)

    torch._utils._rebuild_tensor_v2 = empty_rebuild
    try:
        class Recorder(pickle.Unpickler):
            def persistent_load(self, pid):
                if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
                    raise RuntimeError("unsupported optimizer storage record")
                storage_type, key, _location, numel = pid[1], pid[2], pid[3], int(pid[4])
                records.append((str(key), storage_type, numel))
                return storage_type(0)

        Recorder(io.BytesIO(blob)).load()
    finally:
        torch._utils._rebuild_tensor_v2 = rebuild
    return records


def verify_optimizer_file(path, torch, rank, world):
    if not zipfile.is_zipfile(path):
        if rank != 0:
            return True
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        return _tree_finite(torch, loaded)
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        records = collect_zip_storages(torch, archive.read(pickle_name))
        items = []
        info = {}
        for key, storage_type, numel in records:
            if not _storage_is_float(storage_type):
                continue
            entry = _zip_entry(archive, key)
            size = archive.getinfo(entry).file_size
            items.append((key, size))
            info[key] = (storage_type, entry, numel)
        mine = set(assign_by_size(items, world, rank))
        for key in mine:
            storage_type, entry, numel = info[key]
            blob = archive.read(entry)
            width = _storage_width(storage_type)
            if numel and len(blob) < numel * width:
                return False
            tensor = torch.frombuffer(bytearray(blob[: numel * width]), dtype=_dtype_for_storage(torch, storage_type))
            if not bool(torch.isfinite(tensor).all().item()):
                return False
    return True


def _zip_entry(archive, key):
    suffix = "data/" + key
    for name in archive.namelist():
        if name.endswith(suffix):
            return name
    raise RuntimeError("optimizer storage %s is missing" % key)


def _tree_finite(torch, value):
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            return False
        return True
    if isinstance(value, dict):
        return all(_tree_finite(torch, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_finite(torch, item) for item in value)
    return True


def verify_directory(directory, torch, rank, world):
    directory = Path(directory)
    model = directory / "model.safetensors"
    optimizer = directory / "optimizer_state_dict.pt"
    if not model.is_file() or not optimizer.is_file():
        return False
    if not verify_safetensors(str(model), torch, rank, world):
        return False
    if not verify_optimizer_file(str(optimizer), torch, rank, world):
        return False
    return True


def _learning_rates(trainer):
    rates = []
    for optimizer in getattr(trainer, "optimizers", None) or []:
        for group in getattr(optimizer, "param_groups", None) or []:
            if "lr" in group:
                rates.append(float(group["lr"]))
    return rates


def _marker(trainer, epoch, local_step, directory):
    files = {}
    for name in ("model.safetensors", "optimizer_state_dict.pt", "scheduler_state_dict.pt",
                 "config.json", "training_state.json"):
        path = Path(directory) / name
        if path.is_file():
            files[name] = path.stat().st_size
    return {
        "finite": True,
        "global_step": int(trainer.global_step),
        "epoch": epoch if isinstance(epoch, int) else None,
        "local_step": int(local_step),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "learning_rates": _learning_rates(trainer),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "files": files,
    }


def _pre_save(trainer, torch):
    guard = getattr(trainer, "_beverin_numerical_guard", None)
    if guard is None:
        return
    guard.check(guard.parameters(), "pre-save-parameters")
    state = guard.optimizer_state()
    if state:
        guard.check(state, "pre-save-optimizer")


def _relink(root, epoch, local_step):
    if not isinstance(epoch, int):
        return
    root = Path(root)
    for old in root.glob("epoch%d_*" % epoch):
        if old.is_symlink():
            old.unlink()
    suffix = "step%d" % int(local_step) if int(local_step) else "end"
    link = root / ("epoch%d_%s" % (epoch, suffix))
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("0"))


def install(Trainer, torch):
    if os.environ.get("BEVERIN_CHECKPOINT_RING", "0") != "1":
        return
    original = Trainer.maybe_save_checkpoint
    keep = int(os.environ.get("BEVERIN_RING_KEEP", "2"))
    milestone = int(os.environ.get("BEVERIN_MILESTONE_INTERVAL", "512"))

    def save(self, epoch, local_step=0):
        rank, world = _rank_world(self, torch)
        if epoch == "interrupted" and os.environ.get("BEVERIN_DISABLE_INTERRUPT_SAVE", "0") == "1":
            if rank == 0:
                print("BEVERIN_INTERRUPT_SAVE_DISABLED", flush=True)
            _barrier(self, torch)
            return None
        if epoch == "interrupted":
            epoch = getattr(self, "current_epoch", 0)
            if not isinstance(epoch, int):
                epoch = 0
            # Step within the epoch (multi_epoch.py records where it began).
            local_step = int(getattr(self, "global_step", 0)) - int(
                getattr(self, "_beverin_epoch_start_global", 0))
            if rank == 0:
                print("BEVERIN_INTERRUPT_SAVE_VERIFIED " + json.dumps({
                    "epoch": epoch, "local_step": local_step, "global_step": self.global_step,
                }), flush=True)
        _pre_save(self, torch)
        root = Path(self.checkpointer.path)
        step = int(self.global_step)
        staging_parent = root / (".staging-step-%d" % step)
        if rank == 0 and staging_parent.exists():
            import shutil
            shutil.rmtree(str(staging_parent))
        _barrier(self, torch)
        staging_parent.mkdir(parents=True, exist_ok=True)
        real_path = self.checkpointer.path
        self.checkpointer.path = staging_parent
        try:
            original(self, epoch, local_step=local_step)
            written = staging_parent / str(epoch)
            if os.environ.get("BEVERIN_FAULT_CORRUPT_WRITE", "0") == "1":
                from fault_inject import corrupt_checkpoint, injection_allowed
                if injection_allowed() and rank == 0:
                    corrupt_checkpoint(written)
            _barrier(self, torch)
            local_ok = verify_directory(written, torch, rank, world) if written.is_dir() else False
            ok = not _all_bad(self, torch, not local_ok)
            if rank == 0:
                if ok:
                    payload = _marker(self, epoch, local_step, written)
                    payload["world_size"] = world
                    atomic_json(str(written / MARKER), payload)
                    promote(str(root), str(written), keep=keep, milestone=milestone)
                    _relink(root, epoch, local_step)
                    if staging_parent.exists():
                        import shutil
                        shutil.rmtree(str(staging_parent), ignore_errors=True)
                    verified = read_marker(str(root / "0"))
                    print("BEVERIN_SAVE_VERIFIED " + json.dumps({
                        "global_step": None if verified is None else verified["global_step"],
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }), flush=True)
                else:
                    import shutil
                    rejected = root / "evidence"
                    rejected.mkdir(parents=True, exist_ok=True)
                    dest = rejected / ("rejected-save-step-%d" % step)
                    if written.exists():
                        if dest.exists():
                            shutil.rmtree(str(dest))
                        os.rename(str(written), str(dest))
                    atomic_json(str(root / "save-rejected.json"), {
                        "global_step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "reason": "read-back was not finite",
                    })
                    print("BEVERIN_SAVE_REJECTED " + json.dumps({"global_step": step}), flush=True)
                    if staging_parent.exists():
                        shutil.rmtree(str(staging_parent), ignore_errors=True)
            _barrier(self, torch)
            if not ok:
                raise CheckpointRejected(
                    "checkpoint read-back failed at step %s; checkpoints/0 was not replaced" % step)
        finally:
            self.checkpointer.path = real_path
        return None

    Trainer.maybe_save_checkpoint = save
