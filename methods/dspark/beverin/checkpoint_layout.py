"""Rename-only checkpoint ring. Standard library, Python 3.6 safe.

The trainer resumes the highest integer-named directory and ignores symlinks.
Only ``checkpoints/0`` is an integer name. Older verified copies live in
``verified/step-<N>`` and model-only milestones in ``milestones/step-<N>``.
Nothing in the steady-state rotation is copied.
"""
import json
import os
import shutil
import time

MARKER = "verified-finite.json"
KEEP_FILES = (
    "model.safetensors",
    "config.json",
    "training_state.json",
    "scheduler_state_dict.pt",
    "verified-finite.json",
    "train_command.txt",
)


def atomic_json(path, value):
    temporary = path + ".tmp-" + str(os.getpid())
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_marker(directory):
    payload = read_json(os.path.join(directory, MARKER))
    if not isinstance(payload, dict) or payload.get("finite") is not True:
        return None
    try:
        payload["global_step"] = int(payload["global_step"])
    except (KeyError, TypeError, ValueError):
        return None
    return payload


def list_verified(root):
    """Active verified checkpoints as (step, path, marker), oldest first.

    ``retired/`` is diagnosis-only and is not a resume candidate.
    """
    found = []
    zero = os.path.join(root, "0")
    marker = read_marker(zero)
    if marker is not None and os.path.isdir(zero) and not os.path.islink(zero):
        found.append((marker["global_step"], zero, marker))
    verified = os.path.join(root, "verified")
    if os.path.isdir(verified):
        for name in os.listdir(verified):
            path = os.path.join(verified, name)
            if not name.startswith("step-") or not os.path.isdir(path) or os.path.islink(path):
                continue
            marker = read_marker(path)
            if marker is not None:
                found.append((marker["global_step"], path, marker))
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def _reject_name(root):
    folder = os.path.join(root, "rejected")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "unverified-%d-%d" % (int(time.time()), os.getpid()))


def promote(root, staging, keep=2, milestone=512):
    """Rename a verified staging directory to ``checkpoints/0``.

    The previous verified ``0`` moves to ``verified/step-<old>``. An unverified
    ``0`` is parked under ``rejected/`` and is never resumed. Returns the new step.
    """
    marker = read_marker(staging)
    if marker is None:
        raise RuntimeError("refusing to promote an unverified staging directory")
    if os.path.islink(staging):
        raise RuntimeError("refusing to promote a symlink")
    step = marker["global_step"]
    os.makedirs(root, exist_ok=True)
    verified = os.path.join(root, "verified")
    os.makedirs(verified, exist_ok=True)
    zero = os.path.join(root, "0")
    if os.path.lexists(zero):
        old = read_marker(zero) if os.path.isdir(zero) and not os.path.islink(zero) else None
        if old is not None:
            dest = os.path.join(verified, "step-%d" % old["global_step"])
            if os.path.lexists(dest):
                raise RuntimeError("verified destination already exists: %s" % dest)
            os.rename(zero, dest)
        else:
            os.rename(zero, _reject_name(root))
    os.rename(staging, zero)
    prune(root, keep=keep, milestone=milestone)
    return step


def prune(root, keep=2, milestone=512):
    """Keep the newest ``keep`` full checkpoints. Older 512-step saves become milestones."""
    found = list_verified(root)
    if len(found) <= keep:
        return []
    dropped = []
    for step, path, _marker in found[: len(found) - keep]:
        if os.path.basename(path) == "0":
            continue
        if step > 0 and milestone and step % milestone == 0:
            _to_milestone(root, path, step)
        else:
            shutil.rmtree(path)
        dropped.append(step)
    return dropped


def _to_milestone(root, path, step):
    dest = os.path.join(root, "milestones", "step-%d" % step)
    if os.path.lexists(dest):
        shutil.rmtree(path)
        return
    os.makedirs(dest, exist_ok=True)
    for name in KEEP_FILES:
        source = os.path.join(path, name)
        if os.path.isfile(source) and not os.path.islink(source):
            os.rename(source, os.path.join(dest, name))
    for name in os.listdir(path):
        candidate = os.path.join(path, name)
        if name.endswith(".safetensors") and os.path.isfile(candidate):
            os.rename(candidate, os.path.join(dest, name))
    shutil.rmtree(path)


def retire(root, path):
    """Remove a verified checkpoint from the resume set without deleting it."""
    marker = read_marker(path)
    if marker is None:
        raise RuntimeError("cannot retire an unverified directory")
    dest_dir = os.path.join(root, "retired")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "step-%d" % marker["global_step"])
    if os.path.lexists(dest):
        dest = os.path.join(dest_dir, "step-%d-%d" % (marker["global_step"], os.getpid()))
    os.rename(path, dest)
    return dest


def restore_step(root, step):
    """Make ``checkpoints/0`` the verified checkpoint at ``step``, by rename."""
    found = list_verified(root)
    match = [item for item in found if item[0] == step]
    if not match:
        raise RuntimeError("no verified checkpoint at step %s" % step)
    _step, path, _marker = match[-1]
    zero = os.path.join(root, "0")
    if os.path.abspath(path) == os.path.abspath(zero):
        return zero
    if os.path.lexists(zero):
        current = read_marker(zero) if os.path.isdir(zero) and not os.path.islink(zero) else None
        if current is not None and current["global_step"] != step:
            retire(root, zero)
        elif current is not None and current["global_step"] == step:
            return zero
        else:
            os.rename(zero, _reject_name(root))
    os.rename(path, zero)
    return zero


def prepare_for_start(root):
    """Restore ``0`` from the newest verified checkpoint, or allow a fresh start.

    Returns a status dict. ``ok`` false means the caller must not launch torchrun.
    """
    os.makedirs(root, exist_ok=True)
    verified = list_verified(root)
    zero = os.path.join(root, "0")
    zero_exists = os.path.isdir(zero) and not os.path.islink(zero)
    zero_marker = read_marker(zero) if zero_exists else None
    if not verified:
        if zero_exists and _has_weight(zero) and zero_marker is None:
            return {"ok": False, "reason": "unverified checkpoint and no verified copy"}
        if os.path.islink(zero):
            return {"ok": False, "reason": "checkpoint 0 is a symlink and would be ignored on resume"}
        return {"ok": True, "reason": "fresh", "step": None}
    newest_step, newest_path, _marker = verified[-1]
    if zero_marker is not None and zero_marker["global_step"] == newest_step:
        return {"ok": True, "reason": "current", "step": newest_step}
    try:
        restore_step(root, newest_step)
    except OSError as exc:
        return {"ok": False, "reason": "restore failed: %s" % exc, "step": newest_step}
    restored = read_marker(os.path.join(root, "0"))
    if restored is None or restored["global_step"] != newest_step:
        return {"ok": False, "reason": "restore did not produce the newest verified checkpoint",
                "step": newest_step}
    return {"ok": True, "reason": "restored", "step": newest_step, "from": newest_path}


def _has_weight(directory):
    return os.path.isfile(os.path.join(directory, "model.safetensors")) or os.path.isfile(
        os.path.join(directory, "optimizer_state_dict.pt"))
