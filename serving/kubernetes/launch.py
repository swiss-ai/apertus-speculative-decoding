"""Verify frozen assets, preserve per-start provenance, and run OpenTela/vLLM."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(root, files):
    for relative, expected in files.items():
        path = root / relative
        if sha256(path) != expected:
            raise ValueError(f"asset hash mismatch: {path}")


def engine_script(run, argv):
    import shlex

    path = run / "engine.sh"
    path.write_text("#!/bin/sh\nexec " + shlex.join(argv) + "\n")
    path.chmod(0o700)
    return path


def opentela_command(binary, config_dir, spec, labels, seed, entrypoint):
    # The frozen pilot binary is sai-v0.0.7. Its verified CLI supports a private
    # config root and seed, but not the newer ephemeral-identity/datadir flags.
    command = [
        str(binary),
        "--config-dir",
        str(config_dir),
        "start",
        "--seed",
        str(seed),
        "--bootstrap.addr",
        spec["bootstrap_addr"],
        "--service.name",
        "llm",
        "--service.port",
        "8080",
    ]
    for key, value in labels.items():
        command += ["--label", f"{key}={value}"]
    # The frozen wrapper splits on whitespace without interpreting shell quotes.
    # Keep JSON and other complex arguments inside a real shell script instead.
    if any(char.isspace() for char in str(entrypoint)):
        raise ValueError("The OpenTela entrypoint path must not contain whitespace")
    return command + ["--subprocess", "/bin/sh " + str(entrypoint)]


def serve(runtime=Path("/runtime"), assets=Path("/assets"), logs=Path("/logs")):
    spec = json.loads((runtime / "deployment.json").read_text())
    run = (
        logs
        / os.environ["POD_UID"]
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    run.mkdir(parents=True, exist_ok=False)
    for name in ("deployment.json", "launch.py", "chat_template.jinja"):
        shutil.copy2(runtime / name, run / name)
    (run / "identity.json").write_text(
        json.dumps(
            {
                "pod": os.environ["POD_NAME"],
                "uid": os.environ["POD_UID"],
                "node": os.environ["NODE_NAME"],
                "namespace": os.environ["POD_NAMESPACE"],
                "source_revision": spec["source_revision"],
                "image": spec["image"],
                "started_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    revision = subprocess.check_output(
        ["git", "-C", "/workspace/vllm", "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != spec["vllm_revision"]:
        raise ValueError(f"vLLM revision mismatch: {revision}")
    verify_files(assets, spec["asset_hashes"])
    verify_files(runtime, spec["runtime_hashes"])
    # The OCI image has an editable vLLM install at this pinned source path.
    # Preserve the exact benchmark overlays instead of patching shared images.
    for name, destination in spec["overlays"].items():
        shutil.copy2(runtime / name, destination)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv"],
        stdout=(run / "gpu.csv").open("w"),
        check=True,
    )
    binary = run / "otela-arm64"
    shutil.copy2(assets / "opentela/otela-arm64", binary)
    binary.chmod(0o755)
    (run / "opentela-version.txt").write_text(
        subprocess.check_output([str(binary), "version"], text=True, stderr=subprocess.STDOUT)
    )
    # The run directory and seed are unique per pod start. Keep mesh identity
    # files private and exclude this directory when collecting public artifacts.
    # No API credentials are needed by the model worker.
    import shlex

    config_dir = run / "opentela-private"
    config_dir.mkdir(mode=0o700)
    labels = {
        "launched_by": "k8s",
        "framework": "vllm",
        "served_model_name": spec["model_id"],
        "worker_group_id": os.environ["POD_UID"],
        "started_at": datetime.now(UTC).isoformat(),
        "model_revision": spec["target_revision"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "vllm_commit": revision,
        "source_revision": spec["source_revision"],
        "framework_args": shlex.join(spec["engine_argv"][3:]),
    }
    entrypoint = engine_script(run, spec["engine_argv"])
    command = opentela_command(
        binary, config_dir, spec, labels, uuid.uuid4().int % (2**31 - 1) + 1, entrypoint
    )
    (run / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (run / "verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "vllm_revision": revision,
                "asset_hashes": spec["asset_hashes"],
                "runtime_hashes": spec["runtime_hashes"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified deployment; preserving this start in {run}", flush=True)
    with (run / "worker.log").open("wb") as log:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )

        def forward(signum, _frame):
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)

        def copy_output():
            for block in iter(lambda: proc.stdout.read1(65536), b""):
                log.write(block)
                log.flush()
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()

        copier = threading.Thread(target=copy_output, daemon=True)
        copier.start()
        code = proc.wait()
        copier.join(timeout=5)
    (run / "exit.json").write_text(json.dumps({"returncode": code}) + "\n")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(serve())
