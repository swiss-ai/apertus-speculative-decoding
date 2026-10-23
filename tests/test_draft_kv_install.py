"""CPU coverage for packaging the opt-in serving overlay."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from draft_kv import install


def test_overlay_packages_hashed_host_helpers_without_gpu_imports(tmp_path, monkeypatch):
    # A tiny vLLM package fixture exercises the real installer and helper copies
    # without depending on the pinned serving source or any GPU dependencies.
    source = tmp_path / "source"
    (source / "vllm").mkdir(parents=True)
    (source / "vllm/__init__.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(install, "TRANSFORMS", {
        "vllm/__init__.py": [("VALUE = 1\n", "VALUE = 2\n")],
    })
    output = tmp_path / "overlay"
    mounts = install.build(source, output)
    helpers = output / "vllm/apertus_draft_kv"
    manifest = json.loads((output / "draft-kv-overlay-manifest.json").read_text())
    for name in ("__init__.py", "tracing.py"):
        original = Path(install.__file__).parent / name
        assert (helpers / name).read_bytes() == original.read_bytes()
        assert manifest["helper_files"][name] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert f"{helpers}:/workspace/vllm/vllm/apertus_draft_kv" in mounts

    # Import the exact installed namespace in a fresh interpreter. This catches
    # accidental GPU imports in the package initializer as well as missing files.
    subprocess.run(
        [sys.executable, "-I", "-c", """
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from vllm.apertus_draft_kv import tracing

assert Path(tracing.__file__).resolve() == Path(sys.argv[1], "vllm/apertus_draft_kv/tracing.py")
assert not any(name in sys.modules for name in ("torch", "triton"))
assert "vllm.apertus_draft_kv.integration" not in sys.modules
os.environ["VLLM_DRAFT_KV_TRACE"] = "/unused-test-trace"
assert tracing.trace_due(1)
assert not tracing.trace_due(2)
""", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_pin_covers_every_transformed_file():
    pin = Path(install.__file__).with_name("pinned-source.json")
    manifest = json.loads(pin.read_text())
    missing = set(install.TRANSFORMS) - manifest["files"].keys()
    assert not missing, f"transformed sources missing from pin: {sorted(missing)}"
