import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "serving/kubernetes" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_rejects_corrupted_staged_asset(tmp_path):
    launch = module("launch")
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original")
    expected = {path.name: launch.sha256(path)}
    launch.verify_files(tmp_path, expected)
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        launch.verify_files(tmp_path, expected)


def test_rejects_missing_staged_asset(tmp_path):
    with pytest.raises(FileNotFoundError):
        module("launch").verify_files(tmp_path, {"model.safetensors": "0" * 64})


def test_frozen_runtime_survives_configmap_round_trip(tmp_path):
    import json

    prepare = module("prepare")
    output = tmp_path / "manifests"
    prepare.render(output, 0, "test-revision")
    config = yaml.safe_load((output / "configmap.yaml").read_text())
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name, content in config["data"].items():
        (runtime / name).write_text(content)
    spec = json.loads((runtime / "deployment.json").read_text())
    launch = module("launch")
    launch.verify_files(runtime, spec["runtime_hashes"])
    assert len(spec["overlays"]) == 4
    (runtime / next(iter(spec["overlays"]))).write_text("changed")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        launch.verify_files(runtime, spec["runtime_hashes"])
