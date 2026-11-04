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


def test_mesh_arguments_supported_by_frozen_opentela():
    # Flag set from the actual sai-v0.0.7 start --help, checked on Clariden.
    supported = {
        "--config-dir",
        "--seed",
        "--bootstrap.addr",
        "--service.name",
        "--service.port",
        "--label",
        "--subprocess",
    }
    command = module("launch").opentela_command(
        Path("/bin/otela"),
        Path("/private"),
        {
            "bootstrap_addr": "/ip4/127.0.0.1/tcp/1234/p2p/example",
            "engine_argv": ["python3", "-m", "server"],
        },
        {"launched_by": "k8s"},
        123,
        Path("/private/engine.sh"),
    )
    assert {arg for arg in command if arg.startswith("--")} <= supported
    assert command[command.index("--seed") + 1] == "123"


def test_json_survives_mesh_wrapper_field_splitting(tmp_path):
    import json
    import subprocess
    import sys

    launch = module("launch")
    config = {"method": "dspark", "model": "/assets/drafter", "num_speculative_tokens": 3}
    entrypoint = launch.engine_script(
        tmp_path,
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1])",
            json.dumps(config),
        ],
    )
    command = launch.opentela_command(
        Path("/otela"), tmp_path, {"bootstrap_addr": "address"}, {}, 1, entrypoint
    )
    # Reproduce the frozen wrapper's naive split, then actually run its child.
    argv = command[command.index("--subprocess") + 1].split()
    output = subprocess.check_output(argv, text=True)
    assert json.loads(output) == config


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


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def test_plain_render_differs_only_in_drafter_identity(tmp_path):
    import json

    prepare = module("prepare")
    prepare.render(tmp_path / "dspark", 1, "rev", "dspark", "nid006597")
    prepare.render(tmp_path / "plain", 1, "rev", "plain", "nid006597")

    def load(variant):
        root = tmp_path / variant
        deployment = yaml.safe_load((root / "deployment.yaml").read_text())
        provenance = json.loads((root / "deployment-provenance.json").read_text())
        return deployment, provenance

    (dspark, dspark_spec), (plain, plain_spec) = load("dspark"), load("plain")
    argv = dspark_spec["engine_argv"]
    at = argv.index("--speculative-config")
    expected = argv[:at] + argv[at + 2 :]
    names = expected.index("--served-model-name") + 1
    expected[names : names + 2] = [
        prepare.MODEL_IDS["plain"],
        prepare.MODEL_IDS["plain"].removeprefix("SwissAI-Research/"),
    ]
    assert plain_spec["engine_argv"] == expected
    assert plain_spec["variant"] == "plain" and "variant" not in dspark_spec
    # Same frozen assets (including the drafter files, still verified) and runtime overlays.
    assert plain_spec["asset_hashes"] == dspark_spec["asset_hashes"]
    assert plain_spec["overlays"] == dspark_spec["overlays"]
    differing = {
        key
        for key, value in dict(_flatten(dspark)).items()
        if dict(_flatten(plain)).get(key) != value
    } | (set(dict(_flatten(plain))) - set(dict(_flatten(dspark))))
    template = "/spec/template"
    assert differing == {
        template + "/metadata/annotations/apertus.swissai/variant",
        template + "/spec/volumes[1]/configMap/name",
    }
    selector = dspark["spec"]["template"]["spec"]["nodeSelector"]
    assert selector == {"kubernetes.io/arch": "arm64", "kubernetes.io/hostname": "nid006597"}


def test_default_dspark_render_is_unchanged_without_pin(tmp_path):
    prepare = module("prepare")
    prepare.render(tmp_path / "m", 1, "rev")
    deployment = yaml.safe_load((tmp_path / "m" / "deployment.yaml").read_text())
    template = deployment["spec"]["template"]
    assert template["spec"]["nodeSelector"] == {"kubernetes.io/arch": "arm64"}
    assert "apertus.swissai/variant" not in template["metadata"]["annotations"]
