import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "methods/dspark/train"))
binding = importlib.import_module("rebind_verifier")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, target = tmp_path / "original", tmp_path / "shared-target"
    source.mkdir()
    target.mkdir()
    for name, value in {
        "config.json": b"{}",
        "tokenizer.json": b"tokenizer",
        "chat_template.jinja": b"template",
        "model.safetensors": b"target",
    }.items():
        (target / name).write_bytes(value)
    revision = "a" * 40
    config = {
        "transformer_layer_config": {"num_hidden_layers": 1},
        "aux_hidden_state_layer_ids": [1, 8, 15, 22, 29],
        "speculators_config": {
            "verifier": {
                "name_or_path": f"/private-owner/snapshots/{revision}",
                "architectures": ["Apertus1p5ForConditionalGeneration"],
            }
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"unchanged drafter bytes")
    (source / "truncation.json").write_text(
        json.dumps({"sha256": binding.sha256(source / "model.safetensors")})
    )
    contract = {
        "source": {
            "revision": revision,
            "authorized_checkpoint": str(target),
            "config_sha256": binding.sha256(target / "config.json"),
            "tokenizer_json_sha256": binding.sha256(target / "tokenizer.json"),
            "chat_template_sha256": binding.sha256(target / "chat_template.jinja"),
            "weight_manifest": {
                "files": [
                    {
                        "name": "model.safetensors",
                        "sha256": binding.sha256(target / "model.safetensors"),
                    }
                ]
            },
        },
        "target": {"architecture": "Apertus1p5ForConditionalGeneration"},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    return source, path, image


def test_only_nested_verifier_path_changes_and_weights_stay_identical(tmp_path: Path) -> None:
    source, contract_path, image = fixture(tmp_path)
    before = (source / "config.json").read_bytes()
    output = tmp_path / "rebound"
    binding.prepare(source, output, contract_path)
    old = json.loads(before)
    expected = json.loads(before)
    expected["speculators_config"]["verifier"]["name_or_path"] = json.loads(
        contract_path.read_text()
    )["source"]["authorized_checkpoint"]
    assert json.loads((output / "config.json").read_text()) == expected
    assert (source / "config.json").read_bytes() == before
    assert os.path.samefile(source / "model.safetensors", output / "model.safetensors")
    report = binding.verify(output, contract_path, image, "test-source")
    assert report["passed"]
    assert report["weights"]["model.safetensors"]["same_file_as_source"]
    assert report["target"]["revision"] == "a" * 40
    assert report["target"]["verified_sha256"]["tokenizer.json"]
    assert old["speculators_config"]["verifier"]["name_or_path"].startswith("/private-owner")
    with pytest.raises(FileExistsError):
        binding.prepare(source, output, contract_path)


def test_different_verifier_revision_is_rejected(tmp_path: Path) -> None:
    source, contract, _ = fixture(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config["speculators_config"]["verifier"]["name_or_path"] = "/private/snapshots/different"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="snapshot revision"):
        binding.prepare(source, tmp_path / "rebound", contract)


def test_weight_or_other_config_changes_are_rejected(tmp_path: Path) -> None:
    source, contract, image = fixture(tmp_path)
    output = tmp_path / "rebound"
    binding.prepare(source, output, contract)
    original = (output / "config.json").read_text()
    config = json.loads(original)
    config["other_change"] = True
    (output / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="single verifier path"):
        binding.verify(output, contract, image, "source")
    (output / "config.json").write_text(original)
    (source / "model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="original truncation"):
        binding.verify(output, contract, image, "source")


def test_target_weight_mismatch_fails_allocated_gate(tmp_path: Path) -> None:
    source, contract, image = fixture(tmp_path)
    output = tmp_path / "rebound"
    binding.prepare(source, output, contract)
    target = Path(json.loads(contract.read_text())["source"]["authorized_checkpoint"])
    (target / "model.safetensors").write_bytes(b"changed target")
    with pytest.raises(ValueError, match="target contract mismatch"):
        binding.verify(output, contract, image, "source")


@pytest.mark.parametrize("preflight_rc", [0, 42])
def test_training_submission_checks_warm_start_before_training(
    tmp_path: Path, preflight_rc: int
) -> None:
    import shutil

    train = tmp_path / "repo/methods/dspark/train"
    train.mkdir(parents=True)
    serving = tmp_path / "repo/serving"
    serving.mkdir()
    shutil.copyfile(REPO / "methods/dspark/train/submit-train.sh", train / "submit-train.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    image = tmp_path / "image.sqsh"
    image.touch()
    env_source = tmp_path / "env.toml"
    env_source.write_text(f'image = "{image}"\nworkdir = "/unused"\n')
    scripts = {
        serving / "patch-vllm.sh": 'printf "/fake:/fake\\n"',
        serving / "resolve-env.sh": 'cp "$1" "$TEST_RESOLVED"\nprintf "%s\\n" "$TEST_RESOLVED"',
        fake_bin / "git": 'printf "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n"',
        fake_bin / "sbatch": 'cp "${!#}" "$TEST_JOB_SCRIPT"\nprintf "123456\\n"',
        fake_bin / "scontrol": 'printf "node1\\n"',
        fake_bin / "srun": (
            'printf "%s\\n" "$*" >> "$TEST_CALLS"\n'
            'case "$*" in *rebind_verifier.py*) exit "$TEST_PREFLIGHT_RC";; esac'
        ),
    }
    for path, body in scripts.items():
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body + "\n")
        path.chmod(0o755)
    script = tmp_path / "job.sbatch"
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": str(fake_bin) + ":" + os.environ["PATH"],
        "MODE": "train",
        "STAGE": "8b",
        "RUN_ID": "test-recovery",
        "DATA": "/prepared-data",
        "FROM_PRETRAINED": "/rebound-initializer",
        "DSPARK_VALIDATE_WARM_START": "1",
        "DSPARK_DIR": str(tmp_path / "data"),
        "DSPARK_ENV_SOURCE": str(env_source),
        "TEST_RESOLVED": str(tmp_path / "resolved.toml"),
        "TEST_JOB_SCRIPT": str(script),
        "TEST_CALLS": str(calls),
        "TEST_PREFLIGHT_RC": str(preflight_rc),
        "CHAIN": "1",
        "SLURM_JOB_ID": "123456",
        "SLURM_JOB_NODELIST": "node1,node2",
    }
    subprocess.run(
        ["bash", str(train / "submit-train.sh")], env=env, check=True, capture_output=True
    )
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True)
    commands = calls.read_text().splitlines()
    assert "rebind_verifier.py verify" in commands[0]
    assert result.returncode == preflight_rc
    assert any("train-node.sh" in command for command in commands) == (preflight_rc == 0)
