import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "eagle.sh": "methods/eagle/launch/eagle.sh",
    "baseline.sh": "serving/baseline.sh",
    "dspark.sh": "methods/dspark/launch/dspark.sh",
}
EIGHT = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B"


def _run(script: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "ENV_SOURCE": str(REPO / "methods/eagle/configs/train-env.toml"),
        "SML_ENV_DIR": str(tmp_path / "sml"),
        "IMAGE": str(tmp_path / "absent.sqsh"),
        "VALIDATE_ONLY": "1",
        "RUN_SUFFIX": "test",
        **env,
    }
    return subprocess.run(
        ["bash", str(REPO / SCRIPTS[script])], capture_output=True, text=True, env=full, cwd=REPO
    )


def _settings(stdout: str) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in stdout.splitlines()
        if "=" in line and not line.startswith(" ")
    )


def _head(tmp_path: Path) -> Path:
    head = tmp_path / "head"
    head.mkdir()
    (head / "config.json").write_text(
        (REPO / "methods/eagle/configs/8b/draft-e31-config.json").read_text()
    )
    return head


def test_eagle_launcher_propagates_8b_stage(tmp_path: Path) -> None:
    result = _run("eagle.sh", tmp_path, STAGE="8b", EAGLE_HEAD=str(_head(tmp_path)))
    assert result.returncode == 0, result.stderr
    got = _settings(result.stdout)
    assert got["target_model"] == EIGHT
    assert got["target_tensor_parallel_size"] == "1"
    assert got["draft_tensor_parallel_size"] == "1"
    assert got["max_model_len"] == "32768"
    assert got["enable_prefix_caching"] == "0"
    assert got["target_contract"].endswith("targets/8b/contract.json")
    assert got["served_model"] == "swiss-ai/Apertus-v1.5-8B-eagle31-n3-tp1d1-test"


def test_eagle_launcher_requires_a_stage(tmp_path: Path) -> None:
    result = _run("eagle.sh", tmp_path, EAGLE_HEAD=str(_head(tmp_path)), STAGE="")
    assert result.returncode == 2
    assert "STAGE must be 8b or 70b" in result.stderr


def test_eagle_launcher_refuses_mismatched_tp(tmp_path: Path) -> None:
    result = _run("eagle.sh", tmp_path, STAGE="8b", DRAFT_TP="4", EAGLE_HEAD=str(_head(tmp_path)))
    assert result.returncode == 2
    assert "must equal TARGET_TP" in result.stderr


def test_eagle_launcher_refuses_a_target_that_is_not_the_contract(tmp_path: Path) -> None:
    result = _run(
        "eagle.sh",
        tmp_path,
        STAGE="8b",
        TARGET_MODEL=EIGHT.replace("8B", "70B"),
        EAGLE_HEAD=str(_head(tmp_path)),
    )
    assert result.returncode != 0
    assert "not the contract checkpoint" in result.stderr


def test_config_only_head_needs_validate_only(tmp_path: Path) -> None:
    head = _head(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run(
        [
            "python3",
            "-m",
            "apertus_bench.eagle",
            str(head),
            "--contract",
            str(REPO / "targets/8b/contract.json"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "no weight files" in result.stderr


def test_baseline_matches_the_eagle_arm_for_8b(tmp_path: Path) -> None:
    base = _settings(_run("baseline.sh", tmp_path, STAGE="8b").stdout)
    eagle = _settings(
        _run("eagle.sh", tmp_path, STAGE="8b", EAGLE_HEAD=str(_head(tmp_path))).stdout
    )
    for key in ("target_model", "target_tensor_parallel_size", "max_model_len", "target_contract"):
        assert base[key] == eagle[key]
    assert base["enable_prefix_caching"] == eagle["enable_prefix_caching"] == "0"
    assert base["served_model"] == "swiss-ai/Apertus-v1.5-8B-baseline-tp1-test"


def _dspark_checkpoint(tmp_path: Path, kind: str = "dspark") -> Path:
    checkpoint = tmp_path / "checkpoint_best"
    checkpoint.mkdir()
    config = {
        "speculators_model_type": kind,
        "aux_hidden_state_layer_ids": [1, 8, 15, 22, 29],
        "block_size": 8,
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors").write_bytes(b"x")
    return checkpoint


def test_dspark_launcher_serves_like_the_baseline_for_8b(tmp_path: Path) -> None:
    checkpoint = _dspark_checkpoint(tmp_path)
    result = _run("dspark.sh", tmp_path, STAGE="8b", DSPARK_CHECKPOINT=str(checkpoint))
    assert result.returncode == 0, result.stderr
    got = _settings(result.stdout)
    baseline = _settings(_run("baseline.sh", tmp_path, STAGE="8b").stdout)
    for key in ("target_model", "max_model_len", "gpu_memory_utilization"):
        assert got[key] == baseline[key]
    assert got["engine_method"] == "dspark"
    assert got["num_speculative_tokens"] == "7"
    assert got["served_model"] == "swiss-ai/Apertus-v1.5-8B-dspark-n7-tp1-test"
    report = json.loads(got["checkpoint_report"])
    assert report["aux_hidden_state_layer_ids"] == [1, 8, 15, 22, 29]
    assert report["weights"] == {"model.safetensors": 1}


def test_dspark_launcher_refuses_other_checkpoints_and_depths(tmp_path: Path) -> None:
    eagle = _dspark_checkpoint(tmp_path, kind="eagle3")
    result = _run("dspark.sh", tmp_path, STAGE="8b", DSPARK_CHECKPOINT=str(eagle))
    assert result.returncode != 0
    assert "not 'dspark'" in result.stderr
    result = _run(
        "dspark.sh", tmp_path, STAGE="8b", DSPARK_CHECKPOINT=str(eagle), NUM_SPECULATIVE_TOKENS="8"
    )
    assert result.returncode == 2


def test_baseline_without_stage_keeps_the_historical_70b_launch(tmp_path: Path) -> None:
    got = _settings(_run("baseline.sh", tmp_path).stdout)
    assert got["stage"] == "legacy-70b"
    assert got["target_tensor_parallel_size"] == "4"
    assert got["enable_prefix_caching"] == "engine-default"


def test_submit_requires_an_explicit_training_config(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "EAGLE_TRAIN_CONFIG"}
    result = subprocess.run(
        ["bash", str(REPO / "methods/eagle/launch/submit-eagle-train.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "EAGLE_TRAIN_CONFIG" in result.stderr


@pytest.mark.parametrize("name", ["train-e31-overfit.yaml", "train-e31.yaml"])
def test_8b_training_configs_name_the_8b_contract_and_head(name: str) -> None:
    import yaml

    cfg = yaml.safe_load((REPO / "methods/eagle/configs/8b" / name).read_text())
    assert cfg["stage"] == "8b"
    assert cfg["model"]["contract"] == "targets/8b/contract.json"
    assert cfg["model"]["draft_model_config"] == "methods/eagle/configs/8b/draft-e31-config.json"
    draft = json.loads((REPO / cfg["model"]["draft_model_config"]).read_text())
    assert draft["fc_norm"] is True and draft["norm_output"] is True
    assert cfg["training"]["ttt_length"] == 7
    assert cfg["training"]["draft_accumulation_steps"] == 16
    assert cfg["training"]["learning_rate"] == 1.0e-4
