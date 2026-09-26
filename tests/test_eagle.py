import json
from pathlib import Path

import pytest

from apertus_bench.eagle import (
    EagleHeadError,
    check_target_model,
    infer_algorithm,
    validate_eagle_head,
)

REPO = Path(__file__).resolve().parents[1]
CONTRACT_8B = REPO / "targets/8b/contract.json"
CONTRACT_70B = REPO / "targets/70b/contract.json"


def _write_head(tmp_path: Path, config: dict[str, object], *, name: str = "head") -> Path:
    head = tmp_path / name
    head.mkdir()
    (head / "config.json").write_text(json.dumps(config))
    return head


def _load(path: str) -> dict[str, object]:
    return json.loads((REPO / "methods/eagle/configs" / path).read_text())


def test_8b_e31_template_validates_against_the_8b_contract(tmp_path: Path) -> None:
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    report = validate_eagle_head(
        head, expected_algorithm="eagle31", require_weights=False, target_contract_path=CONTRACT_8B
    )
    assert report["algorithm"] == "eagle31"
    assert report["aux_hidden_state_layer_ids"] == [2, 16, 29]
    assert report["hidden_size"] == 4096
    assert report["provenance"]["checked"] is True
    assert report["provenance"]["revision"] == "a411d838600baf0e3635a3daf66fb7c55fc97bb6"
    assert report["trained_head"] is False


def test_70b_templates_still_validate_against_the_recorded_70b_contract(tmp_path: Path) -> None:
    e31 = validate_eagle_head(
        _write_head(tmp_path, _load("70b/draft-e31-config.json"), name="e31"),
        expected_algorithm="eagle31",
        require_weights=False,
        target_contract_path=CONTRACT_70B,
    )
    e3 = validate_eagle_head(
        _write_head(tmp_path, _load("70b/draft-e3-config.json"), name="e3"),
        expected_algorithm="eagle3",
        require_weights=False,
        target_contract_path=CONTRACT_70B,
    )
    assert e31["aux_hidden_state_layer_ids"] == [2, 40, 77]
    assert e3["fc_norm"] is False and e3["norm_output"] is False


def test_8b_head_is_rejected_by_the_70b_contract(tmp_path: Path) -> None:
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    with pytest.raises(EagleHeadError, match="target hidden"):
        validate_eagle_head(head, require_weights=False, target_contract_path=CONTRACT_70B)


def test_70b_head_is_rejected_by_the_8b_contract(tmp_path: Path) -> None:
    head = _write_head(tmp_path, _load("70b/draft-e31-config.json"))
    with pytest.raises(EagleHeadError):
        validate_eagle_head(head, require_weights=False, target_contract_path=CONTRACT_8B)


def test_same_shape_head_for_another_revision_is_rejected(tmp_path: Path) -> None:
    config = _load("8b/draft-e31-config.json")
    config["apertus_target"] = {**config["apertus_target"], "revision": "0" * 40}
    head = _write_head(tmp_path, config)
    with pytest.raises(EagleHeadError, match="different target"):
        validate_eagle_head(head, require_weights=False, target_contract_path=CONTRACT_8B)


def test_same_shape_head_for_another_tokenizer_is_rejected(tmp_path: Path) -> None:
    config = _load("8b/draft-e31-config.json")
    config["apertus_target"] = {**config["apertus_target"], "tokenizer_sha256": "f" * 64}
    head = _write_head(tmp_path, config)
    with pytest.raises(EagleHeadError, match="tokenizer_sha256"):
        validate_eagle_head(head, require_weights=False, target_contract_path=CONTRACT_8B)


def test_head_without_provenance_is_rejected(tmp_path: Path) -> None:
    config = _load("8b/draft-e31-config.json")
    config.pop("apertus_target")
    head = _write_head(tmp_path, config)
    with pytest.raises(EagleHeadError, match="no apertus_target provenance"):
        validate_eagle_head(head, require_weights=False, target_contract_path=CONTRACT_8B)


def test_no_default_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APERTUS_EAGLE_CONTRACT", raising=False)
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    with pytest.raises(EagleHeadError, match="no target contract selected"):
        validate_eagle_head(head, require_weights=False)


def test_contract_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APERTUS_EAGLE_CONTRACT", str(CONTRACT_8B))
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    assert validate_eagle_head(head, require_weights=False)["hidden_size"] == 4096


def test_public_2509_head_is_rejected_by_path_and_by_config(tmp_path: Path) -> None:
    by_path = _write_head(
        tmp_path, _load("8b/draft-e31-config.json"), name="EAGLE3-Apertus-8B-Instruct-2509"
    )
    with pytest.raises(EagleHeadError, match="not a substitute"):
        validate_eagle_head(by_path, require_weights=False, target_contract_path=CONTRACT_8B)
    config = _load("8b/draft-e31-config.json")
    config["_name_or_path"] = "thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509"
    by_config = _write_head(tmp_path, config, name="renamed")
    with pytest.raises(EagleHeadError, match="2509"):
        validate_eagle_head(by_config, require_weights=False, target_contract_path=CONTRACT_8B)


def test_inconsistent_norm_flags_are_rejected() -> None:
    with pytest.raises(EagleHeadError, match="inconsistent"):
        infer_algorithm({"fc_norm": True, "norm_output": False})


def test_config_only_directory_is_never_a_trained_head(tmp_path: Path) -> None:
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    with pytest.raises(EagleHeadError, match="no weight files"):
        validate_eagle_head(head, require_weights=True, target_contract_path=CONTRACT_8B)


def test_manifest_digest_covers_weights(tmp_path: Path) -> None:
    head = _write_head(tmp_path, _load("8b/draft-e31-config.json"))
    (head / "model.safetensors").write_bytes(b"first")
    first = validate_eagle_head(head, target_contract_path=CONTRACT_8B)
    (head / "model.safetensors").write_bytes(b"second")
    second = validate_eagle_head(head, target_contract_path=CONTRACT_8B)
    assert first["trained_head"] is True
    assert first["config_sha256"] == second["config_sha256"]
    assert first["checkpoint_manifest_sha256"] != second["checkpoint_manifest_sha256"]


def test_target_model_must_be_the_contract_checkpoint() -> None:
    eight = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B"
    assert check_target_model(eight, CONTRACT_8B) == eight
    with pytest.raises(EagleHeadError, match="not the contract checkpoint"):
        check_target_model(eight.replace("8B", "70B"), CONTRACT_8B)
