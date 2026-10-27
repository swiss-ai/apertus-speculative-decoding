import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "methods/dspark/train/depth_recovery.py"
SPEC = importlib.util.spec_from_file_location("depth_recovery", PATH)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_plan_covers_only_two_trained_variants_and_existing_references() -> None:
    a, b = recovery.specs("candidate", "control", "e10", "eagle", "prompts")
    assert len(a["arms"]) == 25
    assert a["arms"][0]["name"] == b["arms"][0]["name"] == "preflight"
    assert b["arms"][1:] == list(reversed(a["arms"][1:]))
    for depth in ("depth-l1", "depth-l5"):
        assert {r["k"] for r in a["arms"] if r["name"].startswith(depth)} == set(range(1, 8))
    assert a["loadtests"][0]["concurrencies"] == [1, 8, 32, 64, 128, 256]


def test_preflight_rejects_incomplete_training(tmp_path: Path) -> None:
    (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 5000}))
    with pytest.raises(ValueError, match="6200"):
        recovery.checkpoint_evidence(tmp_path, 1)


def test_preflight_rejects_nonfinite_validation(tmp_path: Path) -> None:
    (tmp_path / "training_state.json").write_text(json.dumps({"global_step": 6200}))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "transformer_layer_config": {"num_hidden_layers": 1},
                "aux_hidden_state_layer_ids": [1, 8, 15, 22, 29],
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    (tmp_path / "val_metrics.json").write_text(json.dumps({"eal_epoch": float("nan")}))
    with pytest.raises(ValueError, match="nonfinite"):
        recovery.checkpoint_evidence(tmp_path, 1)


def test_evidence_outputs_cannot_overwrite_prior_reports(tmp_path: Path) -> None:
    p = tmp_path / "report.json"
    recovery.write_new(p, {"first": True})
    with pytest.raises(FileExistsError):
        recovery.write_new(p, {"first": False})
    assert json.loads(p.read_text()) == {"first": True}


def _sweep(root: Path, ratio: float) -> None:
    a, _ = recovery.specs("candidate", "control", "e10", "eagle", "prompts")
    root.mkdir()
    (root / "sweep-status.json").write_text(json.dumps({r["name"]: "ok" for r in a["arms"]}))
    for arm in a["arms"]:
        if arm["name"] == "preflight":
            continue
        d = root / arm["name"]
        d.mkdir()
        (d / "status.json").write_text(json.dumps({"status": "ok", "gpus": [0]}))
        for c in recovery.LEVELS:
            p = d / "opb/opb" / f"c{c}"
            p.mkdir(parents=True)
            speed = 100 * ratio if arm["name"].startswith("depth-l1") else 100
            (p / "summary.json").write_text(
                json.dumps(
                    {"requests": {"success_rate": 1}, "tokens": {"output_tokens_per_second": speed}}
                )
            )


def test_decision_requires_repeatable_high_load_gain(tmp_path: Path) -> None:
    _sweep(tmp_path / "a", 1.06)
    _sweep(tmp_path / "b", 1.01)
    assert not recovery.decision(tmp_path / "a", tmp_path / "b")[
        "gate_g3_repeatable_throughput_passed"
    ]


def test_failed_cells_cannot_be_used_for_the_gate(tmp_path: Path) -> None:
    _sweep(tmp_path / "a", 1.1)
    p = tmp_path / "a/depth-l1-k1/opb/opb/c256/summary.json"
    data = json.loads(p.read_text())
    data["requests"]["success_rate"] = 0.98
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="99%"):
        recovery.read_sweep(tmp_path / "a")
