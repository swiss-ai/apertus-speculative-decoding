import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/train"))
campaign = importlib.import_module("depth_campaign")


def test_frozen_source_changes_block_campaign(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "example.py"
    script.write_text("original")
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(
            {
                "files": {"example.py": campaign.recovery.sha256(script)},
                "training_files": {},
            }
        )
    )
    campaign.verify_sources(tmp_path)
    script.write_text("changed")
    with pytest.raises(ValueError, match="frozen source changed"):
        campaign.verify_sources(tmp_path)


def test_recovery_identity_comes_from_exact_new_training_submission(tmp_path: Path) -> None:
    script = tmp_path / "train.sbatch"
    script.write_text(
        "export DSPARK_DIR=/task/data\nexport RUN_ID=fresh-run\n"
        "export DATA=/task/data/data\nexport FROM_PRETRAINED=/task/rebound\n"
        'echo "START git=abcdef123"\n'
    )
    values, revision = campaign.submitted_training(script)
    assert values["RUN_ID"] == "fresh-run"
    assert values["FROM_PRETRAINED"] == "/task/rebound"
    assert revision == "abcdef123"


async def test_preflight_failure_never_starts_serving(tmp_path: Path, monkeypatch) -> None:
    from apertus_bench import node_sweep

    async def failed_preflight(*_):
        return {"status": "failed"}

    async def forbidden_serving(*_):
        pytest.fail("serving started despite failed preflight")

    monkeypatch.setattr(node_sweep, "run_arm", failed_preflight)
    monkeypatch.setattr(node_sweep, "run_plan", forbidden_serving)
    spec = {
        "arms": [
            {"name": "preflight", "method": "command", "gpus": 4, "command": ["false"]},
            {"name": "plain", "method": "baseline"},
        ],
        "loadtests": [{"workload": "opb", "concurrencies": [1]}],
    }
    with pytest.raises(ValueError, match="no serving arms started"):
        await campaign.run_spec(spec, tmp_path / "run")
    assert json.loads((tmp_path / "run/sweep-status.json").read_text()) == {"preflight": "failed"}


def _captures(root: Path, arm: str, text: str) -> None:
    directory = root / arm / "correctness"
    directory.mkdir(parents=True)
    for repeat in (1, 2):
        value = {
            "model": arm,
            "base_url": "http://127.0.0.1:8100",
            "deployment_id": f"{root.name}-{arm}",
            "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 1},
            "cases": [
                {
                    "prompt_id": str(n),
                    "workload": "opb",
                    "measurement": {
                        "success": True,
                        "output": text,
                        "completion_tokens": 4,
                    },
                }
                for n in range(6)
            ],
        }
        (directory / f"capture-{repeat}.json").write_text(json.dumps(value))


def test_calibration_requires_success_and_does_not_borrow_other_configs(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        _captures(root, "plain", "target")
        _captures(root, "depth-l1-k7", "target")
    reports = campaign.calibrated_checks(a, b, ["depth-l1-k7"], tmp_path / "pass")
    assert all(report["passed"] for report in reports)
    assert all(report["control_envelope"]["cross_deployment_pairs"] == 4 for report in reports)
    # Identical wrong candidate output has a tight candidate/candidate control;
    # it must still fail when it diverges from the target beyond fixed margins.
    for root in (a, b):
        for path in campaign.capture_paths(root, "depth-l1-k7"):
            value = json.loads(path.read_text())
            for case in value["cases"]:
                case["measurement"]["output"] = "WRONG"
            path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="existing protocol margins"):
        campaign.calibrated_checks(a, b, ["depth-l1-k7"], tmp_path / "fail")


def test_training_recipe_checks_actual_command_and_patch(tmp_path: Path) -> None:
    original = Path(__file__).resolve().parents[1] / (
        "results/8b/small-drafter/recovery-20261027/evidence/recovery-20261027/"
        "depth-l5-scratch/checkpoints"
    )
    checkpoint = tmp_path / "epoch0_end"
    checkpoint.mkdir()
    (tmp_path / "run.yaml").write_text((original / "run.yaml").read_text())
    command = (
        (original / "train_command.txt")
        .read_text()
        .replace("--num-layers 5", "--from-pretrained /warm-start")
    )
    (tmp_path / "train_command.txt").write_text(command)
    (tmp_path / "speculators.patch").write_text("the exact training patch")
    manifest = {
        "data": "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/data",
        "warm_start": "/warm-start",
        "speculators_revision": "7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00",
        "speculators_patch_sha256": campaign.recovery.sha256(tmp_path / "speculators.patch"),
        "training_files": {},
    }
    campaign.check_training_recipe(checkpoint, manifest)
    (tmp_path / "speculators.patch").write_text("different source")
    with pytest.raises(ValueError, match="source diff changed"):
        campaign.check_training_recipe(checkpoint, manifest)
