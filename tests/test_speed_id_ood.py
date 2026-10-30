import copy
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/serving-pilot/report-speed-id-ood.py"
SPEC = importlib.util.spec_from_file_location("speed_report", SOURCE)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def pair():
    plain = {
        "requests": {"success_rate": 1, "successful": 32},
        "warmup": {"failed": 0},
        "tokens": {"completion": 8192},
        "speculative_decoding": {"enabled": False},
    }
    draft = copy.deepcopy(plain)
    draft["speculative_decoding"] = {"enabled": True, "draft_tokens": 100}
    metadata = {
        "workload_file_sha256": "corpus",
        "prompt_ids": ["one", "two"],
        "cell": {
            "workload": "id-opb",
            "concurrency": 1,
            "requests": 32,
            "warmup_requests": 1,
            "generation": {"ignore_eos": True},
        },
        "extra": dict.fromkeys(
            ["source_revision", "physical_gpu", "target_revision", "corpus_sha256", "engine"],
            "same",
        ),
    }
    return plain, draft, metadata, copy.deepcopy(metadata)


@pytest.mark.parametrize(
    "mismatch", ["prompt_ids", "source_revision", "physical_gpu", "generation"]
)
def test_rejects_unmatched_measurements(mismatch):
    plain, draft, pm, dm = pair()
    if mismatch == "prompt_ids":
        dm[mismatch] = ["two", "one"]
    elif mismatch == "generation":
        dm["cell"][mismatch] = {"ignore_eos": False}
    else:
        dm["extra"][mismatch] = "different"
    with pytest.raises(ValueError, match="unmatched"):
        REPORT.validate_pair(plain, draft, pm, dm, True)


def test_rejects_short_fixed_generations_and_inactive_drafter():
    plain, draft, pm, dm = pair()
    draft["tokens"]["completion"] -= 1
    with pytest.raises(ValueError, match="exactly 256"):
        REPORT.validate_pair(plain, draft, pm, dm, True)
    draft["tokens"]["completion"] += 1
    draft["speculative_decoding"]["draft_tokens"] = 0
    with pytest.raises(ValueError, match="inactive"):
        REPORT.validate_pair(plain, draft, pm, dm, True)


def test_bootstrap_uses_paired_deployments_and_retains_slowdown():
    low, high = REPORT.interval([0.8, 0.85, 0.9])
    assert 0.8 <= low <= high <= 0.9
    assert high < 1
    assert REPORT.interval([2.0, 2.0, 2.0]) == [2.0, 2.0]


def test_missing_cells_are_not_reported_as_no_speedup(tmp_path):
    result = REPORT.report(tmp_path)
    assert result["complete"] is False
    assert result["rows"] == []
    assert len(result["missing"]) == 45
