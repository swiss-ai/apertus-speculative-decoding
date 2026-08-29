import json
from pathlib import Path

import pytest

from apertus_bench.correctness import compare_capture_files


def capture(model: str, output: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": model,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 1},
        "cases": [
            {
                "prompt_id": "p1",
                "workload": "chat",
                "measurement": {
                    "success": True,
                    "output": output,
                    "output_sha256": output,
                },
            }
        ],
    }


def test_compare_capture_files(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(capture("baseline", "same")))
    candidate.write_text(json.dumps(capture("candidate", "same")))
    report = compare_capture_files(baseline, candidate, output)
    assert report["all_exact"] is True
    assert json.loads(output.read_text())["exact_matches"] == 1


def test_compare_capture_files_rejects_different_sampling(tmp_path: Path) -> None:
    baseline_record = capture("baseline", "same")
    candidate_record = capture("candidate", "same")
    candidate_record["sampling"]["seed"] = 2  # type: ignore[index]
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(baseline_record))
    candidate.write_text(json.dumps(candidate_record))
    with pytest.raises(ValueError, match="sampling"):
        compare_capture_files(baseline, candidate, tmp_path / "out.json")
