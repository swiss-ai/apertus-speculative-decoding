import json
from pathlib import Path

import pytest

from apertus_bench.correctness import (
    GateTolerance,
    calibrate_gate,
    case_divergence,
    common_prefix_length,
    compare_capture_files,
    compare_captures,
    edit_distance,
    run_correctness_gate,
)


def capture(
    model: str, output: str, *, base_url: str = "http://node:8080", tokens: int = 4
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": model,
        "base_url": base_url,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 1},
        "cases": [
            {
                "prompt_id": "p1",
                "workload": "chat",
                "measurement": {
                    "success": True,
                    "output": output,
                    "output_sha256": output,
                    "completion_tokens": tokens,
                },
            }
        ],
    }


def write(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record))
    return path


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


def test_edit_distance_ignores_shared_prefix_and_suffix() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("the same start XY the same end", "the same start Z the same end") == 2
    assert common_prefix_length("the same start XY", "the same start Z") == 15


def test_case_divergence_measures_a_late_divergence() -> None:
    divergence = case_divergence(
        {"success": True, "output": "a" * 100 + "left", "completion_tokens": 30},
        {"success": True, "output": "a" * 100 + "right", "completion_tokens": 31},
    )
    assert divergence["exact_match"] is False
    assert divergence["common_prefix_characters"] == 100
    assert divergence["common_prefix_fraction"] == pytest.approx(100 / 104)
    assert divergence["normalized_edit_distance"] == pytest.approx(4 / 105)
    assert divergence["completion_token_difference"] == 1


def test_case_divergence_reports_a_failed_request_as_incomparable() -> None:
    divergence = case_divergence(
        {"success": True, "output": "text", "completion_tokens": 2},
        {"success": False, "output": None, "completion_tokens": None},
    )
    assert divergence["comparable"] is False
    assert divergence["normalized_edit_distance"] is None


def test_two_deployments_of_one_configuration_are_not_a_correctness_violation() -> None:
    """The failure that motivated the gate: identical configurations diverging on both arms."""
    prefix = "identical opening sentence. "
    treatment = compare_captures(
        capture("baseline", prefix + "the baseline continues here"),
        capture("draft-repeat-1", prefix + "the speculative arm continues"),
    )
    control = compare_captures(
        capture("draft-repeat-1", prefix + "the speculative arm continues", base_url="http://a"),
        capture("draft-repeat-2", prefix + "the very same config diverges", base_url="http://b"),
    )
    assert treatment["all_exact"] is False
    assert control["all_exact"] is False

    report = calibrate_gate(treatment, [control])
    assert report["passed"] is True
    assert report["verdict"] == "within_control_envelope"
    assert "not a proof" in str(report["claim"])


def test_divergence_beyond_the_control_envelope_fails_the_gate() -> None:
    control = compare_captures(
        capture("draft-repeat-1", "a stable shared opening, then a small tail difference one"),
        capture("draft-repeat-2", "a stable shared opening, then a small tail difference two"),
    )
    treatment = compare_captures(
        capture("baseline", "a stable shared opening, then a small tail difference one"),
        capture("suspect-draft", "wholly different text from the very first token onwards"),
    )
    report = calibrate_gate(treatment, [control])
    assert report["passed"] is False
    assert report["verdict"] == "exceeds_control_envelope"
    failed = [check["name"] for check in report["checks"] if not check["passed"]]
    assert "minimum_common_prefix_fraction" in failed
    assert "maximum_normalized_edit_distance" in failed


def test_gate_requires_a_calibration_control() -> None:
    treatment = compare_captures(capture("baseline", "same"), capture("candidate", "same"))
    with pytest.raises(ValueError, match="calibration control"):
        calibrate_gate(treatment, [])


def test_gate_requires_a_cross_deployment_control_for_a_cross_deployment_pair() -> None:
    treatment = compare_captures(
        capture("baseline", "text", base_url="http://a"),
        capture("draft", "text", base_url="http://b"),
    )
    same_deployment_control = compare_captures(
        capture("draft", "text", base_url="http://b"),
        capture("draft", "text", base_url="http://b"),
    )
    assert same_deployment_control["same_deployment"] is True
    with pytest.raises(ValueError, match="cross-deployment"):
        calibrate_gate(treatment, [same_deployment_control])


def test_run_correctness_gate_writes_an_artifact(tmp_path: Path) -> None:
    baseline = write(tmp_path / "baseline.json", capture("baseline", "shared opening one"))
    repeat_one = write(
        tmp_path / "draft-1.json", capture("draft-1", "shared opening two", base_url="http://a")
    )
    repeat_two = write(
        tmp_path / "draft-2.json", capture("draft-2", "shared opening three", base_url="http://b")
    )
    output = tmp_path / "gate.json"
    report = run_correctness_gate(
        (baseline, repeat_one),
        [(repeat_one, repeat_two)],
        output,
        GateTolerance(completion_token_difference_margin=0),
    )
    written = json.loads(output.read_text())
    assert written["criterion"] == "calibrated_greedy_divergence"
    assert written["treatment"]["captures"] == [str(baseline), str(repeat_one)]
    assert written["controls"][0]["captures"] == [str(repeat_one), str(repeat_two)]
    assert written["passed"] is report["passed"]
