from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apertus_bench.client import StreamingChatClient
from apertus_bench.stats import distribution
from apertus_bench.workloads import Prompt

GATE_CRITERION = "calibrated_greedy_divergence"

PASS_CLAIM = (
    "greedy divergence between the two arms stays inside the divergence measured between "
    "independently launched deployments of a single configuration, so this run shows no "
    "divergence beyond deployment-level numerical nondeterminism; that is consistent with "
    "losslessness and is not a proof of it"
)
FAIL_CLAIM = (
    "greedy divergence between the two arms exceeds the calibrated same-configuration "
    "envelope, which is evidence of a real distributional violation rather than "
    "deployment-level numerical nondeterminism"
)


@dataclass(frozen=True)
class GateTolerance:
    """How far a compared pair may fall outside the calibration control and still pass."""

    normalized_edit_distance_margin: float = 0.05
    common_prefix_fraction_margin: float = 0.05
    completion_token_difference_margin: int = 2


DEFAULT_TOLERANCE = GateTolerance()


def common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        length += 1
    return length


def edit_distance(left: str, right: str) -> int:
    """Character-level Levenshtein distance, trimming the shared prefix and suffix first."""
    prefix = common_prefix_length(left, right)
    left, right = left[prefix:], right[prefix:]
    suffix = common_prefix_length(left[::-1], right[::-1])
    if suffix:
        left, right = left[:-suffix], right[:-suffix]
    if not left or not right:
        return len(left) + len(right)
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def case_divergence(
    baseline_measurement: dict[str, Any], candidate_measurement: dict[str, Any]
) -> dict[str, Any]:
    """Per-prompt divergence between two greedy generations of the same prompt."""
    comparable = bool(baseline_measurement.get("success")) and bool(
        candidate_measurement.get("success")
    )
    if not comparable:
        return {
            "comparable": False,
            "exact_match": False,
            "common_prefix_characters": None,
            "common_prefix_fraction": None,
            "normalized_edit_distance": None,
            "completion_token_difference": None,
        }
    baseline_output = baseline_measurement.get("output") or ""
    candidate_output = candidate_measurement.get("output") or ""
    prefix = common_prefix_length(baseline_output, candidate_output)
    shortest = min(len(baseline_output), len(candidate_output))
    longest = max(len(baseline_output), len(candidate_output))
    baseline_tokens = baseline_measurement.get("completion_tokens")
    candidate_tokens = candidate_measurement.get("completion_tokens")
    token_difference = (
        abs(int(baseline_tokens) - int(candidate_tokens))
        if baseline_tokens is not None and candidate_tokens is not None
        else None
    )
    return {
        "comparable": True,
        "exact_match": baseline_output == candidate_output,
        "common_prefix_characters": prefix,
        "common_prefix_fraction": prefix / shortest if shortest else 1.0,
        "normalized_edit_distance": (
            edit_distance(baseline_output, candidate_output) / longest if longest else 0.0
        ),
        "completion_token_difference": token_difference,
    }


def _values(divergences: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(divergence[key])
        for divergence in divergences
        if divergence["comparable"] and divergence[key] is not None
    ]


def divergence_summary(divergences: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-prompt divergences into distributions and worst observed values."""
    prefix_characters = _values(divergences, "common_prefix_characters")
    prefix_fractions = _values(divergences, "common_prefix_fraction")
    edit_distances = _values(divergences, "normalized_edit_distance")
    token_differences = _values(divergences, "completion_token_difference")
    return {
        "comparable_cases": sum(1 for divergence in divergences if divergence["comparable"]),
        "divergence": {
            "common_prefix_characters": distribution(prefix_characters),
            "common_prefix_fraction": distribution(prefix_fractions),
            "normalized_edit_distance": distribution(edit_distances),
            "completion_token_difference": distribution(token_differences),
        },
        "worst_case": {
            "minimum_common_prefix_characters": min(prefix_characters, default=None),
            "minimum_common_prefix_fraction": min(prefix_fractions, default=None),
            "maximum_normalized_edit_distance": max(edit_distances, default=None),
            "maximum_completion_token_difference": max(token_differences, default=None),
        },
    }


async def capture_greedy_outputs(
    client: StreamingChatClient,
    prompts: list[Prompt],
    output_path: Path,
    *,
    max_tokens: int | None,
    seed: int,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        result = await client.request(
            prompt,
            index,
            temperature=0.0,
            top_p=1.0,
            seed=seed + index,
            max_tokens=max_tokens,
            ignore_eos=False,
            store_output=True,
        )
        cases.append(
            {
                "prompt_id": prompt.id,
                "workload": prompt.workload,
                "measurement": result.to_dict(),
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "model": client.model,
        "base_url": client.base_url,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": seed},
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_captures(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare two greedy captures, keeping exact match alongside the divergence statistics."""
    if baseline["sampling"] != candidate["sampling"]:
        raise ValueError("capture sampling settings do not match")
    baseline_cases = {case["prompt_id"]: case for case in baseline["cases"]}
    candidate_cases = {case["prompt_id"]: case for case in candidate["cases"]}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("capture prompt ids do not match")

    cases: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    for prompt_id in baseline_cases:
        baseline_measurement = baseline_cases[prompt_id]["measurement"]
        candidate_measurement = candidate_cases[prompt_id]["measurement"]
        divergence = case_divergence(baseline_measurement, candidate_measurement)
        divergences.append(divergence)
        cases.append(
            {
                "prompt_id": prompt_id,
                "workload": baseline_cases[prompt_id]["workload"],
                "exact_match": divergence["exact_match"],
                "baseline_sha256": baseline_measurement.get("output_sha256"),
                "candidate_sha256": candidate_measurement.get("output_sha256"),
                "divergence": divergence,
            }
        )
    exact_matches = sum(bool(case["exact_match"]) for case in cases)
    same_deployment = baseline["model"] == candidate["model"] and baseline.get(
        "base_url"
    ) == candidate.get("base_url")
    return {
        "schema_version": 2,
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "baseline_base_url": baseline.get("base_url"),
        "candidate_base_url": candidate.get("base_url"),
        "same_deployment": same_deployment,
        "sampling": baseline["sampling"],
        "total_cases": len(cases),
        "exact_matches": exact_matches,
        "all_exact": exact_matches == len(cases),
        **divergence_summary(divergences),
        "cases": cases,
    }


def compare_capture_files(
    baseline_path: Path, candidate_path: Path, output_path: Path
) -> dict[str, object]:
    report = compare_captures(
        json.loads(baseline_path.read_text()), json.loads(candidate_path.read_text())
    )
    report["baseline_capture"] = str(baseline_path)
    report["candidate_capture"] = str(candidate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _pair_label(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_model": comparison["baseline_model"],
        "candidate_model": comparison["candidate_model"],
        "same_deployment": comparison["same_deployment"],
        "total_cases": comparison["total_cases"],
        "comparable_cases": comparison["comparable_cases"],
        "exact_matches": comparison["exact_matches"],
        "worst_case": comparison["worst_case"],
    }


def _control_envelope(controls: list[dict[str, Any]]) -> dict[str, Any]:
    prefix_fractions = [
        control["worst_case"]["minimum_common_prefix_fraction"]
        for control in controls
        if control["worst_case"]["minimum_common_prefix_fraction"] is not None
    ]
    edit_distances = [
        control["worst_case"]["maximum_normalized_edit_distance"]
        for control in controls
        if control["worst_case"]["maximum_normalized_edit_distance"] is not None
    ]
    token_differences = [
        control["worst_case"]["maximum_completion_token_difference"]
        for control in controls
        if control["worst_case"]["maximum_completion_token_difference"] is not None
    ]
    return {
        "pairs": len(controls),
        "cross_deployment_pairs": sum(1 for control in controls if not control["same_deployment"]),
        "comparable_cases": sum(control["comparable_cases"] for control in controls),
        "exact_matches": sum(control["exact_matches"] for control in controls),
        "minimum_common_prefix_fraction": min(prefix_fractions, default=None),
        "maximum_normalized_edit_distance": max(edit_distances, default=None),
        "maximum_completion_token_difference": max(token_differences, default=None),
    }


def _check(
    name: str, treatment: float | None, allowance: float | None, *, passed: bool
) -> dict[str, Any]:
    return {"name": name, "treatment": treatment, "allowance": allowance, "passed": passed}


def calibrate_gate(
    treatment: dict[str, Any],
    controls: list[dict[str, Any]],
    tolerance: GateTolerance = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Judge one arm-vs-arm comparison against same-configuration control comparisons.

    Exact greedy agreement cannot be required across independently launched deployments, so the
    criterion is relative: the compared pair may diverge as much as two deployments of one
    configuration diverge, plus a declared margin. Without a control the gate has no null
    distribution and refuses to decide.
    """
    if not controls:
        raise ValueError(
            "a calibration control comparison is required: compare two captures of one "
            "configuration so cross-deployment nondeterminism is measured rather than assumed"
        )
    envelope = _control_envelope(controls)
    if not treatment["same_deployment"] and not envelope["cross_deployment_pairs"]:
        raise ValueError(
            "a cross-deployment calibration control is required to judge a cross-deployment "
            "comparison; same-deployment controls do not measure node-to-node nondeterminism"
        )

    worst = treatment["worst_case"]
    prefix_allowance = envelope["minimum_common_prefix_fraction"]
    if prefix_allowance is not None:
        prefix_allowance = max(0.0, prefix_allowance - tolerance.common_prefix_fraction_margin)
    edit_allowance = envelope["maximum_normalized_edit_distance"]
    if edit_allowance is not None:
        edit_allowance += tolerance.normalized_edit_distance_margin
    token_allowance = envelope["maximum_completion_token_difference"]
    if token_allowance is not None:
        token_allowance += tolerance.completion_token_difference_margin

    checks = [
        _check(
            "all_cases_comparable",
            treatment["comparable_cases"],
            treatment["total_cases"],
            passed=treatment["comparable_cases"] == treatment["total_cases"],
        ),
        _check(
            "minimum_common_prefix_fraction",
            worst["minimum_common_prefix_fraction"],
            prefix_allowance,
            passed=(
                worst["minimum_common_prefix_fraction"] is not None
                and prefix_allowance is not None
                and worst["minimum_common_prefix_fraction"] >= prefix_allowance
            ),
        ),
        _check(
            "maximum_normalized_edit_distance",
            worst["maximum_normalized_edit_distance"],
            edit_allowance,
            passed=(
                worst["maximum_normalized_edit_distance"] is not None
                and edit_allowance is not None
                and worst["maximum_normalized_edit_distance"] <= edit_allowance
            ),
        ),
        _check(
            "maximum_completion_token_difference",
            worst["maximum_completion_token_difference"],
            token_allowance,
            passed=(
                worst["maximum_completion_token_difference"] is not None
                and token_allowance is not None
                and worst["maximum_completion_token_difference"] <= token_allowance
            ),
        ),
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "criterion": GATE_CRITERION,
        "tolerance": asdict(tolerance),
        "treatment": _pair_label(treatment),
        "controls": [_pair_label(control) for control in controls],
        "control_envelope": envelope,
        "checks": checks,
        "passed": passed,
        "verdict": "within_control_envelope" if passed else "exceeds_control_envelope",
        "claim": PASS_CLAIM if passed else FAIL_CLAIM,
    }


def run_correctness_gate(
    treatment: tuple[Path, Path],
    controls: list[tuple[Path, Path]],
    output_path: Path,
    tolerance: GateTolerance = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the comparisons behind the gate from captures, then judge them."""

    def comparison(pair: tuple[Path, Path]) -> dict[str, Any]:
        baseline_path, candidate_path = pair
        report = compare_captures(
            json.loads(baseline_path.read_text()), json.loads(candidate_path.read_text())
        )
        report["baseline_capture"] = str(baseline_path)
        report["candidate_capture"] = str(candidate_path)
        return report

    treatment_comparison = comparison(treatment)
    control_comparisons = [comparison(pair) for pair in controls]
    report = calibrate_gate(treatment_comparison, control_comparisons, tolerance)
    report["treatment"]["captures"] = [str(path) for path in treatment]
    for control, pair in zip(report["controls"], controls, strict=True):
        control["captures"] = [str(path) for path in pair]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


async def compare_greedy_outputs(
    baseline: StreamingChatClient,
    candidate: StreamingChatClient,
    prompts: list[Prompt],
    output_path: Path,
    *,
    max_tokens: int | None,
    seed: int,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    divergences: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        request_seed = seed + index
        baseline_result = await baseline.request(
            prompt,
            index,
            temperature=0.0,
            top_p=1.0,
            seed=request_seed,
            max_tokens=max_tokens,
            ignore_eos=False,
            store_output=True,
        )
        candidate_result = await candidate.request(
            prompt,
            index,
            temperature=0.0,
            top_p=1.0,
            seed=request_seed,
            max_tokens=max_tokens,
            ignore_eos=False,
            store_output=True,
        )
        divergence = case_divergence(baseline_result.to_dict(), candidate_result.to_dict())
        divergences.append(divergence)
        cases.append(
            {
                "prompt_id": prompt.id,
                "workload": prompt.workload,
                "exact_match": divergence["exact_match"],
                "divergence": divergence,
                "baseline": baseline_result.to_dict(),
                "candidate": candidate_result.to_dict(),
            }
        )

    exact_matches = sum(bool(case["exact_match"]) for case in cases)
    report: dict[str, object] = {
        "schema_version": 2,
        "baseline_model": baseline.model,
        "candidate_model": candidate.model,
        "baseline_base_url": baseline.base_url,
        "candidate_base_url": candidate.base_url,
        "same_deployment": baseline.model == candidate.model
        and baseline.base_url == candidate.base_url,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": seed},
        "total_cases": len(cases),
        "exact_matches": exact_matches,
        "all_exact": exact_matches == len(cases),
        **divergence_summary(divergences),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
