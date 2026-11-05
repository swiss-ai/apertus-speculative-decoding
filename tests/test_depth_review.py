import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "methods/dspark/train/depth_review.py"
SPEC = importlib.util.spec_from_file_location("depth_review", PATH)
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


def make_runs(
    tmp_path: Path,
    speeds: list[dict] | None = None,
    tpot: tuple = (1.0, 1.0),
    request: tuple = (1.0, 1.0),
    ttft: float = 1.0,
    tail: float = 1.0,
) -> tuple[Path, Path, Path]:
    arms = {"preflight", "plain", "eagle-e10-k3", "eagle-e10-k7"} | {
        f"{name}-k{k}" for name in ("dspark-e10", "depth-l1", "depth-l5") for k in reviewer.K
    }
    roots = [tmp_path / "a", tmp_path / "b"]
    speeds = speeds or [{"depth-l1-k1": 1060}, {"depth-l1-k1": 1060}]
    for number, root in enumerate(roots):
        root.mkdir()
        (root / "sweep-status.json").write_text(json.dumps(dict.fromkeys(arms, "ok")))
        for arm in arms - {"preflight"}:
            (root / arm).mkdir()
            (root / arm / "status.json").write_text(
                json.dumps({"status": "ok", "gpus": [0], "deployment_id": f"{number}-{arm}"})
            )
            speed = speeds[number].get(arm, 1000 if arm.startswith("dspark-e10") else 500)
            for concurrency in reviewer.LEVELS:
                cell = root / arm / "opb/opb" / f"c{concurrency}"
                cell.mkdir(parents=True)
                candidate = arm == "depth-l1-k1"
                latencies = {}
                for metric, values in {
                    "tpot": (2, 2, 3, 5),
                    "e2e": (1000, 900, 1200, 1400),
                    "ttft": (10, 10, 15, 20),
                }.items():
                    factor = 1
                    if candidate:
                        factor = {"tpot": tpot[number], "e2e": request[number], "ttft": ttft}[
                            metric
                        ]
                    latencies[metric] = {
                        "count": 128,
                        **{
                            key: value
                            * factor
                            * (tail if candidate and key in {"p95", "p99"} else 1)
                            for key, value in zip(reviewer.STATS, values, strict=True)
                        },
                    }
                summary = {
                    "requests": {
                        "success_rate": 1,
                        "successful": 128,
                        "requests_per_second": speed / 500,
                    },
                    "tokens": {"output_tokens_per_second": speed, "completion": 64000},
                    "latency_ms": latencies,
                }
                (cell / "summary.json").write_text(json.dumps(summary))
    # Recompute the scientific result with the unchanged original criterion.
    sweeps = [reviewer.read_sweep(root) for root in roots]
    passed = any(row["passed"] for row in reviewer.original_g3(sweeps))
    raw = tmp_path / "raw-decision.json"
    raw.write_text(
        json.dumps(
            {
                "gate_g3_repeatable_throughput_passed": passed,
                "calibrated_correctness_passed": True,
                "verdict": "substantial_gain_on_declared_workload" if passed else "negative",
            }
        )
    )
    return *roots, raw


def test_five_percent_scientific_pass_is_not_an_implementation_endorsement(tmp_path: Path) -> None:
    a, b, raw = make_runs(tmp_path, tpot=(0.94, 0.94), request=(0.94, 0.94))
    before = raw.read_bytes()
    report = reviewer.review(a, b, raw)
    assert report["original_decision"]["original_G3_passed"]
    assert not report["implementation_evidence_passed"]
    assert (
        report["implementation_verdict"]
        == "original_G3_pass_only_no_substantial_implementation_evidence"
    )
    assert raw.read_bytes() == before


def test_switching_winning_k_between_repeats_cannot_pass_fixed_pair_screen(tmp_path: Path) -> None:
    a, b, raw = make_runs(
        tmp_path,
        [{"depth-l1-k1": 1120, "depth-l1-k2": 1040}, {"depth-l1-k1": 1040, "depth-l1-k2": 1120}],
        tpot=(0.8, 0.8),
        request=(0.8, 0.8),
    )
    report = reviewer.review(a, b, raw)
    assert report["original_decision"]["original_G3_passed"]
    assert not report["implementation_evidence_passed"]
    assert all(row["same_fixed_k_pair_in_both_repeats"] for row in report["fixed_pair_rows"])


def test_substantial_gains_with_ttft_and_tail_costs_are_conditional(tmp_path: Path) -> None:
    a, b, raw = make_runs(
        tmp_path, [{"depth-l1-k1": 1250}] * 2, tpot=(0.8, 0.8), ttft=1.25, tail=1.5
    )
    report = reviewer.review(a, b, raw)
    assert report["implementation_evidence_passed"]
    assert (
        report["implementation_verdict"]
        == "substantial_but_conditional_on_recorded_TTFT_or_tail_costs"
    )
    row = next(
        row
        for row in report["fixed_pair_rows"]
        if row["depth"] == "depth-l1" and row["concurrency"] == 128 and row["candidate_k"] == 1
    )
    assert row["ttft_and_tail_tradeoffs"]
    delta = row["repeats"][0]["deltas"]
    assert delta["ttft_ms.mean"]["delta_candidate_minus_reference"] == 2.5
    assert delta["gpu_seconds_per_million_tokens"]["delta_candidate_minus_reference"] == -200
    assert delta["throughput_tokens_per_s"]["delta_candidate_minus_reference"] == 250


def test_latency_gain_must_use_same_metric_in_both_repeats(tmp_path: Path) -> None:
    a, b, raw = make_runs(
        tmp_path, [{"depth-l1-k1": 1250}] * 2, tpot=(0.8, 0.95), request=(0.95, 0.8)
    )
    assert not reviewer.review(a, b, raw)["implementation_evidence_passed"]


def test_weak_fixed_reference_cannot_manufacture_a_gain(tmp_path: Path) -> None:
    a, b, raw = make_runs(
        tmp_path,
        [
            {
                **{f"dspark-e10-k{k}": 800 for k in reviewer.K},
                "dspark-e10-k1": 1000,
                "depth-l1-k1": 1200,
                "depth-l1-k2": 1300,
            },
            {
                **{f"dspark-e10-k{k}": 800 for k in reviewer.K},
                "dspark-e10-k2": 1000,
                "depth-l1-k1": 950,
                "depth-l1-k2": 1300,
            },
        ],
        tpot=(0.8, 0.8),
    )
    report = reviewer.review(a, b, raw)
    row = next(
        row
        for row in report["fixed_pair_rows"]
        if row["depth"] == "depth-l1" and row["concurrency"] == 128 and row["candidate_k"] == 1
    )
    assert row["reference_k"] == 1
    assert row["throughput_10pct_both_repeats"]
    assert not row["also_10pct_vs_best_reference_in_each_repeat"]
    assert not row["implementation_evidence_passed"]


def test_original_negative_result_stays_negative(tmp_path: Path) -> None:
    a, b, raw = make_runs(tmp_path, [{"depth-l1-k1": 1010}] * 2)
    report = reviewer.review(a, b, raw)
    assert not report["original_decision"]["original_G3_passed"]
    assert report["implementation_verdict"] == "negative_original_G3_no_implementation_endorsement"


def test_missing_correctness_cannot_be_endorsed(tmp_path: Path) -> None:
    a, b, raw = make_runs(tmp_path)
    value = json.loads(raw.read_text())
    value["calibrated_correctness_passed"] = False
    raw.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="correctness evidence"):
        reviewer.review(a, b, raw)


def test_review_artifacts_are_write_once(tmp_path: Path, monkeypatch) -> None:
    a, b, raw = make_runs(
        tmp_path, [{"depth-l1-k1": 1250}] * 2, tpot=(0.8, 0.8), request=(0.8, 0.8)
    )
    output = tmp_path / "review/report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PATH), str(a), str(b), "--raw-decision", str(raw), "--output", str(output)],
    )
    reviewer.main()
    assert json.loads(output.read_text())["implementation_evidence_passed"]
    assert output.with_suffix(".md").is_file()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        reviewer.main()
    assert output.read_bytes() == before
