"""CPU-only implementation review, separate from the frozen scientific G3 gate.

Read completed repeats and the unchanged raw pipeline decision. Select one fixed
reference k per concurrency by geometric-mean throughput across both repeats;
evaluate every fixed candidate k against that same reference in both repeats.
Reports are write-once. This script never launches or modifies an experiment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

LEVELS = (1, 8, 32, 64, 128, 256)
K = range(1, 8)
STATS = ("mean", "p50", "p95", "p99")
GAIN = 0.10
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def positive(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"missing or invalid positive metric: {label}")
    return float(value)


def read_sweep(root: Path) -> dict:
    expected = {"preflight", "plain", "eagle-e10-k3", "eagle-e10-k7"} | {
        f"{name}-k{k}" for name in ("dspark-e10", "depth-l1", "depth-l5") for k in K
    }
    status_path = root / "sweep-status.json"
    status = json.loads(status_path.read_text())
    if set(status) != expected or any(value != "ok" for value in status.values()):
        raise ValueError(f"{root}: incomplete or failed sweep")
    cells = {}
    for arm in sorted(expected - {"preflight"}):
        arm_path = root / arm / "status.json"
        arm_status = json.loads(arm_path.read_text())
        if arm_status.get("status") != "ok" or len(arm_status.get("gpus", [])) != 1:
            raise ValueError(f"{arm_path}: expected one successful one-GPU deployment")
        for concurrency in LEVELS:
            path = root / arm / "opb" / "opb" / f"c{concurrency}" / "summary.json"
            summary = json.loads(path.read_text())
            success = summary["requests"]["success_rate"]
            if (
                not isinstance(success, (int, float))
                or not math.isfinite(success)
                or not 0.99 <= success <= 1
            ):
                raise ValueError(f"{path}: below 99% success or nonfinite success rate")
            speed = positive(summary["tokens"]["output_tokens_per_second"], str(path))
            metrics = {
                "throughput_tokens_per_s": speed,
                "gpu_seconds_per_million_tokens": 1e6 / speed,
                "requests_per_s": positive(summary["requests"]["requests_per_second"], str(path)),
                "mean_completion_tokens_per_successful_request": (
                    positive(summary["tokens"]["completion"], str(path))
                    / positive(summary["requests"]["successful"], str(path))
                ),
            }
            for name, source in (
                ("tpot_ms_per_token", "tpot"),
                ("request_ms", "e2e"),
                ("ttft_ms", "ttft"),
            ):
                distribution = summary["latency_ms"][source]
                positive(distribution["count"], f"{path}: {source} sample count")
                metrics.update(
                    {f"{name}.{stat}": positive(distribution[stat], str(path)) for stat in STATS}
                )
            cells[arm, concurrency] = {
                "metrics": metrics,
                "mean_acceptance_length": (summary.get("speculative_decoding") or {}).get(
                    "mean_acceptance_length"
                ),
                "source": str(path),
                "sha256": sha256(path),
                "deployment_id": arm_status.get("deployment_id"),
                "gpu_ids": arm_status["gpus"],
                "arm_status_sha256": sha256(arm_path),
            }
    return cells


def compare(reference: dict, candidate: dict) -> dict:
    return {
        metric: {
            "reference": before,
            "candidate": candidate["metrics"][metric],
            "delta_candidate_minus_reference": candidate["metrics"][metric] - before,
            "percent_change": 100 * (candidate["metrics"][metric] / before - 1),
        }
        for metric, before in reference["metrics"].items()
    }


def original_g3(sweeps: list[dict]) -> list[dict]:
    rows = []
    for concurrency in LEVELS:
        repeats = []
        for cells in sweeps:
            ref_k = max(
                K,
                key=lambda k: cells[f"dspark-e10-k{k}", concurrency]["metrics"][
                    "throughput_tokens_per_s"
                ],
            )
            candidate_k = max(
                K,
                key=lambda k: cells[f"depth-l1-k{k}", concurrency]["metrics"][
                    "throughput_tokens_per_s"
                ],
            )
            ratio = (
                cells[f"depth-l1-k{candidate_k}", concurrency]["metrics"]["throughput_tokens_per_s"]
                / cells[f"dspark-e10-k{ref_k}", concurrency]["metrics"]["throughput_tokens_per_s"]
            )
            repeats.append({"reference_k": ref_k, "candidate_k": candidate_k, "ratio": ratio})
        rows.append(
            {
                "concurrency": concurrency,
                "repeats": repeats,
                "passed": concurrency >= 128 and all(r["ratio"] >= 1.05 for r in repeats),
            }
        )
    return rows


def review(a: Path, b: Path, raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    if raw.get("calibrated_correctness_passed") is not True:
        raise ValueError("completed calibrated correctness evidence is required")
    sweeps = [read_sweep(a), read_sweep(b)]
    g3 = original_g3(sweeps)
    scientific_pass = any(row["passed"] for row in g3)
    if raw.get("gate_g3_repeatable_throughput_passed") is not scientific_pass:
        raise ValueError("raw G3 result does not match unchanged original scientific criterion")
    rows = []
    for concurrency in LEVELS:
        reference_k = max(
            K,
            key=lambda k: (
                sum(
                    math.log(
                        cells[f"dspark-e10-k{k}", concurrency]["metrics"]["throughput_tokens_per_s"]
                    )
                    for cells in sweeps
                ),
                -k,
            ),
        )
        for depth in ("depth-l1", "depth-l5"):
            for candidate_k in K:
                repeats, tradeoffs = [], []
                for repeat, cells in enumerate(sweeps, 1):
                    reference = cells[f"dspark-e10-k{reference_k}", concurrency]
                    candidate = cells[f"{depth}-k{candidate_k}", concurrency]
                    deltas = compare(reference, candidate)
                    best_ref = max(
                        cells[f"dspark-e10-k{k}", concurrency]["metrics"]["throughput_tokens_per_s"]
                        for k in K
                    )
                    ratio_to_best = candidate["metrics"]["throughput_tokens_per_s"] / best_ref
                    repeats.append(
                        {
                            "repeat": repeat,
                            "reference": reference,
                            "candidate": candidate,
                            "deltas": deltas,
                            "ratio_vs_this_repeat_best_reference": ratio_to_best,
                        }
                    )
                    for metric in (
                        "ttft_ms.mean",
                        "ttft_ms.p50",
                        "ttft_ms.p95",
                        "ttft_ms.p99",
                        "request_ms.p95",
                        "request_ms.p99",
                        "tpot_ms_per_token.p95",
                        "tpot_ms_per_token.p99",
                    ):
                        if deltas[metric]["percent_change"] >= 100 * GAIN - EPS:
                            tradeoffs.append({"repeat": repeat, "metric": metric, **deltas[metric]})
                throughput_pass = all(
                    r["deltas"]["throughput_tokens_per_s"]["percent_change"] >= 100 * GAIN - EPS
                    for r in repeats
                )
                # Prevent a stable but weak reference k from manufacturing a
                # substantial gain against the strongest measured alternative.
                best_reference_guard = all(
                    r["ratio_vs_this_repeat_best_reference"] >= 1 + GAIN - EPS for r in repeats
                )
                latency_metrics = [
                    metric
                    for metric in ("tpot_ms_per_token.mean", "request_ms.mean")
                    if all(
                        r["deltas"][metric]["percent_change"] <= -100 * GAIN + EPS for r in repeats
                    )
                ]
                numerical_pass = throughput_pass and best_reference_guard and bool(latency_metrics)
                eligible = (
                    depth == "depth-l1"
                    and concurrency >= 128
                    and next(r for r in g3 if r["concurrency"] == concurrency)["passed"]
                )
                rows.append(
                    {
                        "depth": depth,
                        "concurrency": concurrency,
                        "reference_k": reference_k,
                        "candidate_k": candidate_k,
                        "same_fixed_k_pair_in_both_repeats": True,
                        "throughput_10pct_both_repeats": throughput_pass,
                        "also_10pct_vs_best_reference_in_each_repeat": best_reference_guard,
                        "latency_10pct_both_repeats_metrics": latency_metrics,
                        "numerical_substantial_screen_passed": numerical_pass,
                        "implementation_evidence_passed": numerical_pass and eligible,
                        "ttft_and_tail_tradeoffs": tradeoffs,
                        "repeats": repeats,
                    }
                )
    qualified = [row for row in rows if row["implementation_evidence_passed"]]
    clean = [row for row in qualified if not row["ttft_and_tail_tradeoffs"]]
    if clean:
        verdict = "substantial_implementation_evidence_for_measured_settings"
    elif qualified:
        verdict = "substantial_but_conditional_on_recorded_TTFT_or_tail_costs"
    elif scientific_pass:
        verdict = "original_G3_pass_only_no_substantial_implementation_evidence"
    else:
        verdict = "negative_original_G3_no_implementation_endorsement"
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "helper_sha256": sha256(Path(__file__)),
        "original_decision": {
            "path": str(raw_path),
            "sha256": sha256(raw_path),
            "raw_label": raw.get("verdict"),
            "original_G3_passed": scientific_pass,
            "calibrated_correctness_passed": True,
            "label_limit": "raw substantial label means original G3 only; "
            "it is not an implementation recommendation",
        },
        "implementation_verdict": verdict,
        "implementation_evidence_passed": bool(qualified),
        "policy": {
            "original_scientific_threshold_unchanged": (
                "G3 >=5%, original per-repeat best k, C>=128"
            ),
            "fixed_reference_selection": "one k maximizing geometric-mean reference throughput "
            "across both repeats, separately per C; ties choose smaller k",
            "candidate_selection": "all seven fixed k values shown, each unchanged across repeats",
            "substantial_screen": ">=10% throughput and >=10% mean TPOT or mean request latency "
            "improvement, same latency metric in both repeats; also >=10% throughput versus each "
            "repeat's strongest reference; original G3 must pass at same C>=128",
            "tradeoff_flag": ">=10% regression of mean/p50 TTFT or p95/p99 TTFT/request/TPOT "
            "in either repeat; conditional evidence only when such costs are present",
            "delta_sign": "candidate minus reference; positive throughput is better, positive "
            "latency/GPU-seconds is worse; absolute values and percent changes are both retained",
            "gpu_accounting": "one GPU per deployment; GPU-seconds/M = 1e6 / output_tokens_per_s",
            "scope_limit": "two counterbalanced repeats and fixed OPB workload; exploratory "
            "implementation screen; no capacity, SLO, confidence-interval or other workload claim",
        },
        "original_G3_by_concurrency": g3,
        "qualifying_fixed_pairs": [
            {
                key: row[key]
                for key in (
                    "depth",
                    "concurrency",
                    "candidate_k",
                    "reference_k",
                    "latency_10pct_both_repeats_metrics",
                    "ttft_and_tail_tradeoffs",
                )
            }
            for row in qualified
        ],
        "fixed_pair_rows": rows,
    }


def markdown(report: dict) -> str:
    original_result = "pass" if report["original_decision"]["original_G3_passed"] else "fail"
    lines = [
        f"Verdict: **{report['implementation_verdict']}**.",
        "",
        f"Original G3: **{original_result}**; "
        "its 5% threshold remains unchanged. The raw decision is preserved.",
        "",
        "Same fixed pair in both repeats. Values are A / B; latency deltas are candidate minus "
        "reference. Full absolute values, tails, control-depth results and source hashes are "
        "in the JSON report.",
        "",
        "| C | candidate/reference k | output tok/s candidate | output tok/s reference | "
        "throughput change | mean TPOT delta ms/token | mean request delta ms | "
        "mean TTFT delta ms | GPU-s/M saved | evidence |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for concurrency in LEVELS:
        candidates = [
            r
            for r in report["fixed_pair_rows"]
            if r["depth"] == "depth-l1" and r["concurrency"] == concurrency
        ]
        eligible = [r for r in candidates if r["implementation_evidence_passed"]]
        row = max(
            eligible or candidates,
            key=lambda r: min(
                repeat["deltas"]["throughput_tokens_per_s"]["percent_change"]
                for repeat in r["repeats"]
            ),
        )

        def pair(metric: str, field: str, digits: int = 2, current: dict = row) -> str:
            return " / ".join(
                f"{repeat['deltas'][metric][field]:.{digits}f}" for repeat in current["repeats"]
            )

        costs = (
            "conditional: TTFT/tail cost" if row["ttft_and_tail_tradeoffs"] else "measured settings"
        )
        label = costs if row["implementation_evidence_passed"] else "does not qualify"
        saved = " / ".join(
            f"{-r['deltas']['gpu_seconds_per_million_tokens']['delta_candidate_minus_reference']:.2f}"
            for r in row["repeats"]
        )
        lines.append(
            f"| {concurrency} | {row['candidate_k']}/{row['reference_k']} | "
            f"{pair('throughput_tokens_per_s', 'candidate')} | "
            f"{pair('throughput_tokens_per_s', 'reference')} | "
            f"{pair('throughput_tokens_per_s', 'percent_change')}% | "
            f"{pair('tpot_ms_per_token.mean', 'delta_candidate_minus_reference')} | "
            f"{pair('request_ms.mean', 'delta_candidate_minus_reference')} | "
            f"{pair('ttft_ms.mean', 'delta_candidate_minus_reference')} | {saved} | {label} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--raw-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text_path = args.output.with_suffix(".md")
    if args.output.exists() or text_path.exists():
        raise FileExistsError("review artifacts already exist; choose a fresh output name")
    report = review(args.a, args.b, args.raw_decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    with text_path.open("x") as out:
        out.write(markdown(report))


if __name__ == "__main__":
    main()
