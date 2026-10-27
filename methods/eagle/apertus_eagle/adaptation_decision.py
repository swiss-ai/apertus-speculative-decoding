"""Preregistered development-only selection and compact T5 serving verdicts.

This module uses stdlib only so the durable login-node controller can close a
negative result without importing training or serving dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

BASE_HEAD = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/heads/e31-opb-thinkoff-stage1-se"
)
COVERAGE = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-"
    "coverage/workloads/coverage/coverage.jsonl"
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def immutable_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError(f"Refusing to change frozen artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(payload)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def select_heads(matrix: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    minimum = policy["selection"]["minimum_absolute_development_acceptance_gain"]
    candidates, excluded = [], {}
    for name, row in sorted(matrix.items()):
        match = re.fullmatch(r"adapt-(chat|summ|mult)-n(500|2k|8k)-r(0|20)(-lr1e4)?", name)
        if not match:
            raise ValueError(f"Unexpected adaptation arm: {name}")
        if row["state"] != "complete":
            raise ValueError(f"Infrastructure is incomplete: {name}")
        metrics = [
            row[k]
            for k in (
                "domain_acceptance_initial",
                "domain_acceptance_final",
                "opb_acceptance_initial",
                "opb_acceptance_final",
            )
        ]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in metrics):
            raise ValueError(f"Missing/nonfinite development metrics: {name}")
        gain = metrics[1] - metrics[0]
        reasons = []
        if not row.get("scientific_gates") or not all(
            v is True for v in row["scientific_gates"].values()
        ):
            reasons.append("original_scientific_gate_failed")
        if gain + 1e-12 < minimum:
            reasons.append("development_acceptance_gain_below_0.20")
        if reasons:
            excluded[name] = reasons
            continue
        domain, size, replay, high_lr = match.groups()
        candidates.append(
            {
                "name": name,
                "domain": domain,
                "head_dir": row["head_dir"],
                "run_dir": row["run_dir"],
                "domain_gain": gain,
                "opb_gain": metrics[3] - metrics[2],
                "opb_ratio": metrics[3] / metrics[2],
                "traces": {"500": 500, "2k": 2000, "8k": 8000}[size],
                "replay_fraction": 0.2 if replay == "20" else 0.0,
                "learning_rate": 1e-4 if high_lr else 3e-5,
                "training_wall_seconds": row["wall_seconds"],
            }
        )
    candidates.sort(
        key=lambda r: (
            -r["domain_gain"],
            r["traces"],
            -r["replay_fraction"],
            r["learning_rate"],
            r["name"],
        )
    )
    selected = []
    for row in candidates:
        if any(r["domain"] == row["domain"] for r in selected):
            excluded[row["name"]] = ["lower_development_rank_in_domain"]
        else:
            selected.append(row)
    assert len(selected) <= policy["selection"]["maximum_heads"]
    return {
        "selected": selected,
        "excluded": excluded,
        "selection_uses": (
            "Final development acceptance and original scientific gates only; no "
            "serving test results."
        ),
    }


def workload_entries(repo: Path, root: Path, domain: str) -> list[dict[str, str]]:
    entries = [
        {
            "label": "opb",
            "workload": "opb",
            "workloads_file": str(repo / "workloads/8b/opb-heldout.jsonl"),
            "role": "forgetting",
        }
    ]
    if domain == "mult":
        entries.append(
            {
                "label": "multilingual",
                "workload": "multilingual",
                "workloads_file": str(root / "data/pools/multilingual/test.jsonl"),
                "role": "domain",
            }
        )
        for label in (
            "lang-deu",
            "lang-fra",
            "lang-ita",
            "lang-spa",
            "translate-from-en",
            "multi-turn",
            "code-multi",
            "math-gsm8k",
        ):
            entries.append(
                {
                    "label": label,
                    "workload": label,
                    "workloads_file": COVERAGE,
                    "role": "forgetting"
                    if label in {"multi-turn", "code-multi", "math-gsm8k"}
                    else "independent_domain",
                }
            )
    else:
        label = "summarization" if domain == "summ" else "chat"
        entries.append(
            {
                "label": label,
                "workload": label,
                "workloads_file": str(repo / "workloads/8b/eagle-8b-test.jsonl"),
                "role": "domain",
            }
        )
    return entries


def paired_metrics(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, Any]:
    required = (
        "output_tokens_per_second",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "e2e_p95_ms",
        "mean_acceptance_length",
    )
    for row in (base, adapted):
        if row.get("success_rate") != 1 or row.get("warmup_failed") != 0:
            raise ValueError("Failed measured requests or warmups")
        if not all(
            isinstance(row.get(k), (float, int)) and math.isfinite(row[k]) and row[k] > 0
            for k in required
        ):
            raise ValueError("Missing/nonfinite serving metric")
    ratio = {k: adapted[k] / base[k] for k in required}
    absolute = {k: adapted[k] - base[k] for k in required}
    cost_base = 1e6 / base["output_tokens_per_second"]
    cost_adapt = 1e6 / adapted["output_tokens_per_second"]
    return {
        "baseline": base,
        "adapted": adapted,
        "ratios": ratio,
        "absolute_changes": absolute,
        "gpu_seconds_per_million_tokens": {
            "baseline": cost_base,
            "adapted": cost_adapt,
            "saved": cost_base - cost_adapt,
        },
    }


def classify_cell(pairs: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    rule = policy["recommendation"]
    substantial = len(pairs) == 2 and all(
        p["ratios"]["output_tokens_per_second"] >= rule["minimum_throughput_ratio_each_repeat"]
        and p["ratios"]["tpot_p50_ms"] <= rule["maximum_tpot_p50_ratio_each_repeat"]
        for p in pairs
    )
    regressions = sorted(
        {
            k
            for p in pairs
            for k in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")
            if p["ratios"][k] > rule["maximum_tail_or_ttft_ratio_for_unconditional_cell"]
        }
    )
    return {
        "substantial_repeatable_benefit": substantial,
        "latency_regressions": regressions,
        "verdict": "conditional_tradeoff"
        if substantial and regressions
        else "worth_considering_for_this_cell"
        if substantial
        else "no_substantial_repeatable_gain",
    }


def final_verdict(
    selection: dict[str, Any],
    confirmations: list[dict[str, Any]],
    policy: dict[str, Any],
    compute: dict[str, Any],
) -> dict[str, Any]:
    if len(confirmations) != len(selection["selected"]):
        raise ValueError("Every selected head must have a confirmation")
    heads, meaningful = [], []
    for selected, report in zip(selection["selected"], confirmations, strict=True):
        if report["name"] != selected["name"] or not report["correctness_passed"]:
            raise ValueError("Missing or failed calibrated correctness")
        cells = []
        for cell in report["cells"]:
            classified = classify_cell(cell["pairs"], policy)
            cells.append({**cell, **classified})
        benefits = [
            c for c in cells if c["role"] != "forgetting" and c["substantial_repeatable_benefit"]
        ]
        opb_regression = (
            selected["opb_ratio"]
            < 1
            - policy["recommendation"][
                "maximum_opb_development_acceptance_regression_fraction_for_broad_replacement"
            ]
        )
        for cell in cells:
            if cell["role"] == "forgetting":
                opb_regression |= any(
                    p["ratios"]["output_tokens_per_second"] < 0.95
                    or any(
                        p["ratios"][k] > 1.05
                        for k in (
                            "tpot_p50_ms",
                            "tpot_p95_ms",
                            "ttft_p50_ms",
                            "ttft_p95_ms",
                            "e2e_p95_ms",
                        )
                    )
                    for p in cell["pairs"]
                )
        setup_gpu_seconds = (
            compute["trace_allocated_gpu_seconds"]
            + compute["domain_training_allocated_gpu_seconds"][selected["domain"]]
        )
        for cell in benefits:
            saved_per_token = (
                min(p["gpu_seconds_per_million_tokens"]["saved"] for p in cell["pairs"]) / 1e6
            )
            cell["conservative_break_even_output_tokens"] = math.ceil(
                setup_gpu_seconds / saved_per_token
            )
            cell["setup_gpu_seconds"] = setup_gpu_seconds
            cell["selected_training_gpu_seconds_lower_bound"] = (
                4 * selected["training_wall_seconds"]
            )
            cell["whole_recovery_break_even_output_tokens"] = math.ceil(
                compute["whole_campaign_allocated_gpu_seconds"] / saved_per_token
            )
            cell["cost_basis"] = (
                "Full trace allocation plus all training allocations of the selected "
                "domain (including independent arms/startup/failures), a conservative "
                "recovery cost; selected training-loop lower bound and entire-recovery "
                "break-even also reported. Engineering/production capture costs excluded."
            )
        qualified = [c for c in benefits if not c["latency_regressions"]]
        heads.append(
            {
                "name": selected["name"],
                "development": selected,
                "forgetting_regression": opb_regression,
                "cells": cells,
                "recommendation": "conditional_specialist_tradeoff"
                if benefits and (opb_regression or not qualified)
                else "consider_only_measured_workload_k_concurrency_cells"
                if qualified
                else "do_not_implement_from_this_experiment",
            }
        )
        meaningful.extend(benefits)
    return {
        "state": "complete" if meaningful else "negative_complete",
        "verdict": "substantial_benefit_in_recorded_conditions"
        if meaningful
        else "no_substantial_repeatable_benefit",
        "selection": selection,
        "heads": heads,
        "compute": compute,
        "scope_completed": "Original bounded EAGLE recovery only, including serving confirmations.",
        "conditional_dspark_followup": {
            "promising_eagle_result": bool(meaningful),
            "submitted": False,
            "next_action": (
                "Verify the original DSpark-compatible data/port and its provenance before "
                "the original conditional best-setting repeat; no automatic scope expansion."
                if meaningful
                else "No qualifying EAGLE serving result; "
                "the original conditional DSpark repeat is not triggered."
            ),
        },
        "claim_limits": [
            "Simulated acceptance is development evidence, not serving speedup.",
            (
                "Calibrated greedy controls show no divergence beyond measured numerical "
                "noise; they do not prove losslessness."
            ),
            (
                "No all-workload/general-production claim; see each cell's latency and "
                "forgetting tradeoffs."
            ),
            (
                "GPU time is a compute-cost proxy; no monetary tariff, production traffic"
                " rate or update cadence is assumed."
            ),
        ],
    }
