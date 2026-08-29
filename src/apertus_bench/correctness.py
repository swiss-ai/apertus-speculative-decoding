from __future__ import annotations

import json
from pathlib import Path

from apertus_bench.client import StreamingChatClient
from apertus_bench.workloads import Prompt


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


def compare_capture_files(
    baseline_path: Path, candidate_path: Path, output_path: Path
) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if baseline["sampling"] != candidate["sampling"]:
        raise ValueError("capture sampling settings do not match")
    baseline_cases = {case["prompt_id"]: case for case in baseline["cases"]}
    candidate_cases = {case["prompt_id"]: case for case in candidate["cases"]}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("capture prompt ids do not match")

    cases: list[dict[str, object]] = []
    for prompt_id in baseline_cases:
        baseline_measurement = baseline_cases[prompt_id]["measurement"]
        candidate_measurement = candidate_cases[prompt_id]["measurement"]
        exact = (
            baseline_measurement["success"]
            and candidate_measurement["success"]
            and baseline_measurement["output"] == candidate_measurement["output"]
        )
        cases.append(
            {
                "prompt_id": prompt_id,
                "workload": baseline_cases[prompt_id]["workload"],
                "exact_match": exact,
                "baseline_sha256": baseline_measurement["output_sha256"],
                "candidate_sha256": candidate_measurement["output_sha256"],
            }
        )
    exact_matches = sum(bool(case["exact_match"]) for case in cases)
    report: dict[str, object] = {
        "schema_version": 1,
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "total_cases": len(cases),
        "exact_matches": exact_matches,
        "all_exact": exact_matches == len(cases),
        "cases": cases,
    }
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
        matches = (
            baseline_result.success
            and candidate_result.success
            and baseline_result.output == candidate_result.output
        )
        cases.append(
            {
                "prompt_id": prompt.id,
                "workload": prompt.workload,
                "exact_match": matches,
                "baseline": baseline_result.to_dict(),
                "candidate": candidate_result.to_dict(),
            }
        )

    successful_pairs = 0
    for case in cases:
        baseline_case = case["baseline"]
        candidate_case = case["candidate"]
        if isinstance(baseline_case, dict) and isinstance(candidate_case, dict):
            successful_pairs += bool(baseline_case["success"] and candidate_case["success"])
    exact_matches = sum(bool(case["exact_match"]) for case in cases)
    report: dict[str, object] = {
        "schema_version": 1,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": seed},
        "total_cases": len(cases),
        "successful_pairs": successful_pairs,
        "exact_matches": exact_matches,
        "all_exact": exact_matches == len(cases),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
