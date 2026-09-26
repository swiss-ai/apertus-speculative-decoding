import pytest

from apertus_bench.analysis import add_inferred_round_costs, add_speedups
from apertus_bench.diagnostics import (
    BLOCK_ORDERS,
    SCREENING_IDS,
    break_even_report,
    env_exports,
    extract_effective_scheduled_tokens,
    generate_prompts,
    get_config,
    inferred_round_metrics,
    normalize_scheduled_tokens,
    write_corpus,
)


def test_speedup_uses_mean_baseline_for_matching_cell() -> None:
    rows = [
        {
            "method": "none",
            "workload": "code",
            "concurrency": 1,
            "output_tokens_per_second": 10.0,
        },
        {
            "method": "none",
            "workload": "code",
            "concurrency": 1,
            "output_tokens_per_second": 14.0,
        },
        {
            "method": "draft_model",
            "workload": "code",
            "concurrency": 1,
            "output_tokens_per_second": 18.0,
        },
    ]
    add_speedups(rows)
    assert rows[2]["speedup_vs_baseline"] == 1.5


def test_incompatible_ignore_eos_is_not_a_baseline() -> None:
    rows = [
        {
            "method": "none",
            "variant": "ops",
            "workload": "code",
            "concurrency": 1,
            "ignore_eos": False,
            "output_tokens_per_second": 70.0,
        },
        {
            "method": "eagle3",
            "variant": "e31",
            "workload": "code",
            "concurrency": 1,
            "ignore_eos": True,
            "output_tokens_per_second": 20.0,
        },
    ]
    with pytest.raises(ValueError, match="no compatible operational baseline"):
        add_speedups(rows)


def test_matched_baseline_is_not_pooled_with_operational_baseline() -> None:
    shared = {
        "workload": "mechanistic_fixed256",
        "concurrency": 1,
        "workload_file_sha256": "abc",
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "container_image": "img",
        "vllm_revision": "rev",
        "max_model_len": "131072",
        "prefix_caching": "0",
    }
    rows = [
        {
            **shared,
            "variant": "B0",
            "method": "none",
            "baseline_role": "operational",
            "async_scheduling": "1",
            "effective_scheduled_tokens": "8192",
            "output_tokens_per_second": 70.0,
        },
        {
            **shared,
            "variant": "B2",
            "method": "none",
            "baseline_role": "matched",
            "async_scheduling": "0",
            "effective_scheduled_tokens": "7168",
            "output_tokens_per_second": 60.0,
        },
        {
            **shared,
            "variant": "D3",
            "method": "draft_model",
            "baseline_role": None,
            "async_scheduling": "0",
            "effective_scheduled_tokens": "7168",
            "output_tokens_per_second": 30.0,
        },
    ]
    add_speedups(rows)
    assert rows[2]["speedup_vs_baseline"] == 30.0 / 70.0
    assert rows[2]["speedup_vs_matched_baseline"] == 0.5


def test_inferred_round_metrics_match_the_plan_formula() -> None:
    metrics = inferred_round_metrics(
        completion_tokens=3264,
        accepted_tokens=2124,
        tpot_ms=31.86,
        drafts=1128,
    )
    assert metrics["inferred_steps"] == 1140
    assert metrics["inferred_g"] == 3264 / 1140
    assert metrics["inferred_t_round_ms"] == 31.86 * 3264 / 1140
    assert metrics["prometheus_g"] == 1 + 2124 / 1128


def test_break_even_uses_t0_from_the_named_variant() -> None:
    rows = [
        {
            "variant": "B2",
            "workload": "mechanistic_fixed256",
            "concurrency": 1,
            "repeat": 1,
            "tpot_p50_ms": 14.0,
            "completion_tokens": 256,
            "accepted_tokens": 0,
            "drafts": None,
            "num_speculative_tokens": None,
            "speedup_vs_baseline": 1.0,
        },
        {
            "variant": "D3",
            "workload": "mechanistic_fixed256",
            "concurrency": 1,
            "repeat": 1,
            "tpot_p50_ms": 32.0,
            "completion_tokens": 256,
            "accepted_tokens": 192,
            "drafts": 64,
            "num_speculative_tokens": 3,
            "speedup_vs_baseline": 0.4,
        },
    ]
    add_inferred_round_costs(rows)
    report = break_even_report(rows, t0_variant="B2", spec_variant="D3")
    comparison = report["comparisons"][0]
    assert report["t0_ms"] == 14.0
    assert comparison["inferred_g"] == 4.0
    assert comparison["inferred_t_round_ms"] == 128.0
    assert comparison["predicted_speedup"] == 14.0 * 4.0 / 128.0
    assert comparison["perfect_g"] == 4
    assert comparison["break_even_t_round_ms"] == 56.0


def test_extract_effective_scheduled_tokens_prefers_reservation_line() -> None:
    draft_log = (
        "Chunked prefill is enabled with max_num_batched_tokens=8192.\n"
        "max_num_scheduled_tokens is set to 7168 based on the speculative decoding settings.\n"
    )
    baseline_log = "Chunked prefill is enabled with max_num_batched_tokens=8192.\n"
    assert extract_effective_scheduled_tokens(draft_log, requested_batched_tokens=8192) == "7168"
    assert extract_effective_scheduled_tokens(baseline_log, requested_batched_tokens=8192) == "8192"
    assert extract_effective_scheduled_tokens("", requested_batched_tokens=7168) == "7168"
    assert normalize_scheduled_tokens(
        "max_num_scheduled_tokens is set to 7168 based on the speculative decoding settings."
    ) == "7168"
    assert normalize_scheduled_tokens("unknown") == "unknown"


def test_screening_block_has_six_configs_and_b0_leads_each_order() -> None:
    assert SCREENING_IDS == ("B0", "B1", "B2", "N3", "N3m", "D3")
    for order in BLOCK_ORDERS.values():
        assert order[0] == "B0"
        assert set(order) == set(SCREENING_IDS)
    d3 = get_config("D3")
    assert d3.max_num_batched_tokens == 8192
    assert d3.apply_image_token_patch is True
    b2 = env_exports("B2")
    assert b2["ASYNC_SCHEDULING"] == "0"
    assert b2["MAX_NUM_BATCHED_TOKENS"] == "7168"
    assert b2["ENABLE_PREFIX_CACHING"] == "0"


def test_diagnostics_corpus_has_32_unique_fixed_output_prompts(tmp_path) -> None:
    prompts = generate_prompts()
    assert len(prompts) == 32
    assert len({prompt.id for prompt in prompts}) == 32
    assert {prompt.workload for prompt in prompts} == {"mechanistic_fixed256"}
    assert {prompt.max_tokens for prompt in prompts} == {256}
    contents = [prompt.messages[0]["content"] for prompt in prompts]
    assert len(set(contents)) == 32
    path = tmp_path / "diagnostics-fixed-256.jsonl"
    report = write_corpus(path)
    assert report["prompts"] == 32
    assert path.is_file()
    assert path.with_suffix(".manifest.json").is_file()


def _cell(method: str, rate: float, **fields: object) -> dict[str, object]:
    base = {
        "method": method,
        "variant": "baseline" if method == "none" else "e31-k3",
        "workload": "code",
        "concurrency": 1,
        "ignore_eos": False,
        "precision": "bfloat16",
        "tokenizer_sha256": "1f2f",
        "output_tokens_per_second": rate,
    }
    return {**base, **fields}


EIGHT = {"target_model": "swiss-ai/Apertus-v1.5-8B", "target_revision": "a411d83", "target_tensor_parallel_size": 1}
SEVENTY = {"target_model": "swiss-ai/Apertus-v1.5-70B", "target_revision": "70b-rev", "target_tensor_parallel_size": 4}


def test_8b_candidate_never_uses_a_70b_baseline() -> None:
    rows = [_cell("none", 70.0, **SEVENTY), _cell("eagle3", 200.0, draft_tensor_parallel_size=1, **EIGHT)]
    with pytest.raises(ValueError, match="no compatible operational baseline"):
        add_speedups(rows)


def test_baseline_at_another_target_tp_is_not_a_denominator() -> None:
    rows = [
        _cell("none", 180.0, **{**EIGHT, "target_tensor_parallel_size": 4}),
        _cell("eagle3", 200.0, draft_tensor_parallel_size=1, **EIGHT),
    ]
    with pytest.raises(ValueError, match="no compatible operational baseline"):
        add_speedups(rows)


def test_mixed_target_baselines_do_not_pool() -> None:
    rows = [
        _cell("none", 100.0, **EIGHT),
        _cell("none", 70.0, **SEVENTY),
        _cell("eagle3", 150.0, draft_tensor_parallel_size=1, **EIGHT),
    ]
    add_speedups(rows)
    assert rows[2]["speedup_vs_baseline"] == 1.5


def test_speculative_repeats_must_share_draft_tp() -> None:
    rows = [
        _cell("none", 100.0, **EIGHT),
        _cell("eagle3", 150.0, draft_tensor_parallel_size=1, **EIGHT),
        _cell("eagle3", 160.0, draft_tensor_parallel_size=4, **EIGHT),
    ]
    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        add_speedups(rows)
