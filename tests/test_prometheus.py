import pytest

from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta

BEFORE = """
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{model_name="target"} 10
vllm:spec_decode_num_draft_tokens_total{model_name="target"} 30
vllm:spec_decode_num_accepted_tokens_total{model_name="target"} 18
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="target",position="0"} 9
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="target",position="1"} 6
vllm:spec_decode_num_drafts_total{model_name="other"} 999
"""

AFTER = """
vllm:spec_decode_num_drafts_total{model_name="target"} 20
vllm:spec_decode_num_draft_tokens_total{model_name="target"} 60
vllm:spec_decode_num_accepted_tokens_total{model_name="target"} 42
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="target",position="0"} 19
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="target",position="1"} 14
"""


def test_snapshot_filters_model_and_computes_delta() -> None:
    before = parse_speculative_snapshot(BEFORE, model="target")
    after = parse_speculative_snapshot(AFTER, model="target")
    delta = speculative_delta(before, after)
    assert before.drafts == 10
    assert delta["drafts"] == 10
    assert delta["draft_tokens"] == 30
    assert delta["accepted_tokens"] == 24
    assert delta["acceptance_rate"] == pytest.approx(0.8)
    assert delta["mean_acceptance_length"] == pytest.approx(3.4)
    assert delta["acceptance_rate_per_position"] == {"0": 1.0, "1": 0.8}


def test_counter_reset_is_rejected() -> None:
    with pytest.raises(ValueError, match="decreased"):
        speculative_delta(
            parse_speculative_snapshot(AFTER, model="target"),
            parse_speculative_snapshot(BEFORE, model="target"),
        )


def test_baseline_without_spec_metrics_is_disabled() -> None:
    snapshot = parse_speculative_snapshot("unrelated_total 1\n")
    assert speculative_delta(snapshot, snapshot) == {"enabled": False}
