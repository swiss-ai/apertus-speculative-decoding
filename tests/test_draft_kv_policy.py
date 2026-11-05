from __future__ import annotations

import pytest
from draft_kv import policy


def test_importance_preserves_old_important_pages_and_unaligned_recent_boundary():
    config = policy.PolicyConfig(policy="importance", budget=512)
    scores = dict.fromkeys(range(129), 0.0)
    scores[1] = 100.0
    scores[2] = 50.0
    kept = policy.select_pages(
        scores, config=config, oldest_visible_position=1, newest_position=2048,
    )
    assert len(kept) == 32
    assert {1, 2}.issubset(kept)
    # Positions 1921..2048 cross nine physical pages, not eight.
    assert set(range(120, 129)).issubset(kept)


def test_expired_and_future_pages_cannot_survive_regardless_of_score():
    config = policy.PolicyConfig(policy="importance", budget=512)
    kept = policy.select_pages(
        {0: 1e9, 10: 1.0, 11: 2.0, 12: 1e9}, config=config,
        oldest_visible_position=160, newest_position=191,
    )
    assert kept == (10, 11)


def test_ties_choose_newer_pages_and_recent_policy_ignores_importance():
    scores = dict.fromkeys(range(128), 1.0)
    config = policy.PolicyConfig(policy="importance", budget=512)
    kwargs = dict(oldest_visible_position=0, newest_position=2047)
    assert policy.select_pages(scores, config=config, **kwargs) == tuple(range(96, 128))
    scores[0] = 1e9
    recent = policy.PolicyConfig(policy="recent", budget=512)
    assert policy.select_pages(scores, config=recent, **kwargs) == tuple(range(96, 128))


def test_no_eviction_reference_keeps_all_native_pages_at_boundary():
    config = policy.PolicyConfig(policy="off")
    scores = dict.fromkeys(range(130), 0.0)
    kept = policy.select_pages(
        scores, config=config, oldest_visible_position=1, newest_position=2048,
    )
    assert kept == tuple(range(129))
    assert config.max_bootstrap_pages == 129


def test_evicted_pages_do_not_resurrect_on_a_later_step():
    config = policy.PolicyConfig(policy="importance", budget=512)
    scores = dict.fromkeys(range(128), 1.0)
    kept = policy.select_pages(
        scores, config=config, oldest_visible_position=0, newest_position=2047,
    )
    surviving = {page: scores[page] for page in kept}
    surviving[128] = 0.0
    next_kept = policy.select_pages(
        surviving, config=config, oldest_visible_position=16, newest_position=2063,
    )
    assert set(next_kept).issubset(set(kept) | {128})
    assert len(next_kept) == 32


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0])
def test_invalid_scores_fail_closed(score):
    with pytest.raises(ValueError):
        policy.select_pages(
            {0: score}, config=policy.PolicyConfig(policy="importance"),
            oldest_visible_position=0, newest_position=10,
        )


def test_budget_accounting_separates_payload_from_runtime_overheads():
    assert policy.retained_bytes(128) == 40 * 1024 * 1024
    assert policy.retained_bytes(64) == 20 * 1024 * 1024
    assert policy.retained_bytes(32) == 10 * 1024 * 1024
    with pytest.raises(ValueError):
        policy.PolicyConfig(policy="importance", budget=128)
