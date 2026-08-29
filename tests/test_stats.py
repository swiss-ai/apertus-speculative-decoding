import pytest

from apertus_bench.stats import distribution, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_distribution_handles_empty_sample() -> None:
    assert distribution([]) == {
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "p99": None,
    }


def test_percentile_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        percentile([1], 1.1)
