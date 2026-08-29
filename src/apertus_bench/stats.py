from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile, or None for an empty sample."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = [float(value) for value in values]
    if not sample:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(sample),
        "mean": sum(sample) / len(sample),
        "p50": percentile(sample, 0.50),
        "p95": percentile(sample, 0.95),
        "p99": percentile(sample, 0.99),
    }
