import pytest

from apertus_bench.client import RequestMeasurement
from apertus_bench.runner import summarize


def measurement(index: int, success: bool = True) -> RequestMeasurement:
    return RequestMeasurement(
        request_index=index,
        prompt_id=f"p{index}",
        workload="code",
        success=success,
        status_code=200 if success else 500,
        e2e_ms=100 + index,
        ttft_ms=20 if success else None,
        tpot_ms=10 if success else None,
        stream_event_gaps_ms=[8, 12] if success else [],
        prompt_tokens=50 if success else None,
        completion_tokens=10 if success else None,
    )


def test_summary_uses_whole_cell_wall_time() -> None:
    summary = summarize(
        [measurement(0), measurement(1), measurement(2, success=False)],
        wall_seconds=2.0,
        spec_metrics={"enabled": False},
    )
    assert summary["requests"]["attempted"] == 3
    assert summary["requests"]["successful"] == 2
    assert summary["tokens"]["completion"] == 20
    assert summary["tokens"]["output_tokens_per_second"] == pytest.approx(10)
    assert summary["latency_ms"]["stream_event_gap"]["mean"] == pytest.approx(10)
