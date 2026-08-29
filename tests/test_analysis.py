from apertus_bench.analysis import add_speedups


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
