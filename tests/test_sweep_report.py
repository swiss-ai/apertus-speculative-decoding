import json
from pathlib import Path

import pytest

from apertus_bench.sweep_report import collect, collect_pooled, table


def _write(sweep: Path, arm: str, label: str, rows: list[dict]) -> None:
    path = sweep / arm / label / "loadtest-summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"workload": label, "variant": arm, "levels": rows}))


def test_table_puts_arms_in_columns_with_ratios(tmp_path: Path) -> None:
    _write(tmp_path, "plain", "chat", [{"concurrency": 1, "output_tokens_per_second": 100.0}])
    _write(
        tmp_path,
        "dspark",
        "chat",
        [
            {"concurrency": 1, "output_tokens_per_second": 150.0},
            {"concurrency": 8, "output_tokens_per_second": 400.0},
        ],
    )
    arms = collect([tmp_path], "chat")
    text = table(arms, "output_tokens_per_second", ["plain", "dspark"], "plain")
    lines = text.splitlines()
    assert lines[2] == "| C | plain | dspark | dspark / plain |"
    assert lines[4] == "| 1 | 100 | 150 | 1.50x |"
    assert lines[5] == "| 8 | - | 400 | - |"


def test_probe_rows_come_from_each_arm(tmp_path: Path) -> None:
    from apertus_bench.sweep_report import collect_probe

    path = tmp_path / "dspark" / "probe" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {"output_tokens_per_second": 5000.0},
                "speculative_decoding": {"mean_acceptance_length": 6.7},
                "latency_ms": {"tpot": {"p50": 1.3}},
            }
        )
    )
    assert collect_probe([tmp_path]) == {
        "dspark": {
            "output_tokens_per_second": 5000.0,
            "mean_acceptance_length": 6.7,
            "tpot_p50_ms": 1.3,
        }
    }


def _cell(root: Path, label: str, c: int, tokens: int, wall: float, drafts: int, acc: int) -> None:
    path = root / label / label / f"c{c}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {"completion": tokens},
                "wall_seconds": wall,
                "speculative_decoding": {"drafts": drafts, "accepted_tokens": acc},
            }
        )
    )


def test_pool_sums_counters_across_labels(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    _cell(sweep / "eagle", "clean", 1, tokens=300, wall=3.0, drafts=100, acc=200)
    _cell(sweep / "eagle", "dirty", 1, tokens=100, wall=1.0, drafts=10, acc=40)
    _cell(sweep / "eagle", "clean", 8, tokens=300, wall=1.0, drafts=100, acc=200)
    pooled = collect_pooled([sweep], ["clean", "dirty"])
    # Only C=1 has both labels; acceptance is 1 + 240/110, not the mean of 3.0 and 5.0.
    assert list(pooled["eagle"]) == [1]
    assert pooled["eagle"][1]["output_tokens_per_second"] == 100.0
    assert pooled["eagle"][1]["mean_acceptance_length"] == pytest.approx(1 + 240 / 110)
