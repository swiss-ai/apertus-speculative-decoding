import json
from pathlib import Path

from apertus_bench.sweep_report import collect, table


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
