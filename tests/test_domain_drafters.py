import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/train"))
dd = importlib.import_module("domain_drafters")


@pytest.mark.parametrize(("front", "total"), [(8000, 8256), (64, 320), (64, 576), (1, 2), (8, 40)])
def test_ratio_splits_exactly(front: int, total: int) -> None:
    assert int(total * dd.ratio_for(front, total)) == front


def test_ratio_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        dd.ratio_for(0, 10)
    with pytest.raises(ValueError):
        dd.ratio_for(10, 10)


def test_mixed_rows_equal_shares_and_deterministic() -> None:
    per = {d: [{"id": f"{d}-{i}"} for i in range(50)] for d in ("a", "b", "c")}
    rows = dd.mixed_rows(per, 31, seed=1)
    assert len(rows) == 31 and len({r["id"] for r in rows}) == 31
    counts = {d: sum(r["id"].startswith(d) for r in rows) for d in per}
    assert sorted(counts.values()) == [10, 10, 11]
    assert rows == dd.mixed_rows(per, 31, seed=1)


def test_mixed_rows_needs_enough_rows() -> None:
    with pytest.raises(ValueError):
        dd.mixed_rows({"a": [{"id": "x"}], "b": [{"id": "y"}]}, 4, seed=0)


def test_pick_checkpoint_first_at_max_steps(tmp_path: Path, capsys) -> None:
    for name, step in (("0", 200), ("1", 384), ("2", 384)):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
        (tmp_path / "checkpoints" / name / "training_state.json").write_text(
            json.dumps({"global_step": step})
        )
    dd.main(["pick-checkpoint", "--run", str(tmp_path), "--max-steps", "384"])
    assert capsys.readouterr().out.strip().endswith("checkpoints/1")
    with pytest.raises(SystemExit):
        dd.main(["pick-checkpoint", "--run", str(tmp_path), "--max-steps", "999"])


def test_convert_head_mixed_keeps_dev_last(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    for d in ("math", "code"):
        for split, n in (("train", 12), ("dev", 4)):
            path = traces / d / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"{d}-{split}-{i}",
                            "prompt_ids": [1, 2],
                            "generated_ids": [5, 68],
                            "finish_reason": "stop",
                        }
                    )
                    + "\n"
                    for i in range(n)
                )
            )
    plan = {
        "domains": ["math", "code"],
        "training": {"seed": 1},
        "data": {
            "traces_root": str(traces),
            "train_rows": None,
            "dev_rows": None,
            "opb": {
                "corpus": str(traces / "math/train.jsonl"),
                "val_ids": str(traces / "math/dev.jsonl"),
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    dd.main(["lock-data", "--plan", str(plan_path), "--output", str(tmp_path / "lock.json")])
    dd.main(
        [
            "convert-head",
            "--plan",
            str(plan_path),
            "--lock",
            str(tmp_path / "lock.json"),
            "--domain",
            "mixed",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    report = json.loads((tmp_path / "out/conversion.json").read_text())
    assert (report["train_rows"], report["dev_rows"]) == (12, 4)
    assert report["train_rows_by_domain"] == {"math": 6, "code": 6}
    rows = [
        json.loads(line) for line in (tmp_path / "out/train-dev.jsonl").read_text().splitlines()
    ]
    assert all("-dev-" in r["id"] for r in rows[12:]) and all(
        "-train-" in r["id"] for r in rows[:12]
    )
    assert int(len(rows) * report["train_data_ratio"]) == 12
