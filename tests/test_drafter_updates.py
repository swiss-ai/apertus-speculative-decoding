import json
from pathlib import Path

from apertus_eagle.build_adaptation_pools import (
    Guard,
    language_quotas,
    read_json_rows,
    sharegpt_messages,
)


def _row(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_sharegpt_rows_become_prompts_without_the_answer() -> None:
    row = {"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]}
    assert sharegpt_messages(row) == [{"role": "user", "content": "Q?"}]


def test_language_quotas_sum_to_the_total() -> None:
    quota = language_quotas(255, {"eng": 0.4, "deu": 0.2, "fra": 0.2, "ita": 0.2})
    assert sum(quota.values()) == 255
    assert quota["deu"] == 51


def test_json_rows_read_arrays_and_lines(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))
    (tmp_path / "b.jsonl").write_text('{"id": 1}\n\n{"id": 2}\n')
    assert [r["id"] for r in read_json_rows(tmp_path / "a.json")] == [1, 2]
    assert [r["id"] for r in read_json_rows(tmp_path / "b.jsonl")] == [1, 2]


def test_guard_rejects_test_prompts_and_near_duplicates() -> None:
    words = " ".join(f"word{i}" for i in range(40))
    guard = Guard([_row(words)], {"digest-of-test"})
    assert not guard.admit(_row("anything"), "digest-of-test")
    assert not guard.admit(_row(words.upper()), "d1")  # same text after normalization
    assert not guard.admit(_row(words + " one more"), "d2")  # 13-gram near duplicate
    assert guard.admit(_row("a different prompt entirely"), "d3")
    assert not guard.admit(_row("a different prompt entirely"), "d3")  # repeat
    assert guard.rejected == {
        "protected_digest": 1,
        "protected_text": 1,
        "near_dup": 1,
        "repeat": 1,
    }
    other = " ".join(f"other{i}" for i in range(40))
    guard.protect([_row(other)])
    assert not guard.admit(_row(other + " tail"), "d4")


def test_trace_seeds_depend_on_the_row_not_its_position() -> None:
    from apertus_eagle.generate_targets import row_seed

    assert row_seed("chat-1", 0) == row_seed("chat-1", 0)
    assert row_seed("chat-1", 0) != row_seed("chat-2", 0)
    assert 0 <= row_seed("x", 2**31 - 1) < 2**31
