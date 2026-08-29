import json
from pathlib import Path

import pytest

from apertus_bench.workloads import available_workloads, load_prompts


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_load_and_filter_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "one",
                "workload": "chat",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 10,
            },
            {
                "id": "two",
                "workload": "code",
                "messages": [{"role": "user", "content": "code"}],
                "max_tokens": 20,
            },
        ],
    )
    prompts = load_prompts(path)
    assert available_workloads(prompts) == ["chat", "code"]
    assert [prompt.id for prompt in load_prompts(path, "code")] == ["two"]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    record = {
        "id": "same",
        "workload": "chat",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    write_jsonl(path, [record, record])
    with pytest.raises(ValueError, match="duplicate prompt id"):
        load_prompts(path)
