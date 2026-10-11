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


def test_chat_template_kwargs_are_loaded_and_derived(tmp_path: Path) -> None:
    from apertus_bench.workloads import derive_workload

    source = tmp_path / "probe.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "p0",
                "workload": "probe",
                "messages": [{"role": "user", "content": "2+2?"}],
                "max_tokens": 384,
            }
        )
        + "\n"
    )
    assert load_prompts(source)[0].chat_template_kwargs is None

    derived = tmp_path / "think.jsonl"
    count = derive_workload(
        source,
        derived,
        chat_template_kwargs={"enable_thinking": True},
        max_tokens=1024,
        id_suffix="-think",
    )
    prompt = load_prompts(derived)[0]
    assert count == 1
    assert (prompt.id, prompt.max_tokens) == ("p0-think", 1024)
    assert prompt.messages == load_prompts(source)[0].messages
    assert prompt.chat_template_kwargs == {"enable_thinking": True}

    derived.write_text(derived.read_text().replace('{"enable_thinking": true}', '"yes"'))
    with pytest.raises(ValueError, match="chat_template_kwargs must be an object"):
        load_prompts(derived)
