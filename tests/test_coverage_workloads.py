import json
import zipfile
from pathlib import Path

import pytest

from apertus_bench import coverage_workloads as cw
from apertus_bench.workloads import load_prompts

COUNT = 8


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for code in cw.LANGUAGES:
        rows = [
            {"input": f"{code} instruction number {i}", "dataset_name": "MRI"} for i in range(12)
        ]
        rows += [{"input": f"{code} task number {i}", "dataset_name": "xP3"} for i in range(12)]
        _jsonl(raw / cw.MURI / code / "test-00000-of-00001.jsonl", rows)
    for language in cw.CODE_LANGUAGES:
        _jsonl(
            raw / "bigcode/humanevalpack" / language / "test-00000-of-00001.jsonl",
            [{"task_id": f"{language}/{i}", "instruction": f"Write {i}"} for i in range(5)],
        )
    _jsonl(
        raw / "openai/gsm8k/main/test-00000-of-00001.jsonl",
        [{"question": f"q{i}", "answer": "1"} for i in range(20)],
    )
    zpath = raw / "zai-org/LongBench/data.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as archive:
        for name in [*cw.LONG_SUMMARY, *cw.LONG_QA]:
            rows = [
                # Lengths alternate in and out of the token window.
                {"_id": f"{name}{i}", "input": "why?", "context": "w " * (9000 if i % 2 else 10)}
                for i in range(10)
            ]
            archive.writestr(f"data/{name}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    hermes = raw / "NousResearch/hermes-function-calling-v1"
    hermes.mkdir(parents=True)
    for file in ("func-calling-singleturn.json", "json-mode-singleturn.json"):
        turns = [{"from": "system", "value": "tools"}, {"from": "human", "value": "call"}]
        turns.append({"from": "gpt", "value": "x"})
        (hermes / file).write_text(
            json.dumps([{"id": f"h{i}", "conversations": turns} for i in range(9)])
        )
    for lp in cw.WMT_PAIRS:
        rows = [
            {"document_id": "d", "segment_id": i, "is_bad_source": i == 0, "domain": "news",
             "source": "s" * 400, "target": "t" * 400}
            for i in range(6)
        ]  # fmt: skip
        _jsonl(raw / "google/wmt24pp" / f"en-{lp}.jsonl", rows)
    conversation = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "answer two"},
    ]
    mt = [
        {"question_id": str(q), "turn": "2", "model_a": "gpt-4", "model_b": "llama-13b",
         "conversation_a": conversation, "conversation_b": conversation}
        for q in range(10)
    ]  # fmt: skip
    _jsonl(
        raw / "lmsys/mt_bench_human_judgments/data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.jsonl",
        mt,
    )
    return raw


def test_every_slice_has_its_count_and_passes_the_client_validation(tmp_path: Path) -> None:
    rows = cw.build(_raw(tmp_path), COUNT, 256, lambda text: len(text.split()))
    labels = {r["workload"] for r in rows}
    assert {f"lang-{code}" for code in cw.LANGUAGES} <= labels
    for label in labels:
        assert sum(r["workload"] == label for r in rows) == COUNT, label
    out = tmp_path / "coverage.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load_prompts(out)) == len(rows)
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in rows)
    # MRI rows only, except where a language has too few of them.
    assert all("MRI" in r["source"] for r in rows if r["workload"] == "lang-deu")
    assert all("xP3" in r["source"] for r in rows if r["workload"] == "lang-yor")
    long = [r for r in rows if r["workload"] == "long-context"]
    assert all(cw.LONG_MIN_TOKENS <= r["prompt_tokens"] <= cw.LONG_MAX_TOKENS for r in long)
    assert [m["role"] for m in rows[-1]["messages"]] == ["user", "assistant", "user"]


def test_selection_does_not_depend_on_source_order() -> None:
    items = [(str(i), i) for i in range(50)]
    assert cw.pick(items, 5) == cw.pick(list(reversed(items)), 5)


def test_a_short_slice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="lang-eng"):
        cw.build(_raw(tmp_path), 20, 256, lambda text: len(text.split()))
