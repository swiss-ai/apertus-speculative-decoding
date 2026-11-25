import json
from pathlib import Path

from apertus_bench.corpus_share import classify, lines_in_range, summarize


def _row(*turns: tuple[str, str], **metadata) -> dict:
    text = "".join(f"<|{role}_start|>{body}<|{role}_end|>" for role, body in turns)
    return {"text": text, "metadata": metadata}


def _lid(text: str) -> str:
    return "de" if "Bitte" in text else "en"


def test_domains_and_languages_of_a_row() -> None:
    row = _row(
        ("user", "Hi"), ("assistant", "Hello"),
        ("user", "Bitte translate this into Rust"),
        ("assistant", "```rust\nfn main() {}\n```\nThe answer is $$x$$"),
        usage={"prompt_tokens": 9000, "completion_tokens": 40},
    )  # fmt: skip
    labels = classify(row, _lid)
    assert labels["prompt_language"] == "de"
    assert labels["answer_language"] == "en"
    assert set(labels["domains"]) == {
        "code", "code-multi", "math", "long-context", "translation", "multi-turn"
    }  # fmt: skip
    assert labels["code_languages"] == ["rust"]


def test_tool_and_json_rows() -> None:
    tool = _row(("user", "<tools>[]</tools> weather?"), ("assistant", "x"), is_tool_call=False)
    assert "tool-call" in classify(tool, _lid)["domains"]
    as_json = _row(("user", "Give me an object"), ("assistant", '```json\n{"a": 1}\n```'))
    domains = classify(as_json, _lid)["domains"]
    assert "json-schema" in domains and "code" not in domains
    untagged = _row(("user", "fix it"), ("assistant", "```\nx = 1\n```"))
    assert "code" in classify(untagged, _lid)["domains"]


def test_a_row_without_a_final_answer_is_not_classified() -> None:
    assert classify(_row(("user", "only a question")), _lid) is None


def test_byte_ranges_cover_every_line_once(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    lines = [json.dumps({"i": i, "pad": "x" * (i * 7 % 23)}) + "\n" for i in range(40)]
    path.write_text("".join(lines))
    size = path.stat().st_size
    for workers in (1, 3, 7, 40):
        bounds = [size * i // workers for i in range(workers + 1)]
        seen = [
            json.loads(line)["i"]
            for a, b in zip(bounds, bounds[1:], strict=False)
            for line in lines_in_range(str(path), a, b)
        ]
        assert seen == list(range(40)), workers


def test_summary_shares_are_per_classified_row() -> None:
    part = {
        "rows": {"classified": 4, "unparsed": 1},
        "prompt_language": {"en": 3, "rm": 1},
        "answer_language": {"en": 4},
        "domains": {"code": 2},
        "code_languages": {},
        "completion_tokens": {"completion:all": 100, "completion:rm": 10},
    }
    result = summarize([part, part])
    assert result["prompt_language"]["rm"] == {"rows": 2, "share": 0.25}
    assert result["domains"]["code"]["share"] == 0.5
    assert result["completion_token_share"]["rm"] == 0.1
