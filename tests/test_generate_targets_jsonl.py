import json

import pytest
from apertus_eagle.generate_targets import main, read_jsonl


def test_jsonl_keeps_unicode_line_separators_inside_prompt_text(tmp_path):
    source = tmp_path / "dev.jsonl"
    rows = [
        {"id": "first", "messages": [{"role": "user", "content": "A\u0085B\u2028C\u2029D"}]},
        {"id": "second", "messages": [{"role": "user", "content": "next prompt"}]},
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    assert len(source.read_text().splitlines()) == 5  # reproduces the failed reader
    assert read_jsonl(source) == rows


def test_malformed_jsonl_fails_before_model_import_or_load(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "invalid.jsonl"
    source.write_text('{"messages": "unfinished\n')
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ValueError, match="Invalid JSONL.*invalid.jsonl:1:"):
        main(["--input", str(source), "--output-dir", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_completed_or_partial_generated_artifacts_are_preserved(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "dev.jsonl"
    source.write_text('{"id":"one"}\n')
    output = tmp_path / "output"
    output.mkdir()
    dest = output / "dev.jsonl"
    original = b"previous generated artifact\n"
    dest.write_bytes(original)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(["--input", str(source), "--output-dir", str(output)])
    assert dest.read_bytes() == original
