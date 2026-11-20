import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/eagle"))

from apertus_eagle import build_domain_pools as b  # noqa: E402


def test_glaive_prompt_keeps_functions_and_first_user_turn():
    system = (
        'SYSTEM: You are a helpful assistant with access to the following functions. {"name": "f"}'
    )
    chat = "USER: Convert 5 EUR.\n\n\nASSISTANT: <functioncall> {} <|endoftext|>\n\n\nUSER: thanks"
    prompt = b.glaive_prompt(system, chat)
    assert prompt.startswith("You are a helpful assistant")
    assert prompt.endswith("Convert 5 EUR.")
    assert "thanks" not in prompt


def test_glaive_prompt_skips_rows_without_functions():
    system = "SYSTEM: You are a helpful assistant, with no access"
    assert b.glaive_prompt(system, "USER: hi") is None


def test_mbpp_prompt_lists_tests():
    item = {"text": "Write a function.", "test_list": ["assert f(1) == 1", "assert f(2) == 2"]}
    assert b.mbpp_prompt(item).endswith("assert f(1) == 1\nassert f(2) == 2")


def test_source_dir_rejects_wrong_revision(tmp_path):
    d = tmp_path / "openai__gsm8k"
    d.mkdir()
    (d / "revision.txt").write_text("deadbeef\n")
    try:
        b.source_dir(tmp_path, "openai/gsm8k")
    except SystemExit as error:
        assert "pinned" in str(error)
    else:
        raise AssertionError("revision mismatch accepted")


def test_user_row_ids_and_revision():
    row = b.user_row("math", "openai/gsm8k/main-train", "3", " q ")
    assert row["source_revision"] == b.REVISIONS["openai/gsm8k"]
    assert row["messages"] == [{"role": "user", "content": "q"}]
    assert row["id"].startswith("math-") and json.dumps(row)
