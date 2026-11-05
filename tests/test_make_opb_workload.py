import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/make-opb-workload.py"
spec = importlib.util.spec_from_file_location("make_opb_workload", SCRIPT)
assert spec and spec.loader
make_opb_workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_opb_workload)


def _row(answer_tokens: int) -> dict:
    text = (
        "<s><|system_start|>You are Apertus.<|system_end|>"
        "<|user_start|>What is 2+2?<|user_end|>"
        "<|assistant_start|><|inner_prefix|>Add them.<|inner_suffix|>4<|assistant_end|>"
        "<|user_start|>And 3+3?<|user_end|>"
        "<|assistant_start|><|inner_prefix|>Add again.<|inner_suffix|>6<|assistant_end|>"
    )
    usage = {"completion_tokens": answer_tokens}
    return {"primary_id": "opb-0000300", "text": text, "metadata": {"usage": usage}}


def test_answer_cut_at_the_generation_cap_keeps_its_prompt() -> None:
    row = _row(8192)
    row["text"] = row["text"].removesuffix("6<|assistant_end|>")
    prompt = make_opb_workload.to_prompt(row, thinking=True, cap=2048)
    assert prompt["max_tokens"] == 2048
    assert prompt["messages"][-1] == {"role": "user", "content": "And 3+3?"}


def test_thinking_off_prompt_is_unchanged() -> None:
    prompt = make_opb_workload.to_prompt(_row(3000))
    assert prompt["id"] == "opb-opb-0000300"
    assert prompt["workload"] == "opb"
    assert prompt["max_tokens"] == 1024
    assert "chat_template_kwargs" not in prompt


def test_thinking_prompt_drops_earlier_reasoning_and_asks_for_thinking() -> None:
    prompt = make_opb_workload.to_prompt(_row(3000), thinking=True, cap=2048)
    assert prompt["id"] == "opb-opb-0000300-think"
    assert prompt["workload"] == "opb-think"
    assert prompt["chat_template_kwargs"] == {"enable_thinking": True}
    assert prompt["max_tokens"] == 2048
    assert prompt["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
