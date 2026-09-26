"""Parse Apertus SFT-mix rendered strings back into chat messages.

The mix stores already-rendered text (``renderer_version=apertus-sft-pretrain-v1``).
Training must drop original assistant tokens and regenerate them with the frozen
70B target. This parser is the prompt-side adapter, not a claim that re-rendering
equals the stored string — that comparison is a T3 gate.
"""

from __future__ import annotations

import re
from typing import Any

SPAN_RE = re.compile(
    r"<\|(system|developer|user|assistant)_start\|>(.*?)<\|\1_end\|>",
    re.DOTALL,
)

ROLE_MAP = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
}


class RenderParseError(ValueError):
    """Raised when a mix row cannot be split into chat turns."""


def parse_rendered_text(text: str) -> list[dict[str, str]]:
    if not text or not text.strip():
        raise RenderParseError("empty rendered text")
    stripped = text[3:] if text.startswith("<s>") else text
    matches = list(SPAN_RE.finditer(stripped))
    if not matches:
        raise RenderParseError("no Apertus control-token spans found")
    leftover = SPAN_RE.sub("", stripped).strip()
    if leftover:
        raise RenderParseError(f"unparsed remainder after control spans: {leftover[:80]!r}")
    messages = [{"role": ROLE_MAP[match.group(1)], "content": match.group(2)} for match in matches]
    if not any(message["role"] == "user" for message in messages):
        raise RenderParseError("conversation has no user turn")
    return messages


def prompt_and_last_assistant(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    """Keep prior turns; return the last assistant content to be regenerated."""
    last_assistant = None
    last_index = None
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            last_assistant = message["content"]
            last_index = index
    if last_index is None or last_assistant is None:
        raise RenderParseError("conversation has no assistant turn to replace")
    return messages[:last_index], last_assistant


def mix_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    messages = parse_rendered_text(str(row["text"]))
    prompt, original_assistant = prompt_and_last_assistant(messages)
    return {
        "id": str(row["id"]),
        "source": str(row.get("dataset_source") or row.get("dataset_name") or "unknown"),
        "dataset_name": str(row.get("dataset_name") or ""),
        "domain": str(row.get("domain") or ""),
        "conversation_id": str(row.get("conversation_id") or row["id"]),
        "messages": prompt,
        "original_assistant": original_assistant,
        "n_turns": len(messages),
        "multi_turn": sum(1 for message in messages if message["role"] == "user") > 1,
        "policy": "prompts only; regenerate last assistant with frozen 70B target",
    }
