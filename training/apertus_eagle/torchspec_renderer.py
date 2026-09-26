"""Apertus renderer copied into TorchSpec as dataset.renderer=apertus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ASSISTANT_START_ID = 67
ASSISTANT_END_ID = 68


def prepare_serving_messages(messages):
    enable_thinking = False
    filtered = []
    for message in messages:
        if message.get("role") == "developer":
            content = str(message.get("content") or "")
            if "Deliberation: enabled" in content:
                enable_thinking = True
            continue
        filtered.append(message)
    return filtered, enable_thinking


def assistant_loss_mask(
    input_ids: Sequence[int],
    *,
    header_id: int = ASSISTANT_START_ID,
    end_id: int = ASSISTANT_END_ID,
) -> list[int]:
    mask = [0] * len(input_ids)
    in_assistant = False
    for index, token_id in enumerate(input_ids):
        if token_id == header_id:
            in_assistant = True
            continue
        if token_id == end_id:
            in_assistant = False
            continue
        if in_assistant:
            mask[index] = 1
    return mask


class ApertusRenderer:
    """Serving chat-template renderer. BOS is owned by the template."""

    CACHE_VERSION = "apertus-v1.5-70b-e2"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def get_assistant_token_ids(self) -> tuple[list[int], list[int], int]:
        return [ASSISTANT_START_ID], [ASSISTANT_END_ID], 0

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_seq_length: int,
        last_turn_only: bool = False,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[list[int], list[int]]:
        del tools, last_turn_only, generation_config
        serving_messages, enable_thinking = prepare_serving_messages(messages)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": enable_thinking,
        }
        try:
            text = self.tokenizer.apply_chat_template(serving_messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            text = self.tokenizer.apply_chat_template(serving_messages, **kwargs)
        if isinstance(text, (list, tuple)):
            text = text[0]
        input_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if hasattr(input_ids, "ids"):
            input_ids = input_ids.ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        elif not isinstance(input_ids, list):
            input_ids = list(input_ids)
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = [int(x) for x in input_ids]
        input_ids = input_ids[:max_seq_length]
        return input_ids, assistant_loss_mask(input_ids)
