"""Apertus 1.5 chat rendering and assistant loss masks.

The serving path owns BOS through the chat template
(`add_special_tokens=False` when a template is present). Training must
reproduce that token sequence, including the default system and developer
blocks in `chat_template.jinja`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_TEMPLATE_PATH = Path(__file__).resolve().parent / "chat_template.jinja"

BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
USER_START_ID = 65
USER_END_ID = 66
ASSISTANT_START_ID = 67
ASSISTANT_END_ID = 68
THINK_START_ID = 69
THINK_END_ID = 70


def assistant_loss_mask(
    input_ids: Sequence[int],
    *,
    header_id: int = ASSISTANT_START_ID,
    end_id: int = ASSISTANT_END_ID,
) -> list[int]:
    """Supervise assistant contents, not the structural start/end tokens.

    Tokens after `<|assistant_start|>` and before `<|assistant_end|>` (or the
    sequence end if the turn is still open) receive 1. Everything else is 0.
    Nested tool spans that sit between the same start/end pair are included;
    T3 must confirm this against serving's actual assistant spans.
    """
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


def serving_tokenize_kwargs() -> dict[str, Any]:
    return {"add_special_tokens": False, "enable_thinking": False}


def coerce_token_ids(token_ids: Any) -> list[int]:
    """Transformers 5 TokenizersBackend returns BatchEncoding, not a Python list."""
    if token_ids is None:
        return []
    if hasattr(token_ids, "input_ids") and not isinstance(token_ids, (list, tuple, dict)):
        token_ids = token_ids.input_ids
    if isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "ids") and not isinstance(token_ids, (list, tuple)):
        token_ids = token_ids.ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if token_ids and hasattr(token_ids[0], "ids"):
        token_ids = token_ids[0].ids
    if token_ids and not isinstance(token_ids[0], int):
        raise TypeError(
            f"chat template tokenize=True returned non-ids {type(token_ids[0]).__name__}; "
            "use tokenize=False then encode"
        )
    return [int(x) for x in token_ids]


def chat_template_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> list[int]:
    """Render with the serving jinja, then encode. Do not trust tokenize=True.

    Transformers 5.3 TokenizersBackend ``apply_chat_template(tokenize=True)``
    returns a BatchEncoding whose ``list(...)`` is ``['input_ids',
    'attention_mask']``. Encoding the template string with
    ``add_special_tokens=False`` keeps BOS ownership on the template.
    """
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(text, (list, tuple)):
        text = text[0]
    encoded = tokenizer.encode(str(text), add_special_tokens=False)
    return coerce_token_ids(encoded)


def prepare_serving_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Drop mix developer spans; the chat template injects that block.

    Mix rows often start with ``<|developer_start|>Deliberation: enabled|disabled``.
    Passing that role into ``apply_chat_template`` raises ``Invalid message role``
    because the template already emits developer text from ``enable_thinking``.
    """
    enable_thinking = False
    filtered: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "developer":
            content = str(message.get("content") or "")
            if "Deliberation: enabled" in content:
                enable_thinking = True
            continue
        filtered.append(message)
    return filtered, enable_thinking


def render_conversation(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    max_seq_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return `(input_ids, loss_mask)` using the tokenizer's chat template."""
    serving_messages, inferred_thinking = prepare_serving_messages(messages)
    if enable_thinking is None:
        enable_thinking = inferred_thinking
    input_ids = chat_template_token_ids(
        tokenizer,
        serving_messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if max_seq_length is not None:
        input_ids = input_ids[:max_seq_length]
    return input_ids, assistant_loss_mask(input_ids)


class ApertusRenderer:
    """TorchSpec-compatible renderer selected with `dataset.renderer=apertus`."""

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
        return render_conversation(
            self.tokenizer,
            messages,
            add_generation_prompt=False,
            enable_thinking=None,
            max_seq_length=max_seq_length,
        )
