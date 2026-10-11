from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prompt:
    id: str
    workload: str
    messages: list[dict[str, str]]
    max_tokens: int
    # Sent as the request's chat_template_kwargs, e.g. {"enable_thinking": True}.
    chat_template_kwargs: dict[str, Any] | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_prompt(record: dict[str, Any], line_number: int) -> Prompt:
    required = {"id", "workload", "messages", "max_tokens"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"line {line_number}: missing fields: {', '.join(sorted(missing))}")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or not {"role", "content"} <= message.keys():
            raise ValueError(f"line {line_number}: every message needs role and content")
    max_tokens = int(record["max_tokens"])
    if max_tokens <= 0:
        raise ValueError(f"line {line_number}: max_tokens must be positive")
    kwargs = record.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ValueError(f"line {line_number}: chat_template_kwargs must be an object")
    return Prompt(
        id=str(record["id"]),
        workload=str(record["workload"]),
        messages=messages,
        max_tokens=max_tokens,
        chat_template_kwargs=kwargs,
    )


def load_prompts(path: Path, workload: str | None = None) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            prompt = _parse_prompt(record, line_number)
            if prompt.id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate prompt id {prompt.id!r}")
            seen_ids.add(prompt.id)
            if workload is None or prompt.workload == workload:
                prompts.append(prompt)
    if not prompts:
        suffix = f" for workload {workload!r}" if workload else ""
        raise ValueError(f"no prompts found in {path}{suffix}")
    return prompts


def available_workloads(prompts: list[Prompt]) -> list[str]:
    return sorted({prompt.workload for prompt in prompts})


def derive_workload(
    source: Path,
    output: Path,
    *,
    chat_template_kwargs: dict[str, Any],
    max_tokens: int | None = None,
    id_suffix: str = "",
) -> int:
    """Copy a workload file with chat_template_kwargs (and optionally max_tokens) set.

    The prompts stay identical, so a run with and without thinking compares like with like.
    """
    count = 0
    with source.open(encoding="utf-8") as handle, output.open("w", encoding="utf-8") as sink:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            record = json.loads(line)
            record["id"] = f"{record['id']}{id_suffix}"
            record["chat_template_kwargs"] = chat_template_kwargs
            if max_tokens is not None:
                record["max_tokens"] = max_tokens
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
