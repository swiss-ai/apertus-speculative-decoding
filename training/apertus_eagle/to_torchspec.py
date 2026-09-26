"""Convert hashed mix splits into TorchSpec conversation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT


def sampled_row_to_conversations(row: dict[str, Any]) -> dict[str, Any]:
    messages = list(row.get("messages") or [])
    assistant = row.get("original_assistant")
    if assistant is not None:
        messages = [*messages, {"role": "assistant", "content": assistant}]
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError(f"row {row.get('id')!r} has no assistant turn")
    return {
        "id": row["id"],
        "conversations": messages,
        "source": row.get("source"),
        "domain": row.get("domain"),
        "split": row.get("split"),
        "label_policy": row.get("policy")
        or "mix original_assistant until frozen-70B regeneration",
    }


def convert_file(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open(encoding="utf-8") as incoming, dest.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            outgoing.write(json.dumps(sampled_row_to_conversations(json.loads(line))) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-to-torchspec")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "results/eagle/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/eagle/data/torchspec",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["overfit", "validation", "train"],
    )
    args = parser.parse_args(argv)
    summary: dict[str, int] = {}
    for name in args.splits:
        src = args.input_dir / f"{name}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        summary[name] = convert_file(src, args.output_dir / f"{name}.jsonl")
    print(json.dumps({"output_dir": str(args.output_dir), "counts": summary}, indent=2))


if __name__ == "__main__":
    main()
