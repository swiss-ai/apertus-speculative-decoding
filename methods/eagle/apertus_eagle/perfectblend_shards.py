"""Shard open-perfectblend prompts for target regeneration.

Input is ``perfectblend_check``'s ``conversations.jsonl`` and ``drop-ids.json``.
A row is kept when its only problem is that it ends with a user turn: that turn
is exactly what the target answers. Rows with benchmark overlap, literal
special tokens or empty turns are dropped.

Only the final assistant turn is regenerated (``generate_targets``), so rows
whose prompt messages are identical would get identical greedy answers. Each
distinct prompt is generated once; ``duplicates.jsonl`` maps every other row to
the row that carries its answer, so training can still use the whole dataset.
Runs in a compute job. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

KEEP_IF_ONLY = {"no_final_assistant"}


def prompt_key(messages: list[dict[str, Any]]) -> str | None:
    """Digest of the messages before the final assistant turn, or None if unusable."""
    prompt = list(messages)
    while prompt and prompt[-1]["role"] == "assistant":
        prompt.pop()
    if not prompt or prompt[-1]["role"] != "user":
        return None
    return hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-perfectblend-shards")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    drop = json.loads((args.data_dir / "drop-ids.json").read_text())
    out_dir = args.data_dir / "shards"
    out_dir.mkdir(exist_ok=True)
    handles = [(out_dir / f"shard-{i:03d}.jsonl").open("w") for i in range(args.shards)]
    duplicates = (args.data_dir / "duplicates.jsonl").open("w")
    first: dict[str, str] = {}
    counts: Counter[str] = Counter()
    per_shard = [0] * args.shards
    by_source: Counter[str] = Counter()
    with (args.data_dir / "conversations.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            problems = set(drop.get(row["id"], ()))
            if problems - KEEP_IF_ONLY:
                counts["dropped"] += 1
                continue
            key = prompt_key(row["messages"])
            if key is None:
                counts["no_user_prompt"] += 1
                continue
            if key in first:
                counts["duplicate_prompt"] += 1
                duplicates.write(json.dumps({"id": row["id"], "answer_from": first[key]}) + "\n")
                continue
            first[key] = row["id"]
            counts["generate"] += 1
            by_source[row["source"]] += 1
            # Round-robin keeps shards similar in size and source mix.
            shard = (counts["generate"] - 1) % args.shards
            per_shard[shard] += 1
            handles[shard].write(json.dumps(row) + "\n")
    for handle in handles:
        handle.close()
    duplicates.close()
    report = {
        "data_dir": str(args.data_dir),
        "counts": dict(counts),
        "shards": args.shards,
        "rows_per_shard": per_shard,
        "generate_by_source": dict(by_source.most_common()),
        "keep_if_only": sorted(KEEP_IF_ONLY),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
