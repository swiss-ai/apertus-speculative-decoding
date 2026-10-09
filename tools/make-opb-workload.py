"""Build a load-test workload from the DSpark Open-PerfectBlend held-out split.

    python3 tools/make-opb-workload.py HELDOUT.jsonl OUT.jsonl [--count 128] [--scan 50000]

Each held-out row is a rendered thinking-off conversation (Yu's regenerated
answers). The prompt is every user/assistant turn before the final answer;
max_tokens is that answer's length, capped at 1024. Rows are taken evenly
over the first --scan lines, one per primary_id. The output holds restricted
text and stays on the cluster (workloads/8b/*.jsonl is not committed).
"""

import argparse
import json
import re
from pathlib import Path

TURN = re.compile(r"<\|(user|assistant)_start\|>(.*?)<\|\1_end\|>", re.S)


def to_prompt(row: dict) -> dict | None:
    turns = [(role, text) for role, text in TURN.findall(row["text"])]
    if len(turns) < 2 or turns[-1][0] != "assistant":
        return None
    messages = [{"role": role, "content": text} for role, text in turns[:-1]]
    completion = (row.get("metadata") or {}).get("usage", {}).get("completion_tokens") or 512
    return {
        "id": f"opb-{row['primary_id']}",
        "workload": "opb",
        "messages": messages,
        "max_tokens": int(min(max(completion, 16), 1024)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--scan", type=int, default=50000)
    args = parser.parse_args()
    step = max(args.scan // args.count, 1)
    seen: set[str] = set()
    prompts = []
    with args.heldout.open() as handle:
        for index, line in enumerate(handle):
            if index >= args.scan or len(prompts) >= args.count:
                break
            if index % step:
                continue
            row = json.loads(line)
            if row["primary_id"] in seen:
                continue
            prompt = to_prompt(row)
            if prompt:
                seen.add(row["primary_id"])
                prompts.append(prompt)
    args.output.write_text("".join(json.dumps(p) + "\n" for p in prompts))
    print(f"{len(prompts)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
