"""Build a load-test workload from the DSpark Open-PerfectBlend held-out split.

    python3 tools/make-opb-workload.py HELDOUT.jsonl OUT.jsonl [--count 128] [--scan 50000]
        [--offset 0] [--exclude WORKLOAD.jsonl ...] [--thinking] [--max-tokens-cap 1024]

Each held-out row is a rendered conversation (Yu's regenerated answers). The
prompt is every user/assistant turn before the final answer, with the
reasoning of earlier answers (<|inner_prefix|>...<|inner_suffix|>) removed;
max_tokens is that answer's length, capped at --max-tokens-cap. --thinking
is for the thinking-on split: the prompts ask for thinking
(chat_template_kwargs) and form workload opb-think, ids suffixed -think. Rows are taken evenly
over the first --scan lines (every scan/count-th, starting at --offset), one
per primary_id, skipping the primary_ids of the --exclude workloads (a
calibration set disjoint from the test prompts). The output holds restricted
text and stays on the cluster (workloads/8b/*.jsonl is not committed).
"""

import argparse
import json
import re
from pathlib import Path

TURN = re.compile(r"<\|(user|assistant)_start\|>(.*?)<\|\1_end\|>", re.S)
REASONING = re.compile(r"<\|inner_prefix\|>.*?<\|inner_suffix\|>", re.S)


def to_prompt(row: dict, thinking: bool = False, cap: int = 1024) -> dict | None:
    turns = [(role, text) for role, text in TURN.findall(row["text"])]
    if len(turns) < 2 or turns[-1][0] != "assistant":
        return None
    messages = [{"role": role, "content": REASONING.sub("", text)} for role, text in turns[:-1]]
    completion = (row.get("metadata") or {}).get("usage", {}).get("completion_tokens") or 512
    prompt = {
        "id": f"opb-{row['primary_id']}",
        "workload": "opb",
        "messages": messages,
        "max_tokens": int(min(max(completion, 16), cap)),
    }
    if thinking:
        prompt["id"] += "-think"
        prompt["workload"] = "opb-think"
        prompt["chat_template_kwargs"] = {"enable_thinking": True}
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--scan", type=int, default=50000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--thinking", action="store_true", help="thinking-on prompts")
    parser.add_argument("--max-tokens-cap", type=int, default=1024)
    args = parser.parse_args()
    step = max(args.scan // args.count, 1)
    seen: set[str] = set()
    for workload in args.exclude:
        with workload.open() as handle:
            seen.update(
                json.loads(line)["id"].removeprefix("opb-").removesuffix("-think")
                for line in handle
            )
    prompts = []
    with args.heldout.open() as handle:
        for index, line in enumerate(handle):
            if index >= args.scan or len(prompts) >= args.count:
                break
            if index % step != args.offset % step:
                continue
            row = json.loads(line)
            if row["primary_id"] in seen:
                continue
            prompt = to_prompt(row, args.thinking, args.max_tokens_cap)
            if prompt:
                seen.add(row["primary_id"])
                prompts.append(prompt)
    args.output.write_text("".join(json.dumps(p) + "\n" for p in prompts))
    print(f"{len(prompts)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
