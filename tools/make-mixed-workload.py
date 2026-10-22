"""Build the heterogeneous "mixed" load-test workload, one task label per row.

    python3 tools/make-mixed-workload.py WORKLOADS_DIR OUT.jsonl [--seed 0]
        [--arena-hard 128]

Per-request draft-length control (docs/per-request-k.md) can only beat one
length per step when the requests of a batch differ, and every earlier sweep
served one workload at a time. This interleaves, from WORKLOADS_DIR
(workloads/8b on the cluster):

- opb: the 128 Open-PerfectBlend held-out test prompts (opb-heldout.jsonl);
- chat, summarization: the 128 test prompts of each (eagle-8b-test.jsonl);
- arena-hard: --arena-hard prompts taken evenly from the clean Arena-Hard
  prompts (arena-hard.jsonl, workload arena-hard-clean);
- probe-math, probe-code: the 32 math-reasoning and 32 HumanEval probe
  prompts (probe-speculator-benchmarks.jsonl, by source).

Each row keeps its source's messages, max_tokens and chat_template_kwargs, is
renamed ``mixed-<label>-<source id>`` with workload ``mixed``, and carries
``vllm_xargs: {"task": <label>}``, sent with the request (the label a serving
gateway would attach). Rows are shuffled with --seed so every concurrency pass
sees the mix. The output holds restricted text and stays on the cluster; the
manifest written next to it (OUT.manifest.json: counts, max_tokens, input and
output digests) is what gets committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def probe_label(row: dict) -> str:
    source = str(row.get("source", ""))
    if "math_reasoning" in source:
        return "probe-math"
    if "HumanEval" in source:
        return "probe-code"
    raise ValueError(f"probe row {row['id']}: unknown source {source!r}")


def select(workloads: Path, arena_hard: int) -> tuple[list[tuple[str, dict]], dict[str, str]]:
    """(label, source row) pairs and the digests of the files they came from."""
    files = {
        "opb": workloads / "opb-heldout.jsonl",
        "test": workloads / "eagle-8b-test.jsonl",
        "arena-hard": workloads / "arena-hard.jsonl",
        "probe": workloads / "probe-speculator-benchmarks.jsonl",
    }
    picked: list[tuple[str, dict]] = [("opb", row) for row in rows(files["opb"])]
    for row in rows(files["test"]):
        if row["workload"] in ("chat", "summarization"):
            picked.append((row["workload"], row))
    clean = [row for row in rows(files["arena-hard"]) if row["workload"] == "arena-hard-clean"]
    step = max(len(clean) // arena_hard, 1)
    picked += [("arena-hard", row) for row in clean[::step][:arena_hard]]
    picked += [(probe_label(row), row) for row in rows(files["probe"])]
    digests = {path.name: sha256(path) for path in files.values()}
    return picked, digests


def build(picked: list[tuple[str, dict]], seed: int) -> list[dict]:
    out = []
    for label, row in picked:
        record = {
            "id": f"mixed-{label}-{row['id']}",
            "workload": "mixed",
            "messages": row["messages"],
            "max_tokens": int(row["max_tokens"]),
            "vllm_xargs": {"task": label},
        }
        if row.get("chat_template_kwargs"):
            record["chat_template_kwargs"] = row["chat_template_kwargs"]
        out.append(record)
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workloads", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arena-hard", type=int, default=128)
    args = parser.parse_args()
    picked, digests = select(args.workloads, args.arena_hard)
    records = build(picked, args.seed)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    max_tokens: dict[str, int] = {}
    for record in records:
        label = record["vllm_xargs"]["task"]
        counts[label] = counts.get(label, 0) + 1
        max_tokens[label] = max(max_tokens.get(label, 0), record["max_tokens"])
    manifest = {
        "file": str(args.output),
        "sha256": sha256(args.output),
        "rows": len(records),
        "builder": "tools/make-mixed-workload.py",
        "seed": args.seed,
        "arena_hard": args.arena_hard,
        "counts": counts,
        "max_tokens_max": max_tokens,
        "sources_sha256": digests,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
