"""Download ``mlabonne/open-perfectblend`` and check it as an EAGLE training corpus.

The DSpark comparison trains on this dataset, so the EAGLE rerun must too. Runs
inside a compute job (network, the serving image's ``huggingface_hub`` and
tokenizer); never on a login node.

Writes, under ``--output-dir`` (scratch, not committed):
  ``conversations.jsonl``  rows as ``{id, source, messages}`` with user/assistant/system roles
  ``drop-ids.json``        rows that overlap the 8B benchmark prompts or cannot be used
and under ``--report`` (committable; counts only, no text):
  dataset revision and file hashes, per-source counts, turn and role structure,
  token lengths under the Apertus chat template (sampled), literal special-token
  strings, and exact / 13-gram near-duplicate overlap with every 8B workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from apertus_eagle.overlap_check import NEAR_DUP, normalize, shingles

DATASET = "mlabonne/open-perfectblend"
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
ROLE_MAP["system"] = "system"
# Literal strings that tokenize to Apertus special ids. <|image|> (131079) in
# prompt text crashed the EAGLE engine once (see docs/eagle-8b-report.md).
SPECIAL_STRINGS = ("<|image|>", "<|video|>", "<s>", "</s>", "<|assistant_start|>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """ShareGPT ``conversations`` to chat messages, plus a list of problems."""
    turns = row.get("conversations") or row.get("messages") or []
    messages, problems = [], []
    for turn in turns:
        role = ROLE_MAP.get(str(turn.get("from") or turn.get("role") or "").lower())
        text = turn.get("value") if "value" in turn else turn.get("content")
        if role is None:
            problems.append("unknown_role")
            continue
        if not isinstance(text, str) or not text.strip():
            problems.append("empty_turn")
        messages.append({"role": role, "content": text or ""})
    chat = [m["role"] for m in messages if m["role"] != "system"]
    if not chat or chat[-1] != "assistant":
        problems.append("no_final_assistant")
    if any(a == b for a, b in zip(chat, chat[1:], strict=False)):
        problems.append("roles_not_alternating")
    if chat and chat[0] != "user":
        problems.append("starts_with_assistant")
    return messages, problems


def user_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def load_workload_index(paths: list[Path]) -> tuple[dict[str, str], list[tuple[str, set[str]]]]:
    exact, grams = {}, []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = user_text(row["messages"])
            exact[hashlib.sha256(normalize(text).encode()).hexdigest()] = row["id"]
            grams.append((row["id"], shingles(text)))
    return exact, grams


def percentiles(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-perfectblend-check")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True, help="Apertus 1.5 checkpoint directory")
    parser.add_argument("--revision", default=None, help="dataset commit; default: current main")
    parser.add_argument("--token-sample", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download

    started = time.time()
    info = HfApi().dataset_info(DATASET, revision=args.revision)
    local = Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=info.sha,
            local_dir=args.output_dir / "hf",
            allow_patterns=["*.parquet", "*.json", "*.jsonl", "README.md"],
        )
    )
    files = sorted(p for p in local.rglob("*") if p.suffix in {".parquet", ".json", ".jsonl"})
    print(f"downloaded {DATASET}@{info.sha}: {len(files)} files in {time.time() - started:.0f}s")

    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(path).to_pylist())
        elif path.suffix == ".jsonl":
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

    exact, workload_grams = load_workload_index(args.workloads)
    gram_index: dict[str, set[int]] = {}
    for i, (_, grams) in enumerate(workload_grams):
        for gram in grams:
            gram_index.setdefault(gram, set()).add(i)

    sources: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    turns: list[int] = []
    specials: Counter[str] = Counter()
    drop: dict[str, list[str]] = {}
    overlap: dict[str, list[str]] = {"exact": [], "near": []}
    seen_prompts: Counter[str] = Counter()
    out_path = args.output_dir / "conversations.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as out:
        for index, row in enumerate(rows):
            row_id = f"perfectblend-{index:07d}"
            source = str(row.get("source") or "unknown")
            sources[source] += 1
            messages, row_problems = to_messages(row)
            turns.append(sum(1 for m in messages if m["role"] != "system"))
            text = user_text(messages)
            key = hashlib.sha256(normalize(text).encode()).hexdigest()
            seen_prompts[key] += 1
            full = "\n".join(m["content"] for m in messages)
            for special in SPECIAL_STRINGS:
                if special in full:
                    specials[special] += 1
                    row_problems.append("literal_special_token")
            if key in exact:
                overlap["exact"].append(f"{row_id}={exact[key]}")
                row_problems.append("benchmark_overlap")
            else:
                grams = shingles(text)
                hits: Counter[int] = Counter()
                for gram in grams:
                    for j in gram_index.get(gram, ()):
                        hits[j] += 1
                for j, shared in hits.items():
                    union = len(grams | workload_grams[j][1])
                    if union and shared / union >= NEAR_DUP:
                        overlap["near"].append(f"{row_id}~{workload_grams[j][0]}")
                        row_problems.append("benchmark_overlap")
                        break
            for problem in dict.fromkeys(row_problems):
                problems[problem] += 1
            if row_problems:
                drop[row_id] = sorted(set(row_problems))
            out.write(json.dumps({"id": row_id, "source": source, "messages": messages}) + "\n")
    tmp_path.replace(out_path)
    (args.output_dir / "drop-ids.json").write_text(json.dumps(drop) + "\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    usable = [i for i in range(len(rows)) if f"perfectblend-{i:07d}" not in drop]
    sample = rng.sample(usable, min(args.token_sample, len(usable)))
    total_tokens, prompt_tokens, answer_tokens = [], [], []
    for i in sample:
        messages, _ = to_messages(rows[i])
        # Same rule as generate_targets: render to text, encode without specials.
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        n_full = len(tokenizer(full, add_special_tokens=False)["input_ids"])
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        total_tokens.append(n_full)
        prompt_tokens.append(n_prompt)
        answer_tokens.append(n_full - n_prompt)

    report = {
        "dataset": DATASET,
        "revision": info.sha,
        "license": (info.card_data or {}).get("license") if info.card_data else None,
        "files": {str(p.relative_to(local)): sha256_file(p) for p in files},
        "rows": len(rows),
        "usable_rows": len(rows) - len(drop),
        "sources": dict(sources.most_common()),
        "problems": dict(problems.most_common()),
        "literal_special_tokens": dict(specials),
        "duplicate_prompts": sum(c - 1 for c in seen_prompts.values() if c > 1),
        "turns_excluding_system": percentiles(turns),
        "benchmark_overlap": {
            "workloads": [str(p) for p in args.workloads],
            "exact": len(overlap["exact"]),
            "near": len(overlap["near"]),
            "pairs": overlap["exact"] + overlap["near"],
        },
        "tokens_apertus_chat_template": {
            "sample_rows": len(sample),
            "seed": args.seed,
            "total": percentiles(total_tokens),
            "last_prompt": percentiles(prompt_tokens),
            "last_answer": percentiles(answer_tokens),
            "share_over_8192": sum(t > 8192 for t in total_tokens) / max(1, len(total_tokens)),
            "share_over_4096": sum(t > 4096 for t in total_tokens) / max(1, len(total_tokens)),
        },
        "seconds": round(time.time() - started),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"files", "benchmark_overlap"}}))


if __name__ == "__main__":
    main()
