#!/usr/bin/env python3
"""Build the expansion-corpus prompt set on a Beverin compute node.

Epoch-1 training data is not read for rewriting and is not reshuffled. This
script only reads the open-perfectblend prompts to drop duplicates, and writes
a new directory.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedupe import BoundedIndex
from quotas import PRIORITY, PROBE_PER_DOMAIN, PROMPT_TOKEN_LIMIT, SUMMARY_TOKEN_LIMIT, allocate, classify


def user_text(value):
    if isinstance(value, str):
        found = re.findall(r"<\|user_start\|>(.*?)<\|user_end\|>", value, re.S)
        return "\n".join(part.strip() for part in found) if found else value
    if isinstance(value, list):
        parts = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("from")
            if role in ("user", "human"):
                content = message.get("content")
                if content is None:
                    content = message.get("value")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return ""


def row_prompt(row):
    for key in ("conversation", "messages", "conversations", "prompt", "text", "question"):
        if key in row and row[key]:
            text = user_text(row[key])
            if text.strip():
                return text
    return ""


def row_source(row):
    for key in ("source", "dataset_source", "dataset", "origin"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def heartbeat(output, message):
    print(message, flush=True)
    if output is None:
        return
    path = Path(output) / "progress.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def load_opb(path, index, limit=None, output=None):
    added = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if limit is not None and added >= limit:
                break
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = user_text(row.get("text") or "")
            if not text.strip():
                continue
            index.add(text, row.get("primary_id") or row.get("id") or added)
            added += 1
            if added % 100000 == 0:
                heartbeat(output, "opb indexed %d" % added)
    heartbeat(output, "opb indexed %d done" % added)
    return added


def token_count(tokenizer, text):
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def iter_parquet(directory):
    import pyarrow.parquet as pq
    for path in sorted(Path(directory).rglob("*.parquet")):
        print("parquet %s" % path, flush=True)
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row


def reservoir(path, plan, seed, output=None):
    rng = random.Random(seed)
    kept = {domain: {source: [] for source in sources} for domain, sources in plan.items()}
    targets = {domain: dict(sources) for domain, sources in plan.items()}
    seen = {domain: {source: 0 for source in sources} for domain, sources in plan.items()}
    scanned = 0
    for row in iter_parquet(path):
        scanned += 1
        if scanned % 200000 == 0:
            heartbeat(output, "reservoir scanned %d" % scanned)
        source = row_source(row)
        domain = classify(source)
        quota = targets.get(domain, {}).get(source)
        if not quota:
            continue
        text = row_prompt(row)
        if not text.strip():
            continue
        bucket = kept[domain][source]
        seen[domain][source] += 1
        item = {"source": source, "domain": domain, "prompt": text,
                "id": str(row.get("id") or row.get("conversation_id") or seen[domain][source])}
        if len(bucket) < quota:
            bucket.append(item)
        else:
            slot = rng.randrange(seen[domain][source])
            if slot < quota:
                bucket[slot] = item
    heartbeat(output, "reservoir done scanned %d" % scanned)
    return kept


def _load_jsonl_prompts(paths, index, prefix):
    added = 0
    for path in paths:
        with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                text = row_prompt(row) or user_text(row.get("question") or row.get("prompt") or "")
                if text.strip():
                    index.add(text, "%s-%d" % (prefix, added))
                    added += 1
    return added


def _finepdf_dirs(root):
    root = Path(root)
    if "finepdf" in root.name.lower():
        return [root]
    if not root.is_dir():
        return []
    return [child for child in sorted(root.iterdir()) if child.is_dir() and "finepdf" in child.name.lower()]


def load_finepdfs(root, limit_rows):
    rows = []
    languages = ("eng_Latn", "deu_Latn", "fra_Latn", "ita_Latn")
    for base in _finepdf_dirs(root):
        print("finepdf %s" % base, flush=True)
        for path in base.rglob("*"):
            if len(rows) >= limit_rows:
                return {"finepdfs-edu": rows}
            if not path.is_file() or not any(lang in str(path) for lang in languages):
                continue
            if path.suffix == ".parquet":
                iterator = _parquet_file(path)
            elif path.suffix == ".jsonl":
                iterator = _jsonl_rows(path)
            else:
                continue
            for row in iterator:
                if len(rows) >= limit_rows:
                    return {"finepdfs-edu": rows}
                text = row_prompt(row)
                if text.strip():
                    rows.append({"source": "finepdfs-edu", "domain": "summarization", "prompt": text,
                                 "id": "pdf-%d" % len(rows)})
    return {"finepdfs-edu": rows}


def _parquet_file(path):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row


def _jsonl_rows(path):
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def write_outputs(kept, output, probes_per_domain):
    output.mkdir(parents=True, exist_ok=True)
    prompt_dir = output / "prompts"
    probe_dir = output / "probes"
    prompt_dir.mkdir(exist_ok=True)
    probe_dir.mkdir(exist_ok=True)
    counts = {}
    for index, domain in enumerate(PRIORITY):
        rows = []
        for bucket in (kept.get(domain) or {}).values():
            rows.extend(bucket)
        rows.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        probe_n = min(probes_per_domain, max(0, len(rows) // 10))
        probes, train = rows[:probe_n], rows[probe_n:]
        train_path = prompt_dir / ("%02d-%s.jsonl" % (index, domain))
        probe_path = probe_dir / ("%s.jsonl" % domain)
        _write_jsonl(train_path, train)
        _write_jsonl(probe_path, probes)
        counts[domain] = {"train": len(train), "probe": len(probes)}
    return counts


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": "%s-%s" % (row["domain"], row["id"]),
                "domain": row["domain"],
                "source": row["source"],
                "messages": [{"role": "user", "content": row["prompt"]}],
            }, ensure_ascii=False) + "\n")


def filter_rows(kept, index, tokenizer, token_limit, output=None):
    dropped = {"exact": 0, "near": 0, "length": 0, "internal": 0}
    accepted = {}
    seen = 0
    for domain, sources in kept.items():
        accepted[domain] = {}
        limit = SUMMARY_TOKEN_LIMIT if domain == "summarization" else token_limit
        for source, rows in sources.items():
            chosen = []
            for row in rows:
                seen += 1
                if seen % 50000 == 0:
                    heartbeat(output, "filtered %d dropped %s" % (seen, dropped))
                kind = index.duplicate(row["prompt"])
                if kind:
                    dropped[kind[0]] += 1
                    continue
                if token_count(tokenizer, row["prompt"]) > limit:
                    dropped["length"] += 1
                    continue
                index.add(row["prompt"], row["id"])
                chosen.append(row)
            accepted[domain][source] = chosen
    heartbeat(output, "filtered %d done dropped %s" % (seen, dropped))
    return accepted, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, required=True)
    parser.add_argument("--opb-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--finepdf-root", type=Path)
    parser.add_argument("--benchmark-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--require-benchmarks", action="store_true")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opb-limit", type=int)
    args = parser.parse_args()
    if args.require_benchmarks and not args.benchmark_jsonl:
        raise SystemExit("benchmark decontamination files are required")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plan_json:
        plan = json.loads(args.plan_json.read_text())
        heartbeat(args.output, "using existing sample plan %s" % args.plan_json)
    else:
        counts = {}
        scanned = 0
        for row in iter_parquet(args.sft_dir):
            counts[row_source(row)] = counts.get(row_source(row), 0) + 1
            scanned += 1
            if scanned % 200000 == 0:
                heartbeat(args.output, "counted %d" % scanned)
        plan = allocate(counts)
        (args.output / "source-counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
        heartbeat(args.output, "counted %d done" % scanned)
    (args.output / "sample-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    sampled = reservoir(args.sft_dir, plan, args.seed, args.output)
    index = BoundedIndex()
    opb_rows = load_opb(args.opb_jsonl, index, args.opb_limit, args.output)
    bench_rows = _load_jsonl_prompts(args.benchmark_jsonl, index, "bench")
    if args.finepdf_root:
        sampled["summarization"] = load_finepdfs(args.finepdf_root, 30000)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    filtered, dropped = filter_rows(sampled, index, tokenizer, PROMPT_TOKEN_LIMIT, args.output)
    written = write_outputs(filtered, args.output, PROBE_PER_DOMAIN)
    provenance = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 3),
        "sft_dir": str(args.sft_dir),
        "opb_jsonl": str(args.opb_jsonl),
        "opb_rows_indexed": opb_rows,
        "benchmark_rows_indexed": bench_rows,
        "tokenizer": str(args.tokenizer) if args.tokenizer else "chars/4",
        "dropped": dropped,
        "written": written,
        "epoch1_training_data": "unchanged",
        "seed": args.seed,
        "near_dup_method": "minhash-8 of 13-gram crc32, Jaccard estimate >= 0.5; exact sha256 unchanged",
        "plan_json": str(args.plan_json) if args.plan_json else "counted",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("EXPANSION_CORPUS " + json.dumps(written))


if __name__ == "__main__":
    main()
