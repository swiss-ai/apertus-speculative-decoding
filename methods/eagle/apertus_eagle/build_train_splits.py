"""A2 (redo): domain-stratified training splits for the 8B pilot.

The first pilot splits (``prepare_data.subsample_pilot``) hash-assigned rows
correctly but then took the first N rows sorted by (source, id), so train,
validation and heldout_test were 100% DeepMath (source names sort
"DeepMath-103K" first). The head then reached 52% acceptance on math and 6% on
the A5 chat/code/summarization strata. This module samples by hash rank within
each domain instead, using the plan's coverage defaults:

- chat 40% (chat_qa 20, multilingual 10, instruction_following 10)
- code 30%
- summarization 30%, finepdfs-edu EN/DE/FR/IT 40/20/20/20

Chat/code rows come only from SFT-mix rows hash-assigned to ``train``
(training) or ``validation`` (training eval). Summarization documents come from
finepdfs files after the ones the A5/A6 workloads read, so no document overlaps.
Login python with pyarrow; standard output format for ``generate_targets``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from apertus_eagle.build_workloads import (
    FINEPDFS,
    MIX_PARQUET,
    SUMMARY_INSTRUCTION,
    SUMMARY_SHARE,
    keep_smallest,
)
from apertus_eagle.contract import REPO_ROOT

SEED = "apertus-eagle-8b-train-mix-v1"
MIX_DOMAINS = {"chat_qa": 0.20, "multilingual": 0.10, "instruction_following": 0.10, "code": 0.30}
SUMMARY_TOTAL_SHARE = 0.30
# Source-token window for training documents: document + <=2048 generated tokens
# must fit the 4096-token training sequence.
SUMMARY_SOURCE_WINDOW = {
    "eng": (900, 2300),
    "deu": (700, 1800),
    "fra": (700, 1800),
    "ita": (700, 1800),
}
# The A5/A6 workloads read the first 2 files per language; training uses the next ones.
WORKLOAD_FILES = 2
TRAIN_FILES = 2


def rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)


def quotas(total: int) -> dict[str, int]:
    q = {domain: int(total * share) for domain, share in MIX_DOMAINS.items()}
    summary = total - sum(q.values())
    per_lang = {lang: int(summary * share) for lang, share in SUMMARY_SHARE.items()}
    per_lang["eng"] += summary - sum(per_lang.values())
    return {**q, **{f"summarization-{lang}": n for lang, n in per_lang.items()}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-build-train-splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/8b/eagle/data-mix")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument(
        "--summary-files",
        type=int,
        default=TRAIN_FILES,
        help="finepdfs files per language after the workload files",
    )
    parser.add_argument(
        "--exclude-overlap-with",
        type=Path,
        nargs="*",
        default=[],
        help="workload JSONL files; drop training rows that duplicate their prompts",
    )
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    from apertus_eagle.parse_rendered import RenderParseError, mix_row_to_record
    from apertus_eagle.prepare_data import lookup_mix_text

    want = {"train": quotas(args.train), "validation": quotas(args.validation)}
    # Oversample mix rows 2x: some fail to parse or do not end with a user turn.
    heaps: dict[tuple[str, str], list] = {
        (split, domain): [] for split in want for domain in MIX_DOMAINS
    }
    with (REPO_ROOT / "results/70b/eagle/data/split-index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["split"], row["domain"])
            if key in heaps:
                keep_smallest(
                    heaps[key],
                    rank(f"{row['domain']}:{row['id']}"),
                    row,
                    2 * want[row["split"]][row["domain"]],
                )
    chosen = {
        key: [entry[2] for entry in sorted(heap, reverse=True)] for key, heap in heaps.items()
    }
    texts = lookup_mix_text(MIX_PARQUET, {r["id"] for rows in chosen.values() for r in rows})

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"seed": SEED, "quotas": want, "counts": {}, "shortfall": {}}
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in want}
    for (split, domain), rows in chosen.items():
        kept = 0
        for row in rows:
            if kept >= want[split][domain] or row["id"] not in texts:
                continue
            try:
                record = mix_row_to_record(
                    {**row, "text": texts[row["id"]], "dataset_source": row["source"]}
                )
            except (RenderParseError, KeyError):
                continue
            messages = [m for m in record["messages"] if m["role"] != "developer"]
            if not messages or messages[-1]["role"] != "user":
                continue
            rows_by_split[split].append(
                {
                    "id": row["id"],
                    "split": split,
                    "domain": domain,
                    "source": row["source"],
                    "dataset_name": record.get("dataset_name"),
                    "messages": messages,
                }
            )
            kept += 1
        report["counts"][f"{split}/{domain}"] = kept
        if kept < want[split][domain]:
            report["shortfall"][f"{split}/{domain}"] = want[split][domain] - kept

    for language, root in FINEPDFS.items():
        files = sorted(root.glob("*.parquet"))[WORKLOAD_FILES : WORKLOAD_FILES + args.summary_files]
        low, high = SUMMARY_SOURCE_WINDOW[language]
        heap: list = []
        need = {split: want[split][f"summarization-{language}"] for split in want}
        for path in files:
            table = pq.read_table(
                path, columns=["id", "text", "token_count", "is_truncated", "fw_edu_scores"]
            )
            for doc in table.to_pylist():
                if doc["is_truncated"] or not low <= int(doc["token_count"]) <= high:
                    continue
                scores = doc["fw_edu_scores"] or []
                if not scores or sum(scores) / len(scores) < 2.5:
                    continue
                keep_smallest(heap, rank(f"summ:{doc['id']}"), doc, sum(need.values()))
        docs = [entry[2] for entry in sorted(heap, reverse=True)]
        cursor = 0
        for split in ("validation", "train"):
            for doc in docs[cursor : cursor + need[split]]:
                rows_by_split[split].append(
                    {
                        "id": "summ-" + hashlib.sha256(doc["id"].encode()).hexdigest()[:32],
                        "split": split,
                        "domain": "summarization",
                        "language": language,
                        "source": f"finepdfs-edu/{language}",
                        "source_id": doc["id"],
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{SUMMARY_INSTRUCTION[language]}\n\n{doc['text']}",
                            }
                        ],
                    }
                )
            got = len(docs[cursor : cursor + need[split]])
            report["counts"][f"{split}/summarization-{language}"] = got
            if got < need[split]:
                report["shortfall"][f"{split}/summarization-{language}"] = need[split] - got
            cursor += need[split]
        report.setdefault("finepdfs_files", {})[language] = [p.name for p in files]

    if args.exclude_overlap_with:
        # OpenCodeReasoning-2 repeats problem statements under different ids, so
        # hash-disjoint ids do not imply disjoint prompts (18 hits in the 1k mix).
        from apertus_eagle.overlap_check import normalize, shingles, user_text

        exact = set()
        grams: list[set[str]] = []
        for path in args.exclude_overlap_with:
            for line in path.read_text().split("\n"):
                if line.strip():
                    text = user_text(json.loads(line))
                    exact.add(normalize(text))
                    grams.append(shingles(text))
        index: dict[str, set[int]] = {}
        for position, gram_set in enumerate(grams):
            for gram in gram_set:
                index.setdefault(gram, set()).add(position)
        dropped = []
        for split in rows_by_split:
            kept_rows = []
            for row in rows_by_split[split]:
                text = user_text(row)
                mine = shingles(text)
                hits: dict[int, int] = {}
                for gram in mine:
                    for position in index.get(gram, ()):
                        hits[position] = hits.get(position, 0) + 1
                near = any(
                    shared / max(1, len(mine | grams[p])) >= 0.5 for p, shared in hits.items()
                )
                if normalize(text) in exact or near:
                    dropped.append({"split": split, "id": row["id"], "domain": row["domain"]})
                else:
                    kept_rows.append(row)
            rows_by_split[split] = kept_rows
        report["dropped_for_workload_overlap"] = dropped

    train_ids = {r["id"] for r in rows_by_split["train"]}
    if train_ids & {r["id"] for r in rows_by_split["validation"]}:
        raise SystemExit("train/validation overlap")
    for split, rows in rows_by_split.items():
        path = out / f"{split}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda r: rank(f"order:{r['id']}")):
                handle.write(json.dumps(row) + "\n")
        tmp.replace(path)
        report[f"{split}_rows"] = len(rows)
        report[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "split-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["shortfall"]:
        raise SystemExit(f"quota shortfall: {report['shortfall']}")


if __name__ == "__main__":
    main()
