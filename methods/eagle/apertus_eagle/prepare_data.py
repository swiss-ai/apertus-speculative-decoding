"""Deterministic source-group splits. Does not generate 70B responses."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT
from apertus_eagle.parse_rendered import RenderParseError, mix_row_to_record

DEFAULT_PARQUET = Path(
    "/capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1/data"
)
PILOT_TRAIN = 1000
PILOT_VALIDATION = 128
PILOT_OVERFIT = 32
HELDOUT_RESERVE = 256

PARQUET_COLUMNS = [
    "id",
    "text",
    "dataset_source",
    "dataset_name",
    "domain",
    "conversation_id",
]


def bucket(key: str, *, seed: int, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest, 16) % modulus


def assign_split(source: str, row_id: str, *, seed: int) -> str:
    """Hash source+id into overfit / validation / train / heldout_test.

    32 overfit + 128 validation + 256 confirmation-reserved + rest train
    is implemented as ranges on a 10_000-wide hash so the assignment is
    stable before any response generation.
    """
    slot = bucket(f"{source}:{row_id}", seed=seed, modulus=10_000)
    if slot < 32:
        return "overfit"
    if slot < 160:
        return "validation"
    if slot < 416:
        return "heldout_test"
    return "train"


def write_assignment(rows: list[dict[str, Any]], output: Path, *, seed: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            split = assign_split(str(row["source"]), str(row["id"]), seed=seed)
            counts[split] = counts.get(split, 0) + 1
            handle.write(json.dumps({**row, "split": split}, sort_keys=True) + "\n")
    return counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _take_sorted(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row.get("source", ""), row["id"]))
    return ordered[:limit]


def subsample_pilot(assigned: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "overfit": _take_sorted(assigned.get("overfit", []), PILOT_OVERFIT),
        "validation": _take_sorted(assigned.get("validation", []), PILOT_VALIDATION),
        "train": _take_sorted(assigned.get("train", []), PILOT_TRAIN),
        "heldout_test": _take_sorted(assigned.get("heldout_test", []), HELDOUT_RESERVE),
    }


class _MaxKey:
    """Max-heap item so each split keeps only the smallest ``limit`` keys."""

    def __init__(self, key: tuple[str, str], record: dict[str, Any]) -> None:
        self.key = key
        self.record = record

    def __lt__(self, other: _MaxKey) -> bool:
        return self.key > other.key


def _consider(heap: list[_MaxKey], record: dict[str, Any], limit: int) -> None:
    item = _MaxKey((str(record.get("source", "")), record["id"]), record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item.key < heap[0].key:
        heapq.heappushpop(heap, item)


def _heap_rows(heap: list[_MaxKey]) -> list[dict[str, Any]]:
    return sorted(
        (item.record for item in heap),
        key=lambda row: (row.get("source", ""), row["id"]),
    )


def iter_parquet_rows(parquet_dir: Path):
    import pyarrow.parquet as pq

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no part-*.parquet under {parquet_dir}")
    for path in files:
        table = pq.read_table(path, columns=PARQUET_COLUMNS)
        yield from table.to_pylist()


def materialize_from_parquet(
    parquet_dir: Path,
    output_dir: Path,
    *,
    seed: int = 1,
) -> dict[str, Any]:
    """Hash-split the SFT mix and write prompt-only JSONL samples.

    Does not call the 70B teacher. Assistant fields in the output are empty
    placeholders plus the original mix text kept under ``original_assistant``
    for audit, not for training labels.
    """
    limits = {
        "overfit": PILOT_OVERFIT,
        "validation": PILOT_VALIDATION,
        "heldout_test": HELDOUT_RESERVE,
        "train": PILOT_TRAIN,
    }
    heaps: dict[str, list[_MaxKey]] = {name: [] for name in limits}
    assigned_counts: Counter[str] = Counter()
    skipped = 0
    seen = 0
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    index_path = output_dir / "split-index.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with index_path.open("w", encoding="utf-8") as index_handle:
        for raw in iter_parquet_rows(parquet_dir):
            try:
                record = mix_row_to_record(raw)
            except (RenderParseError, KeyError):
                skipped += 1
                continue
            split = assign_split(record["source"], record["id"], seed=seed)
            record["split"] = split
            assigned_counts[split] += 1
            domains[record["domain"]] += 1
            sources[record["source"]] += 1
            index_handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "domain": record["domain"],
                        "split": split,
                        "multi_turn": record["multi_turn"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            _consider(heaps[split], record, limits[split])
            seen += 1
            if seen % 50_000 == 0:
                index_handle.flush()
                print(json.dumps({"seen": seen, "skipped": skipped, **dict(assigned_counts)}))

    sampled = {name: _heap_rows(heap) for name, heap in heaps.items()}
    for name, rows in sampled.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)

    summary = {
        "parquet_dir": str(parquet_dir),
        "seed": seed,
        "skipped_unparsed": skipped,
        "assigned_counts": dict(assigned_counts),
        "sampled_counts": {name: len(rows) for name, rows in sampled.items()},
        "domain_counts_all_parsed": dict(domains.most_common()),
        "source_counts_all_parsed": dict(sources.most_common()),
        "language_mix": "not declared on the SFT-mix parquet; recorded domain instead",
        "task_mix_note": (
            "plan default 40/30/30 chat/code/summarization is not a column; "
            "pilot samples are hashed then truncated, not rebalanced"
        ),
        "teacher_generation": "not_run",
        "index": str(index_path),
    }
    (output_dir / "split-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def conversations_from_sampled(input_path: Path, output_path: Path) -> int:
    """Restore last-assistant turns for TorchSpec ``prompt_key=messages`` JSONL.

    Tiny-overfit uses mix assistant text as teacher-forced labels until the
    frozen 70B regenerates responses. That is an implementation gate, not the
    substantive training distribution.
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as dest,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = list(row.get("messages") or [])
            assistant = row.get("original_assistant")
            if assistant is not None:
                messages = [*messages, {"role": "assistant", "content": assistant}]
            dest.write(
                json.dumps({"id": row["id"], "messages": messages}, ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def lookup_mix_text(parquet_dir: Path, ids: set[str]) -> dict[str, str]:
    """Return mix ``text`` for the given row ids from the SFT-mix parquet."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    if not ids:
        return {}
    dataset = ds.dataset(str(parquet_dir), format="parquet")
    table = dataset.to_table(
        filter=pc.field("id").isin(list(ids)),
        columns=["id", "text"],
    )
    found: dict[str, str] = {}
    for row in table.to_pylist():
        found[str(row["id"])] = str(row["text"])
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-prepare-splits")
    parser.add_argument("--input", type=Path, help="JSONL with source and id")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Capstor SFT-mix parquet directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/70b/eagle/data/split-assignments.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/70b/eagle/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--from-parquet",
        action="store_true",
        help="stream the SFT-mix parquet and write sampled prompt JSONL",
    )
    parser.add_argument(
        "--to-conversations",
        action="store_true",
        help="convert sampled prompt JSONL to TorchSpec messages JSONL",
    )
    parser.add_argument(
        "--lookup-ids-from",
        type=Path,
        help="JSONL with id fields; write mix text sidecar (login python + pyarrow)",
    )
    parser.add_argument("--lookup-output", type=Path)
    args = parser.parse_args(argv)
    if args.lookup_ids_from:
        ids: set[str] = set()
        for line in args.lookup_ids_from.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id"):
                ids.add(str(row["id"]))
        found = lookup_mix_text(args.parquet_dir, ids)
        output = args.lookup_output or (
            REPO_ROOT / "results/70b/eagle/preflight/mix-text-by-id.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(found, indent=2) + "\n")
        print(
            json.dumps(
                {"requested": len(ids), "found": len(found), "output": str(output)},
                indent=2,
            )
        )
        return
    if args.from_parquet:
        summary = materialize_from_parquet(args.parquet_dir, args.output_dir, seed=args.seed)
        print(json.dumps(summary, indent=2))
        return
    if args.to_conversations:
        if args.input is None:
            parser.error("--to-conversations requires --input")
        count = conversations_from_sampled(args.input, args.output)
        print(json.dumps({"output": str(args.output), "n": count}, indent=2))
        return
    if args.input is None:
        parser.error("pass --input JSONL or --from-parquet")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = write_assignment(rows, args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
