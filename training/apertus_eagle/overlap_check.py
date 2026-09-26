"""A2: check that held-out splits share no prompts with the training splits.

The splits hash ``source:id`` per row, so the same or a near-identical prompt
can land in two splits. This reports exact duplicates of the normalized user
text and 13-gram near-duplicates (Jaccard >= 0.5 on the first 2000 characters),
and writes the ids a held-out split must drop. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

NGRAM = 13
NEAR_DUP = 0.5


def user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or row.get("conversations") or []
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str) -> set[str]:
    words = normalize(text)[:2000].split(" ")
    return {" ".join(words[i : i + NGRAM]) for i in range(max(1, len(words) - NGRAM + 1))}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-overlap-check")
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    train_exact: dict[str, str] = {}
    train_shingles: list[tuple[str, set[str]]] = []
    index: dict[str, set[int]] = {}
    for path in args.train:
        for row in load(path):
            text = user_text(row)
            train_exact[hashlib.sha256(normalize(text).encode()).hexdigest()] = row["id"]
            grams = shingles(text)
            train_shingles.append((row["id"], grams))
            for gram in grams:
                index.setdefault(gram, set()).add(len(train_shingles) - 1)

    report: dict[str, Any] = {"ngram": NGRAM, "near_dup_jaccard": NEAR_DUP, "splits": {}}
    for path in args.heldout:
        exact, near = [], []
        rows = load(path)
        for row in rows:
            text = user_text(row)
            digest = hashlib.sha256(normalize(text).encode()).hexdigest()
            if digest in train_exact:
                exact.append({"id": row["id"], "train_id": train_exact[digest]})
                continue
            grams = shingles(text)
            candidates: dict[int, int] = {}
            for gram in grams:
                for position in index.get(gram, ()):
                    candidates[position] = candidates.get(position, 0) + 1
            for position, shared in candidates.items():
                other = train_shingles[position][1]
                jaccard = shared / max(1, len(grams | other))
                if jaccard >= NEAR_DUP:
                    near.append({"id": row["id"], "train_id": train_shingles[position][0],
                                 "jaccard": round(jaccard, 3)})
                    break
        drop = sorted({item["id"] for item in exact + near})
        report["splits"][path.stem] = {
            "rows": len(rows), "exact": exact, "near": near, "drop_ids": drop,
            "kept": len(rows) - len(drop),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"rows": v["rows"], "drop": len(v["drop_ids"])} for k, v in report["splits"].items()}))


if __name__ == "__main__":
    main()
