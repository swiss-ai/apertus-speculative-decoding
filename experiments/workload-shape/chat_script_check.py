"""Split chat-like slices by prompt script and compare per-sequence acceptance.

Runs on Clariden next to the restricted corpus; prints aggregates only.

    python3 chat_script_check.py RUN_DIR

A prompt counts as non-ASCII when at least 2% of the letters of its last user
turn are outside ASCII (a rough language proxy: it misses most German, French,
Italian and Spanish text, which is largely ASCII).
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

SLICES = ("chat", "summarization", "multi-turn", "arena-hard-clean", "lang-eng")


def non_ascii_share(row: dict) -> float:
    msgs = row.get("messages") or row.get("conversations") or []
    users = [m for m in msgs if m.get("role") in ("user", "human")]
    text = users[-1].get("content", "") if users else ""
    text = text if isinstance(text, str) else json.dumps(text)
    letters = [c for c in text if c.isalpha()]
    return sum(ord(c) > 127 for c in letters) / max(len(letters), 1)


def main() -> None:
    run = Path(sys.argv[1])
    rows = {
        hashlib.sha256(r["id"].encode()).hexdigest()[:16]: r
        for r in map(json.loads, (run / "inputs/corpus-measured.jsonl").open())
    }
    seqs = list(csv.DictReader((run / "sequences.csv").open()))
    out = {}
    for wl in SLICES:
        groups: dict[str, list] = {"ascii": [], "non_ascii": []}
        for s in seqs:
            if s["workload"] != wl:
                continue
            key = "non_ascii" if non_ascii_share(rows[s["seq"]]) >= 0.02 else "ascii"
            groups[key].append(s)
        out[wl] = {
            key: {
                "n": len(g),
                **{
                    f: round(statistics.mean(float(s[f]) for s in g), 3)
                    for f in ("dspark_acc_k3", "target_ent_mean", "dspark_tv1_mean")
                    if g
                },
            }
            for key, g in groups.items()
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
