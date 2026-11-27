"""Aggregate-only check of the captured continuations: script and repetition per slice.

    python3 output_check.py RUN_DIR TOKENIZER_DIR OUT.json

Runs on the cluster (the continuations of restricted prompts stay there). Per
slice: the share of generated letters that are ASCII (an answer-language proxy:
Latin script vs the prompt's script), the share of continuations that hit the
512-token cap, and repetition, as the share of 4-grams of generated token ids that
already occurred earlier in the same continuation (loops push it towards 1).
Only these aggregates leave the cluster, plus per-sequence values keyed by
``seq_sha256`` (sha256 of the prompt id, first 16 hex, as in sequences.csv).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def repeat_share(ids: list[int], n: int = 4) -> float:
    seen, rep, tot = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i : i + n])
        rep += g in seen
        tot += 1
        seen.add(g)
    return rep / tot if tot else 0.0


def main() -> None:
    run, tok_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    rows = {}
    for line in (run / "inputs" / "corpus-measured.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[str(r["id"])] = r["workload"]
    per: dict[str, dict[str, list[float]]] = {}
    seqs = {}
    for shard in sorted((run / "captures").glob("shard-*.json")):
        for case in json.loads(shard.read_text())["cases"]:
            if "@cap" in case["id"]:
                continue
            ids = case["token_ids"]
            text = tok.decode(ids, skip_special_tokens=True)
            letters = [c for c in text if c.isalpha()]
            ascii_share = sum(c.isascii() for c in letters) / len(letters) if letters else 1.0
            s = per.setdefault(rows[case["id"]], {"ascii": [], "rep4": [], "capped": []})
            s["ascii"].append(ascii_share)
            s["rep4"].append(repeat_share(ids))
            s["capped"].append(float(case["finish_reason"] == "length"))
            sha = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
            seqs[sha] = {"ascii": round(ascii_share, 4), "rep4": round(s["rep4"][-1], 4)}
    res = {}
    for w, s in per.items():
        n = len(s["ascii"])
        res[w] = {
            "sequences": n,
            "ascii_letter_share_mean": sum(s["ascii"]) / n,
            "sequences_mostly_ascii": sum(a > 0.9 for a in s["ascii"]),
            "repeat4_share_mean": sum(s["rep4"]) / n,
            "sequences_repeat4_over_0.5": sum(r > 0.5 for r in s["rep4"]),
            "capped_share": sum(s["capped"]) / n,
        }
    out.write_text(json.dumps({"slices": res, "sequences": seqs}, indent=1) + "\n")
    for w, v in res.items():
        print(w, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})


if __name__ == "__main__":
    main()
