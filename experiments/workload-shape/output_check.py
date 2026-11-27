"""Aggregate-only check of the captured continuations: language, script and loops per slice.

    python3 output_check.py RUN_DIR TOKENIZER_DIR OUT.json [--draft-vocab EAGLE3_HEAD]

Runs on the cluster (the continuations of restricted prompts stay there). Decodes each
greedy continuation with TOKENIZER_DIR/tokenizer.json (special tokens dropped) and
reports per slice:

- script: the share of generated letters that are ASCII (Latin script vs the prompt's)
  and of continuations that are mostly ASCII;
- language: the share of English function words among the output's words, and how
  many outputs look English (share >= 0.2), which tells an English answer to a
  Swahili prompt from a Swahili one;
- loops: the share of 4-grams of generated token ids already seen earlier in the
  same continuation (1 - distinct / total; loops push it towards 1) and how many
  continuations exceed 0.5; the share that hit the 512-token cap;
- with ``--draft-vocab`` (a speculators EAGLE-3 head): the share of generated tokens
  outside its draft vocabulary (``t2d``), which that head can never draft.

Only these aggregates leave the cluster, plus per-sequence script and loop values keyed
by ``seq_sha256`` (sha256 of the prompt id, first 16 hex, as in sequences.csv).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from tokenizers import Tokenizer

ENGLISH = set(
    "the a an and or of to in is are was were be been it this that for on with as by at "
    "from you your i we they he she not can will would should have has had do does if "
    "but which what there their its".split()
)


def repeat_share(ids: list[int], n: int = 4) -> float:
    seen, rep, tot = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i : i + n])
        rep += g in seen
        tot += 1
        seen.add(g)
    return rep / tot if tot else 0.0


def draft_vocab(head: Path) -> set[int]:
    """Target ids in the head's draft vocabulary, from the ``t2d`` bool tensor (no torch)."""
    with (head / "model.safetensors").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))["t2d"]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    return {i for i, b in enumerate(raw) if b}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("tokenizer", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--draft-vocab", type=Path, help="speculators EAGLE-3 head with a t2d map")
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    in_draft = draft_vocab(args.draft_vocab) if args.draft_vocab else None
    rows = {}
    for line in (args.run / "inputs" / "corpus-measured.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[str(r["id"])] = r["workload"]
    per: dict[str, dict[str, list[float]]] = {}
    seqs = {}
    for shard in sorted((args.run / "captures").glob("shard-*.json")):
        for case in json.loads(shard.read_text())["cases"]:
            if "@cap" in case["id"]:
                continue
            ids = case["token_ids"]
            text = tok.decode(ids, skip_special_tokens=True)
            letters = [c for c in text if c.isalpha()]
            ascii_share = sum(c.isascii() for c in letters) / len(letters) if letters else 1.0
            words = re.findall(r"[^\W\d_]+", text.lower())
            s = per.setdefault(
                rows[case["id"]], {k: [] for k in ("ascii", "eng", "rep4", "capped", "outside")}
            )
            s["ascii"].append(ascii_share)
            s["eng"].append(sum(w in ENGLISH for w in words) / max(len(words), 1))
            s["rep4"].append(repeat_share(ids))
            s["capped"].append(float(case["finish_reason"] == "length"))
            if in_draft is not None:
                s["outside"].append(sum(t not in in_draft for t in ids) / max(len(ids), 1))
            sha = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
            seqs[sha] = {"ascii": round(ascii_share, 4), "rep4": round(s["rep4"][-1], 4)}
    res = {}
    for w, s in per.items():
        n = len(s["ascii"])
        res[w] = {
            "sequences": n,
            "ascii_letter_share_mean": sum(s["ascii"]) / n,
            "sequences_mostly_ascii": sum(a > 0.9 for a in s["ascii"]),
            "english_function_word_share_mean": sum(s["eng"]) / n,
            "sequences_english_like": sum(e >= 0.2 for e in s["eng"]),
            "repeat4_share_mean": sum(s["rep4"]) / n,
            "sequences_repeat4_over_0.5": sum(r > 0.5 for r in s["rep4"]),
            "capped_share": sum(s["capped"]) / n,
            **(
                {"outside_draft_vocab_share_mean": sum(s["outside"]) / n}
                if in_draft is not None
                else {}
            ),
        }
    args.out.write_text(json.dumps({"slices": res, "sequences": seqs}, indent=1) + "\n")
    for w, v in res.items():
        print(w, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})


if __name__ == "__main__":
    main()
