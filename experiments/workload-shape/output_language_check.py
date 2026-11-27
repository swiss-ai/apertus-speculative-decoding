"""Does the target answer in the prompt's language? Aggregates per slice only.

Runs on Clariden next to the restricted captures (pure Python, no tokenizer library):

    python3 output_language_check.py RUN_DIR TOKENIZER_DIR [EAGLE3_HEAD] > output-language.json

Decodes each greedy continuation with the byte-level BPE ``vocab.json`` of
TOKENIZER_DIR (Qwen/GPT-2 byte mapping; special tokens dropped) and reports, per
slice, the mean share of non-ASCII letters and of English function words among the
words of the output, and how many outputs look English (function-word share >= 0.2);
also the share of repeated token 4-grams (1 - distinct / total, a degeneration proxy)
and of outputs that hit the length cap. With a speculators EAGLE-3 head it adds the share
of generated tokens outside its draft vocabulary (``t2d``), which that head can never draft.
A rough proxy: it tells an English answer to a Swahili prompt from a Swahili one.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ENGLISH = set(
    "the a an and or of to in is are was were be been it this that for on with as by at "
    "from you your i we they he she not can will would should have has had do does if "
    "but which what there their its".split()
)


def byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def draft_vocab(head: Path) -> set[int]:
    """Target ids in the head's draft vocabulary, from the ``t2d`` bool tensor (no torch)."""
    import struct

    with (head / "model.safetensors").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))["t2d"]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    return {i for i, b in enumerate(raw) if b}


def main() -> None:
    run, tok = Path(sys.argv[1]), Path(sys.argv[2])
    in_draft = draft_vocab(Path(sys.argv[3])) if len(sys.argv) > 3 else None
    inv = {i: s for s, i in json.loads((tok / "vocab.json").read_text()).items()}
    dec = byte_decoder()
    rows = {
        str(r["id"]): r["workload"]
        for r in map(json.loads, (run / "inputs/corpus-measured.jsonl").open())
    }
    per: dict[str, list] = {}
    for cap in sorted((run / "captures").glob("shard-*.json")):
        for c in json.loads(cap.read_text())["cases"]:
            if "@cap" in c["id"]:
                continue
            text = bytes(
                dec[ch] for t in c["token_ids"] if t in inv for ch in inv[t]
            ).decode("utf-8", "replace")
            letters = [ch for ch in text if ch.isalpha()]
            words = re.findall(r"[^\W\d_]+", text.lower())
            ids = c["token_ids"]
            grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
            per.setdefault(rows[c["id"]], []).append(
                (
                    sum(ord(ch) > 127 for ch in letters) / max(len(letters), 1),
                    sum(w in ENGLISH for w in words) / max(len(words), 1),
                    1 - len(set(grams)) / max(len(grams), 1),
                    c.get("finish_reason") == "length",
                    sum(t not in in_draft for t in ids) / max(len(ids), 1) if in_draft else None,
                )
            )
    out = {}
    for w, v in per.items():
        out[w] = {
            "n": len(v),
            "non_ascii_letter_share": round(statistics.mean(x[0] for x in v), 3),
            "english_function_word_share": round(statistics.mean(x[1] for x in v), 3),
            "english_like_outputs": sum(x[1] >= 0.2 for x in v),
            "repeated_4gram_share": round(statistics.mean(x[2] for x in v), 3),
            "length_capped": sum(x[3] for x in v),
            **(
                {"outside_draft_vocab_share": round(statistics.mean(x[4] for x in v), 3)}
                if in_draft
                else {}
            ),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
