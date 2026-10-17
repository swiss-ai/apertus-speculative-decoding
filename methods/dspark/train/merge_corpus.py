"""Merge the regenerated parts so the trainer holds out Yu's 8B validation rows.

prepare-data shuffles the rows by position (numpy default_rng(0) permutation),
drops rows whose prompt alone fills the 8,192-token window, shuffles again, and
the trainer holds out the last 10 % (Yu's val_ids.py reproduces this). With the
row count known up front, this script inverts that: it writes the conversations
of the 8B run's val_ids.txt to the lines that end up in the held-out slice, the
other rows to the training slice, and rows prepare-data will drop to the lines
it skips. A few conversations the 70B could not regenerate change nothing but
the count; if the held-out set does not exactly fill its slice, the difference
is moved across and reported. Ids absent from the parts are listed in --missing;
the merge stops there unless --allow-missing.
"""

import argparse
import glob
import re
import sys
import time

import numpy as np

PROMPT_TOKENS = re.compile(rb'"prompt_tokens": (\d+)')


def row_id(line: bytes) -> str | None:
    i = line.find(b'"id": "')
    if i < 0:
        return None
    j = line.find(b'"', i + 7)
    return line[i + 7:j].decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="glob of the regenerated part files")
    ap.add_argument("--order", required=True, help="the 8B corpus jsonl (defines the row set)")
    ap.add_argument("--val-ids", required=True, help="the 8B run's val_ids.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--missing", required=True, help="where to list ids absent from the parts")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--seq-length", type=int, default=8192)
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    # id -> (file, offset, length, prompt tokens)
    index: dict[str, tuple[int, int, int, int]] = {}
    # regenerate-responses writes failed requests next to the rows (*.errors.jsonl).
    files = sorted(p for p in glob.glob(args.parts) if ".errors." not in p)
    torn = 0
    for k, path in enumerate(files):
        with open(path, "rb") as f:
            off = 0
            for line in f:
                # A killed job can leave a row cut short; the next job appends to it,
                # and resume redoes that conversation. Keep only the last row on a line.
                start = line.rfind(b'{"id": "')
                if start > 0:
                    torn += 1
                if start >= 0 and line.endswith(b"\n"):
                    m = PROMPT_TOKENS.search(line, start)
                    index[row_id(line[start:])] = (
                        k, off + start, len(line) - start, int(m.group(1)) if m else 0
                    )
                off += len(line)
    print(f"70B rows {len(index)} from {len(files)} parts ({torn} torn lines) "
          f"in {time.time() - t0:.0f}s", flush=True)

    with open(args.order, "rb") as f:
        order = [row_id(line) for line in f]
    missing = [rid for rid in order if rid not in index]
    with open(args.missing, "w") as f:
        f.writelines(rid + "\n" for rid in missing)
    extra = len(index) - (len(order) - len(missing))
    print(f"8B rows {len(order)}; missing {len(missing)} (listed in {args.missing}); "
          f"not in the 8B corpus {extra}", flush=True)
    if missing and not args.allow_missing:
        print("MERGE_INCOMPLETE")
        return 1

    present = [rid for rid in order if rid in index]
    with open(args.val_ids) as f:
        val_ids = {line.strip() for line in f if line.strip()}
    dropped = [rid for rid in present if index[rid][3] >= args.seq_length]
    dropped_set = set(dropped)
    kept = [rid for rid in present if rid not in dropped_set]
    n, m = len(present), len(kept)
    split = int(m * args.ratio)
    val = [rid for rid in kept if rid in val_ids]
    train = [rid for rid in kept if rid not in val_ids]
    want = m - split
    moved = len(val) - want
    if moved > 0:  # surplus held-out rows go to training
        train += val[want:]
        val = val[:want]
    elif moved < 0:  # fill the held-out slice from the end of the training rows
        val += train[moved:]
        train = train[:moved]
    print(f"rows {n}, dropped by prepare-data {len(dropped)}, kept {m}, split at {split}; "
          f"held out {len(val)} ({len(val_ids & set(val))} of the 8B run's {len(val_ids)}); "
          f"moved across the split {moved:+d}", flush=True)

    # prepare-data: kept order = first permutation without the dropped lines,
    # then a second permutation; prepared row i comes from line s1k[p2[i]].
    s1 = np.random.default_rng(args.seed).permutation(n)
    s1k = s1[:m]  # the dropped rows go to the lines s1[m:]
    p2 = np.random.default_rng(args.seed).permutation(m)
    final = s1k[p2]
    line_of: list[str | None] = [None] * n
    for rid, line in zip(train + val, final.tolist(), strict=True):
        line_of[line] = rid
    for rid, line in zip(dropped, s1[m:].tolist(), strict=True):
        line_of[line] = rid
    assert all(rid is not None for rid in line_of)

    handles = [open(path, "rb") for path in files]  # noqa: SIM115
    with open(args.out, "wb") as out:
        for rid in line_of:
            k, off, length, _ = index[rid]
            handles[k].seek(off)
            out.write(handles[k].read(length))
    for h in handles:
        h.close()
    with open(args.out + ".heldout.txt", "w") as f:
        f.writelines(rid + "\n" for rid in val)
    print(f"wrote {n} rows to {args.out} in {time.time() - t0:.0f}s")
    print("MERGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
