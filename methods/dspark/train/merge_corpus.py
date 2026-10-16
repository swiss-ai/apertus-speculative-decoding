"""Merge the regenerated parts into one corpus in the row order of Yu's 8B corpus.

prepare-data shuffles by row position (seed 0) and the trainer holds out the
last 10 % of the prepared rows. Writing the 70B rows in the 8B corpus's id order,
with the same ids, therefore holds out the same conversations as the 8B run
(checked afterwards with Yu's val_ids.py). Ids missing from the 70B parts are
listed in --missing and the merge stops, unless --allow-missing.
"""

import argparse
import glob
import sys
import time


def row_id(line: bytes) -> str | None:
    i = line.find(b'"id": "')
    if i < 0:
        return None
    j = line.find(b'"', i + 7)
    return line[i + 7:j].decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="glob of the regenerated part files")
    ap.add_argument("--order", required=True, help="the 8B corpus jsonl whose row order to follow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--missing", required=True, help="where to list ids absent from the parts")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    index: dict[str, tuple[int, int, int]] = {}
    files = sorted(glob.glob(args.parts))
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
                    index[row_id(line[start:])] = (k, off + start, len(line) - start)
                off += len(line)
    print(f"70B rows {len(index)} from {len(files)} parts ({torn} torn lines) "
          f"in {time.time() - t0:.0f}s", flush=True)

    with open(args.order, "rb") as f:
        order = [row_id(line) for line in f]
    print(f"8B rows {len(order)}", flush=True)
    missing = [rid for rid in order if rid not in index]
    extra = len(index) - (len(order) - len(missing))
    with open(args.missing, "w") as f:
        f.writelines(rid + "\n" for rid in missing)
    print(f"missing {len(missing)} (listed in {args.missing}), not in the 8B corpus {extra}")
    if missing and not args.allow_missing:
        print("MERGE_INCOMPLETE")
        return 1

    handles = [open(path, "rb") for path in files]  # noqa: SIM115
    written = 0
    with open(args.out, "wb") as out:
        for rid in order:
            if rid not in index:
                continue
            k, off, length = index[rid]
            handles[k].seek(off)
            out.write(handles[k].read(length))
            written += 1
    for h in handles:
        h.close()
    print(f"wrote {written} rows to {args.out} in {time.time() - t0:.0f}s")
    print("MERGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
