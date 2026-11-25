"""Shuffle open-perfectblend once (seed 0) and write the prompt shards.

Yu's make_opb_shards.py (branch DSpark_Apertus_1.5_8B) with paths as arguments:
same filter, same shuffle, same ids, so the 70B regenerates the conversations of
the 8B corpus. --manifest compares the per-shard source counts with Yu's
manifest; --parts additionally cuts the concatenated shards into parts, one per
regeneration job.
"""

import argparse
import collections
import glob
import json
import os
import random
import sys

import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opb-dir", required=True, help="snapshot of the dataset revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--parts", type=int, default=0, help="also write N part files (0: none)")
    ap.add_argument("--manifest", help="Yu's prompts manifest to compare against")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.opb_dir + "/**/*.parquet", recursive=True))
    rows = []
    dropped = collections.Counter()
    for f in files:
        d = pq.read_table(f).to_pydict()
        for conv, src in zip(d["conversations"], d["source"], strict=True):
            msgs = [{"from": m.get("from"), "value": m.get("value")} for m in conv]
            if len(msgs) < 2 or msgs[0]["from"] not in ("human", "user") or not msgs[0]["value"]:
                dropped["short_or_no_human_first"] += 1
                continue
            rows.append({"source": src, "conversations": msgs})
    print("parquet files", len(files), "kept", len(rows), "dropped", dict(dropped))
    random.Random(0).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"opb-{i:07d}"

    per = (len(rows) + args.shards - 1) // args.shards
    manifest = []
    for k in range(args.shards):
        part = rows[k * per:(k + 1) * per]
        path = f"{args.out}/opb_shuffled_seed0_shard{k}of{args.shards}.jsonl"
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest.append({"shard": path, "rows": len(part),
                         "sources": collections.Counter(r["source"] for r in part)})
        print(path, len(part), "rows")
    with open(args.out + "/MANIFEST.json", "w") as f:
        json.dump({"seed": 0, "shards": args.shards, "total": len(rows), "dropped": dropped,
                   "manifest": manifest}, f, indent=1)

    if args.parts:
        os.makedirs(args.out + "/parts", exist_ok=True)
        per = (len(rows) + args.parts - 1) // args.parts
        for k in range(args.parts):
            with open(f"{args.out}/parts/part-{k:03d}.jsonl", "w") as f:
                for r in rows[k * per:(k + 1) * per]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("parts", args.parts, "of", per, "conversations")

    if args.manifest:
        with open(args.manifest) as f:
            ref = json.load(f)
        ok = ref["total"] == len(rows) and ref["dropped"] == dict(dropped)
        for mine, theirs in zip(manifest, ref["manifest"], strict=True):
            ok &= mine["rows"] == theirs["rows"] and dict(mine["sources"]) == theirs["sources"]
        print("MANIFEST_MATCH" if ok else "MANIFEST_MISMATCH")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
