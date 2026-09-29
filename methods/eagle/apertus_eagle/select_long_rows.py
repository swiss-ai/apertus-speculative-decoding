"""Pick the longest generate_targets rows, for memory tests at the maximum length.

Most conversations are short (open-perfectblend p99 2.3k tokens), so a random
sample rarely reaches ``max_seq_length``. This writes the ``--count`` longest
rows (prompt + generated tokens) of the inputs, longest first. Standard library.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-select-long-rows")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args(argv)
    heap: list[tuple[int, int, str]] = []
    seen = 0
    for path in args.input:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                n = len(row["prompt_ids"]) + len(row["generated_ids"])
                item = (n, seen, line)
                seen += 1
                if len(heap) < args.count:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
    longest = sorted(heap, reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for _n, _i, line in longest:
            out.write(line if line.endswith("\n") else line + "\n")
    lengths = [n for n, _i, _l in longest]
    print(
        json.dumps(
            {"rows_scanned": seen, "kept": len(lengths), "max": lengths[0], "min": lengths[-1]}
        )
    )


if __name__ == "__main__":
    main()
