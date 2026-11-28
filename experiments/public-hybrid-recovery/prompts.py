"""Extract the eight public DAPO prompts used by this campaign (requires PyArrow)."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

PARQUET_SHA = "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
PROMPTS_SHA = "26e517de15c81e21f8af22e4b484a2502a2b8d3b9fbae13a5ded00db45a57546"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.parquet.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != PARQUET_SHA:
            raise ValueError("unexpected DAPO parquet digest")
    rows = next(pq.ParquetFile(args.parquet).iter_batches(batch_size=8, columns=["prompt"]))
    payload = (json.dumps([r["prompt"] for r in rows.to_pylist()]) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != PROMPTS_SHA:
        raise ValueError("unexpected rendered prompt digest")
    with args.output.open("xb") as f:
        f.write(payload)


if __name__ == "__main__":
    main()
