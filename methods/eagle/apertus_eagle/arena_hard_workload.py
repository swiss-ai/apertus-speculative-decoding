"""Build the Arena-Hard v0.1 load-test workload, split by training-corpus contamination.

    python -m apertus_eagle.arena_hard_workload --overlap-report REPORT.json \\
        --cache DIR --out workloads/8b/arena-hard.jsonl [--manifest M.json] [--max-tokens 1024]

The overlap report (``benchmark_overlap``, docs/benchmark-contamination.md)
names contaminated prompts by position (``arena-hard-00000`` ...) in the
de-duplicated prompt order of one revision of ``lmarena-ai/arena-hard-auto``.
This rebuilds that order from the same pinned revision (``benchmark_texts``)
and labels every prompt ``arena-hard-clean`` or ``arena-hard-contaminated``,
so a sweep measures the two as separate cells; the full set is pooled from
both (``sweep_report --pool``). Rows keep Arena-Hard's own ``question_id``.
Runs in a compute job (Hugging Face download).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apertus_eagle.benchmark_overlap import BENCHMARKS, benchmark_texts, prompt_text, read_rows

NAME = "arena-hard"
LABELS = ("arena-hard-clean", "arena-hard-contaminated")


def build_rows(files: list[Path], contaminated: set[str], max_tokens: int) -> list[dict[str, Any]]:
    texts = benchmark_texts(files)
    ids = {f"{NAME}-{i:05d}" for i in range(len(texts))}
    unknown = contaminated - ids
    if unknown:
        raise SystemExit(f"report ids not in the prompt list: {sorted(unknown)[:5]}")
    question_ids: dict[str, str] = {}
    for f in files:
        for row in read_rows(f):
            text = prompt_text(row)
            if text and text not in question_ids:
                question_ids[text] = str(row.get("question_id") or row.get("uid") or "")
    rows = []
    for i, text in enumerate(texts):
        prompt_id = f"{NAME}-{i:05d}"
        rows.append(
            {
                "id": prompt_id,
                "workload": LABELS[prompt_id in contaminated],
                "question_id": question_ids[text],
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-arena-hard-workload")
    parser.add_argument("--overlap-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--expect", type=int, nargs=2, default=(500, 129), metavar=("ALL", "BAD"))
    args = parser.parse_args(argv)

    entry = json.loads(args.overlap_report.read_text())["benchmarks"][NAME]
    repo, patterns = BENCHMARKS[NAME][0]
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=entry["revision"],
            allow_patterns=entry["files"],
            local_dir=args.cache / repo.replace("/", "__"),
        )
    )
    files = [local / name for name in entry["files"]]
    contaminated = set(entry["contaminated_ids"])
    rows = build_rows(files, contaminated, args.max_tokens)
    counts = {label: sum(r["workload"] == label for r in rows) for label in LABELS}
    expected = {"all": args.expect[0], "contaminated": args.expect[1]}
    if (len(rows), counts[LABELS[1]]) != tuple(expected.values()):
        raise SystemExit(f"expected {expected}, built {len(rows)} rows, {counts}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    from apertus_bench.workloads import file_sha256

    manifest = {
        "file": str(args.out),
        "sha256": file_sha256(args.out),
        "rows": len(rows),
        "dataset": repo,
        "revision": entry["revision"],
        "files": entry["files"],
        "selection": (
            "every distinct prompt in file order (benchmark_texts); workload "
            "arena-hard-contaminated for the overlap report's contaminated_ids, "
            "else arena-hard-clean"
        ),
        "overlap_report": str(args.overlap_report),
        "counts": counts,
        "max_tokens": args.max_tokens,
    }
    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
