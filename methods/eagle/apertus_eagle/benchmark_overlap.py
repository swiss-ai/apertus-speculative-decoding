"""Check the DSpark evaluation benchmarks against the open-perfectblend training corpus.

Both drafters train on open-perfectblend, whose sources (MetaMathQA,
orca-math, evol-codealpaca, UltraInteract, lmsys arena, ...) are built from
or near public benchmarks. A benchmark prompt the drafter saw in training
inflates its acceptance on that benchmark. For each benchmark this reports how
many prompts have an exact (normalized) or 13-gram near duplicate
(Jaccard >= 0.5 on the first 2,000 characters, as ``overlap_check``) among the
corpus's user turns, and from which corpus sources.

Benchmarks come from their public Hugging Face copies (``BENCHMARKS``); the
DSpark harness may load different copies, so each report row records the repo,
revision and files used. Runs in a compute job (network, 1.4M rows).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apertus_eagle.overlap_check import NEAR_DUP, normalize, shingles

# name -> candidate (repo, allow_patterns); the first that yields prompts is used.
BENCHMARKS: dict[str, list[tuple[str, list[str]]]] = {
    "gsm8k": [("openai/gsm8k", ["main/test-*.parquet"])],
    "math500": [("HuggingFaceH4/MATH-500", ["test.jsonl"])],
    "aime25": [
        ("opencompass/AIME2025", ["aime2025-*.jsonl"]),
        ("yentinglin/aime_2025", ["data/*.parquet"]),
    ],
    "mbpp": [("google-research-datasets/mbpp", ["full/test-*.parquet"])],
    "humaneval": [("openai/openai_humaneval", ["openai_humaneval/test-*.parquet"])],
    # release_v6: test.jsonl .. test6.jsonl
    "livecodebench": [("livecodebench/code_generation_lite", ["test*.jsonl"])],
    "mt-bench": [("HuggingFaceH4/mt_bench_prompts", ["raw/question.jsonl"])],
    "alpaca": [("tatsu-lab/alpaca_eval", ["alpaca_eval.json"])],
    # v0.1: the 500-prompt set; v2.0 is a different benchmark.
    "arena-hard": [("lmarena-ai/arena-hard-auto", ["data/arena-hard-v0.1/question.jsonl"])],
}
# Fields that hold the prompt, in order of preference.
TEXT_KEYS = ("question", "problem", "prompt", "instruction", "question_content", "text", "turns")


def prompt_text(row: dict[str, Any]) -> str | None:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            parts = [v if isinstance(v, str) else (v or {}).get("content", "") for v in value]
            joined = "\n".join(p for p in parts if p)
            if joined.strip():
                return joined
    return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else list(data.values())[0]


def fetch_benchmarks(cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    prompts: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for name, candidates in BENCHMARKS.items():
        errors = []
        for repo, patterns in candidates:
            try:
                info = api.dataset_info(repo)
                local = Path(
                    snapshot_download(
                        repo,
                        repo_type="dataset",
                        revision=info.sha,
                        allow_patterns=patterns,
                        local_dir=cache / repo.replace("/", "__"),
                    )
                )
                # Only the pinned files: the cache also holds the downloader's
                # metadata and files from earlier, wider patterns (Arena-Hard v2.0).
                files = sorted(
                    p
                    for p in local.rglob("*")
                    if p.is_file()
                    and ".cache" not in p.parts
                    and any(fnmatch.fnmatch(str(p.relative_to(local)), g) for g in patterns)
                )
                found = [t for f in files for t in map(prompt_text, read_rows(f)) if t]
                texts = list(dict.fromkeys(found))  # one entry per distinct prompt
            except Exception as error:  # noqa: BLE001 - try the next copy, report all
                errors.append(f"{repo}: {type(error).__name__}: {error}")
                continue
            if not texts:
                errors.append(f"{repo}: no prompt field in {[f.name for f in files]}")
                continue
            provenance[name] = {
                "repo": repo,
                "revision": info.sha,
                "files": [str(f.relative_to(local)) for f in files],
                "prompts": len(texts),
            }
            prompts += [
                {"benchmark": name, "id": f"{name}-{i:05d}", "text": t} for i, t in enumerate(texts)
            ]
            break
        if name not in provenance:
            provenance[name] = {"error": errors}
    return prompts, provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-benchmark-overlap")
    parser.add_argument("--corpus", type=Path, required=True, help="perfectblend conversations")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    prompts, provenance = fetch_benchmarks(args.cache)
    exact = {hashlib.sha256(normalize(p["text"]).encode()).hexdigest(): p for p in prompts}
    grams = [shingles(p["text"]) for p in prompts]
    index: dict[str, set[int]] = defaultdict(set)
    for i, gram_set in enumerate(grams):
        for gram in gram_set:
            index[gram].add(i)

    hit_exact: dict[str, set[str]] = defaultdict(set)
    hit_near: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    rows = 0
    with args.corpus.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            text = "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")
            key = hashlib.sha256(normalize(text).encode()).hexdigest()
            if key in exact:
                p = exact[key]
                hit_exact[p["benchmark"]].add(p["id"])
                sources[p["benchmark"]][row["source"]] += 1
                continue
            mine = shingles(text)
            counts: Counter[int] = Counter()
            for gram in mine:
                for i in index.get(gram, ()):
                    counts[i] += 1
            for i, shared in counts.items():
                union = len(mine | grams[i])
                if union and shared / union >= NEAR_DUP:
                    p = prompts[i]
                    hit_near[p["benchmark"]].add(p["id"])
                    sources[p["benchmark"]][row["source"]] += 1

    report: dict[str, Any] = {
        "corpus": str(args.corpus),
        "corpus_rows": rows,
        "rule": f"exact normalized user text, or 13-gram Jaccard >= {NEAR_DUP}",
        "benchmarks": {},
    }
    for name, meta in provenance.items():
        total = meta.get("prompts", 0)
        contaminated = hit_exact[name] | hit_near[name]
        report["benchmarks"][name] = {
            **meta,
            "exact": len(hit_exact[name]),
            "near_only": len(hit_near[name] - hit_exact[name]),
            "contaminated": len(contaminated),
            "share": len(contaminated) / total if total else None,
            "corpus_rows_matching_by_source": dict(sources[name].most_common()),
            "contaminated_ids": sorted(contaminated),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in report["benchmarks"].items():
        print(name, {k: row.get(k) for k in ("prompts", "exact", "near_only", "share", "error")})


if __name__ == "__main__":
    main()
