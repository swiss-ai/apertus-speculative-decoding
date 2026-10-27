"""Drafter updates (T5): per-domain prompt pools standing in for served traffic.

For each weak domain, a ``pool`` of adaptation prompts (the "captured
traces" once the target has answered them) and a ``dev`` split for the
trainer's simulated acceptance, both disjoint from the 8B test workloads:

- ``chat``: SFT-mix rows hash-assigned to ``train`` with domain chat_qa,
  multilingual or instruction_following (the chat test workload uses the
  ``heldout_test`` rows of the same domains), 128-2048 input tokens
- ``summarization``: finepdfs-edu eng/deu/fra/ita documents from files after
  the ones the workloads and the earlier training mix read, with the
  workload's instruction, document window and 2048-4096 input-token band,
  EN/DE/FR/IT 40/20/20/20
- ``multilingual``: the fr/de/it/es rows of Infomaniak-AI's
  speculators-multilingual set (machine-translated open-perfectblend
  prompts); pool from its train file, dev and a 128-prompt ``test`` workload
  from its test file, 16-2048 input tokens, equal language shares

Pools are written in hash order, so the first N rows are a nested subset for
every N. Every pool, dev and test row is checked against the committed test
and validation workloads by rendered prompt-token digest, normalized text
and 13-gram near-duplicates (``overlap_check``), and against each other.

Run inside the serving image on a compute node (pyarrow + transformers).
Output JSONL holds restricted source text and stays on the cluster; the
manifest holds only counts and digests.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from apertus_eagle.build_workloads import (
    FINEPDFS,
    MIX_PARQUET,
    SOURCE_TOKEN_WINDOW,
    STRATA,
    SUMMARY_INSTRUCTION,
    SUMMARY_SHARE,
    digest16,
)
from apertus_eagle.contract import REPO_ROOT

SEED = "apertus-drafter-updates-pools-v1"
# finepdfs files 0-1 feed the workloads, 2-3 the earlier 8B training mix.
SUMMARY_FIRST_FILE = 4
ML_LANGUAGES = ("fr", "de", "it", "es")
BANDS = {"chat": (128, 2048), "summarization": (2048, 4096), "multilingual": (16, 2048)}
MAX_TOKENS = {"chat": 256, "summarization": 512, "multilingual": 256}


def rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)


class RankedCandidates:
    """Keep the lowest ranks without ever ordering source dictionaries.

    FinePDFs can repeat an ID within or across shards. A monotonic source
    position breaks equal-rank/equal-ID ties reproducibly; later prompt
    deduplication still decides whether repeated content is eligible.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("candidate limit must be positive")
        self.limit = limit
        self.position = 0
        self.heap: list[tuple[int, str, int, dict[str, Any]]] = []

    def add(self, key: int, row: dict[str, Any]) -> None:
        entry = (-key, str(row["id"]), self.position, row)
        self.position += 1
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, entry)
        elif entry[:3] > self.heap[0][:3]:
            heapq.heapreplace(self.heap, entry)

    def rows(self) -> list[dict[str, Any]]:
        return [entry[3] for entry in sorted(self.heap, reverse=True)]


def language_quotas(total: int, shares: dict[str, float]) -> dict[str, int]:
    quota = {lang: int(total * share) for lang, share in shares.items()}
    first = next(iter(shares))
    quota[first] += total - sum(quota.values())
    return quota


def sharegpt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """User turns of a ShareGPT row up to (not including) its last assistant turn."""
    roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    messages = [
        {"role": roles[turn["from"]], "content": turn["value"]}
        for turn in row["conversations"]
        if turn.get("from") in roles
    ]
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages


def read_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Rows of a JSON array or JSON-lines file."""
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
        while first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


class Guard:
    """Rejects prompts that duplicate a protected workload prompt or an earlier row."""

    def __init__(self, protected: list[dict[str, Any]], protected_digests: set[str]):
        from apertus_eagle.overlap_check import normalize, shingles, user_text

        self._normalize, self._shingles, self._user_text = normalize, shingles, user_text
        self.exact = {normalize(user_text(row)) for row in protected}
        self.grams = [shingles(user_text(row)) for row in protected]
        self.index: dict[str, set[int]] = {}
        for position, gram_set in enumerate(self.grams):
            for gram in gram_set:
                self.index.setdefault(gram, set()).add(position)
        self.protected_digests = set(protected_digests)
        self.seen_digests: set[str] = set()
        self.rejected = {"protected_digest": 0, "protected_text": 0, "near_dup": 0, "repeat": 0}

    def protect(self, rows: list[dict[str, Any]]) -> None:
        """Also keep later rows from near-duplicating these (a new test workload)."""
        for row in rows:
            text = self._user_text(row)
            self.exact.add(self._normalize(text))
            self.grams.append(self._shingles(text))
            for gram in self.grams[-1]:
                self.index.setdefault(gram, set()).add(len(self.grams) - 1)

    def unprotected(self, row: dict[str, Any]) -> bool:
        """Recheck prior candidates after a dev/test split becomes protected."""
        from apertus_eagle.overlap_check import NEAR_DUP

        text = self._user_text(row)
        if self._normalize(text) in self.exact:
            self.rejected["protected_text"] += 1
            return False
        mine = self._shingles(text)
        hits: dict[int, int] = {}
        for gram in mine:
            for position in self.index.get(gram, ()):
                hits[position] = hits.get(position, 0) + 1
        if any(n / max(1, len(mine | self.grams[p])) >= NEAR_DUP for p, n in hits.items()):
            self.rejected["near_dup"] += 1
            return False
        return True

    def admit(self, row: dict[str, Any], digest: str) -> bool:
        if digest in self.protected_digests:
            self.rejected["protected_digest"] += 1
            return False
        if digest in self.seen_digests:
            self.rejected["repeat"] += 1
            return False
        if not self.unprotected(row):
            return False
        self.seen_digests.add(digest)
        return True


def chat_candidates(split_index: Path, limit: int) -> list[dict[str, Any]]:
    from apertus_eagle.parse_rendered import RenderParseError, mix_row_to_record
    from apertus_eagle.prepare_data import lookup_mix_text

    candidates = RankedCandidates(limit)
    domains = STRATA["chat"]["domains"]
    with split_index.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train" and row["domain"] in domains:
                candidates.add(rank(f"chat:{row['id']}"), row)
    rows = candidates.rows()
    texts = lookup_mix_text(MIX_PARQUET, {row["id"] for row in rows})
    out = []
    for row in rows:
        if row["id"] not in texts:
            continue
        try:
            record = mix_row_to_record(
                {**row, "text": texts[row["id"]], "dataset_source": row["source"]}
            )
        except (RenderParseError, KeyError):
            continue
        messages = [m for m in record["messages"] if m["role"] != "developer"]
        if messages and messages[-1]["role"] == "user":
            out.append(
                {
                    "id": f"chat-{row['id'][:16]}",
                    "source_id": row["id"],
                    "source": row["source"],
                    "domain": row["domain"],
                    "messages": messages,
                }
            )
    return out


def summarization_candidates(
    total: int, oversample: float, files: int
) -> dict[str, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    by_language: dict[str, list[dict[str, Any]]] = {}
    for language, root in FINEPDFS.items():
        # About 80% of eng candidates pass the Apertus band, 40% of deu/fra/ita
        # (workload manifest): the other languages get twice the candidates.
        factor = 1 if language == "eng" else 2
        want = int(total * SUMMARY_SHARE[language] * oversample * factor)
        candidates = RankedCandidates(want)
        low, high = SOURCE_TOKEN_WINDOW[language]
        paths = sorted(root.glob("*.parquet"))[SUMMARY_FIRST_FILE : SUMMARY_FIRST_FILE + files]
        for path in paths:
            columns = ["id", "text", "token_count", "is_truncated", "fw_edu_scores"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
                for doc in batch.to_pylist():
                    if doc["is_truncated"] or not low <= int(doc["token_count"]) <= high:
                        continue
                    scores = doc["fw_edu_scores"] or []
                    if not scores or sum(scores) / len(scores) < 2.5:
                        continue
                    candidates.add(rank(f"summ:{doc['id']}"), {**doc, "source_file": path.name})
        by_language[language] = [
            {
                "id": f"summarization-{language}-{digest16(doc['id'])}",
                "source_id": doc["id"],
                "source": f"finepdfs-edu/{language}",
                "source_file": doc["source_file"],
                "language": language,
                "messages": [
                    {"role": "user", "content": f"{SUMMARY_INSTRUCTION[language]}\n\n{doc['text']}"}
                ],
            }
            for doc in candidates.rows()
        ]
    return by_language


def multilingual_candidates(
    path: Path, limit: int, exclude_sources: set[str] | None = None
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    """Candidates per language and every source id in the file.

    Rows whose source id is in ``exclude_sources`` are skipped: the same
    source row translated into another language is not a new prompt.
    """
    candidates = {lang: RankedCandidates(limit) for lang in ML_LANGUAGES}
    sources: set[str] = set()
    for position, row in enumerate(read_json_rows(path)):
        sources.add(str(row.get("id")))
        language = row.get("language")
        if language not in candidates or str(row.get("id")) in (exclude_sources or ()):
            continue
        key = f"{row.get('id')}:{position}"
        candidates[language].add(rank(f"ml:{key}"), {**row, "id": key})
    out = {}
    for language, candidate_pool in candidates.items():
        out[language] = []
        for row in candidate_pool.rows():
            messages = sharegpt_messages(row)
            if messages and messages[-1]["role"] == "user":
                out[language].append(
                    {
                        "id": f"multilingual-{language}-{digest16(row['id'])}",
                        "source_id": row["id"].rsplit(":", 1)[0],
                        "source_position": int(row["id"].rsplit(":", 1)[1]),
                        "source": f"Infomaniak-AI/speculators-multilingual/{path.name}",
                        "language": language,
                        "messages": messages,
                    }
                )
    return out, sources


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    tmp = path.with_suffix(f".jsonl.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-build-adaptation-pools")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--dev", type=int, default=256)
    parser.add_argument("--test", type=int, default=128, help="multilingual test workload")
    parser.add_argument(
        "--protect",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "workloads/8b/eagle-8b-test.jsonl",
            REPO_ROOT / "workloads/8b/eagle-8b-validation.jsonl",
        ],
        help="workload JSONL whose prompts no pool/dev row may duplicate",
    )
    parser.add_argument(
        "--split-index", type=Path, default=REPO_ROOT / "results/70b/eagle/data/split-index.jsonl"
    )
    parser.add_argument("--infomaniak-dir", type=Path, required=True)
    parser.add_argument("--summary-files", type=int, default=4)
    parser.add_argument("--oversample", type=float, default=2.6)
    parser.add_argument("--domains", default="chat,summarization,multilingual")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from apertus_eagle.contract import load_contract
    from apertus_eagle.generate_targets import render_prompt

    contract = load_contract(args.contract)
    tokenizer = AutoTokenizer.from_pretrained(contract["source"]["authorized_checkpoint"])

    def digest_of(ids: list[int]) -> str:
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    protected = [json.loads(line) for path in args.protect for line in path.open() if line.strip()]
    protected_digests = {digest_of(render_prompt(tokenizer, row["messages"])) for row in protected}
    guard = Guard(protected, protected_digests)
    manifest: dict[str, Any] = {
        "seed": SEED,
        "target": contract["target"]["identity"],
        "renderer": "served chat template, enable_thinking=False, add_generation_prompt=True",
        "protected": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.protect
        },
        "protected_prompts": len(protected),
        "infomaniak_revision": (args.infomaniak_dir / "revision.txt").read_text().strip()
        if (args.infomaniak_dir / "revision.txt").is_file()
        else None,
        "domains": {},
    }
    total = args.pool + args.dev
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def eligible(rows: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
        low, high = BANDS[domain]
        kept = []
        for row in rows:
            try:
                ids = render_prompt(tokenizer, row["messages"])
            except (ValueError, TypeError):
                continue
            if not low <= len(ids) <= high:
                continue
            digest = digest_of(ids)
            if guard.admit(row, digest):
                kept.append(
                    {
                        **row,
                        "workload": domain,
                        "max_tokens": MAX_TOKENS[domain],
                        "input_tokens": len(ids),
                        "prompt_ids_sha256": digest,
                    }
                )
        return kept

    def by_quota(
        pools: dict[str, list[dict[str, Any]]], shares: dict[str, float], need: int
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        quota = language_quotas(need, shares)
        chosen, short = [], {}
        for language, n in quota.items():
            chosen += pools[language][:n]
            if len(pools[language]) < n:
                short[language] = n - len(pools[language])
        return chosen, short

    domains = [d for d in args.domains.split(",") if d]
    for domain in domains:
        rejected_before = dict(guard.rejected)
        report: dict[str, Any] = {"band": list(BANDS[domain]), "max_tokens": MAX_TOKENS[domain]}
        splits: dict[str, list[dict[str, Any]]] = {}
        if domain == "chat":
            rows = eligible(chat_candidates(args.split_index, int(total * args.oversample)), domain)
            rows.sort(key=lambda r: rank(f"split:{r['id']}"))
            report["eligible"] = len(rows)
            if len(rows) < total:
                raise SystemExit(f"chat: {len(rows)} eligible prompts < {total}")
            dev = rows[: args.dev]
            guard.protect(dev)
            pool = [row for row in rows[args.dev :] if guard.unprotected(row)][: args.pool]
            if len(pool) < args.pool:
                raise SystemExit(
                    f"chat: {len(pool)} pool prompts after dev protection < {args.pool}"
                )
            splits = {"dev": dev, "pool": pool}
        elif domain == "summarization":
            candidates = summarization_candidates(total, args.oversample, args.summary_files)
            pools = {lang: eligible(rows, domain) for lang, rows in candidates.items()}
            report["eligible_by_language"] = {lang: len(rows) for lang, rows in pools.items()}
            dev, short_dev = by_quota(pools, SUMMARY_SHARE, args.dev)
            guard.protect(dev)
            used = {r["id"] for r in dev}
            rest = {
                lang: [r for r in rows if r["id"] not in used and guard.unprotected(r)]
                for lang, rows in pools.items()
            }
            pool, short_pool = by_quota(rest, SUMMARY_SHARE, args.pool)
            if short_dev or short_pool:
                raise SystemExit(f"summarization shortfall: dev {short_dev}, pool {short_pool}")
            splits = {"dev": dev, "pool": pool}
        elif domain == "multilingual":
            shares = {lang: 1 / len(ML_LANGUAGES) for lang in ML_LANGUAGES}
            test_c, test_sources = multilingual_candidates(
                args.infomaniak_dir / "test_final.json",
                int((args.dev + args.test) * args.oversample / 4),
            )
            # Test and dev are chosen first and protected, so no pool row repeats,
            # near-duplicates or translates the same source row as one of them.
            held = {lang: eligible(rows, domain) for lang, rows in test_c.items()}
            test, short_test = by_quota(held, shares, args.test)
            used = {r["id"] for r in test}
            rest = {lang: [r for r in rows if r["id"] not in used] for lang, rows in held.items()}
            dev, short_dev = by_quota(rest, shares, args.dev)
            guard.protect(test + dev)
            train_c, _ = multilingual_candidates(
                args.infomaniak_dir / "train_final.json",
                int(args.pool * args.oversample / 4),
                exclude_sources=test_sources,
            )
            pools = {lang: eligible(rows, domain) for lang, rows in train_c.items()}
            pool, short_pool = by_quota(pools, shares, args.pool)
            report["eligible_by_language"] = {
                "test_file": {lang: len(rows) for lang, rows in held.items()},
                "train_file": {lang: len(rows) for lang, rows in pools.items()},
            }
            if short_test or short_dev or short_pool:
                raise SystemExit(
                    f"multilingual shortfall: test {short_test}, dev {short_dev}, pool {short_pool}"
                )
            splits = {"test": test, "dev": dev, "pool": pool}
        else:
            raise SystemExit(f"unknown domain {domain}")
        directory = args.output_dir / domain
        directory.mkdir(parents=True, exist_ok=True)
        ids = [r["id"] for rows in splits.values() for r in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{domain}: a row id appears in two splits")
        for split, rows in splits.items():
            # Hash order: the first N pool rows are the N-trace subset.
            rows.sort(key=lambda r: rank(f"order:{r['id']}"))
            report[split] = {
                "rows": len(rows),
                "sha256": write_jsonl(directory / f"{split}.jsonl", rows),
                "input_tokens": [
                    min(r["input_tokens"] for r in rows),
                    max(r["input_tokens"] for r in rows),
                ],
                "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows), 1),
                "languages": {
                    lang: sum(1 for r in rows if r.get("language") == lang)
                    for lang in sorted({r.get("language") for r in rows if r.get("language")})
                },
            }
        report["rejected"] = {k: guard.rejected[k] - rejected_before[k] for k in guard.rejected}
        manifest["domains"][domain] = report
        print(json.dumps({domain: report}), flush=True)
    name = "manifest.json" if len(domains) == 3 else f"manifest-{'-'.join(domains)}.json"
    (args.output_dir / name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
