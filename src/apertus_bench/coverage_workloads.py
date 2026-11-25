"""Build the workload-coverage slices: small prompt sets for under-represented
languages and domains, one workload label per slice, in one file.

    python3 -m apertus_bench.coverage_workloads --raw RAW --tokenizer TARGET_DIR \\
        --out workloads/coverage/coverage.jsonl --manifest workloads/coverage/manifest.json

RAW holds pinned public datasets as downloaded from Hugging Face (``RAW/<repo>/<file>``
and ``RAW/revisions.txt``, one ``repo sha`` per line). Parquet files are read from a
``.jsonl`` copy next to them (one row per line, as ``pyarrow`` gives them), because the
serving image has no pyarrow. Every slice has --count prompts (64), drawn in a fixed
hash order, so a rebuild from the same revisions gives the same file. Every row turns
thinking off (``enable_thinking: false``), as the drafters were trained.

Slices (label: source):

- ``lang-<iso639-3>``: MURI-IT (akoksal/muri-it-language-split, Apache-2.0), the
  reverse-instruction (``MRI``) rows of the test split, so every language asks for the
  same kind of task. ``lang-eng`` is the English control. Yoruba has 7 MRI rows, so
  ``lang-yor`` takes its xP3/SuperNatural-Instructions rows instead (a different task mix).
- ``code-multi``: HumanEvalPack (bigcode/humanevalpack, MIT) instructions in C++, Go,
  Java, JavaScript and Rust. Python is in the existing probe.
- ``math-gsm8k``: GSM8K test questions (openai/gsm8k, MIT).
- ``long-context``: LongBench (zai-org/LongBench, MIT) documents of 8k to 24k prompt
  tokens. Half are summarization (gov_report, qmsum), half QA (qasper,
  multifieldqa_en, narrativeqa, musique).
- ``tool-call`` / ``json-schema``: Hermes function-calling v1 (NousResearch, Apache-2.0).
  Single-turn function calling with the tools in the system prompt, and JSON output
  that follows a schema.
- ``translate-from-en`` / ``translate-to-en``: WMT24++ (google/wmt24pp, Apache-2.0).
  Paragraphs between English and German, French, Italian and Spanish, 16 per language.
  The reverse direction translates the human reference.
- ``multi-turn``: MT-Bench second turns (lmsys/mt_bench_human_judgments, CC-BY-4.0),
  with GPT-4's first-turn answer as the assistant turn.

The output holds third-party text. It stays on the cluster; the repository keeps this
builder and the manifest (digests, revisions, counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

SEED = "workload-coverage-v1"
NO_THINKING = {"enable_thinking": False}
MURI = "akoksal/muri-it-language-split"
LANGUAGES = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "ita": "Italian",
    "spa": "Spanish",
    "roh": "Romansh",
    "zho": "Chinese",
    "jpn": "Japanese",
    "hin": "Hindi",
    "ara": "Arabic",
    "swa": "Swahili",
    "yor": "Yoruba",
    "amh": "Amharic",
}
# Too few MRI rows: these languages take the other MURI-IT sources.
NON_MRI_LANGUAGES = {"yor"}
CODE_LANGUAGES = ("cpp", "go", "java", "js", "rust")
WMT_PAIRS = {"de_DE": "German", "fr_FR": "French", "it_IT": "Italian", "es_MX": "Spanish"}
LONG_SUMMARY = {
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report."
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more paragraphs.\n\nTranscript:\n"
        "{context}\n\nQuery: {input}"
    ),
}
LONG_QA = ("qasper", "multifieldqa_en", "narrativeqa", "musique")
LONG_QA_TEMPLATE = (
    "Read the following text and answer the question. Explain your answer in a short "
    "paragraph.\n\n{context}\n\nQuestion: {input}"
)
LONG_MIN_TOKENS = 8192
LONG_MAX_TOKENS = 24576


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def pick(items: Iterable[tuple[str, Any]], count: int) -> list[tuple[str, Any]]:
    """The first `count` distinct-key items in hash order (deterministic, source-order free)."""
    unique = dict(items)
    return [(key, unique[key]) for key in sorted(unique, key=rank)[:count]]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row(label: str, key: str, messages: list[dict[str, str]], max_tokens: int, **extra: Any):
    return {
        "id": f"{label}-{rank(key)[:12]}",
        "workload": label,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": NO_THINKING,
        **extra,
    }


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def language_rows(raw: Path, count: int, max_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for code in LANGUAGES:
        records = read_jsonl(raw / MURI / code / "test-00000-of-00001.jsonl")
        keep = (
            (r["input"].strip(), r)
            for r in records
            if (r["dataset_name"] == "MRI") != (code in NON_MRI_LANGUAGES)
            and 16 <= len(r["input"].strip()) <= 2000
        )
        chosen = pick(keep, count)
        if len(chosen) < count:
            raise SystemExit(f"lang-{code}: {len(chosen)} prompts, need {count}")
        for text, record in chosen:
            rows.append(
                row(
                    f"lang-{code}", f"{code}:{text}", user(text), max_tokens,
                    source=f"{MURI}:{record['dataset_name']}",
                )
            )  # fmt: skip
    return rows


def split_evenly(total: int, parts: int) -> list[int]:
    return [total // parts + (i < total % parts) for i in range(parts)]


def code_rows(raw: Path, count: int, max_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for language, n in zip(CODE_LANGUAGES, split_evenly(count, len(CODE_LANGUAGES)), strict=True):
        records = read_jsonl(raw / "bigcode/humanevalpack" / language / "test-00000-of-00001.jsonl")
        for task_id, record in pick(((r["task_id"], r) for r in records), n):
            rows.append(
                row("code-multi", task_id, user(record["instruction"]), max_tokens, source=task_id)
            )
    return rows


def math_rows(raw: Path, count: int, max_tokens: int) -> list[dict[str, Any]]:
    records = read_jsonl(raw / "openai/gsm8k/main/test-00000-of-00001.jsonl")
    chosen = pick(((r["question"], r) for r in records), count)
    return [row("math-gsm8k", q, user(q), max_tokens) for q, _ in chosen]


def long_rows(
    raw: Path, count: int, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[dict[str, Any]]:
    archive = zipfile.ZipFile(raw / "zai-org/LongBench/data.zip")
    # Half summarization, half QA, split evenly over the subsets of each.
    plan = list(zip(LONG_SUMMARY.items(), split_evenly(count // 2, len(LONG_SUMMARY)), strict=True))
    plan += zip(
        ((name, LONG_QA_TEMPLATE) for name in LONG_QA),
        split_evenly(count - count // 2, len(LONG_QA)),
        strict=True,
    )
    rows = []
    for (name, template), wanted in plan:
        with archive.open(f"data/{name}.jsonl") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        taken = 0
        for key, record in sorted(((r["_id"], r) for r in records), key=lambda kv: rank(kv[0])):
            text = template.format(context=record["context"], input=record.get("input", ""))
            tokens = count_tokens(text)
            if not LONG_MIN_TOKENS <= tokens <= LONG_MAX_TOKENS:
                continue
            rows.append(
                row(
                    "long-context", f"{name}:{key}", user(text), max_tokens,
                    source=f"LongBench:{name}:{key}", prompt_tokens=tokens,
                )
            )  # fmt: skip
            taken += 1
            if taken == wanted:
                break
        if taken < wanted:
            raise SystemExit(f"long-context {name}: {taken} prompts in range")
    return rows


MT_BENCH_FILE = "gpt4_pair-00000-of-00001-c0b431264a82ddc0.jsonl"
HERMES_ROLES = {"system": "system", "human": "user"}


def hermes_rows(raw: Path, label: str, file: str, count: int, max_tokens: int):
    records = json.loads((raw / "NousResearch/hermes-function-calling-v1" / file).read_text())
    rows = []
    for key, record in pick(((r["id"], r) for r in records), count):
        turns = record["conversations"]
        if [t["from"] for t in turns[:2]] != ["system", "human"]:
            raise SystemExit(f"{label} {key}: expected system then human, got {turns[:2]}")
        messages = [{"role": HERMES_ROLES[t["from"]], "content": t["value"]} for t in turns[:2]]
        rows.append(row(label, key, messages, max_tokens, source=f"hermes:{file}:{key}"))
    return rows


def translation_rows(raw: Path, count: int, max_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for pair, (lp, language) in zip(
        split_evenly(count, len(WMT_PAIRS)), WMT_PAIRS.items(), strict=True
    ):
        records = read_jsonl(raw / "google/wmt24pp" / f"en-{lp}.jsonl")
        keep = [
            (f"{r['document_id']}:{r['segment_id']}", r)
            for r in records
            if not r["is_bad_source"] and r["domain"] != "canary"
            and 300 <= len(r["source"]) <= 2000
        ]  # fmt: skip
        chosen = pick(keep, 2 * pair)
        for key, record in chosen[:pair]:
            text = (
                f"Translate the following English text into {language}. Reply with the "
                f"translation only.\n\n{record['source']}"
            )
            rows.append(
                row(
                    "translate-from-en",
                    f"{lp}:{key}",
                    user(text),
                    max_tokens,
                    source=f"wmt24pp:en-{lp}:{key}",
                )
            )
        for key, record in chosen[pair:]:
            text = (
                f"Translate the following {language} text into English. Reply with the "
                f"translation only.\n\n{record['target']}"
            )
            rows.append(
                row(
                    "translate-to-en",
                    f"{lp}:{key}",
                    user(text),
                    max_tokens,
                    source=f"wmt24pp:{lp}-en:{key}",
                )
            )
    return rows


def multi_turn_rows(raw: Path, count: int, max_tokens: int) -> list[dict[str, Any]]:
    path = raw / "lmsys/mt_bench_human_judgments/data" / MT_BENCH_FILE
    conversations: dict[str, list[dict[str, str]]] = {}
    for record in read_jsonl(path):
        if str(record["turn"]) != "2":
            continue
        for side in ("a", "b"):
            if record[f"model_{side}"] == "gpt-4":
                conversation = record[f"conversation_{side}"]
                if [m["role"] for m in conversation] == ["user", "assistant"] * 2:
                    conversations.setdefault(str(record["question_id"]), conversation[:3])
    return [
        row("multi-turn", f"mt-bench:{qid}", messages, max_tokens, source=f"mt-bench:{qid}")
        for qid, messages in pick(conversations.items(), count)
    ]


def build(raw: Path, count: int, max_tokens: int, count_tokens: Callable[[str], int]):
    rows = language_rows(raw, count, max_tokens)
    rows += code_rows(raw, count, max_tokens)
    rows += math_rows(raw, count, max_tokens)
    rows += long_rows(raw, count, max_tokens, count_tokens)
    rows += hermes_rows(raw, "tool-call", "func-calling-singleturn.json", count, max_tokens)
    rows += hermes_rows(raw, "json-schema", "json-mode-singleturn.json", count, max_tokens)
    rows += translation_rows(raw, count, max_tokens)
    rows += multi_turn_rows(raw, count, max_tokens)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate prompt ids")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-bench-coverage-workloads")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from apertus_bench.workloads import file_sha256, load_prompts

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = build(
        args.raw, args.count, args.max_tokens,
        lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]),
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    load_prompts(args.out)  # the bench client's own validation
    slices: dict[str, dict[str, Any]] = {}
    for record in rows:
        entry = slices.setdefault(
            record["workload"], {"rows": 0, "prompt_sha256": hashlib.sha256()}
        )
        entry["rows"] += 1
        entry["prompt_sha256"].update(json.dumps(record["messages"], ensure_ascii=False).encode())
    revisions = dict(
        line.split()
        for line in (args.raw / "revisions.txt").read_text().splitlines()
        if line.strip()
    )
    manifest = {
        "file": str(args.out),
        "sha256": file_sha256(args.out),
        "rows": len(rows),
        "seed": SEED,
        "count_per_slice": args.count,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": NO_THINKING,
        "tokenizer": str(args.tokenizer),
        "long_context_prompt_tokens": [LONG_MIN_TOKENS, LONG_MAX_TOKENS],
        "revisions": revisions,
        "slices": {
            label: {"rows": e["rows"], "prompt_sha256": e["prompt_sha256"].hexdigest()}
            for label, e in slices.items()
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
