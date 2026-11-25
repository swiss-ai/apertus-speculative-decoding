"""Estimate how much of the drafters' training corpus looks like each coverage slice.

    python3 -m apertus_bench.corpus_share CORPUS.jsonl --lid lid.176.bin --out share.json \\
        [--workers 64] [--limit-bytes N]

Every row of the regenerated Open-PerfectBlend corpus (Yu's thinking-off answers,
``text`` rendered with ``<|user_start|>`` / ``<|assistant_start|>`` turns) is classified
by language and by domain heuristics:

- language: fastText lid.176, applied to the last user turn and separately to the final
  answer (first 2000 characters, newlines removed), top label only;
- code: a fenced block in the answer. Its tag gives the language, and Python and the
  code-multi languages are counted apart;
- math: LaTeX or a GSM8K-style final answer in the answer (``\\boxed``, ``\\frac``,
  ``$$``, ``\\(``, ``####``);
- long-context: prompt of at least 8192 tokens (the row's own ``usage``);
- tool-call: ``metadata.is_tool_call``, or a ``<tools>`` / function schema in the prompt;
- json-schema: a JSON schema in the prompt, or an answer that is a JSON object;
- translation: a translation request in the last user turn (en/de/fr/it/es keywords);
- multi-turn: two or more user turns.

Domains overlap; each is a share of all rows. The output holds counts only, no text.
Runs in a compute job (``methods/eagle/launch/submit-node-command.sh``); the file is
split by byte ranges over the workers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

TURN = re.compile(r"<\|(user|assistant)_start\|>(.*?)<\|\1_end\|>", re.S)
FENCE = re.compile(r"```([A-Za-z0-9_+#.-]*)")
MATH = re.compile(r"\\boxed|\\frac|\$\$|\\\(|\\\[|^#### ", re.M)
TOOLS = re.compile(r"<tools>|\"parameters\"\s*:\s*\{|\bfunction[_ ]call", re.I)
SCHEMA = re.compile(r"json[ _-]?schema|<schema>|\"\$schema\"|\"properties\"\s*:\s*\{", re.I)
TRANSLATE = re.compile(
    r"\btranslat|übersetz|\btradu[icz]|tradurr|traduzione|traducción|traduction", re.I
)
CODE_MULTI = {
    "cpp": "cpp", "c++": "cpp", "go": "go", "golang": "go", "java": "java",
    "js": "js", "javascript": "js", "jsx": "js", "typescript": "js", "ts": "js",
    "rust": "rust", "rs": "rust",
}  # fmt: skip
PYTHON = {"python", "py", "python3"}
PROSE_FENCES = {"json", "text", "markdown", "md", "plaintext", "txt"}
LONG_PROMPT_TOKENS = 8192
LANGUAGE_LABELS = (
    "en", "de", "fr", "it", "es", "rm", "als", "zh", "ja", "hi", "ar", "sw", "yo", "am",
)  # fmt: skip


def turns(text: str) -> list[tuple[str, str]]:
    return [(role, body) for role, body in TURN.findall(text)]


def _is_json_object(text: str) -> bool:
    body = text.strip()
    if body.startswith("```json"):
        body = body.removeprefix("```json").rsplit("```", 1)[0].strip()
    if not body.startswith("{"):
        return False
    try:
        return isinstance(json.loads(body), dict)
    except ValueError:
        return False


def classify(row: dict[str, Any], lid: Callable[[str], str]) -> dict[str, Any] | None:
    """Labels of one corpus row, or None if its text has no user/answer turns."""
    parsed = turns(row.get("text", ""))
    users = [body for role, body in parsed if role == "user"]
    if not users or parsed[-1][0] != "assistant":
        return None
    answer = parsed[-1][1]
    prompt = "\n".join(body for role, body in parsed[:-1])
    metadata = row.get("metadata") or {}
    prompt_tokens = (metadata.get("usage") or {}).get("prompt_tokens") or 0
    fences = {tag.lower() for tag in FENCE.findall(answer) if tag}
    has_fence = "```" in answer
    domains = {
        # An untagged fence counts as code; a fence of only prose or JSON does not.
        "code": has_fence and (not fences or not fences <= PROSE_FENCES),
        "code-python": bool(fences & PYTHON),
        "code-multi": any(tag in CODE_MULTI for tag in fences),
        "math": bool(MATH.search(answer)),
        "long-context": prompt_tokens >= LONG_PROMPT_TOKENS,
        "tool-call": bool(metadata.get("is_tool_call")) or bool(TOOLS.search(prompt)),
        "json-schema": bool(SCHEMA.search(prompt)) or _is_json_object(answer),
        "translation": bool(TRANSLATE.search(users[-1])),
        "multi-turn": len(users) >= 2,
    }
    return {
        "prompt_language": lid(users[-1]),
        "answer_language": lid(answer),
        "domains": [name for name, hit in domains.items() if hit],
        "code_languages": sorted(CODE_MULTI[tag] for tag in fences if tag in CODE_MULTI),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": (metadata.get("usage") or {}).get("completion_tokens") or 0,
    }


def _fasttext_lid(model_path: str) -> Callable[[str], str]:
    import fasttext

    model = fasttext.load_model(model_path)

    def lid(text: str) -> str:
        sample = " ".join(text[:2000].split())
        if not sample:
            return "empty"
        labels, _ = model.predict(sample, k=1)
        return labels[0].removeprefix("__label__")

    return lid


def lines_in_range(path: str, start: int, end: int) -> Iterator[bytes]:
    """The lines that start in [start, end): adjacent ranges cover every line once."""
    with open(path, "rb") as handle:
        if start:
            handle.seek(start - 1)
            handle.readline()  # finish the line that straddles the boundary
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                return
            yield line


def _chunk(args: tuple[str, int, int, str]) -> dict[str, Any]:
    path, start, end, model_path = args
    lid = _fasttext_lid(model_path)
    counts: dict[str, Counter] = {
        key: Counter()
        for key in ("rows", "prompt_language", "answer_language", "domains", "code_languages")
    }
    tokens: Counter = Counter()
    for line in lines_in_range(path, start, end):
        labels = classify(json.loads(line), lid)
        if labels is None:
            counts["rows"]["unparsed"] += 1
            continue
        counts["rows"]["classified"] += 1
        counts["prompt_language"][labels["prompt_language"]] += 1
        counts["answer_language"][labels["answer_language"]] += 1
        counts["domains"].update(labels["domains"])
        counts["code_languages"].update(set(labels["code_languages"]))
        tokens[f"completion:{labels['prompt_language']}"] += labels["completion_tokens"]
        tokens["completion:all"] += labels["completion_tokens"]
        for domain in labels["domains"]:
            tokens[f"completion:{domain}"] += labels["completion_tokens"]
    return {**{k: dict(v) for k, v in counts.items()}, "completion_tokens": dict(tokens)}


def summarize(parts: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Counter] = {}
    for part in parts:
        for key, values in part.items():
            total.setdefault(key, Counter()).update(values)
    rows = total["rows"]["classified"]
    tokens = total["completion_tokens"]

    def share(counter: Counter) -> dict[str, dict[str, float]]:
        return {key: {"rows": count, "share": count / rows} for key, count in counter.most_common()}

    return {
        "rows": dict(total["rows"]),
        "prompt_language": share(total["prompt_language"]),
        "answer_language": share(total["answer_language"]),
        "domains": share(total["domains"]),
        "code_languages": share(total["code_languages"]),
        # Share of supervised (answer) tokens: what the drafter actually trains on.
        "completion_token_share": {
            key.removeprefix("completion:"): value / tokens["completion:all"]
            for key, value in sorted(tokens.items())
            if key != "completion:all"
        },
        "completion_tokens": tokens["completion:all"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-bench-corpus-share")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--lid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--limit-bytes", type=int, help="read only the first N bytes (a test)")
    args = parser.parse_args(argv)
    size = args.corpus.stat().st_size
    if args.limit_bytes:
        size = min(size, args.limit_bytes)
    bounds = [size * i // args.workers for i in range(args.workers + 1)]
    jobs = [
        (str(args.corpus), a, b, str(args.lid)) for a, b in zip(bounds, bounds[1:], strict=False)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(_chunk, jobs))
    result = {
        "corpus": str(args.corpus),
        "bytes_read": size,
        "corpus_bytes": args.corpus.stat().st_size,
        "lid_model": str(args.lid),
        "long_prompt_tokens": LONG_PROMPT_TOKENS,
        **summarize(parts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("rows", "completion_tokens")}, indent=2))
    for key in ("prompt_language", "domains", "code_languages"):
        print(key, {k: round(v["share"], 5) for k, v in list(result[key].items())[:20]})
    print("tracked languages", {
        label: round(result["prompt_language"].get(label, {"share": 0})["share"], 6)
        for label in LANGUAGE_LABELS
    })  # fmt: skip


if __name__ == "__main__":
    main()
