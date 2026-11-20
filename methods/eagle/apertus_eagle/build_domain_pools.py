"""Domain drafters (A3): prompt pools for math, code and tool use.

Companion of ``build_adaptation_pools`` (chat, summarization, multilingual).
Each domain gets a ``dev`` split and a ``pool`` from public datasets at
pinned revisions (pre-downloaded under ``--source-root/<owner>__<name>``
with a ``revision.txt``), disjoint from the protected evaluation workloads by
rendered prompt-token digest, normalized text and 13-gram near-duplicates
(the same ``Guard`` as the T5 pools):

- ``math``: GSM8K ``main/train`` and the seven Hendrycks MATH subjects'
  ``train`` splits; the problem text is the user turn
- ``code``: MBPP ``full`` train/validation/prompt (not test) and
  Magicoder-OSS-Instruct-75K (decontaminated) problems
- ``tool``: glaive-function-calling-v2 rows whose system prompt defines
  functions; the function definitions and the first user turn form one
  user message (the served template gets no tool schema)

Rows are ranked by a seeded hash, so the first N pool rows are a nested
subset for every N. Output JSONL holds source text and stays on the cluster;
the manifest holds only counts, digests and revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from apertus_eagle.build_adaptation_pools import Guard, rank, write_jsonl

SEED = "apertus-domain-drafters-pools-v1"
BAND = (16, 2048)
REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "ise-uiuc/Magicoder-OSS-Instruct-75K": "5f839b1f368a76b161028bb9edff055db34022b2",
    "glaiveai/glaive-function-calling-v2": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
}
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def domain_rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)


def repo_of(source: str) -> str:
    owner, name, _rest = source.split("/", 2)
    return f"{owner}/{name}"


def user_row(domain: str, source: str, source_id: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"{domain}-{hashlib.sha256(f'{source}:{source_id}'.encode()).hexdigest()[:16]}",
        "source": source,
        "source_revision": REVISIONS[repo_of(source)],
        "source_id": source_id,
        "domain": domain,
        "messages": [{"role": "user", "content": text.strip()}],
        **extra,
    }


def source_dir(root: Path, repo: str) -> Path:
    directory = root / repo.replace("/", "__")
    pinned = (directory / "revision.txt").read_text().strip()
    if pinned != REVISIONS[repo]:
        raise SystemExit(f"{repo}: downloaded revision {pinned} != pinned {REVISIONS[repo]}")
    return directory


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def math_candidates(root: Path) -> list[dict[str, Any]]:
    rows = []
    gsm = source_dir(root, "openai/gsm8k") / "main/train-00000-of-00001.parquet"
    for index, item in enumerate(parquet_rows(gsm)):
        rows.append(user_row("math", "openai/gsm8k/main-train", str(index), item["question"]))
    base = source_dir(root, "EleutherAI/hendrycks_math")
    for subject in MATH_SUBJECTS:
        path = base / f"{subject}/train-00000-of-00001.parquet"
        for index, item in enumerate(parquet_rows(path)):
            rows.append(
                user_row(
                    "math",
                    f"EleutherAI/hendrycks_math/{subject}-train",
                    str(index),
                    item["problem"],
                    level=item.get("level"),
                )
            )
    return rows


def mbpp_prompt(item: dict[str, Any]) -> str:
    tests = "\n".join(item["test_list"])
    return f"{item['text']}\nYour code should pass these tests:\n\n{tests}"


def code_candidates(root: Path) -> list[dict[str, Any]]:
    rows = []
    base = source_dir(root, "google-research-datasets/mbpp")
    for split in ("train", "validation", "prompt"):
        for item in parquet_rows(base / f"full/{split}-00000-of-00001.parquet"):
            rows.append(
                user_row(
                    "code",
                    f"google-research-datasets/mbpp/full-{split}",
                    str(item["task_id"]),
                    mbpp_prompt(item),
                )
            )
    path = (
        source_dir(root, "ise-uiuc/Magicoder-OSS-Instruct-75K")
        / "data-oss_instruct-decontaminated.jsonl"
    )
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            rows.append(
                user_row(
                    "code",
                    "ise-uiuc/Magicoder-OSS-Instruct-75K/train",
                    str(index),
                    item["problem"],
                    language=item.get("lang"),
                )
            )
    return rows


GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):", re.MULTILINE)


def glaive_prompt(system: str, chat: str) -> str | None:
    """Function definitions plus the first user turn, or None if there are none."""
    system = system.removeprefix("SYSTEM:").strip()
    if "{" not in system or "function" not in system.lower():
        return None
    parts = GLAIVE_TURN.split(chat)
    # parts: [prefix, role, text, role, text, ...]
    for role, text in zip(parts[1::2], parts[2::2], strict=False):
        if role == "USER":
            text = text.replace("<|endoftext|>", "").strip()
            return f"{system}\n\n{text}" if text else None
    return None


def tool_candidates(root: Path) -> list[dict[str, Any]]:
    path = (
        source_dir(root, "glaiveai/glaive-function-calling-v2") / "glaive-function-calling-v2.json"
    )
    rows = []
    for index, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        prompt = glaive_prompt(item["system"], item["chat"])
        if prompt:
            rows.append(
                user_row("tool", "glaiveai/glaive-function-calling-v2/train", str(index), prompt)
            )
    return rows


CANDIDATES = {"math": math_candidates, "code": code_candidates, "tool": tool_candidates}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-build-domain-pools")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pool", type=int, default=8000)
    parser.add_argument("--dev", type=int, default=256)
    parser.add_argument("--protect", type=Path, nargs="+", required=True)
    parser.add_argument("--domains", default="math,code,tool")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from apertus_eagle.contract import load_contract
    from apertus_eagle.generate_targets import render_prompt

    contract = load_contract(args.contract)
    tokenizer = AutoTokenizer.from_pretrained(contract["source"]["authorized_checkpoint"])

    def digest_of(ids: list[int]) -> str:
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    protected = []
    for path in args.protect:
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                if row.get("messages"):
                    protected.append(row)
    protected_digests = {digest_of(render_prompt(tokenizer, row["messages"])) for row in protected}
    guard = Guard(protected, protected_digests)
    manifest: dict[str, Any] = {
        "seed": SEED,
        "revisions": REVISIONS,
        "source_root": str(args.source_root),
        "target": contract["target"]["identity"],
        "renderer": "served chat template, enable_thinking=False, add_generation_prompt=True",
        "protected": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.protect
        },
        "protected_prompts": len(protected),
        "band": list(BAND),
        "domains": {},
    }
    need = args.pool + args.dev
    for domain in [d for d in args.domains.split(",") if d]:
        before = dict(guard.rejected)
        candidates = CANDIDATES[domain](args.source_root)
        candidates.sort(key=lambda r: domain_rank(f"{domain}:{r['id']}"))
        kept, out_of_band, render_errors = [], 0, 0
        for row in candidates:
            if len(kept) >= need:
                break
            try:
                ids = render_prompt(tokenizer, row["messages"])
            except (ValueError, TypeError):
                render_errors += 1
                continue
            if not BAND[0] <= len(ids) <= BAND[1]:
                out_of_band += 1
                continue
            digest = digest_of(ids)
            if guard.admit(row, digest):
                kept.append(
                    {
                        **row,
                        "workload": domain,
                        "input_tokens": len(ids),
                        "prompt_ids_sha256": digest,
                    }
                )
        if len(kept) < need:
            raise SystemExit(f"{domain}: {len(kept)} eligible prompts < {need}")
        splits = {"dev": kept[: args.dev], "pool": kept[args.dev :]}
        directory = args.output_dir / domain
        directory.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "candidates": len(candidates),
            "out_of_band": out_of_band,
            "render_errors": render_errors,
        }
        for split, rows in splits.items():
            rows.sort(key=lambda r: rank(f"order:{r['id']}"))
            sources: dict[str, int] = {}
            for r in rows:
                sources[r["source"]] = sources.get(r["source"], 0) + 1
            report[split] = {
                "rows": len(rows),
                "sha256": write_jsonl(directory / f"{split}.jsonl", rows),
                "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows), 1),
                "sources": sources,
            }
        report["rejected"] = {k: guard.rejected[k] - before[k] for k in guard.rejected}
        manifest["domains"][domain] = report
        print(json.dumps({domain: report}), flush=True)
    (args.output_dir / "manifest-domain-pools.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
