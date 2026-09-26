"""A5/A6: build disjoint validation / test / profile workloads for the 8B pilot.

Two steps on a Clariden login node:

``collect`` (login python with pyarrow): candidate prompts for three strata.
  - chat: SFT-mix rows hash-assigned to ``heldout_test`` with domain chat_qa,
    multilingual or instruction_following; prompts only (last assistant dropped)
  - code: the same pool, domain code
  - summarization: finepdfs-edu documents (eng/deu/fra/ita) wrapped in a
    same-language summarization instruction
  The 256 ``heldout_test`` rows used by A4 and every training/validation row are
  excluded, so these prompts never touched training, early stopping or A4.

``finalize`` (python with transformers): exact Apertus token counts with the
served chat template (``Deliberation: disabled``), the plan's input-length
bands, then a hash split into validation (screening, depth selection) and test
(untouched confirmation). Also a 32-prompt profile set (512-1024 input tokens,
256 fixed output) from the validation pool.

| stratum | input tokens | output cap |
| chat | 128-2048 | 256 |
| code | 128-4096 | 512 |
| summarization | 2048-4096 | 512 |

Output JSONL uses the harness format (id, workload, messages, max_tokens) and
holds restricted source text: keep it on the cluster; commit only the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import heapq
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT

SEED = "apertus-eagle-8b-workloads-v1"
MIX_PARQUET = Path("/capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1/data")
FINEPDFS = {
    "eng": Path("/capstor/store/cscs/swissai/infra01/datasets/finepdfs-edu_eng_spp_annotated/eng_Latn/train"),
    "deu": Path("/capstor/store/cscs/swissai/infra01/datasets/finepdfs-edu_multilingual_spp_annotated/deu_Latn/train"),
    "fra": Path("/capstor/store/cscs/swissai/infra01/datasets/finepdfs-edu_multilingual_spp_annotated/fra_Latn/train"),
    "ita": Path("/capstor/store/cscs/swissai/infra01/datasets/finepdfs-edu_multilingual_spp_annotated/ita_Latn/train"),
}
# Plan default language coverage for summarization: EN/DE/FR/IT 40/20/20/20.
SUMMARY_SHARE = {"eng": 0.4, "deu": 0.2, "fra": 0.2, "ita": 0.2}
# finepdfs token_count is not Apertus tokens. Apertus encodes English more
# compactly, so a 1400-3600 window left no English document above 2048 Apertus
# tokens (first finalize, 2026-09-24). Exact bands are applied in finalize.
SOURCE_TOKEN_WINDOW = {"eng": (2000, 4800), "deu": (1400, 3600), "fra": (1400, 3600), "ita": (1400, 3600)}
SUMMARY_INSTRUCTION = {
    "eng": "Summarize the following document in about 150 words. Write the summary in English.",
    "deu": "Fasse das folgende Dokument in etwa 150 Wörtern zusammen. Schreibe die Zusammenfassung auf Deutsch.",
    "fra": "Résume le document suivant en environ 150 mots. Rédige le résumé en français.",
    "ita": "Riassumi il seguente documento in circa 150 parole. Scrivi il riassunto in italiano.",
}
STRATA = {
    "chat": {"domains": {"chat_qa", "multilingual", "instruction_following"}, "band": (128, 2048), "max_tokens": 256},
    "code": {"domains": {"code"}, "band": (128, 4096), "max_tokens": 512},
    "summarization": {"band": (2048, 4096), "max_tokens": 512},
}
PER_STRATUM = {"validation": 64, "test": 128}
PROFILE = {"count": 32, "band": (512, 1024), "max_tokens": 256}


def rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)


def keep_smallest(heap: list, key: int, item: Any, limit: int) -> None:
    entry = (-key, item["id"], item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif -key > heap[0][0]:
        heapq.heapreplace(heap, entry)


def excluded_ids() -> set[str]:
    ids: set[str] = set()
    for name in ("overfit", "validation", "train", "heldout_test"):
        path = REPO_ROOT / "results/eagle/data" / f"{name}.jsonl"
        if path.is_file():
            ids.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
    return ids


def collect(args: argparse.Namespace) -> None:
    from apertus_eagle.parse_rendered import RenderParseError, mix_row_to_record
    from apertus_eagle.prepare_data import lookup_mix_text

    exclude = excluded_ids()
    heaps: dict[str, list] = {"chat": [], "code": []}
    pool = 0
    with (REPO_ROOT / "results/eagle/data/split-index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "heldout_test" or row["id"] in exclude:
                continue
            pool += 1
            for stratum in heaps:
                if row["domain"] in STRATA[stratum]["domains"]:
                    keep_smallest(heaps[stratum], rank(f"{stratum}:{row['id']}"), row, args.candidates)
    chosen = {s: [entry[2] for entry in heap] for s, heap in heaps.items()}
    texts = lookup_mix_text(MIX_PARQUET, {row["id"] for rows in chosen.values() for row in rows})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {"heldout_pool_rows": pool, "excluded_ids": len(exclude)}
    for stratum, rows in chosen.items():
        written = 0
        with (out / f"candidates-{stratum}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["id"] not in texts:
                    continue
                try:
                    record = mix_row_to_record({**row, "text": texts[row["id"]], "dataset_source": row["source"]})
                except (RenderParseError, KeyError):
                    continue
                messages = [m for m in record["messages"] if m["role"] != "developer"]
                if not messages or messages[-1]["role"] != "user":
                    continue
                handle.write(json.dumps({
                    "id": f"{stratum}-{row['id'][:16]}", "source_id": row["id"], "stratum": stratum,
                    "source": row["source"], "domain": row["domain"], "multi_turn": row.get("multi_turn"),
                    "messages": messages,
                }, ensure_ascii=False) + "\n")
                written += 1
        counts[stratum] = written

    import pyarrow.parquet as pq

    total = args.candidates
    summaries = 0
    with (out / "candidates-summarization.jsonl").open("w", encoding="utf-8") as handle:
        for language, root in FINEPDFS.items():
            want = int(total * SUMMARY_SHARE[language])
            heap: list = []
            for path in sorted(root.glob("*.parquet"))[: args.finepdfs_files]:
                table = pq.read_table(path, columns=["id", "text", "url", "token_count", "is_truncated", "fw_edu_scores", "language"])
                for doc in table.to_pylist():
                    low, high = SOURCE_TOKEN_WINDOW[language]
                    if doc["is_truncated"] or not low <= int(doc["token_count"]) <= high:
                        continue
                    scores = doc["fw_edu_scores"] or []
                    if not scores or sum(scores) / len(scores) < 2.5:
                        continue
                    keep_smallest(heap, rank(f"summ:{doc['id']}"), doc, want)
            if not heap:
                raise SystemExit(f"no {language} summarization documents under {root}")
            for _key, _id, doc in sorted(heap, reverse=True):
                prompt = f"{SUMMARY_INSTRUCTION[language]}\n\n{doc['text']}"
                handle.write(json.dumps({
                    "id": f"summarization-{language}-{hashlib.sha256(doc['id'].encode()).hexdigest()[:16]}",
                    "source_id": doc["id"], "stratum": "summarization", "language": language,
                    "source": f"finepdfs-edu/{language}", "url": doc["url"],
                    "messages": [{"role": "user", "content": prompt}],
                }, ensure_ascii=False) + "\n")
                summaries += 1
    counts["summarization"] = summaries
    (out / "collect-summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


def finalize(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from apertus_eagle.contract import load_contract
    from apertus_eagle.generate_targets import render_prompt

    contract = load_contract(args.contract)
    tokenizer = AutoTokenizer.from_pretrained(contract["source"]["authorized_checkpoint"])
    out = args.output_dir
    manifest: dict[str, Any] = {
        "seed": SEED,
        "target": contract["target"]["identity"],
        "renderer": "served chat template, enable_thinking=False, add_generation_prompt=True",
        "strata": {},
        "sources": {
            "chat/code": f"{MIX_PARQUET} (swiss-ai/Apertus-1.5-SFT-mix, revision d38f5b98b1da36405f6d55b939f096f154883103), heldout_test hash pool",
            "summarization": {k: str(v) for k, v in FINEPDFS.items()},
        },
        "licenses": "per-source licences of the SFT-mix components and finepdfs-edu (ODC-By); text is not committed",
    }
    splits: dict[str, list[dict[str, Any]]] = {"validation": [], "test": [], "profile": []}
    for stratum, spec in STRATA.items():
        low, high = spec["band"]
        rows = [json.loads(l) for l in (out / f"candidates-{stratum}.jsonl").read_text().split("\n") if l.strip()]
        eligible = []
        seen_prompts: set[str] = set()
        for row in rows:
            ids = render_prompt(tokenizer, row["messages"])
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if not low <= len(ids) <= high or digest in seen_prompts:
                continue
            seen_prompts.add(digest)
            eligible.append({**row, "input_tokens": len(ids), "prompt_ids_sha256": digest})
        # Hash split: each prompt lands in exactly one of validation / test.
        eligible.sort(key=lambda r: rank(f"split:{r['id']}"))
        need_val, need_test = PER_STRATUM["validation"], PER_STRATUM["test"]
        if len(eligible) < need_val + need_test:
            raise SystemExit(f"{stratum}: only {len(eligible)} eligible prompts in [{low}, {high}]")
        if stratum == "summarization":
            # Plan coverage default EN/DE/FR/IT 40/20/20/20, per split.
            validation, test, shortfall = [], [], {}

            def quotas(total: int) -> dict[str, int]:
                q = {lang: int(total * share) for lang, share in SUMMARY_SHARE.items()}
                q["eng"] += total - sum(q.values())
                return q

            q_val, q_test = quotas(need_val), quotas(need_test)
            for language in SUMMARY_SHARE:
                pool = [r for r in eligible if r.get("language") == language]
                want_val, want_test = q_val[language], q_test[language]
                validation += pool[:want_val]
                test += pool[want_val : want_val + want_test]
                missing = want_val + want_test - len(pool[: want_val + want_test])
                if missing > 0:
                    shortfall[language] = missing
            if shortfall or len(validation) != need_val or len(test) != need_test:
                raise SystemExit(f"summarization language quotas not met: {shortfall}, "
                                 f"validation {len(validation)}, test {len(test)}")
            used = {r["id"] for r in validation + test}
            leftover_validation_pool = [r for r in eligible if r["id"] not in used]
        else:
            validation = eligible[:need_val]
            test = eligible[need_val : need_val + need_test]
            leftover_validation_pool = eligible[need_val + need_test :]
        for split, chosen in (("validation", validation), ("test", test)):
            for row in chosen:
                splits[split].append({
                    "id": row["id"], "workload": stratum, "messages": row["messages"],
                    "max_tokens": spec["max_tokens"], "input_tokens": row["input_tokens"],
                    "source": row["source"], "language": row.get("language"),
                })
        if stratum in ("chat", "code"):
            for row in leftover_validation_pool:
                if PROFILE["band"][0] <= row["input_tokens"] <= PROFILE["band"][1]:
                    splits["profile"].append(row)
        manifest["strata"][stratum] = {
            "eligible_by_language": {
                lang: sum(1 for r in eligible if r.get("language") == lang) for lang in SUMMARY_SHARE
            } if stratum == "summarization" else None,
            "band": [low, high], "max_tokens": spec["max_tokens"], "candidates": len(rows),
            "eligible": len(eligible), "validation": len(validation), "test": len(test),
            "validation_input_tokens": [min(r["input_tokens"] for r in validation), max(r["input_tokens"] for r in validation)],
            "test_input_tokens": [min(r["input_tokens"] for r in test), max(r["input_tokens"] for r in test)],
        }
        if stratum == "summarization":
            manifest["strata"][stratum]["languages"] = {
                split: {lang: sum(1 for r in chosen if r.get("language") == lang) for lang in SUMMARY_SHARE}
                for split, chosen in (("validation", validation), ("test", test))
            }
    profile = sorted(splits["profile"], key=lambda r: rank(f"profile:{r['id']}"))[: PROFILE["count"]]
    splits["profile"] = [
        {"id": r["id"], "workload": "profile_fixed256", "messages": r["messages"],
         "max_tokens": PROFILE["max_tokens"], "input_tokens": r["input_tokens"], "source": r["source"]}
        for r in profile
    ]
    # One writer only: an overlapping earlier finalize left fragment lines in
    # eagle-8b-validation.jsonl (196 lines for 192 rows) and every A5 screening
    # cell failed to load it. Write to a temp file, re-read, then rename.
    ids = {split: {r["id"] for r in rows} for split, rows in splits.items()}
    if ids["validation"] & ids["test"] or ids["profile"] & ids["test"]:
        raise SystemExit("validation/profile and test overlap")
    for split, rows in splits.items():
        path = out / f"eagle-8b-{split}.jsonl"
        tmp = path.with_suffix(f".jsonl.tmp-{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with tmp.open(encoding="utf-8") as handle:
            reread = [json.loads(line) for line in handle]
        if [r["id"] for r in reread] != [r["id"] for r in rows]:
            raise SystemExit(f"{tmp} does not re-read as {len(rows)} rows")
        os.replace(tmp, path)
        manifest[f"{split}_file"] = str(path)
        manifest[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[f"{split}_rows"] = len(rows)
    manifest["profile_note"] = "chat/code prompts outside validation and test, 512-1024 input tokens; ignore_eos=true mechanics check only"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-build-workloads")
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/eagle/8b/workloads")
    col.add_argument("--candidates", type=int, default=1200)
    col.add_argument("--finepdfs-files", type=int, default=2)
    fin = sub.add_parser("finalize")
    fin.add_argument("--contract", type=Path)
    fin.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/eagle/8b/workloads")
    args = parser.parse_args(argv)
    collect(args) if args.command == "collect" else finalize(args)


if __name__ == "__main__":
    main()
