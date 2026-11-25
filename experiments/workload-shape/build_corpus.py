"""Rebuild the 28-slice task-speed corpus byte for byte and check it.

The corpus of the controlled task-speed run (corpus_sha256 7d8e60ef...) was purged
from scratch with its inputs. This re-implements ``apertus_bench.task_speed.build_corpus``
(branch experiment/dspark-e10-task-speed, f37f2678) without its other imports, then
refuses to continue unless every slice selects the frozen manifest's ids and the
written file has the frozen digest.

    python3 build_corpus.py SUITE MANIFEST INPUT_DIR OUTPUT

INPUT_DIR holds the source files under the suite's ``file`` names.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

PREWARM = "prewarm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rank(seed: str, prompt_id: str) -> str:
    return hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()


def main() -> None:
    suite = yaml.safe_load(Path(sys.argv[1]).read_text())
    frozen = json.loads(Path(sys.argv[2]).read_text())
    inputs, output = Path(sys.argv[3]), Path(sys.argv[4])
    sources = {}
    for name, source in suite["sources"].items():
        path = inputs / source["file"]
        if sha256(path) != source["sha256"]:
            raise SystemExit(f"source {name}: digest differs from the frozen suite")
        sources[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    count = int(suite["prompts_per_slice"])
    records, prewarm = [], []
    for entry in suite["slices"]:
        pool = [
            row
            for row in sources[entry["source"]]
            if row["workload"] == entry["workload"]
            and row["id"].startswith(entry.get("id_prefix", ""))
        ]
        pool.sort(key=lambda row: rank(suite["seed"], row["id"]))
        chosen = pool[:count]
        if [r["id"] for r in chosen] != frozen["slices"][entry["label"]]["ids"]:
            raise SystemExit(f"{entry['label']}: selection differs from the frozen manifest")
        kwargs = suite["chat_template_kwargs"]
        records += [
            {**row, "workload": entry["label"], "chat_template_kwargs": kwargs} for row in chosen
        ]
        prewarm.append(
            {
                **chosen[-1],
                "id": f"{PREWARM}-{chosen[-1]['id']}",
                "workload": PREWARM,
                "chat_template_kwargs": kwargs,
            }
        )
    output.write_text("".join(json.dumps(row) + "\n" for row in records + prewarm))
    digest = sha256(output)
    if digest != frozen["corpus_sha256"]:
        raise SystemExit(f"corpus digest {digest} != frozen {frozen['corpus_sha256']}")
    # The measured rows only (no pre-warm pass), for the captures.
    measured = output.with_name(output.stem + "-measured.jsonl")
    measured.write_text("".join(json.dumps(row) + "\n" for row in records))
    print(json.dumps({"corpus": str(output), "sha256": digest, "rows": len(records)}))


if __name__ == "__main__":
    main()
