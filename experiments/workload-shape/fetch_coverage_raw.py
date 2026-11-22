"""Re-download the pinned public sources of the workload-coverage slices.

The coverage file (sha256 ddb921fe...7804) and its raw inputs were purged from
Clariden scratch on 2026-11-20. ``apertus_bench.coverage_workloads`` rebuilds the
file byte for byte from the same dataset revisions; this script fetches exactly
those revisions into ``RAW/<repo>/<file>`` and writes the ``.jsonl`` copies the
builder reads next to every parquet file (the serving image has no pyarrow).

    ~/miniconda/bin/python experiments/workload-shape/fetch_coverage_raw.py RAW

Runs on a login node (network only, a few hundred MB). Needs huggingface_hub and
pyarrow. The revisions are those of ``workloads/coverage/manifest.json`` on the
``workload-coverage`` branch (ba1533b0).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REVISIONS = {
    "akoksal/muri-it-language-split": "4987fc82a54145caf778efec6a682863591be1f2",
    "NousResearch/hermes-function-calling-v1": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
    "lmsys/mt_bench_human_judgments": "f7d2896d2cc5d80f8b55c2bbc722613555233c25",
    "bigcode/humanevalpack": "9a41762f73a8cb23bb5811b73d5aab164efcf378",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google/wmt24pp": "fd7405c06494bc66a57b25f55d217a72f96e60dc",
    "zai-org/LongBench": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
}
LANGS = ("eng", "deu", "fra", "ita", "spa", "roh", "zho", "jpn", "hin", "ara", "swa", "yor", "amh")
FILES = {
    "akoksal/muri-it-language-split": [f"{c}/test-00000-of-00001.parquet" for c in LANGS],
    "NousResearch/hermes-function-calling-v1": [
        "func-calling-singleturn.json",
        "json-mode-singleturn.json",
    ],
    "lmsys/mt_bench_human_judgments": ["data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet"],
    "bigcode/humanevalpack": [
        f"{c}/test-00000-of-00001.parquet" for c in ("cpp", "go", "java", "js", "rust")
    ],
    "openai/gsm8k": ["main/test-00000-of-00001.parquet"],
    "google/wmt24pp": [f"en-{lp}.jsonl" for lp in ("de_DE", "fr_FR", "it_IT", "es_MX")],
    "zai-org/LongBench": ["data.zip"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    raw = Path(sys.argv[1])
    raw.mkdir(parents=True, exist_ok=True)
    digests = {}
    for repo, files in FILES.items():
        for name in files:
            local = Path(
                hf_hub_download(
                    repo, name, repo_type="dataset", revision=REVISIONS[repo],
                    local_dir=raw / repo,
                )
            )  # fmt: skip
            digests[f"{repo}/{name}"] = sha256(local)
            if name.endswith(".parquet"):
                out = local.with_suffix(".jsonl")
                rows = pq.read_table(local).to_pylist()
                with out.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                digests[f"{repo}/{out.name}"] = sha256(out)
            print(repo, name, digests[f"{repo}/{name}"], flush=True)
    (raw / "revisions.txt").write_text("".join(f"{r} {s}\n" for r, s in REVISIONS.items()))
    (raw / "fetch-sha256.json").write_text(json.dumps(digests, indent=1) + "\n")


if __name__ == "__main__":
    main()
