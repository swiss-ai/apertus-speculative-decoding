"""Per-slice acceptance, corpus share and best draft length under load (docs/workload-coverage.md).

    PYTHONPATH=src python3 tools/coverage-best-k.py --cost-table COST.json \\
        --dspark results/8b/sweeps/coverage-a-* --eagle results/8b/sweeps/coverage-b-* \\
        --share results/8b/coverage/corpus-share.json --out results/8b/coverage/best-k.json

Acceptance comes from each slice's C=1 cells. The predicted speedup over the plain target
at B running requests per GPU comes from ``waste_report.predicted_speedup`` (pure decode
steps; prefill is left out). The measured acceptance length is used for k = 1/3/5/7, and
the k=7 per-position rates for the other lengths. DSpark drafting comes from the cost
table. EAGLE has no table entry: its draft cost at B=1 is fitted from the C=1 cells
(first pass plus one cost per later pass) and scaled with B like the table's DSpark
draft step of length 1. A load is skipped when B requests of the slice's mean length do
not fit in the arm's KV cache.
"""

import argparse
import json
import statistics
from pathlib import Path

from apertus_bench.waste_report import CostModel, predicted_speedup

LOADS = (1, 32, 64, 128, 256)
MEASURED_K = (1, 3, 5, 7)
LANGUAGE_CODES = {
    "eng": "en", "deu": "de", "fra": "fr", "ita": "it", "spa": "es", "roh": "rm", "zho": "zh",
    "jpn": "ja", "hin": "hi", "ara": "ar", "swa": "sw", "yor": "yo", "amh": "am",
}  # fmt: skip
DOMAINS = {
    "code-multi": "code-multi", "math-gsm8k": "math", "long-context": "long-context",
    "tool-call": "tool-call", "json-schema": "json-schema", "translate-from-en": "translation",
    "translate-to-en": "translation", "multi-turn": "multi-turn",
}  # fmt: skip


def cell(sweep: Path, method: str, k: int, label: str, c: int = 1) -> dict:
    return json.loads(
        (sweep / f"{method}-e10-k{k}" / label / label / f"c{c}" / "summary.json").read_text()
    )


def context(summary: dict) -> tuple[float, float]:
    """Mean cached context during decode (prompt + half the answer) and full length."""
    n = summary["requests"]["successful"]
    prompt, completion = summary["tokens"]["prompt"] / n, summary["tokens"]["completion"] / n
    return prompt + completion / 2, prompt + completion


def per_position(summary: dict) -> list[float]:
    rates = summary["speculative_decoding"]["acceptance_rate_per_position"]
    return [float(rates[str(i)]) for i in range(len(rates))]


def eagle_draft_fit(model: CostModel, sweep: Path, labels: list[str]) -> tuple[float, float]:
    """EAGLE draft time at B=1 as a + b (k - 1): measured step (TPOT x acceptance length)
    minus the table's verify time, over every slice and length."""
    points = []
    for label in labels:
        for k in MEASURED_K:
            s = cell(sweep, "eagle", k, label)
            step = (
                s["latency_ms"]["tpot"]["mean"]
                * s["speculative_decoding"]["mean_acceptance_length"]
            )
            points.append((k - 1, step - model.verify_ms(1 + k, context(s)[0])))
    mx = statistics.mean(x for x, _ in points)
    my = statistics.mean(y for _, y in points)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return my - b * mx, b


def corpus_share(shares: dict, label: str) -> float:
    if label.startswith("lang-"):
        key = LANGUAGE_CODES[label.removeprefix("lang-")]
        return shares["prompt_language"].get(key, {"share": 0.0})["share"]
    return shares["domains"].get(DOMAINS[label], {"share": 0.0})["share"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--dspark", type=Path, required=True)
    parser.add_argument("--eagle", type=Path, required=True)
    parser.add_argument("--share", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = CostModel.from_cost_table(args.cost_table)
    sweeps = {"dspark": args.dspark, "eagle": args.eagle}
    labels = sorted(p.name for p in (args.dspark / "dspark-e10-k7").iterdir() if p.is_dir())
    shares = json.loads(args.share.read_text())
    a, b = eagle_draft_fit(model, args.eagle, labels)

    def scale(requests: float) -> float:
        return model.dspark_draft_ms(requests, 1) / model.dspark_draft_ms(1, 1)

    drafts = {
        "dspark": model.dspark_draft_ms,
        "eagle": lambda requests, k: (a + b * (k - 1)) * scale(requests),
    }
    slices = {}
    for label in labels:
        entry: dict = {"corpus_share": corpus_share(shares, label)}
        for method, sweep in sweeps.items():
            k7 = cell(sweep, method, 7, label)
            ctx, full = context(k7)
            kv_requests = k7["load"]["kv_cache_size_tokens"] / full
            lengths = {
                k: cell(sweep, method, k, label)["speculative_decoding"]["mean_acceptance_length"]
                for k in MEASURED_K
            }
            loads = {}
            for requests in LOADS:
                if requests > kv_requests:
                    continue
                total_ctx = requests * ctx
                draft = drafts[method]
                speedup = {
                    row["k"]: row["speedup"]
                    for row in predicted_speedup(
                        model, requests, per_position(k7), total_ctx, draft_ms=draft
                    )
                }
                plain = requests / model.verify_ms(requests, total_ctx)
                for k, length in lengths.items():
                    step = draft(requests, k) + model.verify_ms(requests * (1 + k), total_ctx)
                    speedup[k] = requests * length / step / plain
                best = max(speedup, key=speedup.get)
                loads[str(requests)] = {
                    "best_k": best if speedup[best] > 1 else 0,
                    "best_speculative_k": best,
                    "speedup": speedup[best],
                    "speedup_by_k": speedup,
                }
            entry[method] = {
                "acceptance_length": lengths,
                "acceptance_length_c128": {
                    k: cell(sweep, method, k, label, 128)["speculative_decoding"][
                        "mean_acceptance_length"
                    ]
                    for k in MEASURED_K
                    if (sweep / f"{method}-e10-k{k}" / label / label / "c128").is_dir()
                },
                "mean_context_tokens": ctx,
                "kv_cache_requests": kv_requests,
                "loads": loads,
            }
        slices[label] = entry
    result = {
        "cost_table": str(args.cost_table),
        "sweeps": {method: str(path) for method, path in sweeps.items()},
        "corpus_share": str(args.share),
        "eagle_draft_ms_b1": {"first_pass": a, "per_later_pass": b},
        "eagle_draft_scaling": "dspark_draft_ms(B, 1) / dspark_draft_ms(1, 1)",
        "loads": list(LOADS),
        "slices": slices,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"EAGLE draft at B=1: {a:.3f} + {b:.3f} (k-1) ms")
    for label in sorted(slices, key=lambda s: -slices[s]["dspark"]["acceptance_length"][7]):
        e = slices[label]
        cells = []
        for method in sweeps:
            m = e[method]
            picks = {r: m["loads"].get(str(r), {}).get("best_k", "-") for r in LOADS}
            best = " ".join(f"{r}:{k}" for r, k in picks.items())
            cells.append(
                f"{m['acceptance_length'][3]:.2f}/{m['acceptance_length'][7]:.2f} [{best}]"
            )
        print(f"{label:18s} {e['corpus_share']:.5%}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
