"""Report paired deployment effects, validating comparisons before computing speedup."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def interval(values):
    """Exact percentile bootstrap over deployment pairs, never requests."""
    samples = sorted(
        statistics.geometric_mean(s) for s in itertools.product(values, repeat=len(values))
    )

    def quantile(p):
        index = (len(samples) - 1) * p
        low = math.floor(index)
        return samples[low] + (samples[math.ceil(index)] - samples[low]) * (index - low)

    return [quantile(0.025), quantile(0.975)]


def validate_pair(plain, draft, pm, dm, fixed):
    for key in ("workload_file_sha256", "prompt_ids"):
        if pm[key] != dm[key]:
            raise ValueError(f"unmatched {key}")
    for key in ("workload", "concurrency", "requests", "warmup_requests", "generation"):
        if pm["cell"][key] != dm["cell"][key]:
            raise ValueError(f"unmatched cell {key}")
    for key in ("source_revision", "physical_gpu", "target_revision", "corpus_sha256", "engine"):
        if pm["extra"][key] != dm["extra"][key]:
            raise ValueError(f"unmatched {key}")
    for summary in (plain, draft):
        if summary["requests"]["success_rate"] < 0.99:
            raise ValueError("request success below 99%")
        if summary["warmup"]["failed"]:
            raise ValueError("failed warmup")
        if fixed and summary["tokens"]["completion"] != 256 * summary["requests"]["successful"]:
            raise ValueError("fixed-length cell did not generate exactly 256 tokens per request")
    if plain["speculative_decoding"].get("enabled"):
        raise ValueError("baseline unexpectedly speculates")
    counters = draft["speculative_decoding"]
    if not counters.get("enabled") or not (counters.get("draft_tokens", 0) > 0):
        raise ValueError("drafter inactive or counters reset")


def report(root):
    rows = []
    incomplete = []
    for label in ("id-opb", "ood-chat", "ood-summary"):
        for fixed in (False, True):
            mode = "fixed" if fixed else "natural"
            for concurrency in [1, 128] if fixed else [1, 8, 128]:
                pairs = []
                for repeat in (1, 2, 3):
                    base = root / f"repeat-{repeat}"
                    paths = [
                        base / arm / f"{label}-{mode}" / label / f"c{concurrency}"
                        for arm in ("plain", "dspark")
                    ]
                    if not all((p / "summary.json").exists() for p in paths):
                        incomplete.append(f"pair {repeat} {label} {mode} c{concurrency}")
                        continue
                    plain, draft = [read(p / "summary.json") for p in paths]
                    pm, dm = [read(p / "metadata.json") for p in paths]
                    validate_pair(plain, draft, pm, dm, fixed)
                    rates = [s["tokens"]["output_tokens_per_second"] for s in (plain, draft)]
                    pairs.append(
                        {
                            "repeat": repeat,
                            "speedup": rates[1] / rates[0],
                            "plain_tokens_s": rates[0],
                            "draft_tokens_s": rates[1],
                            "plain_tpot_ms": plain["latency_ms"]["tpot"]["p50"],
                            "draft_tpot_ms": draft["latency_ms"]["tpot"]["p50"],
                            "plain_ttft_p95_ms": plain["latency_ms"]["ttft"]["p95"],
                            "draft_ttft_p95_ms": draft["latency_ms"]["ttft"]["p95"],
                            "plain_e2e_p50_ms": plain["latency_ms"]["e2e"]["p50"],
                            "draft_e2e_p50_ms": draft["latency_ms"]["e2e"]["p50"],
                            "acceptance": draft["speculative_decoding"].get("acceptance_rate"),
                            "requests_each": plain["requests"]["attempted"],
                            "plain_completion_tokens": plain["tokens"]["completion"],
                            "draft_completion_tokens": draft["tokens"]["completion"],
                        }
                    )
                if not pairs:
                    continue
                ratios = [p["speedup"] for p in pairs]
                ci = interval(ratios) if len(pairs) == 3 else None
                verdict = "incomplete"
                if ci:
                    verdict = (
                        "speedup"
                        if ci[0] > 1.05
                        else "slowdown"
                        if ci[1] < 0.95
                        else "within 5%"
                        if ci[0] >= 0.95 and ci[1] <= 1.05
                        else "inconclusive"
                    )
                rows.append(
                    {
                        "workload": label,
                        "mode": mode,
                        "concurrency": concurrency,
                        "speedup_geomean": statistics.geometric_mean(ratios),
                        "bootstrap_95": ci,
                        "verdict": verdict,
                        "pairs": pairs,
                    }
                )
    return {"complete": not incomplete, "missing": incomplete, "rows": rows}


def render(result):
    lines = [
        "# Controlled DSpark e10 k=3 speed test",
        "",
        "Same physical GH200 GPU, target, image, overlays, template, settings and loopback client. "
        "Only the speculative configuration changes. Three fresh deployment pairs use "
        "AB, BA, AB order.",
        "",
        "32 deterministically selected prompts per stratum; natural EOS with a "
        "256-token cap is primary. "
        "The separate fixed-length check generates exactly 256 tokens. C=128 is a "
        "single 128-request burst "
        "over four corpus passes, not a sustained-capacity measurement.",
        "",
        "ID is the existing DSpark OpenPerfectBlend held-out validation workload. OOD "
        "chat and long-document "
        "summarization are shifts relative to that training mixture; they do not prove "
        "every individual topic "
        "was absent from training. Corpus hashes and chosen IDs are retained on cluster.",
        "",
        "Speedup is DSpark output tokens/s divided by baseline output tokens/s. Verdicts require a "
        "95% paired-deployment bootstrap interval entirely above 1.05 or below 0.95. "
        "Three pairs on one node give limited uncertainty estimates and do not "
        "establish universal speedup, "
        "losslessness, answer quality, multi-node performance, or gateway latency.",
        "",
        "| Stratum | Generation | C | Paired speedups | Geomean [95% CI] | Verdict |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in result["rows"]:
        ratios = ", ".join(f"{p['speedup']:.2f}x" for p in row["pairs"])
        ci = row["bootstrap_95"]
        effect = f"{row['speedup_geomean']:.2f}x" + (
            f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else " [pending]"
        )
        lines.append(
            f"| {row['workload']} | {row['mode']} | {row['concurrency']} | "
            f"{ratios} | {effect} | {row['verdict']} |"
        )
    lines += [
        "",
        "Absolute throughput, TTFT p95, TPOT p50, E2E p50, acceptance and completion lengths "
        "for every deployment pair are in `paired-results.json`. Raw cells retain "
        "request records and "
        "metric snapshots. Natural output-length differences must be interpreted "
        "alongside the fixed-length check.",
        "",
    ]
    if not result["complete"]:
        lines += [f"**Incomplete: {len(result['missing'])} paired cells remain.**", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    (args.root / "paired-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.root / "REPORT.md").write_text(render(result))
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "rows": len(result["rows"]),
                "missing": len(result["missing"]),
            }
        )
    )


if __name__ == "__main__":
    main()
