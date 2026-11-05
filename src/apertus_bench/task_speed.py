"""Per-task DSpark speed: suite, corpus, paired controlled runs and the report.

The suite (experiments/task-speed/suite.yaml) fixes the slices, their ID/OOD
class and the load grid before anything is measured. Controlled runs pair a
plain and a DSpark deployment on the same GPU; live runs pair windows of the
deployed pod in plain and DSpark mode. Both reduce to the same paired cells,
so one report covers them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from .load import parse_load_sample
from .workloads import file_sha256

CLASSES = ("id", "near-id", "ood-task", "ood-language")
PREWARM = "prewarm"
# Verdict band and rule of the 2026-10-30 ID/OOD campaign.
BAND = (0.95, 1.05)


def load_suite(path: Path) -> dict[str, Any]:
    suite = yaml.safe_load(Path(path).read_text())
    labels = [entry["label"] for entry in suite["slices"]]
    if len(set(labels)) != len(labels):
        raise ValueError("slice labels must be unique")
    if PREWARM in labels:
        raise ValueError(f"{PREWARM!r} is reserved")
    for entry in suite["slices"]:
        if entry["class"] not in CLASSES:
            raise ValueError(f"{entry['label']}: class must be one of {CLASSES}")
        if entry["source"] not in suite["sources"]:
            raise ValueError(f"{entry['label']}: unknown source {entry['source']!r}")
    if sorted(suite["concurrencies"]) != suite["concurrencies"]:
        raise ValueError("concurrencies must be ascending")
    return suite


def slice_classes(suite: dict[str, Any]) -> dict[str, str]:
    return {entry["label"]: entry["class"] for entry in suite["slices"]}


def _rank(seed: str, prompt_id: str) -> str:
    return hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()


def build_corpus(suite: dict[str, Any], inputs: Path, output: Path) -> dict[str, Any]:
    """Write the measured corpus and a pre-warm pass (one prompt per slice).

    Returns the manifest: source hashes, the chosen IDs per slice and the
    corpus digest. Prompt text stays in ``output`` (restricted, not committed).
    """
    sources = {}
    for name, source in suite["sources"].items():
        path = Path(inputs) / source["file"]
        digest = file_sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"source {name}: sha256 {digest} != frozen {source['sha256']}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources[name] = (path, digest, rows)
    count = int(suite["prompts_per_slice"])
    records, prewarm, manifest, seen = [], [], {}, set()
    for entry in suite["slices"]:
        path, digest, rows = sources[entry["source"]]
        pool = [
            row
            for row in rows
            if row["workload"] == entry["workload"]
            and row["id"].startswith(entry.get("id_prefix", ""))
        ]
        pool.sort(key=lambda row: _rank(suite["seed"], row["id"]))
        chosen = pool[:count]
        if len(chosen) != count:
            raise ValueError(f"{entry['label']}: need {count} prompts, pool has {len(pool)}")
        for row in chosen:
            if row["id"] in seen:
                raise ValueError(f"prompt {row['id']} selected twice")
            seen.add(row["id"])
            records.append(
                {
                    **row,
                    "workload": entry["label"],
                    "chat_template_kwargs": suite["chat_template_kwargs"],
                }
            )
        # The pre-warm pass reaches every slice's shapes (and the drafter's JIT
        # kernels) before the first measured cell; its requests are not reported.
        prewarm.append(
            {
                **chosen[-1],
                "id": f"{PREWARM}-{chosen[-1]['id']}",
                "workload": PREWARM,
                "chat_template_kwargs": suite["chat_template_kwargs"],
            }
        )
        manifest[entry["label"]] = {
            "class": entry["class"],
            "source": entry["source"],
            "source_file": str(path),
            "source_sha256": digest,
            "pool": len(pool),
            "max_tokens": sorted({row["max_tokens"] for row in chosen}),
            "ids": [row["id"] for row in chosen],
        }
    output = Path(output)
    output.write_text("".join(json.dumps(row) + "\n" for row in records + prewarm))
    return {
        "suite": suite["name"],
        "seed": suite["seed"],
        "prompts_per_slice": count,
        "slices": manifest,
        "corpus_sha256": file_sha256(output),
    }


def requests_for(suite: dict[str, Any], concurrency: int) -> int:
    """The client's whole-pass rule (cli.loadtest_requests) for one slice."""
    count = int(suite["prompts_per_slice"])
    wanted = max(int(suite["min_requests"]), int(suite["requests_per_slot"]) * concurrency)
    return -(-wanted // count) * count


def controlled_spec(
    suite: dict[str, Any],
    corpus: Path,
    target_model: str,
    drafter: str,
    chat_template: str,
    select: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """A node_sweep spec for one plain/DSpark pair with the live engine settings.

    ``select`` limits the measured cells (slice -> concurrencies), for extra
    pairs on comparisons that the first pairs left inconclusive.
    """
    generation = dict(suite["generation"])
    labels = [entry["label"] for entry in suite["slices"]]
    loadtests = [
        {
            "label": PREWARM,
            "workload": PREWARM,
            "workloads_file": str(corpus),
            "concurrencies": [1, 32],
            "min_requests": len(labels),
            "requests_per_slot": 1,
            "generation": generation,
        }
    ]
    for label in labels:
        concurrencies = suite["concurrencies"] if select is None else select.get(label)
        if not concurrencies:
            continue
        loadtests.append(
            {
                "label": label,
                "workload": label,
                "workloads_file": str(corpus),
                "concurrencies": list(concurrencies),
                "min_requests": int(suite["min_requests"]),
                "requests_per_slot": int(suite["requests_per_slot"]),
                "generation": generation,
            }
        )
    vllm_args = ["--dtype", "bfloat16", "--no-enable-log-requests"]
    vllm_args += ["--chat-template", chat_template, "--chat-template-content-format", "string"]
    return {
        "name": suite["name"],
        "stage": "8b",
        "probe": False,
        "target_model": target_model,
        "node_gpus": 4,
        # The live Deployment's engine flags (serving/kubernetes/prepare.py).
        "engine": {
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 256,
        },
        "loadtests": loadtests,
        "arms": [
            {"name": "plain", "method": "baseline", "vllm_args": vllm_args},
            {
                "name": "dspark",
                "method": "dspark",
                "drafter": drafter,
                "k": 3,
                "vllm_args": vllm_args,
            },
        ],
    }


def interval(values: list[float]) -> list[float]:
    """Exact percentile bootstrap of the geometric mean over deployment pairs."""
    samples = sorted(
        statistics.geometric_mean(s) for s in itertools.product(values, repeat=len(values))
    )

    def quantile(p: float) -> float:
        index = (len(samples) - 1) * p
        low = math.floor(index)
        return samples[low] + (samples[math.ceil(index)] - samples[low]) * (index - low)

    return [quantile(0.025), quantile(0.975)]


def verdict(ci: list[float] | None) -> str:
    if ci is None:
        return "incomplete"
    if ci[0] > BAND[1]:
        return "speedup"
    if ci[1] < BAND[0]:
        return "slowdown"
    if ci[0] >= BAND[0] and ci[1] <= BAND[1]:
        return "within 5%"
    return "inconclusive"


def validate_pair(
    plain: dict,
    draft: dict,
    pm: dict,
    dm: dict,
    matched_extra: tuple,
    guarded: bool = False,
) -> None:
    """Refuse to compute a speedup from cells that differ in anything but the arm."""
    for key in ("workload_file_sha256", "prompt_ids"):
        if pm[key] != dm[key]:
            raise ValueError(f"unmatched {key}")
    for key in ("workload", "concurrency", "requests", "generation"):
        if pm["cell"][key] != dm["cell"][key]:
            raise ValueError(f"unmatched cell {key}")
    # Live cells run their warmup inline or as a separate cell just before (same
    # requests on the same prompts); compare the effective warmup.
    if _warmup(pm) != _warmup(dm):
        raise ValueError("unmatched cell warmup_requests")
    for key in matched_extra:
        if pm["extra"].get(key) != dm["extra"].get(key):
            raise ValueError(f"unmatched {key}")
    for summary in (plain, draft):
        if summary["requests"]["success_rate"] < 0.99:
            raise ValueError("request success below 99%")
        # A live cell's guard proves no leftover warmup request ran inside it.
        if summary["warmup"]["failed"] and not guarded:
            raise ValueError("failed warmup")
    if plain["speculative_decoding"].get("enabled"):
        raise ValueError("baseline unexpectedly speculates")
    counters = draft["speculative_decoding"]
    if not counters.get("enabled") or not (counters.get("draft_tokens", 0) > 0):
        raise ValueError("drafter inactive or counters reset")


def _warmup(metadata: dict) -> int:
    return metadata["cell"]["warmup_requests"] or int(metadata["extra"].get("separate_warmup", 0))


def pair_row(plain: dict, draft: dict) -> dict[str, Any]:
    rates = [s["tokens"]["output_tokens_per_second"] for s in (plain, draft)]
    spec = draft["speculative_decoding"]
    latency = {}
    for arm, summary in (("plain", plain), ("dspark", draft)):
        for metric in ("ttft", "tpot", "e2e"):
            for quantile in ("p50", "p95"):
                latency[f"{arm}_{metric}_{quantile}_ms"] = summary["latency_ms"][metric][quantile]
    tpot = [plain["latency_ms"]["tpot"]["p50"], draft["latency_ms"]["tpot"]["p50"]]
    return {
        "speedup": rates[1] / rates[0],
        # Per-request decode speedup: insensitive to a cell's tail (one stalled stream).
        "tpot_speedup": tpot[0] / tpot[1] if tpot[0] and tpot[1] else None,
        "plain_tokens_s": rates[0],
        "dspark_tokens_s": rates[1],
        **latency,
        "acceptance_rate": spec.get("acceptance_rate"),
        "acceptance_length": spec.get("mean_acceptance_length"),
        "plain_completion_tokens": plain["tokens"]["completion"],
        "dspark_completion_tokens": draft["tokens"]["completion"],
        "requests": plain["requests"]["attempted"],
        "plain_load": plain.get("load"),
        "dspark_load": draft.get("load"),
    }


def cell_rows(
    suite: dict[str, Any],
    pairs: dict[str, dict[str, Path]],
    matched_extra: tuple,
    guarded: bool = False,
) -> dict[str, Any]:
    """Pair every (slice, C) cell over the deployment pairs.

    ``pairs`` maps a pair name to {"plain": dir, "dspark": dir}; a cell lives at
    ``<dir>/<label>/<label>/c<C>``. Invalid pairs are reported, not dropped silently.
    """
    rows, missing, rejected = [], [], []
    for entry in suite["slices"]:
        label = entry["label"]
        for concurrency in suite["concurrencies"]:
            measured = []
            for name, arms in pairs.items():
                cells = [arms[a] / label / label / f"c{concurrency}" for a in ("plain", "dspark")]
                if not all((cell / "summary.json").exists() for cell in cells):
                    missing.append(f"{name} {label} c{concurrency}")
                    continue
                plain, draft = [json.loads((c / "summary.json").read_text()) for c in cells]
                pm, dm = [json.loads((c / "metadata.json").read_text()) for c in cells]
                try:
                    if guarded and not all(
                        json.loads((c / "guard.json").read_text())["valid"] for c in cells
                    ):
                        raise ValueError("traffic guard not valid")
                    validate_pair(plain, draft, pm, dm, matched_extra, guarded)
                except ValueError as error:
                    rejected.append(f"{name} {label} c{concurrency}: {error}")
                    continue
                measured.append({"pair": name, **pair_row(plain, draft)})
            if not measured:
                continue
            ratios = [p["speedup"] for p in measured]
            ci = interval(ratios) if len(measured) >= 2 else None
            lengths = [p["acceptance_length"] for p in measured if p["acceptance_length"]]
            rows.append(
                {
                    "slice": label,
                    "class": entry["class"],
                    "concurrency": concurrency,
                    "speedup_geomean": statistics.geometric_mean(ratios),
                    "tpot_speedup_geomean": _geomean([p["tpot_speedup"] for p in measured]),
                    "bootstrap_95": ci,
                    "verdict": verdict(ci),
                    "acceptance_length": statistics.fmean(lengths) if lengths else None,
                    "pairs": measured,
                }
            )
    return {"rows": rows, "missing": missing, "rejected": rejected, "complete": not missing}


def _geomean(values: list[float | None]) -> float | None:
    values = [v for v in values if v]
    return statistics.geometric_mean(values) if values else None


def class_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Geometric mean of slice speedups per class and concurrency (slices weigh equally)."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["class"], row["concurrency"]), []).append(row)
    summary = []
    for cls in CLASSES:
        for concurrency in sorted({c for k, c in groups if k == cls}):
            members = groups[(cls, concurrency)]
            lengths = [r["acceptance_length"] for r in members if r["acceptance_length"]]
            summary.append(
                {
                    "class": cls,
                    "concurrency": concurrency,
                    "slices": len(members),
                    "speedup_geomean": statistics.geometric_mean(
                        [r["speedup_geomean"] for r in members]
                    ),
                    "speedup_min": min(r["speedup_geomean"] for r in members),
                    "speedup_max": max(r["speedup_geomean"] for r in members),
                    "acceptance_length_mean": statistics.fmean(lengths) if lengths else None,
                    "verdicts": {
                        v: sum(r["verdict"] == v for r in members)
                        for v in sorted({r["verdict"] for r in members})
                    },
                }
            )
    return summary


def acceptance_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per C, a least-squares line speedup = a + b * acceptance length.

    The break-even length (speedup 1) tells which tasks gain at that load, and a
    new task's speedup can be read off from its acceptance alone.
    """
    curves = []
    for concurrency in sorted({r["concurrency"] for r in rows}):
        points = [
            (r["acceptance_length"], r["speedup_geomean"])
            for r in rows
            if r["concurrency"] == concurrency and r["acceptance_length"]
        ]
        if len(points) < 3:
            continue
        xs, ys = zip(*points, strict=True)
        slope, intercept = statistics.linear_regression(xs, ys)
        fitted = [intercept + slope * x for x in xs]
        mean = statistics.fmean(ys)
        total = sum((y - mean) ** 2 for y in ys)
        r2 = 1 - sum((y - f) ** 2 for y, f in zip(ys, fitted, strict=True)) / total if total else 1
        curves.append(
            {
                "concurrency": concurrency,
                "intercept": intercept,
                "slope": slope,
                "r2": r2,
                "break_even_acceptance_length": (1 - intercept) / slope if slope else None,
                "points": len(points),
            }
        )
    return curves


# Other users' requests finishing inside a live cell are tolerated up to this share
# of the cell's generated tokens (their load can shift throughput by about as much).
FOREIGN_TOKEN_SHARE = 0.02


def traffic_guard(
    before: str, after: str, model: str, requests: int, completion_tokens: int | None = None
) -> tuple[bool, str | None, dict[str, Any]]:
    """Whether a live cell measured (essentially) only our requests.

    The engine must be idle when the cell starts, and its finished-request
    counter must advance by the measured requests. Extra finished requests are
    tolerated when they generated at most FOREIGN_TOKEN_SHARE of the cell's
    tokens (``completion_tokens`` known); fewer finished requests than sent, or a
    counter that goes down (engine restart), always invalidate the cell.
    """
    first, last = parse_load_sample(before, model), parse_load_sample(after, model)
    detail = {"before": first, "after": last, "requests": requests}
    busy = (first.get("running") or 0) + (first.get("waiting") or 0)
    if busy:
        return False, f"engine busy at start ({busy:g} requests)", detail
    done = _counter(after, "vllm:request_success_total", model) - _counter(
        before, "vllm:request_success_total", model
    )
    detail["finished_delta"] = done
    if done < 0:
        return False, "counters went down (engine restart)", detail
    if done == requests:
        return True, None, detail
    if done > requests and completion_tokens:
        generated = _counter(after, "vllm:generation_tokens_total", model) - _counter(
            before, "vllm:generation_tokens_total", model
        )
        share = max(0.0, generated - completion_tokens) / completion_tokens
        detail["foreign_requests"] = done - requests
        detail["foreign_token_share"] = share
        if share <= FOREIGN_TOKEN_SHARE:
            return True, f"tolerated {done - requests:g} foreign requests ({share:.1%})", detail
    return False, f"{done:g} requests finished, {requests} sent (foreign traffic)", detail


def _counter(text: str, name: str, model: str) -> float:
    from prometheus_client.parser import text_string_to_metric_families

    from .prometheus import _matches_model

    total = 0.0
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and _matches_model(sample.labels, model):
                total += float(sample.value)
    return total
