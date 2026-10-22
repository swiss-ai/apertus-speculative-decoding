"""Wasted compute and cost per token of speculative decoding (docs/waste-profiling.md).

Three sources, all already in sweep records:

- unprofiled cells (``summary.json``): throughput, draft and accepted counters,
  per-position acceptance; ``gpu-memory.csv`` of the job: power per GPU;
- ``cost-table.json`` of a DSpark scheduler arm (kv-cost, paper sweeps):
  target verify time by verified tokens, DSpark draft step by requests and
  draft length, KV read per context token;
- profiled windows (``nsys-stats/`` of an arm with a ``profile`` block):
  NVTX ranges projected onto the GPU per engine step, kernel and GPU-metric
  summaries, and the window's server counters.

Python API (what other analyses reuse):

    model = CostModel.from_cost_table("…/cost-table.json")
    model.verify_ms(tokens, context_tokens=0)       # target forward of one step
    model.marginal_verify_ms(tokens)                # d verify / d token at that size
    model.dspark_draft_ms(requests, k)              # DSpark draft step (table)
    model.step_ms(requests, k, context_tokens=…, draft_ms=…)
    marginal_rule(model, requests, k, accept, …)    # does slot j pay?
    step_waste(model, requests, k, accept, …)       # analytic per-step waste
    DSparkComponents / fit_dspark_components(...)   # per-layer drafter model
    decompose_window(stats_dir, stem, window, model)  # measured per-step split

CLI:

    python -m apertus_bench.waste_report cost SWEEP_DIR... [--label opb]
    python -m apertus_bench.waste_report decompose SWEEP_DIR... --cost-table PATH
    python -m apertus_bench.waste_report marginal --cost-table PATH --summary CELL/summary.json
        --requests 128 [--context 1000]

Waste is defined against the same step without the slots that were rejected:
rejected verify = rejected tokens x marginal verify cost at the operating
step size (not the average cost: the step's fixed part is paid anyway), and
wasted drafting = the per-slot share of the draft step for slots never
accepted (DSpark drafts every slot; EAGLE one pass per slot).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Piecewise-linear, clamped below, linearly extrapolated above the table."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        if len(xs) < 2:
            return ys[-1]
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _slope(xs: list[float], ys: list[float], x: float, span: float | None = None) -> float:
    """Least-squares slope of the table over [x - span, x + span]; the table is
    noisy, so the default span widens with x (64 tokens or 15%)."""
    span = max(64.0, 0.15 * x) if span is None else span
    points = [(a, b) for a, b in zip(xs, ys, strict=True) if abs(a - x) <= span]
    if len(points) < 3:
        points = list(zip(xs, ys, strict=True))[-3:] if x >= xs[-1] else points
    if len(points) < 2:
        lo, hi = max(xs[0], x - span), min(xs[-1], x + span)
        return (_interp(xs, ys, hi) - _interp(xs, ys, lo)) / (hi - lo) if hi > lo else 0.0
    mx = sum(a for a, _ in points) / len(points)
    my = sum(b for _, b in points) / len(points)
    var = sum((a - mx) ** 2 for a, _ in points)
    return sum((a - mx) * (b - my) for a, b in points) / var if var else 0.0


@dataclass
class CostModel:
    """Step costs of the 8B target (and the DSpark drafter) on one GH200."""

    verify_tokens: list[float]
    verify_ms_table: list[float]
    draft_ms_by_len: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    kv_ms_per_token: float = 0.0
    max_cudagraph_capture_size: int | None = None
    source: str = ""

    @classmethod
    def from_cost_table(cls, path: str | Path) -> CostModel:
        data = json.loads(Path(path).read_text())
        reqs = [float(r) for r in data.get("draft_reqs") or []]
        by_len = {
            int(length): (reqs, [float(v) for v in values])
            for length, values in (data.get("draft_ms_by_len") or {}).items()
        }
        return cls(
            verify_tokens=[float(t) for t in data["verify_tokens"]],
            verify_ms_table=[float(v) for v in data["verify_ms"]],
            draft_ms_by_len=by_len,
            kv_ms_per_token=float(data.get("kv_ms_per_token") or 0.0),
            max_cudagraph_capture_size=data.get("max_cudagraph_capture_size"),
            source=str(path),
        )

    def verify_ms(self, tokens: float, context_tokens: float = 0.0) -> float:
        """Target forward of a step with ``tokens`` query tokens over
        ``context_tokens`` cached tokens in total (the table was measured at
        short contexts; the KV read is added from its measured bandwidth)."""
        return (
            _interp(self.verify_tokens, self.verify_ms_table, tokens)
            + context_tokens * self.kv_ms_per_token
        )

    def marginal_verify_ms(self, tokens: float) -> float:
        """Extra verify time per extra query token at a step of ``tokens``."""
        return max(_slope(self.verify_tokens, self.verify_ms_table, tokens), 0.0)

    def dspark_draft_ms(self, requests: float, k: int) -> float:
        if k not in self.draft_ms_by_len:
            raise KeyError(f"cost table has no DSpark draft length {k}")
        xs, ys = self.draft_ms_by_len[k]
        return _interp(xs, ys, requests)

    def step_ms(
        self, requests: float, k: int, context_tokens: float = 0.0, draft_ms: float | None = None
    ) -> float:
        """A pure decode step: draft (DSpark table unless given) plus verify of
        ``requests x (1 + k)`` tokens."""
        draft = (
            draft_ms if draft_ms is not None else (self.dspark_draft_ms(requests, k) if k else 0)
        )
        return draft + self.verify_ms(requests * (1 + k), context_tokens)


def accepted_per_step(accept: list[float]) -> float:
    """Tokens emitted per request and step: the bonus/corrected token plus the
    accepted drafts (``accept[j]`` = P(slot j accepted))."""
    return 1.0 + sum(accept)


def acceptance(summary: dict[str, Any]) -> list[float]:
    """Per-position acceptance (P(slot j accepted)) of a cell's summary."""
    per = (summary.get("speculative_decoding") or {}).get("acceptance_rate_per_position") or {}
    return [float(per[key]) for key in sorted(per, key=int) if per[key] is not None]


def marginal_draft_ms(model: CostModel, requests: float, k: int) -> list[float]:
    """DSpark: extra draft time of slot j (draft length j+1 against j)."""
    out = []
    previous = 0.0
    for length in range(1, k + 1):
        now = model.dspark_draft_ms(requests, length)
        out.append(now - previous)
        previous = now
    return out


def marginal_rule(
    model: CostModel,
    requests: float,
    k: int,
    accept: list[float],
    context_tokens: float = 0.0,
    draft_slot_ms: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Yu's rule per draft slot: slot j pays iff
    ``p_accept(j) x time per emitted token > marginal draft + verify cost``
    at this load, where time per emitted token is the step time over the
    tokens a request emits per step (1 + sum of acceptances) and the cost is
    the whole step's extra time for the slot: its draft share plus
    ``requests`` extra verify tokens. (Throughput B n / t rises with the slot
    iff a_j t > n c_j.) ``draft_slot_ms`` overrides the DSpark table (e.g. an
    EAGLE pass: the same cost for every slot)."""
    slots = draft_slot_ms if draft_slot_ms is not None else marginal_draft_ms(model, requests, k)
    step = model.step_ms(requests, k, context_tokens, draft_ms=sum(slots))
    per_token = step / accepted_per_step(accept[:k])
    verify_slot = requests * model.marginal_verify_ms(requests * (1 + k))
    rows = []
    for j in range(k):
        cost = slots[j] + verify_slot
        gain = accept[j] * per_token
        rows.append(
            {
                "slot": j + 1,
                "p_accept": accept[j],
                "gain_ms": gain,
                "cost_ms": cost,
                "pays": gain > cost,
            }
        )
    return rows


def predicted_speedup(
    model: CostModel,
    requests: float,
    accept: list[float],
    context_tokens: float = 0.0,
    draft_ms: Any = None,
) -> list[dict[str, float]]:
    """Predicted decode throughput of draft lengths 1..len(accept) against the
    plain target at ``requests`` (pure decode steps, no prefill). The marginal
    rule judges one slot; this judges a whole length, so the drafter's fixed
    cost (slot 1) is spread over all slots. ``draft_ms(requests, k)`` defaults
    to the DSpark table."""
    draft = draft_ms or model.dspark_draft_ms
    plain = requests / model.verify_ms(requests, context_tokens)
    rows = []
    for k in range(1, len(accept) + 1):
        step = draft(requests, k) + model.verify_ms(requests * (1 + k), context_tokens)
        rate = requests * accepted_per_step(accept[:k]) / step
        rows.append({"k": k, "step_ms": step, "speedup": rate / plain})
    return rows


def step_waste(
    model: CostModel,
    requests: float,
    k: int,
    accept: list[float],
    context_tokens: float = 0.0,
    draft_ms: float | None = None,
) -> dict[str, float]:
    """Analytic split of one decode step: useful/rejected verify, draft of slots
    later accepted / never accepted, KV read, fixed part."""
    draft = draft_ms if draft_ms is not None else model.dspark_draft_ms(requests, k)
    tokens = requests * (1 + k)
    verify = model.verify_ms(tokens, context_tokens)
    kv = context_tokens * model.kv_ms_per_token
    rejected_tokens = requests * (k - sum(accept[:k]))
    rejected_verify = min(rejected_tokens * model.marginal_verify_ms(tokens), verify - kv)
    accepted_share = sum(accept[:k]) / k if k else 0.0
    step = draft + verify
    return {
        "step_ms": step,
        "verify_ms": verify,
        "kv_read_ms": kv,
        "rejected_verify_ms": rejected_verify,
        "useful_verify_ms": verify - rejected_verify,
        "draft_ms": draft,
        "draft_accepted_ms": draft * accepted_share,
        "draft_wasted_ms": draft * (1 - accepted_share),
        "wasted_fraction": (rejected_verify + draft * (1 - accepted_share)) / step,
        "tokens_per_step": requests * accepted_per_step(accept[:k]),
        "ms_per_token": step / (requests * accepted_per_step(accept[:k])),
    }


def oracle_verify(
    model: CostModel, requests: float, k: int, accept: list[float], context_tokens: float = 0.0,
    draft_ms: float | None = None,
) -> dict[str, float]:  # fmt: skip
    """Per-request oracle: verify exactly the drafts that will be accepted (plus
    the bonus token), drafting unchanged. Returns the step-time ratio."""
    draft = draft_ms if draft_ms is not None else model.dspark_draft_ms(requests, k)
    fixed = draft + model.verify_ms(requests * (1 + k), context_tokens)
    oracle = draft + model.verify_ms(requests * accepted_per_step(accept[:k]), context_tokens)
    return {"fixed_ms": fixed, "oracle_ms": oracle, "speedup": fixed / oracle}


# --- per-component drafter model -------------------------------------------


@dataclass
class DSparkComponents:
    """DSpark draft step as components, each linear in its token count:
    ``fc`` and ``context_kv`` over the target's step tokens (context KV of
    every layer), ``backbone`` over the query block (requests x k) per layer,
    ``lm_head`` over requests x k rows of the draft vocabulary, ``markov``
    k sequential embed/bias passes over the requests."""

    layers: int
    fixed_ms: float = 0.0
    fc_ms_per_token: float = 0.0
    context_kv_ms_per_token_layer: float = 0.0
    backbone_ms_per_token_layer: float = 0.0
    lm_head_ms_per_row: float = 0.0
    markov_ms_per_request_slot: float = 0.0
    markov_ms_per_slot: float = 0.0
    source: str = ""

    def draft_ms(self, requests: float, k: int, target_tokens: float | None = None,
                 layers: int | None = None) -> dict[str, float]:  # fmt: skip
        """Components of one draft step (``layers`` extrapolates the depth)."""
        n = self.layers if layers is None else layers
        target = requests * (1 + k) if target_tokens is None else target_tokens
        parts = {
            "fixed": self.fixed_ms,
            "fc": self.fc_ms_per_token * target,
            "context_kv": self.context_kv_ms_per_token_layer * target * n,
            "backbone": self.backbone_ms_per_token_layer * requests * k * n,
            "lm_head": self.lm_head_ms_per_row * requests * k,
            "markov": (self.markov_ms_per_request_slot * requests + self.markov_ms_per_slot) * k,
        }
        parts["total"] = sum(parts.values())
        return parts


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares y = a + b x; (0, y/x) through the origin with one point."""
    if not points:
        return 0.0, 0.0
    if len(points) == 1 or len({x for x, _ in points}) == 1:
        x, y = points[0]
        return 0.0, (y / x if x else 0.0)
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return my - b * mx, b


def fit_dspark_components(samples: list[dict[str, float]], layers: int = 5) -> DSparkComponents:
    """Fit per-component costs to measured draft steps.

    Each sample: ``requests``, ``k``, ``target_tokens`` (per step) and the mean
    projected ms per step of ``draft:fc``, ``draft:context_kv``,
    ``draft:backbone``, ``draft:lm_head``, ``draft:markov`` (summed over the
    step's k calls); samples without a component (graph mode) are skipped
    for it."""

    def points(key: str, x: Any) -> list[tuple[float, float]]:
        return [(x(s), s[key]) for s in samples if s.get(key) is not None and x(s) > 0]

    _, fc = _fit_line(points("draft:fc", lambda s: s["target_tokens"]))
    _, ctx = _fit_line(points("draft:context_kv", lambda s: s["target_tokens"] * layers))
    _, backbone = _fit_line(points("draft:backbone", lambda s: s["requests"] * s["k"] * layers))
    _, head = _fit_line(points("draft:lm_head", lambda s: s["requests"] * s["k"]))
    markov_points = [
        (s["requests"], s["draft:markov"] / s["k"])
        for s in samples
        if s.get("draft:markov") is not None and s["k"]
    ]
    markov_fixed, markov_req = _fit_line(markov_points)
    return DSparkComponents(
        layers=layers,
        fc_ms_per_token=fc,
        context_kv_ms_per_token_layer=ctx,
        backbone_ms_per_token_layer=backbone,
        lm_head_ms_per_row=head,
        markov_ms_per_request_slot=markov_req,
        markov_ms_per_slot=max(markov_fixed, 0.0),
        source=f"{len(samples)} profiled windows",
    )


# --- measured windows ------------------------------------------------------

STEP_RANGE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
ATTENTION_KERNEL = re.compile(r"flash|attn|attention|paged|reshape_and_cache", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _range_name(raw: str) -> str:
    return raw.lstrip(":")


def nvtx_steps(trace_csv: Path) -> list[dict[str, Any]]:
    """Per engine step: projected GPU ms of each child range, plus the batch
    composition from the worker's execute_context annotation."""
    steps: dict[str, dict[str, Any]] = {}
    rows = read_csv(trace_csv)
    by_id = {row["RangeId"]: row for row in rows}
    for row in rows:
        name = _range_name(row["Name"])
        stack = [part for part in row["RangeStack"].split(":") if part]
        root = stack[0] if stack else row["RangeId"]
        root_row = by_id.get(root)
        if root_row is None or _range_name(root_row["Name"]) != "engine:step":
            continue
        step = steps.setdefault(root, {"ranges": {}})
        duration = float(row["Projected Duration (ns)"] or 0) / 1e6
        if name == "engine:step":
            step["step_ms"] = duration
            step["start_ns"] = float(row["Projected Start (ns)"] or 0)
            continue
        match = STEP_RANGE.fullmatch(name)
        if match:
            ctx_reqs, ctx_tokens, gen_reqs, gen_tokens = map(int, match.groups())
            step.update(prefill_tokens=ctx_tokens, prefill_reqs=ctx_reqs,
                        decode_tokens=gen_tokens, decode_reqs=gen_reqs)  # fmt: skip
            continue
        step["ranges"][name] = step["ranges"].get(name, 0.0) + duration
    return [s for s in steps.values() if "step_ms" in s]


def kernel_split(kern_csv: Path) -> dict[str, float]:
    """Kernel time (ms) by class: attention/KV-cache kernels against the rest."""
    split = {"attention_ms": 0.0, "other_ms": 0.0}
    for row in read_csv(kern_csv):
        ms = float(row["Total Time (ns)"]) / 1e6
        key = "attention_ms" if ATTENTION_KERNEL.search(row["Name"]) else "other_ms"
        split[key] += ms
    split["kernel_ms"] = split["attention_ms"] + split["other_ms"]
    return split


def gpu_metrics(stats_dir: Path, stem: str) -> dict[str, float]:
    path = stats_dir / f"{stem}_gpu_metrics_summary.csv"
    if not path.exists():
        return {}
    return {row["metric"]: float(row["mean"]) for row in read_csv(path)}


def decompose_window(
    stats_dir: Path, stem: str, window: dict[str, Any], model: CostModel | None = None
) -> dict[str, Any]:
    """Measured GPU-time split of one capture window.

    Target time of mixed steps is split into prefill and verify by token
    share; verify into useful and rejected with the window's rejected drafts
    x the marginal verify cost at the window's mean decode step (cost table)
    or, without a table, the average per-token verify cost; drafting into
    slots accepted and not by the accepted share of drafted tokens (DSpark
    ``draft:fc``/``draft:context_kv`` stay a fixed per-step part)."""
    steps = nvtx_steps(stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv")
    counters = window.get("counters") or {}
    spec = counters.get("speculative_decoding") or {}
    out_tokens = float(counters.get("generation_tokens") or 0)
    draft_tokens = float(spec.get("draft_tokens") or 0)
    accepted = float(spec.get("accepted_tokens") or 0)
    rejected = max(draft_tokens - accepted, 0.0)

    totals = {
        key: 0.0
        for key in ("step", "prefill", "verify", "draft_fixed", "draft_slots", "sample", "other")
    }
    decode_tokens = prefill_tokens = 0.0
    decode_step_tokens: list[float] = []
    for step in steps:
        ranges = step["ranges"]
        target = sum(v for k, v in ranges.items() if k.startswith("target:execute"))
        p_tok, d_tok = step.get("prefill_tokens", 0), step.get("decode_tokens", 0)
        share = p_tok / (p_tok + d_tok) if p_tok + d_tok else 0.0
        totals["step"] += step["step_ms"]
        totals["prefill"] += target * share
        totals["verify"] += target * (1 - share)
        fixed = ranges.get("draft:fc", 0.0) + ranges.get("draft:context_kv", 0.0)
        totals["draft_fixed"] += fixed
        totals["draft_slots"] += max(ranges.get("draft:propose", 0.0) - fixed, 0.0)
        totals["sample"] += ranges.get("target:sample", 0.0) + ranges.get("target:postprocess", 0)
        decode_tokens += d_tok
        prefill_tokens += p_tok
        if d_tok:
            decode_step_tokens.append(d_tok)
    accounted = sum(v for k, v in totals.items() if k != "step")
    totals["other"] = max(totals["step"] - accounted, 0.0)

    mean_decode = sum(decode_step_tokens) / len(decode_step_tokens) if decode_step_tokens else 0
    if model is not None and mean_decode:
        per_token = model.marginal_verify_ms(mean_decode)
    else:
        per_token = totals["verify"] / decode_tokens if decode_tokens else 0.0
    rejected_verify = min(rejected * per_token, totals["verify"])
    accepted_share = accepted / draft_tokens if draft_tokens else 0.0
    window_ms = 1000.0 * float(window.get("window_s") or 0)
    kernels = {}
    kern = stats_dir / f"{stem}_cuda_gpu_kern_sum.csv"
    if kern.exists():
        kernels = kernel_split(kern)
    parts = {
        "prefill_ms": totals["prefill"],
        "useful_verify_ms": totals["verify"] - rejected_verify,
        "rejected_verify_ms": rejected_verify,
        "draft_accepted_ms": totals["draft_slots"] * accepted_share,
        "draft_wasted_ms": totals["draft_slots"] * (1 - accepted_share),
        "draft_fixed_ms": totals["draft_fixed"],
        "sample_ms": totals["sample"],
        "other_in_step_ms": totals["other"],
    }
    span = (
        steps[-1]["start_ns"] / 1e6 + steps[-1]["step_ms"] - steps[0]["start_ns"] / 1e6
        if steps
        else 0
    )
    parts["between_steps_ms"] = max(span - totals["step"], 0.0)
    total = sum(parts.values())
    result: dict[str, Any] = {
        "steps": len(steps),
        "trace_span_ms": span,
        "window_ms": window_ms,
        "output_tokens": out_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "mean_decode_step_tokens": mean_decode,
        "rejected_tokens": rejected,
        "acceptance_length": spec.get("mean_acceptance_length"),
        "marginal_verify_ms_per_token": per_token,
        "parts_ms": parts,
        "fractions": {k: v / total for k, v in parts.items()} if total else {},
        "kernels": kernels,
        "attention_fraction_of_kernels": (
            kernels["attention_ms"] / kernels["kernel_ms"] if kernels.get("kernel_ms") else None
        ),
        "gpu_busy_fraction": (
            kernels["kernel_ms"] / span if kernels.get("kernel_ms") and span else None
        ),  # fmt: skip
        "gpu_metrics": gpu_metrics(stats_dir, stem),
    }
    if out_tokens and span:
        # GPU-ms per output token, using the token rate inside the window.
        rate = out_tokens / (float(window.get("open_s") or window.get("window_s") or 1) * 1000)
        result["gpu_ms_per_output_token"] = {k: v / span / rate for k, v in parts.items()}
        result["wasted_gpu_ms_per_output_token"] = (
            (parts["rejected_verify_ms"] + parts["draft_wasted_ms"]) / span / rate
        )
    return result


def dspark_samples(stats_dir: Path, stem: str, k: int) -> dict[str, float] | None:
    """Mean per-step component times of a DSpark window (for the fit)."""
    steps = nvtx_steps(stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv")
    decode = [s for s in steps if not s.get("prefill_tokens")]
    if not decode:
        return None
    mean = lambda key: sum(s["ranges"].get(key, 0.0) for s in decode) / len(decode)  # noqa: E731
    sample: dict[str, float] = {
        "requests": sum(s.get("decode_reqs", 0) for s in decode) / len(decode),
        "k": k,
        "target_tokens": sum(s.get("decode_tokens", 0) for s in decode) / len(decode),
    }
    for key in ("draft:fc", "draft:context_kv", "draft:propose", "draft:graph"):
        sample[key] = mean(key)
    if any("draft:backbone" in s["ranges"] for s in decode):
        for key in ("draft:backbone", "draft:lm_head", "draft:markov"):
            sample[key] = mean(key)
    return sample


def eagle_pass_ms(stats_dir: Path, stem: str) -> dict[str, float] | None:
    """Mean projected ms of the first and of each later EAGLE pass in decode steps."""
    steps = [s for s in nvtx_steps(stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv")
             if not s.get("prefill_tokens")]  # fmt: skip
    if not steps:
        return None
    rows = read_csv(stats_dir / f"{stem}_nvtx_gpu_proj_sum.csv")
    counts = {_range_name(r["Range"]): float(r["Range Instances"]) for r in rows}
    totals = {_range_name(r["Range"]): float(r["Total Proj Time (ns)"]) / 1e6 for r in rows}
    out = {"requests": sum(s.get("decode_reqs", 0) for s in steps) / len(steps)}
    for key in ("draft:pass0", "draft:pass"):
        if counts.get(key):
            out[key] = totals[key] / counts[key]
    return out


# --- cost per token --------------------------------------------------------


def _parse_time(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


# nvidia-smi stamps gpu-memory.csv in the compute node's local time.
CLUSTER_TZ = ZoneInfo("Europe/Zurich")


def power_samples(csv_path: Path) -> list[tuple[float, int, float]]:
    """(unix time, GPU index, watts) from a sweep job's gpu-memory.csv."""
    out = []
    for line in csv_path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=CLUSTER_TZ)
            out.append((stamp.timestamp(), int(parts[1]), float(parts[5])))
        except ValueError:
            continue
    return out


def mean_power(samples: list[tuple[float, int, float]], gpus: list[int], start: float,
               end: float) -> float | None:  # fmt: skip
    """Mean summed power of ``gpus`` between ``start`` and ``end``."""
    per_gpu = []
    for gpu in gpus:
        values = [w for t, g, w in samples if g == gpu and start <= t <= end]
        if not values:
            return None
        per_gpu.append(sum(values) / len(values))
    return sum(per_gpu)


def cell_costs(sweep: Path, label: str | None = None) -> list[dict[str, Any]]:
    """Per arm and level: tok/s, GPU-s per 1M output tokens, W and J/token
    (whole arm's GPUs, nvidia-smi samples within the cell's measured phase)."""
    samples = power_samples(sweep / "gpu-memory.csv") if (sweep / "gpu-memory.csv").exists() else []
    rows = []
    for summary_path in sorted(sweep.glob("*/*/*/c*/summary.json")):
        cell = summary_path.parent
        arm_dir = cell.parent.parent.parent
        test = cell.parent.parent.name
        if label and test != label:
            continue
        status = json.loads((arm_dir / "status.json").read_text())
        meta = json.loads((cell / "metadata.json").read_text())
        summary = json.loads(summary_path.read_text())
        gpus = list(status.get("gpus") or [0])
        rate = summary["tokens"]["output_tokens_per_second"] or 0
        watts = mean_power(samples, gpus, _parse_time(meta["started_at"]),
                           _parse_time(meta["ended_at"])) if samples else None  # fmt: skip
        spec = summary.get("speculative_decoding") or {}
        rows.append(
            {
                "sweep": sweep.name,
                "arm": arm_dir.name,
                "label": test,
                "concurrency": int(cell.name[1:]),
                "profiled": "profile" in (meta.get("extra") or {}),
                "output_tokens_per_second": rate,
                "gpu_seconds_per_million_tokens": len(gpus) * 1e6 / rate if rate else None,
                "watts": watts,
                "joules_per_token": watts / rate if watts and rate else None,
                "acceptance_length": spec.get("mean_acceptance_length"),
                "acceptance_per_position": acceptance(summary),
            }
        )
    return rows


def best_k_ceilings(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Load-dependent k schedule against the best single fixed k over the
    levels, for arms named ``{prefix}{k}`` (e.g. dspark-e10-k): the schedule
    takes the best k at every level; a fixed k must serve all of them."""
    arms: dict[str, dict[int, float]] = {}
    for row in rows:
        suffix = row["arm"][len(prefix) :]
        if row["arm"].startswith(prefix) and suffix.isdigit() and not row["profiled"]:
            arms.setdefault(row["arm"], {})[row["concurrency"]] = row["output_tokens_per_second"]
    levels = sorted(set.intersection(*(set(v) for v in arms.values()))) if arms else []
    if not levels:
        return {}
    schedule = {c: max(arms, key=lambda a: arms[a][c]) for c in levels}

    def geo(values: list[float]) -> float:
        product = 1.0
        for v in values:
            product *= v
        return product ** (1 / len(values))

    fixed = {a: geo([arms[a][c] for c in levels]) for a in arms}
    best_fixed = max(fixed, key=fixed.get)
    sched = geo([arms[schedule[c]][c] for c in levels])
    return {
        "levels": levels,
        "schedule": schedule,
        "best_fixed": best_fixed,
        "schedule_over_best_fixed": sched / fixed[best_fixed],
        "per_level_gain_over_best_fixed": {
            c: arms[schedule[c]][c] / arms[best_fixed][c] for c in levels
        },
    }


def saturation(levels: list[dict[str, Any]], metric: str, threshold: float = 0.9) -> int | None:
    """Smallest concurrency whose ``metric`` reaches ``threshold`` of its maximum."""
    values = [(row["concurrency"], row[metric]) for row in levels if row.get(metric) is not None]
    if not values:
        return None
    peak = max(v for _, v in values)
    for concurrency, value in sorted(values):
        if value >= threshold * peak:
            return concurrency
    return None


def verify_saturation_tokens(model: CostModel, threshold: float = 0.8) -> float:
    """Verify tokens from which the marginal cost per token stays within
    ``threshold`` of its compute-bound value (the slope over the top half of
    the table)."""
    xs, ys = model.verify_tokens, model.verify_ms_table
    top = xs[-1] / 2
    asymptote = _slope(xs, ys, (top + xs[-1]) / 2, span=(xs[-1] - top) / 2)
    candidates = [t for t in xs if t <= top]
    for tokens in candidates:
        if all(
            model.marginal_verify_ms(t) >= threshold * asymptote for t in xs if tokens <= t <= top
        ):
            return tokens
    return top


# --- whole-sweep reports ---------------------------------------------------


def decompose_sweeps(sweeps: list[Path], model: CostModel | None) -> list[dict[str, Any]]:
    """One row per profiled window of every profiled arm in ``sweeps``."""
    rows = []
    for sweep in sweeps:
        for manifest_path in sorted(sweep.glob("*/nsys-stats/manifest.json")):
            stats_dir = manifest_path.parent
            arm = stats_dir.parent.name
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["reports"]:
                window = entry.get("window") or {}
                stem = Path(entry["report"]).name.removesuffix(".nsys-rep")
                if not (stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv").exists():
                    continue
                row = decompose_window(stats_dir, stem, window, model)
                row.update(
                    sweep=sweep.name, arm=arm, report=entry["report"],
                    loadtest=window.get("loadtest"), concurrency=window.get("concurrency"),
                )  # fmt: skip
                rows.append(row)
    return rows


def _fmt(value: Any, spec: str = "{:.1%}") -> str:
    return "-" if value is None else spec.format(value)


def decomposition_table(rows: list[dict[str, Any]]) -> str:
    keys = ["prefill_ms", "useful_verify_ms", "rejected_verify_ms", "draft_accepted_ms",
            "draft_wasted_ms", "draft_fixed_ms", "sample_ms", "other_in_step_ms",
            "between_steps_ms"]  # fmt: skip
    head = ["arm", "load", "C", "acc len", *[k.removesuffix("_ms") for k in keys],
            "wasted ms/tok"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for row in sorted(rows, key=lambda r: (r["arm"], r.get("loadtest") or "", r["concurrency"])):
        frac = row["fractions"]
        lines.append(
            "| " + " | ".join([
                row["arm"], str(row.get("loadtest")), str(row["concurrency"]),
                _fmt(row.get("acceptance_length"), "{:.2f}"),
                *[_fmt(frac.get(k)) for k in keys],
                _fmt(row.get("wasted_gpu_ms_per_output_token"), "{:.3f}"),
            ]) + " |"
        )  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus_bench.waste_report")
    sub = parser.add_subparsers(dest="command", required=True)
    cost = sub.add_parser("cost", help="tok/s, GPU-s per 1M tokens and J/token per cell")
    cost.add_argument("sweeps", type=Path, nargs="+")
    cost.add_argument("--label")
    cost.add_argument("--json", type=Path)
    decompose = sub.add_parser("decompose", help="GPU-time split of every profiled window")
    decompose.add_argument("sweeps", type=Path, nargs="+")
    decompose.add_argument("--cost-table", type=Path)
    decompose.add_argument("--json", type=Path)
    marginal = sub.add_parser("marginal", help="the marginal rule per draft slot")
    marginal.add_argument("--cost-table", type=Path, required=True)
    marginal.add_argument("--summary", type=Path, required=True)
    marginal.add_argument("--requests", type=float, required=True)
    marginal.add_argument("--context", type=float, default=0.0, help="cached tokens, all requests")
    marginal.add_argument("--k", type=int)
    marginal.add_argument("--eagle-pass-ms", type=float, help="EAGLE: cost per drafter pass")
    args = parser.parse_args(argv)

    if args.command == "cost":
        rows = [row for sweep in args.sweeps for row in cell_costs(sweep, args.label)]
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print("| sweep | arm | load | C | tok/s | GPU-s / 1M tok | W | J/tok | acc len |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            print(f"| {r['sweep']} | {r['arm']} | {r['label']} | {r['concurrency']} | "
                  f"{r['output_tokens_per_second']:,.0f} | "
                  f"{_fmt(r['gpu_seconds_per_million_tokens'], '{:,.0f}')} | "
                  f"{_fmt(r['watts'], '{:.0f}')} | {_fmt(r['joules_per_token'], '{:.3f}')} | "
                  f"{_fmt(r['acceptance_length'], '{:.2f}')} |")  # fmt: skip
    elif args.command == "decompose":
        model = CostModel.from_cost_table(args.cost_table) if args.cost_table else None
        rows = decompose_sweeps(args.sweeps, model)
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(decomposition_table(rows))
    else:
        model = CostModel.from_cost_table(args.cost_table)
        summary = json.loads(args.summary.read_text())
        accept = acceptance(summary)
        k = args.k or len(accept)
        slots = [args.eagle_pass_ms] * k if args.eagle_pass_ms else None
        rows = marginal_rule(model, args.requests, k, accept, args.context, slots)
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CostModel", "DSparkComponents", "acceptance", "accepted_per_step",
    "best_k_ceilings", "cell_costs", "decompose_window", "fit_dspark_components",
    "marginal_rule", "oracle_verify", "predicted_speedup", "saturation", "step_waste",
    "verify_saturation_tokens",
]  # fmt: skip
