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
import gzip
import json
import re
from dataclasses import asdict, dataclass, field
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


# --- per-component drafter models ------------------------------------------


def least_squares(rows: list[list[float]], ys: list[float]) -> list[float]:
    """Coefficients c minimising |X c - y|^2 (normal equations, small X)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    v = [sum(r[i] * y for r, y in zip(rows, ys, strict=True)) for i in range(n)]
    for col in range(n):  # Gauss-Jordan with partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot], v[col], v[pivot] = a[pivot], a[col], v[pivot], v[col]
        if abs(a[col][col]) < 1e-30:
            raise ValueError("singular fit: the samples do not vary in every regressor")
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col], strict=True)]
                v[r] -= f * v[col]
    return [v[i] / a[i][i] for i in range(n)]


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares y = a + b x; (0, y/x) through the origin with one x value."""
    if not points:
        return 0.0, 0.0
    if len({x for x, _ in points}) == 1:
        x, y = points[0]
        return 0.0, (y / x if x else 0.0)
    a, b = least_squares([[1.0, x] for x, _ in points], [y for _, y in points])
    return a, b


@dataclass
class DSparkComponents:
    """DSpark draft step as components (ms), for depth extrapolation.

    - ``fc``: combine of the target's aux hidden states, over the step's
      target tokens (eager, outside the graph);
    - ``context_kv``: context K/V projection of every layer over the target
      tokens (eager, per layer);
    - ``backbone``: the query block (requests x k tokens) through each layer,
      plus its attention over the cached context (capped at the drafter's
      sliding window) per layer;
    - ``lm_head``: the draft-vocabulary head over requests x k rows;
    - ``markov``: k sequential embed/bias passes;
    - ``fixed``: the batch-independent floor of the draft graph (weight reads,
      launches), split between the layers and the head by parameter count.
    """

    layers: int = 5
    window: int = 2048
    layer_fixed_ms: float = 0.0
    head_fixed_ms: float = 0.0
    fc_fixed_ms: float = 0.0
    fc_ms_per_token: float = 0.0
    context_kv_fixed_ms_layer: float = 0.0
    context_kv_ms_per_token_layer: float = 0.0
    backbone_ms_per_token_layer: float = 0.0
    attention_ms_per_token_context_layer: float = 0.0
    lm_head_ms_per_row: float = 0.0
    markov_ms_per_slot: float = 0.0
    # Extra time of a step that runs the query block eagerly (no CUDA graph:
    # more than max_cudagraph_capture_size query tokens); not in draft_ms.
    eager_offset_ms: float = 0.0
    propose_overhead_ms: float = 0.0
    layer_params: float = 306e6
    head_params: float = 4096 * 131072
    source: str = ""

    def draft_ms(
        self, requests: float, k: int, target_tokens: float | None = None,
        layers: int | None = None, context_tokens: float = 0.0,
    ) -> dict[str, float]:  # fmt: skip
        """Components of one draft step; ``layers`` extrapolates the depth,
        ``context_tokens`` is the mean cached context per request."""
        n = self.layers if layers is None else layers
        target = requests * (1 + k) if target_tokens is None else target_tokens
        query = requests * k
        parts = {
            "fixed": n * self.layer_fixed_ms + self.head_fixed_ms,
            "fc": self.fc_fixed_ms + self.fc_ms_per_token * target,
            "context_kv": n * (self.context_kv_fixed_ms_layer
                               + self.context_kv_ms_per_token_layer * target),
            "backbone": n * query * (self.backbone_ms_per_token_layer
                                     + self.attention_ms_per_token_context_layer
                                     * min(context_tokens, self.window)),
            "lm_head": self.lm_head_ms_per_row * query,
            "markov": self.markov_ms_per_slot * k,
            "overhead": self.propose_overhead_ms,
        }  # fmt: skip
        parts["total"] = sum(parts.values())
        return parts


def fit_dspark_components(
    samples: list[dict[str, float]], layers: int = 5, window: int = 2048
) -> DSparkComponents:
    """Fit the components to measured draft steps.

    Each sample has ``requests``, ``k``, ``target_tokens`` and
    ``context_tokens`` (per step / per request), the mean projected ms per
    step of ``draft:fc`` and ``draft:context_kv`` (always eager), and the
    query-block time: ``draft:graph`` in graph mode, or ``draft:backbone`` +
    ``draft:lm_head`` + ``draft:markov`` when the step ran eagerly (large
    batches). The query-block time is fitted as
    ``F + a (B k) + c k + d (B k min(ctx, window))``; ``a`` is split between
    the backbone layers and the head by their ratio in the eager windows."""
    fc0, fc = _fit_line([(s["target_tokens"], s["draft:fc"]) for s in samples])
    ctx0, ctx = _fit_line([(s["target_tokens"], s["draft:context_kv"]) for s in samples])
    rows, ys = [], []
    for s in samples:
        eager_step = bool(s.get("draft:backbone"))
        if eager_step:
            y = s["draft:backbone"] + s.get("draft:lm_head", 0.0) + s.get("draft:markov", 0.0)
        elif s.get("draft:graph"):
            y = s["draft:graph"]
        else:
            continue
        query = s["requests"] * s["k"]
        context = min(s.get("context_tokens") or 0.0, window)
        # Weighted by 1/y: relative errors, so the small batches (where the
        # floor and the per-slot latency show) count as much as the large.
        row = [1.0, query, s["k"], query * context, float(eager_step)]
        rows.append([x / y for x in row])
        ys.append(1.0)
    if not any(row[4] for row in rows):
        rows = [row[:4] for row in rows]
    coefficients = least_squares(rows, ys)
    floor, per_query, per_slot, attention = coefficients[:4]
    eager_offset = coefficients[4] if len(coefficients) > 4 else 0.0
    eager = [s for s in samples if s.get("draft:backbone") and s.get("draft:lm_head")]
    head_share = (
        sum(s["draft:lm_head"] for s in eager)
        / sum(s["draft:lm_head"] + s["draft:backbone"] for s in eager)
        if eager
        else 0.2
    )
    model = DSparkComponents(layers=layers, window=window)
    total_params = layers * model.layer_params + model.head_params
    model.layer_fixed_ms = floor * model.layer_params / total_params
    model.head_fixed_ms = floor * model.head_params / total_params
    model.fc_fixed_ms, model.fc_ms_per_token = fc0, fc
    model.context_kv_fixed_ms_layer = ctx0 / layers
    model.context_kv_ms_per_token_layer = ctx / layers
    model.lm_head_ms_per_row = per_query * head_share
    model.backbone_ms_per_token_layer = per_query * (1 - head_share) / layers
    model.attention_ms_per_token_context_layer = attention / layers
    # Per-slot latency of the Markov loop; clamped, a small negative fit is noise.
    model.markov_ms_per_slot = max(per_slot, 0.0)
    model.eager_offset_ms = eager_offset
    # propose() also prepares inputs and attention metadata around the graph.
    overheads = [
        s["draft:propose"] - s["draft:graph"] - s["draft:fc"] - s["draft:context_kv"]
        for s in samples
        if s.get("draft:graph") and s.get("draft:propose")
    ]
    model.propose_overhead_ms = sum(overheads) / len(overheads) if overheads else 0.0
    model.source = f"{len(rows)} windows ({len(eager)} eager), head share {head_share:.2f}"
    return model


@dataclass
class EaglePassCost:
    """EAGLE draft step: the first pass runs the drafter over the step's target
    tokens (draft prefill), each later pass over one token per request."""

    pass0_fixed_ms: float = 0.0
    pass0_ms_per_request: float = 0.0
    pass0_ms_per_target_token: float = 0.0
    pass_fixed_ms: float = 0.0
    pass_ms_per_request: float = 0.0
    source: str = ""

    def pass0_ms(self, requests: float, target_tokens: float) -> float:
        return (self.pass0_fixed_ms + self.pass0_ms_per_request * requests
                + self.pass0_ms_per_target_token * target_tokens)  # fmt: skip

    def pass_ms(self, requests: float) -> float:
        return self.pass_fixed_ms + self.pass_ms_per_request * requests

    def draft_ms(self, requests: float, k: int, target_tokens: float | None = None) -> float:
        target = requests * (1 + k) if target_tokens is None else target_tokens
        return self.pass0_ms(requests, target) + (k - 1) * self.pass_ms(requests)


def fit_eagle_passes(samples: list[dict[str, float]]) -> EaglePassCost:
    """Fit from per-window means: ``requests``, ``target_tokens``,
    ``draft:pass0`` and (k > 1) ``draft:pass``."""
    first = [s for s in samples if s.get("draft:pass0")]
    later = [s for s in samples if s.get("draft:pass")]
    p0 = least_squares(
        [[1.0, s["requests"], s["target_tokens"]] for s in first], [s["draft:pass0"] for s in first]
    )
    p = _fit_line([(s["requests"], s["draft:pass"]) for s in later])
    return EaglePassCost(
        pass0_fixed_ms=p0[0], pass0_ms_per_request=p0[1], pass0_ms_per_target_token=p0[2],
        pass_fixed_ms=p[0], pass_ms_per_request=p[1],
        source=f"{len(first)} windows (pass0), {len(later)} windows (later passes)",
    )  # fmt: skip


# --- measured windows ------------------------------------------------------

STEP_RANGE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
ATTENTION_KERNEL = re.compile(r"flash|attn|attention|paged|reshape_and_cache", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    """A CSV, or its ``.gz`` (committed records gzip the per-range traces)."""
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _has_csv(path: Path) -> bool:
    return path.exists() or path.with_name(path.name + ".gz").exists()


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
        step = steps.setdefault(root, {"ranges": {}, "counts": {}})
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
        step["counts"][name] = step["counts"].get(name, 0) + 1
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
    share. Verify is split into useful and rejected step by step: the step's
    rejected tokens (its decode tokens x the window's rejected share of decode
    tokens) x the marginal cost per token at the step's total size (cost
    table) or, without a table, the step's average cost per token, capped at
    the step's verify share. Drafting is split into
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
    total_decode = sum(step.get("decode_tokens", 0) for step in steps)
    rejected_share = min(rejected / total_decode, 1.0) if total_decode else 0.0
    rejected_verify = 0.0
    for step in steps:
        ranges = step["ranges"]
        target = sum(v for k, v in ranges.items() if k.startswith("target:execute"))
        p_tok, d_tok = step.get("prefill_tokens", 0), step.get("decode_tokens", 0)
        share = p_tok / (p_tok + d_tok) if p_tok + d_tok else 0.0
        totals["step"] += step["step_ms"]
        totals["prefill"] += target * share
        totals["verify"] += target * (1 - share)
        if d_tok:
            size = p_tok + d_tok
            per_token = model.marginal_verify_ms(size) if model is not None else target / size
            rejected_verify += min(rejected_share * d_tok * per_token, target * (1 - share))
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
    per_token = rejected_verify / rejected if rejected else 0.0
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
        "truncated": bool(window.get("truncated")),
        # A window that saw no prompt tokens caught only the level's drain.
        "steady": not window.get("truncated")
        or bool(counters.get("prompt_tokens") or prefill_tokens),
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
    # Yu's framing: of the work speculation adds (verifying the drafted tokens
    # and drafting), how much is wasted? Drafted-token verify is priced like
    # the rejected part (same marginal cost per token).
    drafted_verify = rejected_verify * draft_tokens / rejected if rejected else 0.0
    speculative = drafted_verify + totals["draft_slots"] + totals["draft_fixed"]
    result["speculative_ms"] = speculative
    result["speculative_fraction"] = speculative / total if total else None
    result["speculative_wasted_share"] = (
        (rejected_verify + parts["draft_wasted_ms"]) / speculative if speculative else None
    )
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
    """Mean projected ms of the first and of each later EAGLE pass, over the
    window's decode-only steps (in mixed steps the first pass also covers the
    prompt tokens)."""
    steps = [s for s in nvtx_steps(stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv")
             if not s.get("prefill_tokens") and s["ranges"].get("draft:pass0")]  # fmt: skip
    if not steps:
        return None
    out = {
        "requests": sum(s.get("decode_reqs", 0) for s in steps) / len(steps),
        "target_tokens": sum(s.get("decode_tokens", 0) for s in steps) / len(steps),
        "draft:pass0": sum(s["ranges"]["draft:pass0"] for s in steps) / len(steps),
    }
    later = [s for s in steps if s["counts"].get("draft:pass")]
    if later:
        out["draft:pass"] = sum(
            s["ranges"]["draft:pass"] / s["counts"]["draft:pass"] for s in later
        ) / len(later)
    return out


def mean_context(arm_dir: Path, label: str | None, concurrency: int | None) -> float | None:
    """Mean cached tokens per running request in a cell: its prompt plus half
    its completion, averaged over the cell's requests."""
    if label is None or concurrency is None:
        return None
    for path in (arm_dir / label).glob(f"*/c{concurrency}/summary.json"):
        summary = json.loads(path.read_text())
        requests = summary["requests"]["successful"] or 1
        tokens = summary["tokens"]
        return (tokens["prompt"] + tokens["completion"] / 2) / requests
    return None


def drafter_samples(sweeps: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Per-window drafter samples of every profiled DSpark and EAGLE arm
    (arm names ``dspark-…-k{K}`` / ``eagle-…-k{K}``); drain-only windows
    are left out."""
    out: dict[str, list[dict[str, Any]]] = {"dspark": [], "eagle": []}
    for sweep in sweeps:
        for manifest_path in sorted(sweep.glob("*/nsys-stats/manifest.json")):
            stats_dir = manifest_path.parent
            arm = stats_dir.parent.name
            kind = (
                "dspark"
                if arm.startswith("dspark")
                else "eagle"
                if arm.startswith("eagle")
                else None
            )
            match = re.search(r"-k(\d+)$", arm)
            if kind is None or match is None:
                continue
            for entry in json.loads(manifest_path.read_text())["reports"]:
                window = entry.get("window") or {}
                counters = window.get("counters") or {}
                if window.get("truncated") and not counters.get("prompt_tokens"):
                    continue
                stem = Path(entry["report"]).name.removesuffix(".nsys-rep")
                if kind == "dspark":
                    sample = dspark_samples(stats_dir, stem, int(match.group(1)))
                else:
                    sample = eagle_pass_ms(stats_dir, stem)
                if sample:
                    sample["context_tokens"] = mean_context(
                        stats_dir.parent, window.get("loadtest"), window.get("concurrency")
                    )
                    sample.update(arm=arm, sweep=sweep.name, k=int(match.group(1)),
                                  loadtest=window.get("loadtest"),
                                  concurrency=window.get("concurrency"))  # fmt: skip
                    out[kind].append(sample)
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
                if not _has_csv(stats_dir / f"{stem}_nvtx_gpu_proj_trace.csv"):
                    continue
                row = decompose_window(stats_dir, stem, window, model)
                row.update(
                    sweep=sweep.name, arm=arm, report=entry["report"],
                    loadtest=window.get("loadtest"), concurrency=window.get("concurrency"),
                )  # fmt: skip
                rows.append(row)
    return rows


def fit_drafters(sweeps: list[Path]) -> dict[str, Any]:
    """Fitted DSpark components and EAGLE pass costs, with the samples and
    each sample's prediction (relative error) for checking."""
    samples = drafter_samples(sweeps)
    report: dict[str, Any] = {"sweeps": [sweep.name for sweep in sweeps]}
    if samples["dspark"]:
        dspark = fit_dspark_components(samples["dspark"])
        for sample in samples["dspark"]:
            parts = dspark.draft_ms(
                sample["requests"], sample["k"], sample["target_tokens"],
                context_tokens=sample.get("context_tokens") or 0.0,
            )  # fmt: skip
            predicted = parts["total"] + (
                dspark.eager_offset_ms if sample.get("draft:backbone") else 0
            )
            sample["predicted_propose_ms"] = predicted
            sample["relative_error"] = predicted / sample["draft:propose"] - 1
        report["dspark"] = {"model": asdict(dspark), "samples": samples["dspark"]}
    if samples["eagle"]:
        eagle = fit_eagle_passes(samples["eagle"])
        for sample in samples["eagle"]:
            sample["predicted_pass0_ms"] = eagle.pass0_ms(
                sample["requests"], sample["target_tokens"]
            )
            if sample.get("draft:pass"):
                sample["predicted_pass_ms"] = eagle.pass_ms(sample["requests"])
        report["eagle"] = {"model": asdict(eagle), "samples": samples["eagle"]}
    return report


TENSOR = "Tensor Active [Throughput %]"
SM = "SMs Active [Throughput %]"
DRAM = "DRAM Read Bandwidth [Throughput %]"


def saturation_rows(rows: list[dict[str, Any]], threshold: float = 0.9) -> list[dict[str, Any]]:
    """Per arm and load test: GPU metrics by level and the saturation
    concurrency C* (first level whose tensor-pipe activity reaches
    ``threshold`` of the arm's maximum over the levels)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("steady", True) and row.get("gpu_metrics"):
            groups.setdefault((row["arm"], row.get("loadtest") or ""), []).append(row)
    out = []
    for (arm, test), members in sorted(groups.items()):
        levels = [
            {"concurrency": r["concurrency"], "tensor": r["gpu_metrics"].get(TENSOR),
             "sm": r["gpu_metrics"].get(SM), "dram": r["gpu_metrics"].get(DRAM)}
            for r in sorted(members, key=lambda r: r["concurrency"])
        ]  # fmt: skip
        out.append(
            {"arm": arm, "loadtest": test, "levels": levels,
             "c_star_tensor": saturation(levels, "tensor", threshold),
             "c_star_sm": saturation(levels, "sm", threshold)}
        )  # fmt: skip
    return out


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
    fit = sub.add_parser("fit", help="fit DSpark component and EAGLE pass costs")
    fit.add_argument("sweeps", type=Path, nargs="+")
    fit.add_argument("--json", type=Path)
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
    elif args.command == "fit":
        report = fit_drafters(args.sweeps)
        text = json.dumps(report, indent=2) + "\n"
        if args.json:
            args.json.write_text(text)
        print(text, end="")
    elif args.command == "decompose":
        model = CostModel.from_cost_table(args.cost_table) if args.cost_table else None
        rows = decompose_sweeps(args.sweeps, model)
        saturated = saturation_rows(rows)
        if args.json:
            args.json.write_text(
                json.dumps({"windows": rows, "saturation": saturated}, indent=2) + "\n"
            )
        print(decomposition_table([r for r in rows if r.get("steady", True)]))
        print()
        print("| arm | load | tensor active % by C | C* (tensor) | C* (SM) |")
        print("| --- | --- | --- | --- | --- |")
        for row in saturated:
            levels = ", ".join(f"{lv['concurrency']}: {_fmt(lv['tensor'], '{:.0f}')}"
                               for lv in row["levels"])  # fmt: skip
            print(f"| {row['arm']} | {row['loadtest']} | {levels} | {row['c_star_tensor']} | "
                  f"{row['c_star_sm']} |")  # fmt: skip
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
