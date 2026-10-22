"""Offline headroom of draft-length policies for DSpark, from a confidence dump.

    python3 headroom_sim.py OUT.json DUMP.jsonl [DUMP.jsonl ...] --cost-table COST.json
        --sts STS.json [--measured MEASURED.json] [--concurrency 64 128 256]

Input: the per-verification lines VLLM_DSPARK_CONF_DUMP writes with every slot
verified (k=7, VLLM_DSPARK_VERIFY_MODE=none): request id ``req``, raw head
logits ``z``, verified ``n`` and accepted ``a``; with
vllm-apertus-dspark-conf-dump-task.patch also the context ``ctx`` and the task
``label``. Each request becomes a trajectory: per verification, the prefix
survival of its drafts (STS-calibrated head, ``c_j = sigmoid(z_j / T_j)``,
running product) and the accepted run at k=7; and the tokens it produced.

A closed loop of C slots replays the trajectories. Every step a policy picks
a draft and verify length ``L_r`` per request; the request gains
``1 + min(a, L_r)`` tokens (the run at a shorter length is the k=7 run cut to
it) and moves to its next record. A request ends when it has produced its k=7
token count, and its slot takes the next request and pays its prefill. Step:

    verify_ms(R + sum L_r) + draft + kv_ms_per_token * context + overhead

``verify_ms``, the draft times per length and ``kv_ms_per_token`` come from the
cost table the paper scheduler profiles at engine start. Drafting is costed
three ways: all 7 slots (the paper), the step's longest length (the per-step
draft cut, ``draft_ms_by_len``), or per (request, slot) pair, from a
least-squares fit of the same table, ``a0 + aR * R + sum_j (o + c * n_j)``
with ``n_j`` the requests drafting slot j (a per-request cut that runs the
Markov step and LM head only on the kept rows). ``overhead`` (ms per step) and
the prefill cost (us per prompt token) are fitted to measured plain and
fixed-k throughput (``--measured``).

Length rules (``plan``): fixed k; ``topk``, the paper's greedy admission by
descending survival with the best prefix for tokens per ms under the chosen
draft cost (per step, on the current survival: an upper bound for the rule);
``budget``, the implementable per-request form: per-slot row counts ``n_j``
from the ``topk`` plan of two steps earlier (as the paper lags its budget),
applied to the current survival, slot by slot, to the rows still drafting
(the slot-j confidence is known before slot j's token is drawn); and
``threshold``. Signals: the head (``conf``), the task label's mean survival
(``label``: a per-task length, what a gateway label alone gives), the head
recalibrated per label (``conf-label``), and the truth (``oracle``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SLOTS = 7
LAG = 2


@dataclass
class Trajectory:
    survival: np.ndarray  # [steps, SLOTS] prefix survival (STS head)
    accepted: np.ndarray  # [steps] accepted run at k=SLOTS
    tokens: int  # tokens produced at k=SLOTS
    label: str
    prompt: float | None  # prompt tokens (first dump line's context), if dumped


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load(paths: list[Path], temperatures: list[float]) -> list[Trajectory]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                # Ids may repeat across servers: key them per file.
                rows[f"{path}:{row['req']}"].append(row)
    temps = np.asarray(temperatures[:SLOTS], dtype=np.float64)
    out = []
    for records in rows.values():
        records = [r for r in records if len(r["z"]) >= SLOTS]
        if len(records) < 2:
            continue
        z = np.array([r["z"][:SLOTS] for r in records], dtype=np.float64)
        acc = np.array([min(r["a"], SLOTS) for r in records])
        survival = np.cumprod(sigmoid(z / temps), axis=1)
        # The last line can be cut short by a stop: it counts for the
        # tokens, not for the replayed acceptance.
        out.append(
            Trajectory(
                survival=survival[:-1],
                accepted=acc[:-1],
                tokens=int((acc + 1).sum()),
                label=str(records[0].get("label", "all")),
                prompt=float(records[0]["ctx"]) if "ctx" in records[0] else None,
            )
        )
    return out


class CostModel:
    def __init__(self, table: dict, overhead_ms: float = 0.0, prefill_us: float = 0.0):
        self.verify_tokens = np.asarray(table["verify_tokens"], dtype=np.float64)
        self.verify_ms = np.asarray(table["verify_ms"], dtype=np.float64)
        self.draft_reqs = np.asarray(table["draft_reqs"], dtype=np.float64)
        self.draft_by_len = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in table["draft_ms_by_len"].items()
        }
        self.kv_ms_per_token = float(table.get("kv_ms_per_token", 0.0))
        self.overhead_ms = overhead_ms
        self.prefill_us = prefill_us
        top = self.verify_tokens[-1]
        tail = self.verify_tokens > 0.75 * top
        if tail.sum() < 3:
            tail[-3:] = True
        self._tail_slope = float(np.polyfit(self.verify_tokens[tail], self.verify_ms[tail], 1)[0])
        # Per-pair fit: draft(L, R) = a0 + aR R + L (o + c R).
        design, target = [], []
        for length, values in self.draft_by_len.items():
            for reqs, ms in zip(self.draft_reqs, values, strict=True):
                design.append([1.0, reqs, length, length * reqs])
                target.append(ms)
        self.pair_coef = np.linalg.lstsq(np.array(design), np.array(target), rcond=None)[0]

    def verify(self, tokens):
        """Target forward over ``tokens`` batch tokens (beyond the table: the
        slope of its last quarter)."""
        tokens = np.asarray(tokens, dtype=np.float64)
        top = self.verify_tokens[-1]
        base = np.interp(np.minimum(tokens, top), self.verify_tokens, self.verify_ms)
        return base + self._tail_slope * np.maximum(tokens - top, 0.0)

    def draft_batch(self, length: int, reqs: int) -> float:
        """Draft step at one length for all ``reqs`` requests (length 0 still
        runs the backbone and the first slot)."""
        length = min(max(int(length), 1), SLOTS)
        return float(np.interp(reqs, self.draft_reqs, self.draft_by_len[length]))

    def draft_pairs(self, reqs: int, max_len, pairs):
        a0, a_r, o, c = self.pair_coef
        return a0 + a_r * reqs + o * np.asarray(max_len) + c * np.asarray(pairs)

    def base(self, context: float) -> float:
        """Per-step cost independent of the lengths: KV read and overhead."""
        return self.kv_ms_per_token * context + self.overhead_ms


def topk_plan(surv: np.ndarray, cost: CostModel, base_ms: float, draft: str) -> np.ndarray:
    """Greedy admission by descending survival (paper Algorithm 1): the prefix
    with the most expected tokens per ms. Returns the length per request."""
    reqs = surv.shape[0]
    order = np.argsort(-surv, axis=None, kind="stable")
    flat = surv.reshape(-1)[order]
    slot = order % SLOTS
    gain = reqs + np.concatenate([[0.0], np.cumsum(flat)])
    count = np.arange(len(flat) + 1, dtype=np.float64)
    max_len = np.concatenate([[0], np.maximum.accumulate(slot + 1)])
    if draft == "all":
        draft_ms = cost.draft_batch(SLOTS, reqs)
    elif draft == "batch":
        draft_ms = np.array([cost.draft_batch(m, reqs) for m in range(SLOTS + 1)])[max_len]
    else:
        draft_ms = cost.draft_pairs(reqs, max_len, count)
    best = int(np.argmax(gain / (base_ms + cost.verify(reqs + count) + draft_ms)))
    lengths = np.zeros(reqs, dtype=int)
    np.add.at(lengths, order[:best] // SLOTS, 1)
    return lengths


def slot_counts(lengths: np.ndarray) -> np.ndarray:
    """Requests drafting each slot, as a fraction of the batch."""
    return np.array([(lengths > j).mean() for j in range(SLOTS)])


def budget_lengths(surv: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Slot by slot, keep the ``n_j`` rows with the highest survival among the
    rows that drafted slot j-1 (nested, as the Markov loop runs)."""
    reqs = surv.shape[0]
    counts = np.minimum.accumulate(np.round(fractions * reqs).astype(int))
    active = np.arange(reqs)
    lengths = np.zeros(reqs, dtype=int)
    for j in range(SLOTS):
        n = min(counts[j], len(active))
        if n <= 0:
            break
        keep = active[np.argsort(-surv[active, j], kind="stable")[:n]]
        lengths[keep] += 1
        active = keep
    return lengths


def simulate(
    trajectories: list[Trajectory],
    cost: CostModel,
    concurrency: int,
    *,
    rule: str,
    draft: str,
    signal: str = "conf",
    param: float | None = None,
    prompt_tokens: float = 563.0,
    label_prior: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> dict:
    """Replay ``trajectories`` in a closed loop of ``concurrency`` slots."""
    rng = np.random.default_rng(seed)
    queue = [int(i) for i in rng.permutation(len(trajectories))]

    def start() -> dict | None:
        if not queue:
            return None
        traj = trajectories[queue.pop()]
        prompt = traj.prompt if traj.prompt is not None else prompt_tokens
        return {"t": traj, "pos": 0, "made": 0, "prompt": prompt}

    slots = [start() for _ in range(concurrency)]
    prefill_ms = sum(cost.prefill_us * s["prompt"] / 1000.0 for s in slots if s)
    time_ms = 0.0
    tokens = steps = verified = 0
    history: deque[np.ndarray] = deque(maxlen=LAG)
    while any(slots):
        live = [s for s in slots if s]
        reqs = len(live)
        index = [s["pos"] % len(s["t"].accepted) for s in live]
        conf = np.array([s["t"].survival[i] for s, i in zip(live, index, strict=True)])
        acc = np.array([s["t"].accepted[i] for s, i in zip(live, index, strict=True)])
        if signal == "conf":
            surv = conf
        elif signal == "label":
            surv = np.array([label_prior[s["t"].label] for s in live])
        elif signal == "conf-label":
            ratio = np.array([label_prior[s["t"].label] for s in live])
            surv = np.minimum.accumulate(np.minimum(conf * ratio, 1.0), axis=1)
        elif signal == "oracle":
            surv = (np.arange(SLOTS)[None, :] < acc[:, None]).astype(np.float64)
        else:
            raise ValueError(signal)
        base = cost.base(sum(s["prompt"] + s["made"] for s in live))
        if rule == "fixed":
            lengths = np.full(reqs, int(param))
        elif rule == "threshold":
            lengths = (surv >= param).sum(axis=1)
        elif rule == "topk":
            lengths = topk_plan(surv, cost, base, draft)
        elif rule == "budget":
            planned = topk_plan(surv, cost, base, draft)
            fractions = history[0] if len(history) == LAG else slot_counts(planned)
            history.append(slot_counts(planned))
            lengths = budget_lengths(surv, fractions)
        else:
            raise ValueError(rule)
        if draft == "all":
            draft_ms = cost.draft_batch(SLOTS, reqs)
        elif draft == "batch":
            draft_ms = cost.draft_batch(int(lengths.max()), reqs)
        else:
            draft_ms = float(cost.draft_pairs(reqs, lengths.max(), lengths.sum()))
        gained = 1 + np.minimum(acc, lengths)
        time_ms += base + float(cost.verify(reqs + lengths.sum())) + draft_ms
        steps += 1
        tokens += int(gained.sum())
        verified += int(lengths.sum())
        for s, g in zip(live, gained, strict=True):
            s["made"] += int(g)
            s["pos"] += 1
        for i, s in enumerate(slots):
            if s and s["made"] >= s["t"].tokens:
                slots[i] = start()
                if slots[i]:
                    prefill_ms += cost.prefill_us * slots[i]["prompt"] / 1000.0
    return {
        "tokens_per_s": tokens / (time_ms + prefill_ms) * 1000.0,
        "tokens_per_step_per_req": tokens / max(steps, 1) / concurrency,
        "verified_per_step": verified / max(steps, 1),
        "step_ms": time_ms / max(steps, 1),
        "steps": steps,
    }


def fixed_rate(
    cost: CostModel,
    concurrency: int,
    prompt_tokens: float,
    out_tokens: float,
    k: int = 0,
    gain: float = 1.0,
) -> float:
    """Steady-state tok/s at a fixed length k (0: plain) with ``gain`` tokens
    per request and step."""
    step = cost.base(concurrency * (prompt_tokens + out_tokens / 2))
    step += float(cost.verify(concurrency * (1 + k)))
    if k:
        step += cost.draft_batch(k, concurrency)
    per_req_ms = out_tokens * step / (concurrency * gain)
    per_req_ms += cost.prefill_us * prompt_tokens / 1000.0
    return out_tokens / per_req_ms * 1000.0


def fit_costs(
    cost: CostModel,
    measured: dict[str, dict[str, float]],
    gains: dict[int, float],
    prompt_tokens: float,
    out_tokens: float,
) -> tuple[float, float, float]:
    """Grid-fit the per-step overhead (ms) and prefill (us per prompt token)
    to measured tok/s of plain ("plain") and fixed lengths ("k3", ...), by
    log error. Returns (rms log error, overhead, prefill) and sets them."""
    points = [
        (0 if arm == "plain" else int(arm[1:]), int(c), v)
        for arm, levels in measured.items()
        for c, v in levels.items()
    ]
    best = (np.inf, 0.0, 0.0)
    for overhead in np.linspace(0.0, 12.0, 61):
        for prefill in np.linspace(0.0, 80.0, 81):
            cost.overhead_ms, cost.prefill_us = overhead, prefill
            err = [
                np.log(fixed_rate(cost, c, prompt_tokens, out_tokens, k, gains.get(k, 1.0)))
                - np.log(v)
                for k, c, v in points
            ]
            rms = float(np.sqrt(np.mean(np.square(err))))
            if rms < best[0]:
                best = (rms, float(overhead), float(prefill))
    cost.overhead_ms, cost.prefill_us = best[1], best[2]
    return best


POLICIES: list[tuple[str, dict]] = [
    *[(f"fixed k={k}", {"rule": "fixed", "param": k, "draft": "batch"}) for k in range(1, 8)],
    ("paper top-K, draft 7", {"rule": "topk", "draft": "all"}),
    ("paper top-K + cut", {"rule": "topk", "draft": "batch"}),
    ("per-request conf, per step (bound)", {"rule": "topk", "draft": "pair"}),
    ("per-request conf, lagged budgets", {"rule": "budget", "draft": "pair"}),
    ("oracle, batch draft", {"rule": "topk", "draft": "batch", "signal": "oracle"}),
    ("oracle, pair draft", {"rule": "topk", "draft": "pair", "signal": "oracle"}),
]
LABEL_POLICIES: list[tuple[str, dict]] = [
    ("per-task length (label only)", {"rule": "topk", "draft": "pair", "signal": "label"}),
    ("per-request conf x label, lagged budgets",
     {"rule": "budget", "draft": "pair", "signal": "conf-label"}),
]  # fmt: skip


def label_priors(trajectories: list[Trajectory]) -> tuple[dict, dict]:
    """Per label: the observed prefix survival per slot, and its ratio to the
    head's mean prediction (a per-task recalibration)."""
    by: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by[t.label].append(t)
    prior, ratio = {}, {}
    for label, group in by.items():
        acc = np.concatenate([t.accepted for t in group])
        surv = np.concatenate([t.survival for t in group])
        observed = np.array([(acc > j).mean() for j in range(SLOTS)])
        prior[label] = observed
        ratio[label] = observed / np.maximum(surv.mean(axis=0), 1e-6)
    return prior, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("dumps", type=Path, nargs="+")
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--sts", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--prompt-tokens", type=float, default=563.0,
        help="prompt length when the dump has no context",
    )  # fmt: skip
    parser.add_argument(
        "--measured", type=Path, default=None,
        help='JSON {"plain": {"64": tok_s, ...}, "k3": {...}}: fit overhead and prefill',
    )  # fmt: skip
    parser.add_argument("--thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=None, help="keep only these labels")
    args = parser.parse_args()
    temperatures = json.loads(args.sts.read_text())["temperatures"]
    trajectories = load(args.dumps, temperatures)
    if args.labels:
        trajectories = [t for t in trajectories if t.label in args.labels]
    cost = CostModel(json.loads(args.cost_table.read_text()))
    out_tokens = float(np.mean([t.tokens for t in trajectories]))
    prompts = [t.prompt for t in trajectories if t.prompt is not None]
    prompt_tokens = float(np.mean(prompts)) if prompts else args.prompt_tokens
    accepted = np.concatenate([t.accepted for t in trajectories])
    gains = {k: 1.0 + float(np.minimum(accepted, k).mean()) for k in range(1, SLOTS + 1)}
    fitted = None
    if args.measured:
        measured = json.loads(args.measured.read_text())
        fitted = fit_costs(cost, measured, gains, prompt_tokens, out_tokens)
    prior, ratio = label_priors(trajectories)
    policies = list(POLICIES) + (list(LABEL_POLICIES) if len(prior) > 1 else [])
    policies += [
        (f"threshold {t}, pair draft", {"rule": "threshold", "param": t, "draft": "pair"})
        for t in args.thresholds
    ]
    results: dict = {
        "dumps": [str(p) for p in args.dumps],
        "requests": len(trajectories),
        "records": int(sum(len(t.accepted) for t in trajectories)),
        "mean_tokens": out_tokens,
        "mean_prompt_tokens": prompt_tokens,
        "labels": {k: int(sum(t.label == k for t in trajectories)) for k in prior},
        "label_prior": {k: v.tolist() for k, v in prior.items()},
        "fixed_gain": gains,
        "fit": None
        if fitted is None
        else {"rms_log_error": fitted[0], "overhead_ms": fitted[1], "prefill_us": fitted[2]},
        "pair_coef_ms": cost.pair_coef.tolist(),
        "concurrency": {},
    }
    for conc in args.concurrency:
        row: dict = {"plain": {"tokens_per_s": fixed_rate(cost, conc, prompt_tokens, out_tokens)}}
        for name, spec in policies:
            spec = dict(spec)
            signal = spec.get("signal", "conf")
            row[name] = simulate(
                trajectories, cost, conc, prompt_tokens=prompt_tokens,
                label_prior=ratio if signal == "conf-label" else prior, **spec,
            )  # fmt: skip
        results["concurrency"][str(conc)] = row
        plain = row["plain"]["tokens_per_s"]
        fixed = max(v["tokens_per_s"] for k, v in row.items() if k.startswith("fixed"))
        paper = row["paper top-K + cut"]["tokens_per_s"]
        print(f"C={conc}: plain {plain:.0f} tok/s, best fixed {fixed:.0f}")
        for name, value in row.items():
            if name == "plain":
                continue
            rate = value["tokens_per_s"]
            print(
                f"  {name:<44} {rate:>7.0f} {rate / plain:5.2f}x  {rate / fixed - 1:+6.1%} fixed"
                f"  {rate / paper - 1:+6.1%} paper+cut  verified/step"
                f" {value['verified_per_step']:5.0f}  step {value['step_ms']:5.1f} ms"
            )
    args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
