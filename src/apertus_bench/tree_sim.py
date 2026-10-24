"""Offline simulator for tree-based and multi-drafter speculative decoding.

Greedy decoding only. At every decode step (an *anchor*: the last token the
target has emitted) each drafter proposes candidate continuations; the
candidates of all drafters are merged into one token trie per anchor (the
records written by ``methods/tree/apertus_tree/candidates.py``). A *policy*
picks at most ``budget`` trie nodes to verify (always ancestor-closed, so the
selection is itself a tree rooted at the anchor). Under greedy verification the
target accepts the longest selected root path that equals its own greedy
continuation and then emits one token of its own, so a step yields
``accepted + 1`` tokens.

Anchor record (one JSON object per anchor)::

    {"seq": str, "wl": str, "a": int, "ctx": int, "truth": [int, ...],
     "p": [parent...], "t": [token...], "f": [flags...],
     "le": [EAGLE edge logprob...], "ld": [DSpark edge logprob...],
     "ng": [n-gram branch rank or -1...], "nm": int}

``p[i]`` is the parent node index (-1 for the anchor itself) and always
smaller than ``i``. ``le``/``ld`` are the log-probabilities of the edge
``parent -> t[i]`` under each drafter (both drafters score every node).
``nm`` is the longest n-gram match length at the anchor (0: none).

``walk`` replays a sequence step by step (the next anchor is ``a + accepted +
1``), so every policy is evaluated on the anchors it would really visit.
``CostModel`` turns the per-step verify and draft work into projected decode
throughput from the measured GH200 cost table.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

E_TREE, D_TREE, N_TREE, E_CHAIN, D_CHAIN, N_CHAIN = 1, 2, 4, 8, 16, 32
MAX_DEPTH = 7


# --------------------------------------------------------------------------- trie helpers


def depths(parents: Sequence[int]) -> list[int]:
    out: list[int] = []
    for parent in parents:
        out.append(1 if parent < 0 else out[parent] + 1)
    return out


def cumulative(parents: Sequence[int], edge: Sequence[float]) -> list[float]:
    """Path log-probability of every node (sum of its edge log-probabilities)."""
    out: list[float] = []
    for i, parent in enumerate(parents):
        out.append(edge[i] + (0.0 if parent < 0 else out[parent]))
    return out


def mixture(a: float, b: float) -> float:
    """log(0.5 e^a + 0.5 e^b): the two drafters as an equal-weight mixture."""
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi)) - math.log(2.0)


def select_top(
    parents: Sequence[int],
    scores: Sequence[float],
    budget: int,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    """Best-first, ancestor-closed selection of at most ``budget`` nodes.

    Nodes are taken in descending score; a node enters together with any of its
    ancestors not yet selected, if they all fit (TreeGraft's pre-verification
    pruning and EAGLE-2's reranking do the same). Ancestors of an eligible node
    are used even when not eligible themselves.
    """
    if budget <= 0:
        return []
    pool = range(len(parents)) if eligible is None else eligible
    order = sorted(pool, key=lambda i: (-scores[i], i))
    chosen: set[int] = set()
    for node in order:
        if node in chosen:
            continue
        chain = []
        cur = node
        while cur >= 0 and cur not in chosen:
            chain.append(cur)
            cur = parents[cur]
        if len(chosen) + len(chain) > budget:
            continue
        chosen.update(chain)
        if len(chosen) >= budget:
            break
    return sorted(chosen)


def accepted_length(
    parents: Sequence[int], tokens: Sequence[int], selected: Iterable[int], truth: Sequence[int]
) -> int:
    """Longest selected root path equal to the target's greedy continuation."""
    chosen = set(selected)
    match: dict[int, int] = {}
    best = 0
    for node in sorted(chosen):  # parents precede children
        parent = parents[node]
        if parent < 0:
            depth = 1
        elif parent in match:
            depth = match[parent] + 1
        else:
            continue  # parent unselected or not on the target's path
        if depth > len(truth) or tokens[node] != truth[depth - 1]:
            continue
        match[node] = depth
        best = max(best, depth)
    return best


def expanded_parents(parents: Sequence[int], selected: Iterable[int]) -> int:
    """Distinct non-root parents of the selected nodes: the drafter forwards an
    autoregressive drafter needs (the anchor's own forward is the first pass)."""
    return len({parents[i] for i in selected if parents[i] >= 0})


def max_depth(parents: Sequence[int], selected: Iterable[int]) -> int:
    d = depths(parents)
    return max((d[i] for i in selected), default=0)


# --------------------------------------------------------------------------- policies


@dataclass
class Step:
    """What one policy does at one anchor."""

    selected: list[int]
    dspark: bool = False  # DSpark block forward
    dspark_len: int = 0  # block slots used (draft-cut patch drafts only these)
    dspark_expansions: int = 0  # tree nodes whose children DSpark must rank
    eagle_passes: int = 0  # sequential EAGLE forwards
    eagle_tokens: int = 0  # EAGLE node forwards in total (over all passes)
    eagle_called: bool = False


Policy = Callable[[dict[str, Any], int], Step]


def _eligible(rec: dict[str, Any], mask: int) -> list[int]:
    return [i for i, f in enumerate(rec["f"]) if f & mask]


def _chain(rec: dict[str, Any], bit: int, k: int) -> list[int]:
    d = depths(rec["p"])
    return [i for i, f in enumerate(rec["f"]) if f & bit and d[i] <= k]


def _dspark_step(rec: dict[str, Any], selected: list[int], *, tree: bool) -> Step:
    depth = max_depth(rec["p"], selected)
    return Step(
        selected,
        dspark=True,
        dspark_len=max(depth, 1),
        dspark_expansions=(expanded_parents(rec["p"], selected) + 1) if tree else 0,
    )


def _eagle_cost(rec: dict[str, Any], step: Step) -> Step:
    depth = max_depth(rec["p"], step.selected)
    step.eagle_passes = max(depth, 1)
    # The first pass runs on the anchor (and the tokens accepted with it); every
    # deeper pass forwards the selected nodes that have selected children.
    step.eagle_tokens = 1 + expanded_parents(rec["p"], step.selected)
    step.eagle_called = True
    return step


def chain_dspark(rec, k):
    return _dspark_step(rec, _chain(rec, D_CHAIN, k), tree=False)


def chain_eagle(rec, k):
    sel = _chain(rec, E_CHAIN, k)
    step = _eagle_cost(rec, Step(sel))
    step.eagle_passes = k  # a chain drafts all k tokens before verification
    step.eagle_tokens = k
    return step


def chain_ngram(rec, k):
    return Step(_chain(rec, N_CHAIN, k))


def tree_dspark(rec, budget):
    scores = cumulative(rec["p"], rec["ld"])
    sel = select_top(rec["p"], scores, budget, _eligible(rec, D_TREE | D_CHAIN))
    return _dspark_step(rec, sel, tree=True)


def tree_eagle(rec, budget):
    scores = cumulative(rec["p"], rec["le"])
    sel = select_top(rec["p"], scores, budget, _eligible(rec, E_TREE | E_CHAIN))
    return _eagle_cost(rec, Step(sel))


def _ngram_order(rec: dict[str, Any]) -> list[float]:
    d = depths(rec["p"])
    return [(-1e9 if r < 0 else -(r * 16 + d[i])) for i, r in enumerate(rec["ng"])]


def tree_ngram(rec, budget):
    sel = select_top(rec["p"], _ngram_order(rec), budget, _eligible(rec, N_TREE | N_CHAIN))
    return Step(sel)


def graft_dspark_ngram(rec, budget):
    """n-gram branches grafted onto the DSpark tree, all ranked by DSpark."""
    scores = cumulative(rec["p"], rec["ld"])
    sel = select_top(rec["p"], scores, budget, _eligible(rec, D_TREE | D_CHAIN | N_TREE | N_CHAIN))
    return _dspark_step(rec, sel, tree=True)


def _mixture_scores(rec: dict[str, Any]) -> list[float]:
    return cumulative(rec["p"], [mixture(a, b) for a, b in zip(rec["le"], rec["ld"], strict=True)])


def graft_dspark_eagle(rec, budget, mask=E_TREE | E_CHAIN | D_TREE | D_CHAIN):
    """Union of both neural trees, every node scored by both drafters (mixture)."""
    sel = select_top(rec["p"], _mixture_scores(rec), budget, _eligible(rec, mask))
    step = _dspark_step(rec, sel, tree=True)
    return _eagle_cost(rec, step)


def graft_all(rec, budget):
    return graft_dspark_eagle(rec, budget, mask=63)


def scheduled(threshold: float, depth: int = 3) -> Policy:
    """TreeGraft-style rule: graft n-gram branches onto the DSpark tree, and call
    the stronger, more expensive drafter (EAGLE: rescoring and its own branches)
    only when DSpark's greedy path is unsure: the probability of its first
    ``depth`` drafts falls below ``threshold``."""

    log_threshold = math.log(threshold) if threshold > 0 else -math.inf

    def policy(rec, budget):
        chain = _chain(rec, D_CHAIN, min(depth, budget))
        confidence = sum(rec["ld"][i] for i in chain)
        if confidence >= log_threshold:
            return graft_dspark_ngram(rec, budget)
        return graft_all(rec, budget)

    return policy


def ngram_first(min_match: int) -> Policy:
    """TreeGraft's own pairing: the free n-gram drafter builds the tree and the
    neural drafter (DSpark) is called only when no n-gram of length
    ``min_match`` matches."""

    def policy(rec, budget):
        if rec.get("nm", 0) >= min_match:
            return tree_ngram(rec, budget)
        return graft_dspark_ngram(rec, budget)

    return policy


def union_all(rec, budget):
    """Diagnostic: every candidate of every drafter, ignoring the budget."""
    return Step(list(range(len(rec["p"]))))


POLICIES: dict[str, Policy] = {
    "chain-dspark": chain_dspark,
    "chain-eagle": chain_eagle,
    "chain-ngram": chain_ngram,
    "tree-dspark": tree_dspark,
    "tree-eagle": tree_eagle,
    "tree-ngram": tree_ngram,
    "graft-dspark+ngram": graft_dspark_ngram,
    "graft-dspark+eagle": graft_dspark_eagle,
    "graft-all": graft_all,
    "sched-0.5": scheduled(0.5),
    "sched-0.8": scheduled(0.8),
    "ngram-first-3": ngram_first(3),
}


# --------------------------------------------------------------------------- replay


@dataclass
class Totals:
    steps: int = 0
    tokens: int = 0
    verified: int = 0  # draft nodes verified (excluding the anchor's own token)
    ctx: int = 0
    dspark_steps: int = 0
    dspark_len: int = 0
    dspark_expansions: int = 0
    eagle_steps: int = 0
    eagle_passes: int = 0
    eagle_tokens: int = 0
    accepted_hist: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, other: Totals) -> None:
        for name in (
            "steps",
            "tokens",
            "verified",
            "ctx",
            "dspark_steps",
            "dspark_len",
            "dspark_expansions",
            "eagle_steps",
            "eagle_passes",
            "eagle_tokens",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key, value in other.accepted_hist.items():
            self.accepted_hist[key] += value

    def summary(self) -> dict[str, Any]:
        n = max(self.steps, 1)
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "accepted_per_step": self.tokens / n,
            "verified_per_step": self.verified / n,
            "ctx_mean": self.ctx / n,
            "dspark_call_rate": self.dspark_steps / n,
            "dspark_len_mean": self.dspark_len / max(self.dspark_steps, 1),
            "dspark_expansions_per_step": self.dspark_expansions / n,
            "eagle_call_rate": self.eagle_steps / n,
            "eagle_passes_per_call": self.eagle_passes / max(self.eagle_steps, 1),
            "eagle_tokens_per_call": self.eagle_tokens / max(self.eagle_steps, 1),
        }


def walk(anchors: dict[int, dict[str, Any]], policy: Policy, budget: int) -> Totals:
    """Replay one sequence: start at its first anchor, jump by accepted + 1."""
    totals = Totals()
    if not anchors:
        return totals
    a = min(anchors)
    last = max(anchors)
    while a <= last:
        rec = anchors.get(a)
        if rec is None:
            break
        step = policy(rec, budget)
        acc = accepted_length(rec["p"], rec["t"], step.selected, rec["truth"])
        totals.steps += 1
        totals.tokens += acc + 1
        totals.verified += len(step.selected)
        totals.ctx += rec["ctx"]
        totals.accepted_hist[acc] += 1
        if step.dspark:
            totals.dspark_steps += 1
            totals.dspark_len += step.dspark_len
            totals.dspark_expansions += step.dspark_expansions
        if step.eagle_called:
            totals.eagle_steps += 1
            totals.eagle_passes += step.eagle_passes
            totals.eagle_tokens += step.eagle_tokens
        a += acc + 1
    return totals


# --------------------------------------------------------------------------- cost model


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation, linear extrapolation past the last point."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class CostModel:
    """Decode-step GPU time on one GH200 from the kv-cost sweep's cost table.

    - verify: ``verify_ms`` over the batch's verified tokens (measured);
    - attention reads: ``kv_ms_per_token`` per request and context token;
    - DSpark: ``draft_ms_by_len[len]`` over the batch's requests (measured);
      tree expansions rank a 131k-row logit vector each (HBM-bound estimate);
    - EAGLE: ``eagle_pass_fixed_ms + eagle_ms_per_token * tokens`` per pass.
      Fitted by the waste-profiling branch from Nsight windows of EAGLE e10
      (origin/waste-profiling b27fd2fe, results/8b/waste/drafter-costs.json;
      within 12% of 34 windows): later passes 0.529 ms + 4.10 us per request
      (the first, 0.509 ms + 3.61 us/request + 0.82 us/target token, is folded
      in). The fit saw one token per request per pass; tree passes charge the
      per-request slope per forwarded node.
    """

    verify_tokens: list[float]
    verify_ms: list[float]
    draft_reqs: list[float]
    draft_ms_by_len: dict[int, list[float]]
    kv_ms_per_token: float
    eagle_pass_fixed_ms: float = 0.529
    eagle_ms_per_token: float = 0.0041
    expansion_ms: float = 131072 * 2 * 4 / 3.5e12 * 1e3
    source: str = ""

    @classmethod
    def from_table(cls, path: Path, **overrides: float) -> CostModel:
        table = json.loads(Path(path).read_text())
        return cls(
            verify_tokens=[float(x) for x in table["verify_tokens"]],
            verify_ms=[float(x) for x in table["verify_ms"]],
            draft_reqs=[float(x) for x in table["draft_reqs"]],
            draft_ms_by_len={
                int(k): [float(x) for x in v] for k, v in table["draft_ms_by_len"].items()
            },
            kv_ms_per_token=float(table["kv_ms_per_token"]),
            source=str(path),
            **overrides,
        )

    def verify(self, tokens: float) -> float:
        return _interp(self.verify_tokens, self.verify_ms, tokens)

    def dspark(self, requests: float, length: float) -> float:
        length = min(max(length, 1.0), float(max(self.draft_ms_by_len)))
        lo, hi = math.floor(length), math.ceil(length)
        at = lambda k: _interp(self.draft_reqs, self.draft_ms_by_len[k], requests)  # noqa: E731
        if lo == hi:
            return at(lo)
        return at(lo) + (at(hi) - at(lo)) * (length - lo)

    def eagle_pass(self, tokens: float) -> float:
        return self.eagle_pass_fixed_ms + self.eagle_ms_per_token * tokens

    def step_ms(self, concurrency: int, s: dict[str, Any]) -> dict[str, float]:
        """GPU time of one decode step for ``concurrency`` requests that all
        behave like the averages in ``s`` (a ``Totals.summary``).

        A drafter's fixed per-pass latency is paid whenever at least one request
        of the batch calls it, i.e. with probability ~min(1, C x call rate).
        """
        c = float(concurrency)
        out = {
            "verify": self.verify(c * (1.0 + s["verified_per_step"])),
            "kv": self.kv_ms_per_token * c * s["ctx_mean"],
        }
        rate = s["dspark_call_rate"]
        if rate > 0:
            share = min(1.0, c * rate)
            requests = max(c * rate, 1.0)
            out["dspark"] = share * self.dspark(requests, s["dspark_len_mean"])
            out["dspark_tree"] = self.expansion_ms * c * s["dspark_expansions_per_step"]
        rate = s["eagle_call_rate"]
        if rate > 0:
            share = min(1.0, c * rate)
            passes = s["eagle_passes_per_call"]
            tokens = c * rate * s["eagle_tokens_per_call"]
            out["eagle"] = (
                share * passes * self.eagle_pass_fixed_ms + self.eagle_ms_per_token * tokens
            )
        out["total"] = sum(out.values())
        return out

    def throughput(self, concurrency: int, s: dict[str, Any]) -> dict[str, Any]:
        step = self.step_ms(concurrency, s)
        tok_s = concurrency * s["accepted_per_step"] / step["total"] * 1e3
        return {"tok_s": tok_s, "step_ms": step}


def plain_summary(ctx_mean: float) -> dict[str, Any]:
    return {
        "accepted_per_step": 1.0,
        "verified_per_step": 0.0,
        "ctx_mean": ctx_mean,
        "dspark_call_rate": 0.0,
        "eagle_call_rate": 0.0,
    }


# --------------------------------------------------------------------------- analysis


def load_anchors(paths: Iterable[Path]) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    """(workload, sequence id) -> anchor -> record."""
    out: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    out[(rec["wl"], rec["seq"])][rec["a"]] = rec
    return out


def _evaluate(args: tuple[dict[int, dict[str, Any]], str, int]) -> Totals:
    anchors, name, budget = args
    return walk(anchors, POLICIES[name], budget)


def analyze(
    sequences: dict[tuple[str, str], dict[int, dict[str, Any]]],
    policies: Sequence[str],
    budgets: Sequence[int],
    cost: CostModel,
    concurrencies: Sequence[int],
    processes: int = 1,
) -> dict[str, Any]:
    jobs = []
    keys = []
    for (wl, _seq), anchors in sorted(sequences.items()):
        for name in policies:
            for budget in budgets:
                if name.startswith("chain") and budget > MAX_DEPTH:
                    continue
                jobs.append((anchors, name, budget))
                keys.append((wl, name, budget))
    if processes > 1:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(processes) as pool:
            results = pool.map(_evaluate, jobs, chunksize=4)
    else:
        results = [_evaluate(job) for job in jobs]
    grouped: dict[tuple[str, str, int], Totals] = defaultdict(Totals)
    for key, totals in zip(keys, results, strict=True):
        grouped[key].add(totals)
        grouped[("all", key[1], key[2])].add(totals)
    rows = []
    workloads = sorted({k[0] for k in grouped})
    plain = {}
    for wl in workloads:
        ctx = [t.summary()["ctx_mean"] for k, t in grouped.items() if k[0] == wl]
        plain[wl] = plain_summary(sum(ctx) / len(ctx))
    for (wl, name, budget), totals in sorted(grouped.items()):
        s = totals.summary()
        projected = {}
        for c in concurrencies:
            base = cost.throughput(c, plain[wl])
            got = cost.throughput(c, s)
            projected[str(c)] = {
                "tok_s": round(got["tok_s"], 1),
                "speedup_vs_plain": round(got["tok_s"] / base["tok_s"], 4),
                "step_ms": {k: round(v, 3) for k, v in got["step_ms"].items()},
            }
        rows.append(
            {
                "workload": wl,
                "policy": name,
                "budget": budget,
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                "accepted_hist": dict(sorted(totals.accepted_hist.items())),
                "projected": projected,
            }
        )
    return {
        "cost_model": {
            "source": cost.source,
            "eagle_pass_fixed_ms": cost.eagle_pass_fixed_ms,
            "eagle_ms_per_token": cost.eagle_ms_per_token,
            "expansion_ms": cost.expansion_ms,
            "kv_ms_per_token": cost.kv_ms_per_token,
        },
        "sequences": {wl: sum(1 for k in sequences if k[0] == wl) for wl in workloads},
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m apertus_bench.tree_sim")
    parser.add_argument("candidates", type=Path, nargs="+", help="anchor JSONL(.gz) shards")
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--budgets", default="1,2,3,4,5,6,7,8,10,12,16,24,32,48")
    parser.add_argument("--concurrency", default="1,8,64,128,256")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sequences = load_anchors(args.candidates)
    report = analyze(
        sequences,
        [p for p in args.policies.split(",") if p],
        [int(b) for b in args.budgets.split(",")],
        CostModel.from_table(args.cost_table),
        [int(c) for c in args.concurrency.split(",")],
        processes=args.processes,
    )
    report["inputs"] = [str(p) for p in args.candidates]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"rows": len(report["rows"]), "sequences": report["sequences"]}))


if __name__ == "__main__":
    main()
