import gzip
import json
import math
from pathlib import Path

import pytest

from apertus_bench import tree_sim as ts

# Anchor trie used below (tokens in brackets, truth = 1 2 3):
#   root -> 0[1] -> 1[2] -> 2[9]
#                 \-> 3[3]          (child of 1)
#        -> 4[7] -> 5[2]
PARENTS = [-1, 0, 1, 1, -1, 4]
TOKENS = [1, 2, 9, 3, 7, 2]
TRUTH = [1, 2, 3, 4]


def record(**overrides):
    rec = {
        "seq": "s",
        "wl": "w",
        "a": 0,
        "ctx": 100,
        "truth": TRUTH,
        "p": PARENTS,
        "t": TOKENS,
        # 0-1-2 is DSpark's chain, 0-1-3 EAGLE's; 4-5 an n-gram branch.
        "f": [
            ts.D_TREE | ts.D_CHAIN | ts.E_TREE | ts.E_CHAIN,
            ts.D_TREE | ts.D_CHAIN | ts.E_TREE | ts.E_CHAIN,
            ts.D_TREE | ts.D_CHAIN,
            ts.E_TREE | ts.E_CHAIN | ts.D_TREE,
            ts.N_TREE | ts.N_CHAIN,
            ts.N_TREE | ts.N_CHAIN,
        ],
        "le": [-0.1, -0.2, -2.0, -0.3, -3.0, -0.5],
        "ld": [-0.1, -0.2, -0.4, -0.9, -3.0, -0.5],
        "ng": [-1, -1, -1, -1, 0, 0],
        "nm": 2,
    }
    rec.update(overrides)
    return rec


def test_depths_and_cumulative():
    assert ts.depths(PARENTS) == [1, 2, 3, 3, 1, 2]
    assert ts.cumulative(PARENTS, [-1, -1, -1, -1, -2, -1]) == [-1, -2, -3, -3, -2, -3]


def test_accepted_length_is_longest_matching_selected_path():
    assert ts.accepted_length(PARENTS, TOKENS, range(6), TRUTH) == 3
    assert ts.accepted_length(PARENTS, TOKENS, [0, 1, 2], TRUTH) == 2
    # A matching node whose parent is not selected does not count.
    assert ts.accepted_length(PARENTS, TOKENS, [0, 3], TRUTH) == 1
    assert ts.accepted_length(PARENTS, TOKENS, [], TRUTH) == 0
    assert ts.accepted_length(PARENTS, TOKENS, [4, 5], TRUTH) == 0
    # Truth shorter than the tree (end of the sequence).
    assert ts.accepted_length(PARENTS, TOKENS, range(6), [1, 2]) == 2


def test_select_top_is_best_first_and_ancestor_closed():
    scores = [-0.1, -0.3, -2.3, -0.6, -3.0, -3.5]
    assert ts.select_top(PARENTS, scores, 2) == [0, 1]
    assert ts.select_top(PARENTS, scores, 3) == [0, 1, 3]
    # Eligible node 5 drags in its ancestor 4 (two slots).
    assert ts.select_top(PARENTS, scores, 2, eligible=[5]) == [4, 5]
    assert ts.select_top(PARENTS, scores, 1, eligible=[5]) == []
    assert ts.select_top(PARENTS, scores, 0) == []


def test_mixture_is_log_of_average_probability():
    assert ts.mixture(math.log(0.2), math.log(0.6)) == pytest.approx(math.log(0.4))
    assert ts.mixture(-1e9, 0.0) == pytest.approx(math.log(0.5))


def test_chain_and_tree_policies():
    rec = record()
    assert ts.chain_dspark(rec, 3).selected == [0, 1, 2]
    assert ts.chain_eagle(rec, 3).selected == [0, 1, 3]
    assert ts.chain_ngram(rec, 7).selected == [4, 5]
    step = ts.tree_dspark(rec, 4)
    assert step.selected == [0, 1, 2, 3]
    assert step.dspark and step.dspark_len == 3 and step.dspark_expansions == 3
    step = ts.tree_eagle(rec, 3)
    assert step.selected == [0, 1, 3]
    assert step.eagle_passes == 3 and step.eagle_tokens == 3  # anchor + nodes 0, 1
    chain = ts.chain_eagle(rec, 5)
    assert chain.eagle_passes == 5 and chain.eagle_tokens == 5
    assert ts.tree_ngram(rec, 1).selected == [4]


def test_graft_mixture_prefers_nodes_both_drafters_like():
    rec = record()
    # Mixture of node 3: log(0.5 e^-0.3 + 0.5 e^-0.9) > node 2's.
    step = ts.graft_dspark_eagle(rec, 3)
    assert step.selected == [0, 1, 3]
    assert step.dspark and step.eagle_called


def test_scheduler_calls_eagle_only_when_dspark_is_unsure():
    sure = record()
    assert not ts.scheduled(0.4)(sure, 3).eagle_called  # chain p = e^-0.7 = 0.50
    unsure = record(ld=[-2.0, -0.2, -0.4, -0.9, -3.0, -0.5])
    assert ts.scheduled(0.4)(unsure, 3).eagle_called
    assert not ts.ngram_first(2)(sure, 3).dspark
    assert ts.ngram_first(3)(sure, 3).dspark


def test_walk_jumps_by_accepted_plus_one():
    # Truth: the sequence 1..; anchor a proposes a+1, a+2 (right) for even a and
    # a wrong token for odd a.
    anchors = {}
    for a in range(10):
        good = a % 2 == 0
        anchors[a] = record(
            a=a,
            truth=[a + 1, a + 2, a + 3],
            p=[-1, 0],
            t=[a + 1, a + 2] if good else [0, 0],
            f=[ts.D_CHAIN | ts.D_TREE] * 2,
            le=[0.0, 0.0],
            ld=[0.0, 0.0],
            ng=[-1, -1],
        )
    totals = ts.walk(anchors, ts.chain_dspark, 2)
    # a=0 accepts 2 -> a=3 accepts 0 -> a=4 accepts 2 -> a=7 -> a=8 -> a=11 stop.
    assert totals.steps == 5
    assert totals.tokens == 3 + 1 + 3 + 1 + 3
    assert dict(totals.accepted_hist) == {2: 3, 0: 2}
    assert totals.summary()["dspark_call_rate"] == 1.0


def cost_table(tmp_path: Path) -> Path:
    table = {
        "verify_tokens": [1, 64, 512, 1024],
        "verify_ms": [5.0, 6.0, 20.0, 34.0],
        "draft_reqs": [1, 64, 256],
        "draft_ms_by_len": {str(k): [1.0 + 0.1 * k, 2.0 + k, 4.0 + 2 * k] for k in range(1, 8)},
        "kv_ms_per_token": 1e-5,
    }
    path = tmp_path / "cost-table.json"
    path.write_text(json.dumps(table))
    return path


def test_cost_model_interpolates_and_projects(tmp_path):
    cost = ts.CostModel.from_table(cost_table(tmp_path))
    assert cost.verify(1) == 5.0
    assert cost.verify(288) == pytest.approx(13.0)
    assert cost.verify(2048) == pytest.approx(62.0)  # extrapolated
    assert cost.dspark(64, 3) == pytest.approx(5.0)
    assert cost.dspark(64, 3.5) == pytest.approx(5.5)
    plain = ts.plain_summary(1000)
    got = cost.throughput(1, plain)
    assert got["step_ms"]["total"] == pytest.approx(5.0 + 0.01)
    spec = {
        **plain,
        "accepted_per_step": 3.0,
        "verified_per_step": 3.0,
        "dspark_call_rate": 1.0,
        "dspark_len_mean": 3.0,
        "dspark_expansions_per_step": 0.0,
    }
    step = cost.step_ms(64, spec)
    assert step["verify"] == pytest.approx(cost.verify(256))
    assert step["dspark"] == pytest.approx(5.0)
    # Half the requests call EAGLE: the fixed pass latency is still paid.
    eagle = {
        **plain,
        "eagle_call_rate": 0.5,
        "eagle_passes_per_call": 3,
        "eagle_tokens_per_call": 4,
    }
    step = cost.step_ms(64, eagle)
    assert step["eagle"] == pytest.approx(3 * 0.55 + 0.0035 * 32 * 4)


def test_analyze_end_to_end(tmp_path):
    path = tmp_path / "anchors.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for a in range(4):
            handle.write(json.dumps(record(a=a)) + "\n")
    sequences = ts.load_anchors([path])
    report = ts.analyze(
        sequences,
        ["chain-dspark", "graft-all"],
        [2, 8],
        ts.CostModel.from_table(cost_table(tmp_path)),
        [1, 64],
    )
    rows = {(r["workload"], r["policy"], r["budget"]): r for r in report["rows"]}
    assert ("w", "chain-dspark", 8) not in rows  # chains stop at depth 7
    chain = rows[("w", "chain-dspark", 2)]
    assert chain["accepted_per_step"] == 3.0  # accepts 2 at a=0, then a=3
    assert rows[("all", "graft-all", 8)]["accepted_per_step"] == 4.0
    assert chain["projected"]["1"]["speedup_vs_plain"] > 1.0
