"""Scientific matching/scope regressions for the isolated read-only helper."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("review.py")
spec = importlib.util.spec_from_file_location("layout_review_v1", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def cell(arm, scope="all_observed_logical_engines", rate=100):
    return {
        "run": "/run",
        "stage": "pd-off",
        "arm": arm,
        "label": "opb",
        "concurrency": 1024,
        "review_eligible": True,
        "review_errors": [],
        "throughput_tok_s": rate,
        "summary": arm + "/summary.json",
        "wall_seconds": 1,
        "completion_tokens": 100,
        "generation": {"temperature": 0},
        "workload_digest": "corpus",
        "prompt_ids_sha256": "prompts",
        "cache_capacity": {"scope": scope, "observed_logical_kv_tokens": 1000},
        "counters_scope": scope,
        "load": {"preemptions": 2},
        "engine": {},
    }


def test_capacity_counts_logical_engines_not_tp_shards(tmp_path):
    p = tmp_path / "snapshot.prom"
    p.write_text(
        'vllm:cache_config_info{engine="0",kv_cache_size_tokens="100"} 1\n'
        'vllm:cache_config_info{engine="1",kv_cache_size_tokens="120"} 1\n'
    )
    configs = review.cache_configs(p)
    assert sum(r["kv_tokens"] for r in configs) == 220
    p.write_text('vllm:cache_config_info{engine="0",kv_cache_size_tokens="400"} 1\n')
    assert sum(r["kv_tokens"] for r in review.cache_configs(p)) == 400


def test_heterogeneous_pool_peak_is_bounded_not_assumed_exact():
    result = review.used_kv_bounds([{"kv_tokens": 100}, {"kv_tokens": 200}], 1.5)
    assert result["lower_bound"] == 200
    assert result["upper_bound"] == 250


def test_pd_has_within_layout_and_dp4_baselines_with_scope_guards():
    rows = [
        cell("plain-dp4"),
        cell("pd-1p3d-plain", "first_decode_instance"),
        cell("pd-1p3d-dspark-off-k3", "first_decode_instance", 150),
    ]
    results = review.comparisons(rows)
    treatment = [r for r in results if r["arm"] == "pd-1p3d-dspark-off-k3"]
    assert {r["baseline"] for r in treatment} == {"plain-dp4", "pd-1p3d-plain"}
    cross = next(r for r in treatment if r["baseline"] == "plain-dp4")
    assert cross["delta_preemptions"] is None
    assert cross["delta_observed_logical_kv_tokens"] is None
    same = next(r for r in treatment if r["baseline"] == "pd-1p3d-plain")
    assert same["speedup"] == 1.5
    assert same["implementation_recommendation"].startswith("unestablished")


def test_failed_and_mismatched_cells_cannot_be_candidates():
    a, b = cell("eagle-k7-tp4", rate=200), cell("plain-tp4")
    a["review_eligible"] = False
    assert not review.compare(a, b, "same_layout_speculation")["screening_pair_eligible"]
    a["review_eligible"] = True
    a["generation"] = {"temperature": 0.7}
    assert not review.compare(a, b, "same_layout_speculation")["screening_pair_eligible"]


def test_helper_never_overwrites_previous_review(tmp_path):
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps({"rows": [{"summary": "/dummy"}]}))
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--analysis", str(p), "--output", str(output)],
        capture_output=True,
    )
    assert result.returncode != 0
    assert marker.read_text() == "original"
