"""Safety gates for resuming the unfinished layout deployments."""

import json
from pathlib import Path

import pytest

from apertus_bench.layout_recovery import ALLOCATOR, cell_gate, collect_rows, recovery_specs
from apertus_bench.node_sweep import build_plan


def test_recovery_plans_preserve_original_scope_and_match_memory_recipe():
    specs = recovery_specs(Path("/fresh/nixl/site"))
    for name, spec in specs.items():
        plan = build_plan(spec, check_heads=False)
        assert plan.arms
        if name.startswith("tp4-") or name == "preflight":
            assert spec["engine"]["gpu_memory_utilization"] == 0.7
            for arm in plan.arms:
                if arm.kind != "command":
                    assert arm.env.items() >= ALLOCATOR.items()
                    assert arm.serve_argv[arm.serve_argv.index("--max-model-len") + 1] == "32768"
        for arm in plan.arms:
            if arm.kind == "pd":
                assert arm.pythonpath == ["/fresh/nixl/site"]
    assert len(specs["tp2dp2"]["arms"]) == 5
    assert len(specs["pd-off"]["arms"]) == 7
    assert [a["name"] for a in specs["tp4-dspark-on-k7"]["arms"]] == [
        "plain-tp4",
        "dspark-on-k7-tp4",
    ]


def spec():
    return {
        "arms": [{"name": "plain-tp4", "method": "baseline"}],
        "loadtests": [{"workload": "opb", "concurrencies": [128]}],
    }


def make_cell(tmp_path, *, status="ok", success=1.0, throughput=14000):
    (tmp_path / "sweep-status.json").write_text(json.dumps({"plain-tp4": status}))
    p = tmp_path / "plain-tp4/opb/opb/c128/summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "requests": {
                    "attempted": 512,
                    "successful": 512,
                    "failed": 0,
                    "success_rate": success,
                },
                "tokens": {"output_tokens_per_second": throughput, "completion": 1000},
                "wall_seconds": 1,
            }
        )
    )
    return p


@pytest.mark.parametrize(
    "status,success,throughput", [("failed", 1.0, 14000), ("ok", 0.98, 14000), ("ok", 1.0, 2000)]
)
def test_gate_rejects_failed_engine_request_errors_and_slow_plain(
    tmp_path, status, success, throughput
):
    make_cell(tmp_path, status=status, success=success, throughput=throughput)
    assert not cell_gate(tmp_path, spec(), "tp4-dspark-off-k7")["passed"]


def test_gate_requires_every_cell_and_native_import(tmp_path):
    assert not cell_gate(tmp_path, spec())["passed"]
    make_cell(tmp_path)
    assert cell_gate(tmp_path, spec(), "tp4-dspark-off-k7")["passed"]
    s = spec()
    s["arms"].append({"name": "nixl-native-import", "method": "command"})
    assert not cell_gate(tmp_path, s)["passed"]


def test_analysis_matches_tp4_to_tp4_when_dp4_is_also_present(tmp_path):
    p = make_cell(tmp_path)
    data = json.loads(p.read_text())
    for arm, rate in [("plain-dp4", 30000), ("dspark-on-k7-tp4", 28000)]:
        q = tmp_path / arm / "opb/opb/c128/summary.json"
        q.parent.mkdir(parents=True)
        data["tokens"]["output_tokens_per_second"] = rate
        q.write_text(json.dumps(data))
    rows = collect_rows(tmp_path, "old-thinking")
    row = next(r for r in rows if r["arm"] == "dspark-on-k7-tp4")
    assert row["speedup_vs_matched_plain"] == 2.0
