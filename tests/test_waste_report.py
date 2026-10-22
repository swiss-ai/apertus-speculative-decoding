import json
from pathlib import Path

import pytest

from apertus_bench import waste_report as wr


def _table(tmp_path: Path) -> Path:
    tokens = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # 5 ms fixed up to 32 tokens, then 0.03 ms per token (compute-bound).
    verify = [5.0 if t <= 32 else 5.0 + 0.03 * (t - 32) for t in tokens]
    reqs = [1, 8, 32, 128, 256]
    table = {
        "verify_tokens": tokens,
        "verify_ms": verify,
        "draft_reqs": reqs,
        "draft_ms_by_len": {
            str(k): [1.0 + 0.1 * k + 0.004 * r * k for r in reqs] for k in range(1, 8)
        },
        "kv_ms_per_token": 3.5e-5,
    }
    path = tmp_path / "cost-table.json"
    path.write_text(json.dumps(table))
    return path


def test_cost_model_interpolates_and_extrapolates(tmp_path: Path) -> None:
    model = wr.CostModel.from_cost_table(_table(tmp_path))
    assert model.verify_ms(16) == pytest.approx(5.0)
    assert model.verify_ms(1024 + 32) == pytest.approx(5.0 + 0.03 * 1024)
    assert model.verify_ms(4096) == pytest.approx(5.0 + 0.03 * (4096 - 32))
    assert model.verify_ms(16, context_tokens=100_000) == pytest.approx(5.0 + 3.5)
    assert model.marginal_verify_ms(1024) == pytest.approx(0.03)
    assert model.marginal_verify_ms(8) < model.marginal_verify_ms(1024)
    assert model.dspark_draft_ms(128, 7) == pytest.approx(1.0 + 0.7 + 0.004 * 128 * 7)
    with pytest.raises(KeyError):
        model.dspark_draft_ms(8, 9)
    assert 32 <= wr.verify_saturation_tokens(model) <= 128


def test_marginal_rule_drops_unlikely_slots_under_load(tmp_path: Path) -> None:
    model = wr.CostModel.from_cost_table(_table(tmp_path))
    accept = [0.85, 0.7, 0.6, 0.5, 0.43, 0.37, 0.33]
    light = wr.marginal_rule(model, 1, 7, accept)
    heavy = wr.marginal_rule(model, 256, 7, accept)
    assert all(row["pays"] for row in light[1:])
    assert not heavy[-1]["pays"]
    # gain = p_accept x step time / tokens per step
    step = model.step_ms(256, 7)
    assert heavy[0]["gain_ms"] == pytest.approx(0.85 * step / (1 + sum(accept)))


def test_step_waste_splits_verify_and_draft(tmp_path: Path) -> None:
    model = wr.CostModel.from_cost_table(_table(tmp_path))
    accept = [0.5] * 4
    waste = wr.step_waste(model, 128, 4, accept)
    assert waste["rejected_verify_ms"] == pytest.approx(128 * 2 * 0.03)
    assert waste["draft_wasted_ms"] == pytest.approx(waste["draft_ms"] / 2)
    assert waste["useful_verify_ms"] + waste["rejected_verify_ms"] == pytest.approx(
        waste["verify_ms"]
    )
    oracle = wr.oracle_verify(model, 128, 4, accept)
    assert oracle["speedup"] > 1
    rows = wr.predicted_speedup(model, 1, accept)
    assert [row["k"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["speedup"] > 1


def test_dspark_component_fit_and_depth_extrapolation() -> None:
    truth = wr.DSparkComponents(
        layers=5, fc_ms_per_token=1e-4, context_kv_ms_per_token_layer=2e-5,
        backbone_ms_per_token_layer=1.4e-3, lm_head_ms_per_row=1.8e-3,
        markov_ms_per_request_slot=1e-3, markov_ms_per_slot=0.02,
    )  # fmt: skip
    samples = []
    for requests, k in ((128, 7), (256, 7), (256, 3)):
        parts = truth.draft_ms(requests, k)
        samples.append(
            {
                "requests": requests, "k": k, "target_tokens": requests * (1 + k),
                "draft:fc": parts["fc"], "draft:context_kv": parts["context_kv"],
                "draft:backbone": parts["backbone"], "draft:lm_head": parts["lm_head"],
                "draft:markov": parts["markov"],
            }
        )  # fmt: skip
    fit = wr.fit_dspark_components(samples)
    for requests, k in ((64, 5), (200, 2)):
        assert fit.draft_ms(requests, k)["total"] == pytest.approx(
            truth.draft_ms(requests, k)["total"], rel=1e-6
        )
    one = fit.draft_ms(128, 7, layers=1)
    five = fit.draft_ms(128, 7)
    assert one["backbone"] == pytest.approx(five["backbone"] / 5)
    assert one["lm_head"] == pytest.approx(five["lm_head"])


TRACE_HEAD = (
    "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),Orig Duration (ns),"
    "Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,ParentId,RangeStack\n"
)


def _trace_row(name, start, dur, range_id, stack):
    return f"{name},{start},{dur},{start},{dur},PushPop,1,1,1,0,0,{range_id},,{stack}\n"


def _write_window(stats: Path) -> dict:
    ms = 1_000_000
    rows = TRACE_HEAD
    # Step 1: decode only, 64 requests x (1 + 3) tokens.
    rows += _trace_row(":engine:step", 0, 20 * ms, 1, ":1")
    rows += _trace_row(":execute_context_0(0)_generation_64(256)", 0, 12 * ms, 2, ":1:2")
    rows += _trace_row(":target:execute[decode]", 0, 12 * ms, 3, ":1:2:3")
    rows += _trace_row(":draft:propose", 13 * ms, 6 * ms, 4, ":1:4")
    rows += _trace_row(":draft:fc", 13 * ms, 1 * ms, 5, ":1:4:5")
    rows += _trace_row(":target:sample", 12 * ms, 1 * ms, 6, ":1:6")
    # Step 2: 256 prefill tokens next to the same decode.
    rows += _trace_row(":engine:step", 21 * ms, 30 * ms, 7, ":7")
    rows += _trace_row(":execute_context_1(256)_generation_64(256)", 21 * ms, 20 * ms, 8, ":7:8")
    rows += _trace_row(":target:execute[mixed]", 21 * ms, 20 * ms, 9, ":7:8:9")
    rows += _trace_row(":draft:propose", 42 * ms, 6 * ms, 10, ":7:10")
    stats.mkdir(parents=True)
    (stats / "a.1_nvtx_gpu_proj_trace.csv").write_text(rows)
    (stats / "a.1_cuda_gpu_kern_sum.csv").write_text(
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
        f"50,{20 * ms},1,1,1,1,1,0,nvjet_gemm\n"
        f'50,{10 * ms},1,1,1,1,1,0,"void flash::FlashAttnFwdSm90<x>"\n'
    )
    return {
        "window_s": 0.051,
        "open_s": 0.051,
        "counters": {
            "generation_tokens": 320.0,
            "speculative_decoding": {"draft_tokens": 384.0, "accepted_tokens": 192.0,
                                     "mean_acceptance_length": 2.5},
        },
    }  # fmt: skip


def test_decompose_window_accounts_every_step_ms(tmp_path: Path) -> None:
    stats = tmp_path / "nsys-stats"
    window = _write_window(stats)
    result = wr.decompose_window(stats, "a.1", window)
    parts = result["parts_ms"]
    assert result["steps"] == 2
    assert parts["prefill_ms"] == pytest.approx(10.0)  # half of the mixed step's 20 ms
    verify = 12 + 10
    # no cost table: rejected tokens x average verify ms per verified token
    assert parts["rejected_verify_ms"] == pytest.approx(192 * verify / 512)
    assert parts["draft_fixed_ms"] == pytest.approx(1.0)
    assert parts["draft_wasted_ms"] == pytest.approx(11.0 / 2)
    assert parts["between_steps_ms"] == pytest.approx(1.0)
    in_steps = sum(v for k, v in parts.items() if k != "between_steps_ms")
    assert in_steps == pytest.approx(50.0)
    assert result["attention_fraction_of_kernels"] == pytest.approx(1 / 3)
    assert sum(result["fractions"].values()) == pytest.approx(1.0)


def test_best_k_ceilings_and_saturation() -> None:
    rows = []
    rates = {"x-k1": {1: 100, 256: 900}, "x-k7": {1: 300, 256: 200}}
    for arm, levels in rates.items():
        for c, rate in levels.items():
            rows.append({"arm": arm, "concurrency": c, "output_tokens_per_second": rate,
                         "profiled": False})  # fmt: skip
    ceilings = wr.best_k_ceilings(rows, "x-k")
    assert ceilings["schedule"] == {1: "x-k7", 256: "x-k1"}
    assert ceilings["best_fixed"] == "x-k1"
    assert ceilings["schedule_over_best_fixed"] == pytest.approx(
        (300 * 900) ** 0.5 / (100 * 900) ** 0.5
    )
    levels = [{"concurrency": c, "sm": v} for c, v in ((1, 10), (32, 60), (64, 95), (128, 100))]
    assert wr.saturation(levels, "sm") == 64


def test_power_and_cell_costs(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    cell = sweep / "arm" / "opb" / "opb" / "c8"
    cell.mkdir(parents=True)
    (sweep / "arm" / "status.json").write_text(json.dumps({"gpus": [1]}))
    (cell / "metadata.json").write_text(json.dumps({
        "started_at": "2026-10-22T15:00:00+00:00", "ended_at": "2026-10-22T15:00:10+00:00",
        "extra": {},
    }))  # fmt: skip
    (cell / "summary.json").write_text(json.dumps({"tokens": {"output_tokens_per_second": 1000.0}}))
    (sweep / "gpu-memory.csv").write_text(
        "2026/10/22 17:00:05.000, 1, 1, 1, 50, 400.0\n"  # 15:00:05 UTC
        "2026/10/22 17:00:06.000, 0, 1, 1, 50, 100.0\n"
        "2026/10/22 17:00:20.000, 1, 1, 1, 50, 900.0\n"  # after the cell
    )
    (row,) = wr.cell_costs(sweep)
    assert row["watts"] == pytest.approx(400.0)
    assert row["joules_per_token"] == pytest.approx(0.4)
    assert row["gpu_seconds_per_million_tokens"] == pytest.approx(1000.0)
