import copy
import hashlib
import json
from pathlib import Path

import pytest

from apertus_bench import task_speed
from apertus_bench.node_sweep import bench_command, build_plan

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/task-speed/suite.yaml"


def test_frozen_suite_classes_and_grid():
    suite = task_speed.load_suite(SUITE)
    classes = task_speed.slice_classes(suite)
    assert len(classes) == 28
    counts = {c: list(classes.values()).count(c) for c in task_speed.CLASSES}
    assert counts == {"id": 5, "near-id": 6, "ood-task": 5, "ood-language": 12}
    assert [task_speed.requests_for(suite, c) for c in suite["concurrencies"]] == [32, 32, 64, 256]


def _suite(tmp_path, **sources):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frozen = {}
    for name, rows in sources.items():
        path = inputs / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        frozen[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return inputs, {
        "name": "t",
        "seed": "s",
        "prompts_per_slice": 2,
        "concurrencies": [1, 4],
        "min_requests": 2,
        "requests_per_slot": 1,
        "generation": {"temperature": 0, "top_p": 1, "seed": 1},
        "chat_template_kwargs": {"enable_thinking": False},
        "sources": frozen,
        "slices": [],
    }


def _row(prompt_id, workload):
    return {
        "id": prompt_id,
        "workload": workload,
        "messages": [{"role": "user", "content": prompt_id}],
        "max_tokens": 16,
    }


def test_corpus_is_hash_selected_relabelled_and_prewarmed(tmp_path):
    rows = [_row(f"probe-math-{i}", "probe") for i in range(4)]
    rows += [_row(f"probe-humaneval-{i}", "probe") for i in range(4)]
    inputs, suite = _suite(tmp_path, probe=rows)
    suite["slices"] = [
        {"label": label, "class": "id", "source": "probe", "workload": "probe", "id_prefix": prefix}
        for label, prefix in (("m", "probe-math-"), ("h", "probe-humaneval-"))
    ]
    first = task_speed.build_corpus(suite, inputs, tmp_path / "a.jsonl")
    again = task_speed.build_corpus(suite, inputs, tmp_path / "b.jsonl")
    assert first == {**again, "corpus_sha256": first["corpus_sha256"]}
    assert first["corpus_sha256"] == again["corpus_sha256"]
    assert all(i.startswith("probe-math-") for i in first["slices"]["m"]["ids"])
    written = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert [r["workload"] for r in written] == ["m", "m", "h", "h", "prewarm", "prewarm"]
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in written)
    assert written[4]["id"].startswith("prewarm-")
    expected = sorted(
        (f"probe-math-{i}" for i in range(4)),
        key=lambda i: hashlib.sha256(f"s:{i}".encode()).hexdigest(),
    )[:2]
    assert first["slices"]["m"]["ids"] == expected


def test_corpus_refuses_changed_source_and_short_pool(tmp_path):
    inputs, suite = _suite(tmp_path, src=[_row("a", "w"), _row("b", "w")])
    suite["slices"] = [{"label": "x", "class": "id", "source": "src", "workload": "w"}]
    task_speed.build_corpus(suite, inputs, tmp_path / "ok.jsonl")
    bad = copy.deepcopy(suite)
    bad["sources"]["src"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen"):
        task_speed.build_corpus(bad, inputs, tmp_path / "bad.jsonl")
    suite["prompts_per_slice"] = 3
    with pytest.raises(ValueError, match="need 3 prompts"):
        task_speed.build_corpus(suite, inputs, tmp_path / "short.jsonl")


def test_controlled_spec_matches_live_engine_and_builds(tmp_path):
    suite = task_speed.load_suite(SUITE)
    spec = task_speed.controlled_spec(
        suite, tmp_path / "c.jsonl", "/target", "/drafter", "/t.jinja"
    )
    plan = build_plan(spec, check_heads=False)
    assert [arm.name for arm in plan.arms] == ["plain", "dspark"]
    assert [t.label for t in plan.loadtests][:2] == ["prewarm", "opb"]
    assert len(plan.loadtests) == 29
    plain, dspark = (" ".join(arm.serve_argv) for arm in plan.arms)
    for flag in ("--max-num-seqs 256", "--max-num-batched-tokens 16384", "--max-model-len 32768"):
        assert flag in plain and flag in dspark
    assert "--speculative-config" in dspark and "--speculative-config" not in plain
    command = bench_command(plan.arms[1], plan.loadtests[1], tmp_path / "out", "d")
    assert command[command.index("--concurrencies") + 1 : command.index("--requests")] == [
        "1",
        "8",
        "32",
        "128",
    ]
    assert "--max-tokens" not in command  # each prompt keeps its own cap
    select = task_speed.controlled_spec(
        suite, tmp_path / "c.jsonl", "/t", "/d", "/t.jinja", select={"chat": [128]}
    )
    assert [(t["label"], t["concurrencies"]) for t in select["loadtests"]] == [
        ("prewarm", [1, 32]),
        ("chat", [128]),
    ]


def _summary(rate, spec=None):
    return {
        "requests": {"success_rate": 1.0, "attempted": 32, "successful": 32},
        "warmup": {"failed": 0},
        "tokens": {"output_tokens_per_second": rate, "completion": 1000},
        "latency_ms": {m: {"p50": 1.0, "p95": 2.0} for m in ("ttft", "tpot", "e2e")},
        "speculative_decoding": spec or {"enabled": False},
    }


def _metadata(**extra):
    return {
        "workload_file_sha256": "c",
        "prompt_ids": ["a"],
        "cell": {"workload": "x", "concurrency": 1, "requests": 32, "warmup_requests": 1,
                 "generation": {}},
        "extra": {"physical_gpu": "0", **extra},
    }  # fmt: skip


def _write(cell, summary, metadata):
    cell.mkdir(parents=True)
    (cell / "summary.json").write_text(json.dumps(summary))
    (cell / "metadata.json").write_text(json.dumps(metadata))


def test_cells_pair_validate_and_bootstrap(tmp_path):
    suite = {"slices": [{"label": "x", "class": "id"}], "concurrencies": [1, 8]}
    spec = {"enabled": True, "draft_tokens": 30, "mean_acceptance_length": 3.0,
            "acceptance_rate": 0.66}  # fmt: skip
    pairs = {}
    for name, speedup in (("p1", 2.0), ("p2", 2.2), ("p3", 1.8)):
        arms = {arm: tmp_path / name / arm for arm in ("plain", "dspark")}
        _write(arms["plain"] / "x/x/c1", _summary(100.0), _metadata())
        _write(arms["dspark"] / "x/x/c1", _summary(100.0 * speedup, spec), _metadata())
        pairs[name] = arms
    # A pair on another GPU and a pair with an inactive drafter are rejected.
    odd = {arm: tmp_path / "p4" / arm for arm in ("plain", "dspark")}
    _write(odd["plain"] / "x/x/c1", _summary(100.0), _metadata())
    _write(odd["dspark"] / "x/x/c1", _summary(300.0, spec), _metadata(physical_gpu="1"))
    pairs["p4"] = odd
    dead = {arm: tmp_path / "p5" / arm for arm in ("plain", "dspark")}
    _write(dead["plain"] / "x/x/c1", _summary(100.0), _metadata())
    _write(dead["dspark"] / "x/x/c1", _summary(300.0, {"enabled": True, "draft_tokens": 0}),
           _metadata())  # fmt: skip
    pairs["p5"] = dead
    result = task_speed.cell_rows(suite, pairs, ("physical_gpu",))
    (row,) = result["rows"]
    assert [p["pair"] for p in row["pairs"]] == ["p1", "p2", "p3"]
    assert row["speedup_geomean"] == pytest.approx((2.0 * 2.2 * 1.8) ** (1 / 3))
    assert 1.8 <= row["bootstrap_95"][0] < row["bootstrap_95"][1] <= 2.2
    assert row["verdict"] == "speedup" and row["acceptance_length"] == 3.0
    assert any("unmatched physical_gpu" in r for r in result["rejected"])
    assert any("inactive" in r for r in result["rejected"])
    assert len(result["missing"]) == 5 and not result["complete"]  # every pair lacks c8


def test_verdicts_and_class_summary():
    assert task_speed.verdict([1.06, 1.2]) == "speedup"
    assert task_speed.verdict([0.9, 0.94]) == "slowdown"
    assert task_speed.verdict([0.96, 1.04]) == "within 5%"
    assert task_speed.verdict([0.9, 1.1]) == "inconclusive"
    assert task_speed.verdict(None) == "incomplete"
    rows = [
        {"class": "id", "concurrency": 1, "speedup_geomean": 2.0, "acceptance_length": 3.4,
         "verdict": "speedup"},
        {"class": "id", "concurrency": 1, "speedup_geomean": 0.5, "acceptance_length": 1.5,
         "verdict": "slowdown"},
    ]  # fmt: skip
    (summary,) = task_speed.class_summary(rows)
    assert summary["speedup_geomean"] == pytest.approx(1.0)
    assert summary["verdicts"] == {"slowdown": 1, "speedup": 1}


def test_acceptance_curve_recovers_break_even():
    rows = [
        {"concurrency": 128, "acceptance_length": x, "speedup_geomean": 0.5 + 0.25 * x}
        for x in (1.5, 2.0, 3.0, 3.5)
    ]
    (curve,) = task_speed.acceptance_curve(rows)
    assert curve["slope"] == pytest.approx(0.25)
    assert curve["break_even_acceptance_length"] == pytest.approx(2.0)
    assert curve["r2"] == pytest.approx(1.0)


MODEL = "SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3"


def _metrics(running=0, waiting=0, done=10):
    label = f'{{engine="0",model_name="{MODEL}"}}'
    stop = f'{{engine="0",finished_reason="stop",model_name="{MODEL}"}}'
    length = f'{{engine="0",finished_reason="length",model_name="{MODEL}"}}'
    return (
        "# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running{label} {running}\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        f"vllm:num_requests_waiting{label} {waiting}\n"
        "# TYPE vllm:request_success counter\n"
        f"vllm:request_success_total{stop} {done - 3}\n"
        f"vllm:request_success_total{length} 3\n"
    )


def test_traffic_guard():
    ok, reason, detail = task_speed.traffic_guard(_metrics(done=10), _metrics(done=42), MODEL, 32)
    assert ok and reason is None and detail["finished_delta"] == 32
    ok, reason, _ = task_speed.traffic_guard(_metrics(running=1), _metrics(done=42), MODEL, 32)
    assert not ok and "busy" in reason
    ok, reason, _ = task_speed.traffic_guard(_metrics(done=10), _metrics(done=45), MODEL, 32)
    assert not ok and "foreign" in reason
    ok, reason, _ = task_speed.traffic_guard(_metrics(done=10), _metrics(done=5), MODEL, 32)
    assert not ok and "restart" in reason
