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


def _report_module():
    import importlib.util

    path = ROOT / "experiments/task-speed/report-task-speed.py"
    spec = importlib.util.spec_from_file_location("report_task_speed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_combines_controlled_and_live(tmp_path, monkeypatch):
    import sys

    import yaml

    suite = yaml.safe_load(SUITE.read_text())
    suite["slices"] = suite["slices"][:1] + suite["slices"][-1:]  # opb (id), lang-amh (ood)
    suite["concurrencies"] = [1]
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    spec = {"enabled": True, "draft_tokens": 30, "mean_acceptance_length": 3.0,
            "acceptance_rate": 0.66}  # fmt: skip
    controlled = tmp_path / "controlled-x"
    for pair, factor in (("pair-1", 1.0), ("pair-2", 1.02)):
        for label, speedup in (("opb", 2.2), ("lang-amh", 1.0)):
            base = controlled / pair
            _write(base / "plain" / label / label / "c1", _summary(100.0), _metadata())
            _write(
                base / "dspark" / label / label / "c1",
                _summary(100.0 * speedup * factor, spec),
                _metadata(),
            )
    live = tmp_path / "live-x"
    windows = []
    for name, variant, rnd in (("D1", "dspark", 1), ("P1", "plain", 1), ("P2", "plain", 2),
                               ("D2", "dspark", 2)):  # fmt: skip
        windows.append(
            {"name": name, "variant": variant, "round": rnd, "status": "ok", "route": "gateway"}
        )
        for label, speedup in (("opb", 2.0), ("lang-amh", 0.9)):
            rate = 100.0 * (speedup if variant == "dspark" else 1.0)
            _write(
                live / name / label / label / "c1",
                _summary(rate, spec if variant == "dspark" else None),
                _metadata(route="gateway"),
            )
    # A direct-route window of D1 must not enter the pairs, only the overhead table.
    windows.append({"name": "D1-direct", "variant": "dspark", "round": 1, "status": "ok",
                    "route": "direct"})  # fmt: skip
    fast = _summary(250.0, spec)
    fast["latency_ms"]["ttft"]["p50"] = 0.25
    _write(live / "D1-direct" / "opb" / "opb" / "c1", fast, _metadata(route="direct"))
    (live / "windows.json").write_text(json.dumps(windows))
    out = tmp_path / "report"
    argv = ["x", "--suite", str(suite_path), "--controlled", str(controlled), "--live", str(live)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(out)])
    _report_module().main()
    result = json.loads((out / "paired-results.json").read_text())
    assert len(result["controlled"]["rows"]) == 2 and len(result["live"]["rows"]) == 2
    ratios = {t["slice"]: t["ratio"] for t in result["transfer"]}
    assert ratios["opb"] == pytest.approx(2.0 / (2.2 * 1.01), rel=1e-3)
    report = (out / "REPORT.md").read_text()
    assert "| opb | id |" in report and "Live ÷ controlled" in report
    (overhead,) = result["gateway_overhead"]
    assert overhead["window"] == "D1" and overhead["ttft_p50_ms_added"] == pytest.approx(0.75)
    assert overhead["tokens_s_gateway_over_direct"] == pytest.approx(0.8)
    # lang-amh's interval [1.0, 1.02] sits inside the ±5% band.
    assert json.loads((out / "select.json").read_text()) == {"select": {}}


def _live_module():
    import importlib.util

    path = ROOT / "experiments/task-speed/run-live.py"
    spec = importlib.util.spec_from_file_location("run_live", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_window_guards_cells_and_records_the_window(tmp_path, monkeypatch):
    import argparse
    import asyncio

    import httpx
    import yaml

    live = _live_module()
    rows = [_row(f"opb-{i}", "opb") for i in range(4)]
    inputs, suite = _suite(tmp_path, src=rows)
    suite["slices"] = [{"label": "opb", "class": "id", "source": "src", "workload": "opb"}]
    suite["concurrencies"] = [1]  # the window covers the whole suite
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    corpus_dir = tmp_path / "controlled"
    corpus_dir.mkdir()
    manifest = task_speed.build_corpus(suite, inputs, corpus_dir / "corpus.jsonl")
    (corpus_dir / "corpus-manifest.json").write_text(json.dumps(manifest))

    model = live.model_ids()["dspark"]
    # One foreign request (100 generated tokens) lands in the first measured cell.
    state = {"done": 0, "generated": 0, "foreign": 1}

    def scrape():
        label = f'{{engine="0",model_name="{model}"}}'
        return (
            f"# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{label} 0\n"
            f"# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{label} 0\n"
            "# TYPE vllm:request_success counter\n"
            f"vllm:request_success_total{label} {state['done']}\n"
            "# TYPE vllm:generation_tokens counter\n"
            f"vllm:generation_tokens_total{label} {state['generated']}\n"
            "# TYPE vllm:spec_decode_num_drafts counter\n"
            f"vllm:spec_decode_num_drafts_total{label} {state['done'] * 2}\n"
            "# TYPE vllm:spec_decode_num_draft_tokens counter\n"
            f"vllm:spec_decode_num_draft_tokens_total{label} {state['done'] * 6}\n"
            "# TYPE vllm:spec_decode_num_accepted_tokens counter\n"
            f"vllm:spec_decode_num_accepted_tokens_total{label} {state['done'] * 4}\n"
        )

    pod = {
        "metadata": {
            "name": "pod-a",
            "uid": "uid-a",
            "annotations": {
                "apertus.swissai/checkpoint-sha256": "7a70",
                "apertus.swissai/source-revision": "rev",
            },
        },
        "spec": {"nodeName": "nid006597"},
        "status": {"startTime": "t", "containerStatuses": [{"restartCount": 0, "imageID": "i"}]},
    }
    scrapes = []

    def kube_run(self, *argv, timeout=60):
        if argv[:2] == ("get", "pods"):
            return json.dumps({"items": [pod]})
        if argv[0] == "exec":
            scrapes.append(1)
            # The foreign request finishes inside the first measured cell: scrapes 1-2
            # are the idle waits around the warmup, 3 is "before", 4 is "after".
            if len(scrapes) == 4 and state["foreign"]:
                state["done"] += state.pop("foreign")
                state["generated"] += 100
            return scrape()
        raise AssertionError(argv)

    async def handler(request):
        state["done"] += 1
        state["generated"] += 2
        events = [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    class Client(live.StreamingChatClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.http.headers["Authorization"] == "Bearer key"
            self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(live.Kube, "run", kube_run)
    monkeypatch.setattr(live, "StreamingChatClient", Client)
    monkeypatch.setenv("CSCS_SERVING_API", "key")
    args = argparse.Namespace(
        suite=suite_path, corpus=corpus_dir / "corpus.jsonl", output=tmp_path / "live",
        window="D1", variant="dspark", round=1, direct=False, slices=None,
        concurrencies=[1], retries=2, budget_minutes=None, metrics_interval=None,
        timeout_seconds=5,
        kubectl="kubectl", context="c", namespace="n", api_key_env="CSCS_SERVING_API",
        sml_python="python3", metrics_url=None, pod_json=None, api_key_file=None,
    )  # fmt: skip
    asyncio.run(live.window(args))
    (record,) = json.loads((tmp_path / "live" / "windows.json").read_text())
    assert record["status"] == "ok" and record["cells"] == [
        {"slice": "opb", "concurrency": 1, "valid": True}
    ]
    window = tmp_path / "live" / "D1"
    (rejected,) = (window / "rejected").iterdir()
    assert (rejected / "guard.json").exists()
    # A second job of the same window skips the measured cell.
    sent = state["done"]
    asyncio.run(live.window(args))
    assert state["done"] - sent == 1 + 2  # only the pre-warm: one prompt at C=1, twice at C=128
    assert len(json.loads((tmp_path / "live" / "windows.json").read_text())) == 1
    guard = json.loads((window / "opb/opb/c1/guard.json").read_text())
    assert guard["valid"] and guard["finished_delta"] == 2  # requests_for(suite, 1)
    summary = json.loads((window / "opb/opb/c1/summary.json").read_text())
    assert summary["speculative_decoding"]["mean_acceptance_length"] == 3.0
    metadata = json.loads((window / "opb/opb/c1/metadata.json").read_text())
    assert metadata["extra"]["route"] == "gateway" and metadata["extra"]["node"] == "nid006597"
    with pytest.raises(RuntimeError, match="toggle first"):
        args.window, args.variant = "P1", "plain"
        asyncio.run(live.window(args))


def test_inline_and_separate_warmups_pair():
    plain, draft = (
        _summary(100.0),
        _summary(200.0, {"enabled": True, "draft_tokens": 3, "mean_acceptance_length": 2.0}),
    )
    pm, dm = _metadata(), _metadata(separate_warmup="1")
    dm["cell"]["warmup_requests"] = 0
    task_speed.validate_pair(plain, draft, pm, dm, ())
    dm["extra"]["separate_warmup"] = "8"
    with pytest.raises(ValueError, match="warmup"):
        task_speed.validate_pair(plain, draft, pm, dm, ())


def test_traffic_guard_tolerates_small_foreign_load():
    def scrape(done, generated):
        label = f'{{engine="0",model_name="{MODEL}"}}'
        return (
            _metrics(done=done)
            + "# TYPE vllm:generation_tokens counter\n"
            + f"vllm:generation_tokens_total{label} {generated}\n"
        )

    ok, reason, detail = task_speed.traffic_guard(
        scrape(10, 0), scrape(43, 10_150), MODEL, 32, completion_tokens=10_000
    )
    assert ok and "tolerated" in reason and detail["foreign_token_share"] == pytest.approx(0.015)
    ok, reason, _ = task_speed.traffic_guard(
        scrape(10, 0), scrape(43, 10_300), MODEL, 32, completion_tokens=10_000
    )
    assert not ok and "foreign" in reason
    ok, _, _ = task_speed.traffic_guard(scrape(10, 0), scrape(41, 9_000), MODEL, 32, 10_000)
    assert not ok  # fewer finished than sent is never tolerated


def test_live_window_in_cluster_mode_reads_service_metrics(tmp_path, monkeypatch):
    """The client pod has no kubectl: pod identity from a file, metrics over HTTP."""
    import argparse
    import asyncio

    import httpx
    import yaml

    live = _live_module()
    inputs, suite = _suite(tmp_path, src=[_row(f"opb-{i}", "opb") for i in range(2)])
    suite["slices"] = [{"label": "opb", "class": "id", "source": "src", "workload": "opb"}]
    suite["concurrencies"] = [1]
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    (tmp_path / "c").mkdir()
    manifest = task_speed.build_corpus(suite, inputs, tmp_path / "c" / "corpus.jsonl")
    (tmp_path / "c" / "corpus-manifest.json").write_text(json.dumps(manifest))
    model = live.model_ids()["plain"]
    state = {"done": 0}
    pod = {
        "metadata": {"name": "p", "uid": "u", "annotations": {
            "apertus.swissai/variant": "plain", "apertus.swissai/checkpoint-sha256": "c",
            "apertus.swissai/source-revision": "r"}},
        "spec": {"nodeName": "nid006597"},
        "status": {"containerStatuses": [{"restartCount": 0}]},
    }  # fmt: skip
    (tmp_path / "pod.json").write_text(json.dumps(pod))
    (tmp_path / "key").write_text("key\n")
    urls = []

    async def metrics_get(self, url):
        urls.append(self.url)
        label = f'{{engine="0",model_name="{model}"}}'
        text = (
            f"# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{label} 0\n"
            "# TYPE vllm:request_success counter\n"
            f"vllm:request_success_total{label} {state['done']}\n"
        )
        return httpx.Response(200, text=text, request=httpx.Request("GET", self.url))

    async def handler(request):
        state["done"] += 1
        events = [
            {"choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    class Client(live.StreamingChatClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.http.headers["Authorization"] == "Bearer key"
            self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(live.HttpMetrics, "get", metrics_get)
    monkeypatch.setattr(live, "StreamingChatClient", Client)
    monkeypatch.setattr(live, "Kube", None)  # must not be used in-cluster
    url = "http://svc.rob-poc.svc:8080/metrics"
    args = argparse.Namespace(
        suite=suite_path, corpus=tmp_path / "c" / "corpus.jsonl", output=tmp_path / "live",
        window="P1", variant="plain", round=1, direct=False, slices=None, concurrencies=None,
        retries=0, budget_minutes=None, metrics_interval=None, timeout_seconds=5,
        kubectl="kubectl", context="c", namespace="n", api_key_env="UNSET_KEY_ENV",
        sml_python="python3", metrics_url=url, pod_json=tmp_path / "pod.json",
        api_key_file=tmp_path / "key",
    )  # fmt: skip
    asyncio.run(live.window(args))
    (record,) = json.loads((tmp_path / "live" / "windows.json").read_text())
    assert record["status"] == "ok" and set(urls) == {url}
    metadata = json.loads((tmp_path / "live/P1/opb/opb/c1/metadata.json").read_text())
    assert metadata["metrics_url"] == url and metadata["extra"]["node"] == "nid006597"
