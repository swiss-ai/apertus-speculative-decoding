import asyncio
from pathlib import Path

from apertus_bench.cli import loadtest_level
from apertus_bench.load import MetricsSampler, parse_load_sample, summarize_load

REPO = Path(__file__).resolve().parents[1]
SCRAPE = (
    REPO / "results/8b/eagle/confirm/apertus15-8b-baseline-confirm-c3-20260926T144421Z"
    "/cells/summarization/c1/metrics_after.prom"
)


def _scrape(kv: float, running: int, waiting: int, preempted: int, generated: int) -> str:
    labels = 'engine="0",model_name="m"'
    info = f'{labels},kv_cache_size_tokens="1000",num_gpu_blocks="62",block_size="16"'
    return (
        f"vllm:kv_cache_usage_perc{{{labels}}} {kv}\n"
        f"vllm:num_requests_running{{{labels}}} {running}\n"
        f"vllm:num_requests_waiting{{{labels}}} {waiting}\n"
        f"vllm:num_preemptions_total{{{labels}}} {preempted}\n"
        f"vllm:generation_tokens_total{{{labels}}} {generated}\n"
        f'vllm:cache_config_info{{{info},gpu_memory_utilization="0.8"}} 1.0\n'
    )


def test_real_scrape_from_the_pinned_image_parses() -> None:
    sample = parse_load_sample(SCRAPE.read_text())
    assert sample["kv_cache_size_tokens"] == 466144
    assert sample["gpu_memory_utilization"] == 0.8
    for key in ("kv_cache_usage", "running", "waiting", "preemptions", "generation_tokens"):
        assert key in sample


def test_other_models_on_the_same_server_are_ignored() -> None:
    assert "kv_cache_usage" not in parse_load_sample(_scrape(0.5, 1, 0, 0, 10), model="other")


def test_summary_reports_peak_kv_tokens_queue_and_deltas() -> None:
    samples = []
    for t, (kv, running, waiting, preempted, generated) in enumerate(
        [(0.1, 2, 0, 0, 100), (0.95, 8, 3, 1, 400), (0.5, 4, 0, 2, 700)]
    ):
        sample = parse_load_sample(_scrape(kv, running, waiting, preempted, generated), "m")
        sample["t"] = float(t)
        samples.append(sample)
    summary = summarize_load(samples, errors=1)
    assert summary["kv_cache_usage"]["max"] == 0.95
    assert summary["kv_cache_tokens_peak"] == 950
    assert summary["waiting"]["max"] == 3
    assert summary["preemptions"] == 2
    assert summary["generation_tokens"] == 600
    assert summary["seconds_kv_above_90pct"] == 1.0
    assert summary["scrape_errors"] == 1
    assert summarize_load([]) == {"samples": 0, "scrape_errors": 0}


async def test_sampler_polls_until_stopped_and_survives_failed_scrapes() -> None:
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("scrape failed")
        return _scrape(0.2, 1, 0, 0, calls)

    sampler = MetricsSampler(fetch, interval=0.01, model="m")
    sampler.start()
    await asyncio.sleep(0.08)
    await sampler.stop()
    assert sampler.errors == 1
    assert len(sampler.samples) >= 3
    assert all("t" in s for s in sampler.samples)


def test_loadtest_level_row() -> None:
    summary = {
        "requests": {"attempted": 32, "success_rate": 1.0},
        "tokens": {"output_tokens_per_second": 900.0},
        "latency_ms": {"ttft": {"p50": 50, "p95": 90}, "tpot": {"p50": 6, "p95": 9}, "e2e": {}},
        "load": {"kv_cache_usage": {"mean": 0.4, "max": 0.7}, "waiting": {"max": 2}},
    }
    row = loadtest_level(8, summary)
    assert row["concurrency"] == 8
    assert row["kv_cache_usage_max"] == 0.7
    assert row["waiting_max"] == 2
    assert row["e2e_p95_ms"] is None
    assert row["preemptions"] is None
    assert row["mean_acceptance_length"] is None


def test_loadtest_level_row_reports_acceptance() -> None:
    summary = {
        "requests": {"attempted": 32, "success_rate": 1.0},
        "tokens": {"output_tokens_per_second": 900.0},
        "latency_ms": {"ttft": {}, "tpot": {}, "e2e": {}},
        "speculative_decoding": {
            "enabled": True,
            "mean_acceptance_length": 4.2,
            "acceptance_rate": 0.46,
        },
    }
    row = loadtest_level(8, summary)
    assert row["mean_acceptance_length"] == 4.2
    assert row["acceptance_rate"] == 0.46


async def test_run_cell_samples_load_during_the_cell(tmp_path: Path) -> None:
    import json

    import httpx

    from apertus_bench.client import StreamingChatClient
    from apertus_bench.runner import CellSettings, Variant, run_cell
    from apertus_bench.workloads import Prompt

    scrapes = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal scrapes
        if request.url.path == "/metrics":
            scrapes += 1
            return httpx.Response(200, text=_scrape(0.3, 1, 0, 0, scrapes))
        await asyncio.sleep(0.05)  # long enough for the sampler to scrape mid-cell
        events = [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    workload = tmp_path / "w.jsonl"
    workload.write_text("{}\n")
    prompts = [
        Prompt(id="p0", workload="chat", messages=[{"role": "user", "content": "x"}], max_tokens=4)
    ]
    async with StreamingChatClient("http://server", "m", None, timeout_seconds=5) as client:
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        summary = await run_cell(
            client,
            prompts,
            workload,
            tmp_path / "cell",
            CellSettings(workload="chat", concurrency=2, requests=4, warmup_requests=0, repeat=1),
            Variant(name="plain", method="none"),
            "http://server/metrics",
            "m",
            metrics_interval=0.01,
        )
    assert summary["requests"]["successful"] == 4
    assert summary["load"]["samples"] >= 2
    assert summary["load"]["kv_cache_size_tokens"] == 1000
    lines = (tmp_path / "cell" / "metrics_timeseries.jsonl").read_text().splitlines()
    assert len(lines) == summary["load"]["samples"]


def test_client_does_not_cap_concurrent_connections() -> None:
    from apertus_bench.client import StreamingChatClient

    client = StreamingChatClient("http://server", "m", None, timeout_seconds=5)
    # httpx keeps the limit on the transport's pool (private, but it is the setting).
    assert client.http._transport._pool._max_connections >= 10**6  # "unlimited" is sys.maxsize
    # Shorter than the server's 5 s keep-alive, so no half-closed connection is reused.
    assert client.http._transport._pool._keepalive_expiry < 5
