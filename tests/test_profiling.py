import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from apertus_bench import profiling
from apertus_bench.node_sweep import SpecError, bench_command, build_plan

DRAFTER = "/capstor/drafters/dspark-epoch10"


def _spec(profile=None, **arm_overrides):
    dspark = {"name": "dspark-k7", "method": "dspark", "drafter": DRAFTER, "k": 7}
    if profile is not None:
        dspark["profile"] = profile
    dspark.update(arm_overrides)
    return {
        "name": "smoke",
        "stage": "8b",
        "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256},
        "loadtests": [{"workload": "opb", "concurrencies": [8, 64]}],
        "arms": [{"name": "plain", "method": "baseline"}, dspark],
    }


def test_profile_defaults_and_validation() -> None:
    spec = profiling.parse_profile({"tool": "nsys", "levels": [64]}, "a")
    assert spec is not None
    assert (spec.window_s, spec.delay_s, spec.levels) == (15.0, 10.0, [64])
    assert spec.captures("opb", 64) and not spec.captures("opb", 8)
    assert profiling.parse_profile(None, "a") is None
    with pytest.raises(ValueError, match="unknown profile settings"):
        profiling.parse_profile({"window": 3}, "a")
    with pytest.raises(ValueError, match="tool must be nsys"):
        profiling.parse_profile({"tool": "ncu"}, "a")
    with pytest.raises(ValueError, match="window_s"):
        profiling.parse_profile({"window_s": 0}, "a")


def test_nsys_argv_wraps_serve_with_a_repeating_capture_range() -> None:
    spec = profiling.parse_profile({"gpu_metrics": True, "extra_args": ["--x", 1]}, "a")
    assert spec is not None
    serve = ["vllm", "serve", "--model", "m", "--port", "8101"]
    argv = profiling.nsys_argv(spec, serve, Path("/r/arm/nsys/arm"))
    assert argv[0] == profiling.DEFAULT_NSYS and argv[1] == "profile"
    assert argv[-len(serve) :] == serve
    flags = dict(zip(argv[2 : -len(serve) : 2], argv[3 : -len(serve) : 2], strict=False))
    assert flags["--capture-range"] == "cudaProfilerApi"
    assert flags["--capture-range-end"] == "repeat"
    assert flags["--cuda-graph-trace"] == "node"
    assert flags["--trace"] == "cuda,nvtx,osrt"
    assert flags["--output"] == "/r/arm/nsys/arm"
    assert flags["--gpu-metrics-devices"] == "cuda-visible"
    assert "--x" in argv and "1" in argv


def test_unprofiled_arms_are_served_exactly_as_before() -> None:
    plain_before, dspark_before = build_plan(_spec(), check_heads=False).arms
    plain, dspark = build_plan(_spec(profile={"levels": [64]}), check_heads=False).arms
    assert plain.serve_argv == plain_before.serve_argv and plain.env == plain_before.env
    assert plain.profile is None and "profile" not in plain.metadata
    plan = build_plan(_spec(), check_heads=False)
    command = bench_command(plan.arms[0], plan.loadtests[0], Path("/o"), "d")
    assert "--profile-levels" not in command
    # The profiled arm: the same argv plus the CUDA-profiler hook, NVTX env, a tag.
    assert dspark.serve_argv == [*dspark_before.serve_argv, "--profiler-config.profiler", "cuda"]
    assert dspark.env == {"VLLM_APERTUS_NVTX": "1"}
    assert json.loads(dspark.metadata["profile"])["levels"] == [64]


def test_profiled_arm_bench_opens_windows_only_in_chosen_levels() -> None:
    plan = build_plan(
        _spec(profile={"levels": [64], "window_s": 5, "delay_s": 3}), check_heads=False
    )
    command = bench_command(plan.arms[1], plan.loadtests[0], Path("/o"), "d")
    start = command.index("--profile-levels")
    assert command[start : start + 6] == [
        "--profile-levels", "64", "--profile-window", "5.0", "--profile-delay", "3.0",
    ]  # fmt: skip
    assert profiling.expected_windows(plan.arms[1].profile, [("opb", [8, 64])]) == 1


def test_profile_spec_errors() -> None:
    with pytest.raises(SpecError, match="not in any load test"):
        build_plan(_spec(profile={"levels": [128]}), check_heads=False)
    with pytest.raises(SpecError, match="serving arms only"):
        spec = _spec()
        spec["arms"].append({"name": "c", "method": "command", "command": ["true"], "profile": {}})
        build_plan(spec, check_heads=False)


def test_arm_env_overrides_profile_env() -> None:
    (_, dspark) = build_plan(
        _spec(profile={"eager_draft": True}, env={"VLLM_APERTUS_NVTX": 0}), check_heads=False
    ).arms
    assert dspark.env == {"VLLM_APERTUS_NVTX": "0", "VLLM_APERTUS_NVTX_EAGER_DRAFT": "1"}


PROM = """# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{model_name="m"}} {gen}
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{model_name="m"}} {prompt}
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{{model_name="m"}} {drafts}
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{{model_name="m"}} {draft_tokens}
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{{model_name="m"}} {accepted}
"""


def test_profile_window_posts_start_and_stop_and_records_counters() -> None:
    calls: list[str] = []
    scrapes = iter(
        [
            PROM.format(gen=100, prompt=50, drafts=10, draft_tokens=70, accepted=30),
            PROM.format(gen=400, prompt=90, drafts=110, draft_tokens=770, accepted=230),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/metrics":
            return httpx.Response(200, text=next(scrapes))
        return httpx.Response(200)

    async def run() -> dict:
        record: dict = {}
        stop = asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await profiling.profile_window(
                http, "http://s", "http://s/metrics", "m",
                profiling.WindowSettings(0.0, 0.05), record, stop,
            )  # fmt: skip
        return record

    record = asyncio.run(run())
    assert calls == ["/metrics", "/start_profile", "/stop_profile", "/metrics"]
    assert record["truncated"] is False
    counters = record["counters"]
    assert counters["generation_tokens"] == 300 and counters["prompt_tokens"] == 40
    assert counters["speculative_decoding"]["mean_acceptance_length"] == 3.0


def test_profile_window_is_skipped_when_the_level_ends_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async def run() -> dict:
        record: dict = {}
        stop = asyncio.Event()
        stop.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await profiling.profile_window(
                http, "http://s", None, None, profiling.WindowSettings(5, 5), record, stop
            )
        return record

    assert "skipped" in asyncio.run(run())


def test_gpu_metrics_summary_reads_the_exported_tables(tmp_path: Path) -> None:
    database = tmp_path / "r.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (typeId, metricId, metricName)")
    connection.execute("CREATE TABLE GPU_METRICS (timestamp, typeId, metricId, value)")
    connection.executemany(
        "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)",
        [(1, 1, "SMs Active [Throughput %]"), (2, 1, "SMs Active [Throughput %]")],
    )
    connection.executemany(
        "INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", [(0, 1, 1, 40), (1, 1, 1, 60)]
    )
    connection.commit()
    connection.close()
    (row,) = profiling.gpu_metrics_summary(database)
    assert row == {"metric": "SMs Active [Throughput %]", "samples": 2, "mean": 50.0,
                   "min": 40, "max": 60}  # fmt: skip
    empty = tmp_path / "e.sqlite"
    sqlite3.connect(empty).close()
    assert profiling.gpu_metrics_summary(empty) == []


def test_collect_windows_in_run_order(tmp_path: Path) -> None:
    for level, opened in ((8, False), (64, True)):
        cell = tmp_path / "opb" / "opb" / f"c{level}"
        cell.mkdir(parents=True)
        record = {"start_requested_at": "t"} if opened else {"skipped": "x"}
        (cell / profiling.WINDOW_FILE).write_text(json.dumps(record))
    windows = profiling.collect_windows(tmp_path, [("opb", "opb", [64, 8])])
    assert [w["concurrency"] for w in windows] == [64]
