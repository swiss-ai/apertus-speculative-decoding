import json
from pathlib import Path

import pytest

from apertus_bench.node_sweep import SpecError, bench_command, build_plan

DRAFTER = "/capstor/drafters/dspark-epoch10"


def _spec(**overrides):
    spec = {
        "name": "k-sweep",
        "stage": "8b",
        "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256},
        "loadtests": [{"workload": "summarization", "concurrencies": [1, 8]}],
        "arms": [
            {"name": "plain", "method": "baseline"},
            {"name": "dspark-k3", "method": "dspark", "drafter": DRAFTER, "k": 3},
        ],
    }
    spec.update(overrides)
    return spec


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_plan_serves_arms_like_the_launchers() -> None:
    plan = build_plan(_spec(), check_heads=False)
    plain, dspark = plan.arms
    assert _flag(plain.serve_argv, "--max-num-batched-tokens") == "16384"
    assert _flag(plain.serve_argv, "--max-model-len") == "32768"
    assert _flag(plain.serve_argv, "--gpu-memory-utilization") == "0.8"
    assert "--no-enable-prefix-caching" in plain.serve_argv
    assert "--speculative-config" not in plain.serve_argv
    config = json.loads(_flag(dspark.serve_argv, "--speculative-config"))
    assert config == {"method": "dspark", "model": DRAFTER, "num_speculative_tokens": 3}
    assert plain.port != dspark.port
    assert plain.served_model.endswith("-plain")


def test_arm_engine_and_speculative_settings_override_the_common_ones() -> None:
    schedule = [[1, 8, 7], [9, 256, 2]]
    spec = _spec(
        arms=[
            {
                "name": "dspark-dyn",
                "method": "dspark",
                "drafter": DRAFTER,
                "k": 7,
                "engine": {"async_scheduling": False, "max_num_seqs": 128},
                "speculative": {"num_speculative_tokens_per_batch_size": schedule},
                "env": {"VLLM_DSPARK_SCHEDULER": 1},
            }
        ]
    )
    (arm,) = build_plan(spec, check_heads=False).arms
    assert _flag(arm.serve_argv, "--max-num-seqs") == "128"
    assert "--no-async-scheduling" in arm.serve_argv
    config = json.loads(_flag(arm.serve_argv, "--speculative-config"))
    assert config["num_speculative_tokens_per_batch_size"] == schedule
    assert arm.env == {"VLLM_DSPARK_SCHEDULER": "1"}


def test_eagle_arm_sets_draft_tp_and_bench_algorithm() -> None:
    spec = _spec(arms=[{"name": "eagle-k3", "method": "eagle", "drafter": "/heads/h", "k": 3}])
    (arm,) = build_plan(spec, check_heads=False).arms
    config = json.loads(_flag(arm.serve_argv, "--speculative-config"))
    assert config["method"] == "eagle3"
    assert config["draft_tensor_parallel_size"] == 1
    assert arm.bench_argv[:4] == ["--method", "eagle3", "--algorithm", "eagle31"]


@pytest.mark.parametrize(
    "arm, message",
    [
        ({"name": "x", "method": "dspark", "drafter": DRAFTER, "k": 8}, "at most 7"),
        ({"name": "x", "method": "dspark", "k": 3}, "needs drafter"),
        ({"name": "x y", "method": "baseline"}, "name must be"),
        ({"name": "x", "method": "medusa"}, "method must be"),
        ({"name": "x", "method": "baseline", "tensor_parallel_size": 8}, "GPUs"),
        ({"name": "x", "method": "baseline", "engine": {"bogus": 1}}, "unknown engine"),
    ],
)
def test_bad_arms_are_refused(arm, message) -> None:
    with pytest.raises(SpecError, match=message):
        build_plan(_spec(arms=[arm]), check_heads=False)


def test_duplicate_arm_names_are_refused() -> None:
    arms = [{"name": "a", "method": "baseline"}, {"name": "a", "method": "baseline"}]
    with pytest.raises(SpecError, match="duplicate"):
        build_plan(_spec(arms=arms), check_heads=False)


def test_data_parallel_arm_takes_its_gpus() -> None:
    spec = _spec(arms=[{"name": "dp4", "method": "baseline", "engine": {"data_parallel_size": 4}}])
    (arm,) = build_plan(spec, check_heads=False).arms
    assert arm.gpus == 4
    assert _flag(arm.serve_argv, "--data-parallel-size") == "4"


def test_bench_command_passes_generation_and_provenance(tmp_path: Path) -> None:
    spec = _spec(
        loadtests=[
            {
                "label": "chat-sampled",
                "workload": "chat",
                "concurrencies": [1, 32],
                "generation": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            }
        ]
    )
    plan = build_plan(spec, check_heads=False)
    arm = plan.arms[1]
    command = bench_command(arm, plan.loadtests[0], tmp_path / "out", "dep-1")
    assert command[3] == "loadtest"
    assert _flag(command, "--base-url") == f"http://127.0.0.1:{arm.port}"
    assert _flag(command, "--top-k") == "20"
    assert _flag(command, "--temperature") == "0.7"
    assert command[command.index("--concurrencies") + 1 : command.index("--concurrencies") + 3] == [
        "1",
        "32",
    ]
    assert f"dspark_checkpoint={DRAFTER}" in command
    assert "deployment_id=dep-1" in command


def test_probe_runs_a_single_cell(tmp_path: Path) -> None:
    plan = build_plan(_spec(probe=True), check_heads=False)
    assert plan.probe is not None
    command = bench_command(plan.arms[1], plan.probe, tmp_path, "dep")
    assert command[3] == "run"
    assert _flag(command, "--concurrency") == "8"
    assert _flag(command, "--max-tokens") == "384"


def test_unknown_generation_setting_is_refused() -> None:
    spec = _spec(
        loadtests=[{"workload": "chat", "concurrencies": [1], "generation": {"min_p": 0.1}}]
    )
    with pytest.raises(SpecError, match="unknown generation"):
        build_plan(spec, check_heads=False)
