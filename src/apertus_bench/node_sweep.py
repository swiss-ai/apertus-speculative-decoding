"""Several deployments side by side on one node, each load-tested at the same time.

A sweep spec (YAML) lists arms: plain target, DSpark or EAGLE drafter, draft
length, engine settings. Each arm gets its own GPUs on the node and its own
`vllm serve` process; the load tests of all running arms proceed in parallel,
so a four-GPU node measures four configurations in the time of one. An arm
that needs more GPUs than are free waits for the earlier arms to finish.
A `command` arm runs one program on its GPUs instead (e.g. a step-time
profile); `{arm_dir}` and `{repo}` in its argv are filled in.

    python3 -m apertus_bench.node_sweep plan SPEC        (print the expanded arms)
    python3 -m apertus_bench.node_sweep run SPEC --output DIR

Runs inside the serving container (the job script of serving/node-sweep.sh),
where `vllm` is on PATH and the GPUs of the allocation are visible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_MODELS_ROOT = Path("/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai")
# Mirrors serving/stage-defaults.sh, so a sweep arm is served like a launcher arm.
STAGES: dict[str, dict[str, Any]] = {
    "8b": {
        "target_model": str(HF_MODELS_ROOT / "Apertus-v1.5-8B"),
        "served_base": "swiss-ai/Apertus-v1.5-8B",
        "max_model_len": 32768,
    },
    "70b": {
        "target_model": str(HF_MODELS_ROOT / "Apertus-v1.5-70B"),
        "served_base": "swiss-ai/Apertus-v1.5-70B",
        "max_model_len": 131072,
    },
}
DEFAULT_ENGINE: dict[str, Any] = {
    "gpu_memory_utilization": 0.8,
    "enable_prefix_caching": False,
}
ENGINE_FLAGS = {
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_num_seqs": "--max-num-seqs",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "data_parallel_size": "--data-parallel-size",
}
BOOL_FLAGS = {
    "enable_prefix_caching": ("--enable-prefix-caching", "--no-enable-prefix-caching"),
    "async_scheduling": ("--async-scheduling", "--no-async-scheduling"),
    "enforce_eager": ("--enforce-eager", None),
}
METHODS = {"baseline", "dspark", "eagle", "command"}
# Same lines serving/loadtest.sh keeps from the engine log.
EXCERPT = re.compile(
    r"non-default args|[Ss]peculative|Loading drafter|Model loading took|Available KV cache "
    r"memory|GPU KV cache size|Maximum concurrency|CUDA graph|Graph capturing|"
    r"gpu_memory_utilization|memory profiling|torch.compile took|padding layers"
)
BASE_PORT = 8100


class SpecError(ValueError):
    pass


@dataclass
class Loadtest:
    label: str
    workload: str
    concurrencies: list[int]
    workloads_file: Path
    requests_per_slot: int = 4
    min_requests: int = 32
    max_tokens: int | None = None
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass
class Arm:
    name: str
    index: int
    gpus: int
    served_model: str
    serve_argv: list[str]
    bench_argv: list[str]
    env: dict[str, str]
    metadata: dict[str, str]
    kind: str = "serve"

    @property
    def port(self) -> int:
        return BASE_PORT + self.index


@dataclass
class Plan:
    name: str
    arms: list[Arm]
    loadtests: list[Loadtest]
    probe: Loadtest | None
    node_gpus: int


def _engine_args(engine: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in engine.items():
        if value is None:
            continue
        if key in BOOL_FLAGS:
            on, off = BOOL_FLAGS[key]
            flag = on if value else off
            if flag:
                argv.append(flag)
        elif key in ENGINE_FLAGS:
            argv += [ENGINE_FLAGS[key], str(value)]
        else:
            raise SpecError(f"unknown engine setting {key!r}")
    return argv


def _speculative_config(arm: dict[str, Any], tp: int) -> dict[str, Any] | None:
    method = arm["method"]
    if method == "baseline":
        if arm.get("speculative"):
            raise SpecError(f"arm {arm['name']}: a baseline arm takes no speculative settings")
        return None
    drafter = arm.get("drafter")
    k = arm.get("k")
    if not drafter or not isinstance(k, int) or k < 1:
        raise SpecError(f"arm {arm['name']}: method {method} needs drafter and k >= 1")
    if method == "dspark" and k > 7:
        raise SpecError(f"arm {arm['name']}: a block-8 DSpark drafter drafts at most 7 tokens")
    config: dict[str, Any] = {
        "method": "dspark" if method == "dspark" else "eagle3",
        "model": str(drafter),
        "num_speculative_tokens": k,
    }
    if method == "eagle":
        config["draft_tensor_parallel_size"] = tp
    config.update(arm.get("speculative") or {})
    return config


def _eagle_metadata(head: Path, stage: str, served_model: str, tp: int) -> dict[str, str]:
    """The provenance the bench client requires for eagle3 cells (see serving/loadtest.sh)."""
    from apertus_bench.eagle import validate_eagle_head

    contract = REPO_ROOT / "targets" / stage / "contract.json"
    report = validate_eagle_head(head, expected_algorithm="eagle31", target_contract_path=contract)
    return {
        "eagle_head": str(head),
        "checkpoint_sha256": report["checkpoint_manifest_sha256"],
        "target_model": served_model,
        "target_revision": str((report.get("provenance") or {}).get("revision")),
        "target_tensor_parallel_size": str(tp),
    }


def build_plan(spec: dict[str, Any], *, check_heads: bool = True) -> Plan:
    stage = spec.get("stage", "8b")
    if stage not in STAGES:
        raise SpecError(f"stage must be one of {sorted(STAGES)}")
    defaults = STAGES[stage]
    target_model = spec.get("target_model", defaults["target_model"])
    served_base = spec.get("served_base", defaults["served_base"])
    node_gpus = int(spec.get("node_gpus", 4))
    common_engine = {**DEFAULT_ENGINE, "max_model_len": defaults["max_model_len"]}
    common_engine.update(spec.get("engine") or {})
    workloads_file = Path(
        spec.get("workloads_file", REPO_ROOT / "workloads" / stage / "eagle-8b-test.jsonl")
    )

    loadtests = []
    for entry in spec.get("loadtests") or []:
        generation = dict(entry.get("generation") or {})
        unknown = set(generation) - {"temperature", "top_p", "top_k", "seed", "ignore_eos"}
        if unknown:
            raise SpecError(f"unknown generation settings {sorted(unknown)}")
        loadtests.append(
            Loadtest(
                label=entry.get("label", entry["workload"]),
                workload=entry["workload"],
                concurrencies=[int(c) for c in entry["concurrencies"]],
                workloads_file=Path(entry.get("workloads_file", workloads_file)),
                requests_per_slot=int(entry.get("requests_per_slot", 4)),
                min_requests=int(entry.get("min_requests", 32)),
                max_tokens=entry.get("max_tokens"),
                generation=generation,
            )
        )
    labels = [test.label for test in loadtests]
    if len(set(labels)) != len(labels):
        raise SpecError(f"loadtest labels must be unique: {labels}")
    if not loadtests:
        raise SpecError("spec has no loadtests")

    probe = None
    if spec.get("probe"):
        probe = Loadtest(
            label="probe",
            workload="probe",
            concurrencies=[8],
            workloads_file=REPO_ROOT / "workloads" / stage / "probe-speculator-benchmarks.jsonl",
            min_requests=64,
            requests_per_slot=1,
            max_tokens=384,
        )

    arms: list[Arm] = []
    names: set[str] = set()
    for index, raw in enumerate(spec.get("arms") or []):
        name = raw.get("name")
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise SpecError(f"arm {index}: name must be [A-Za-z0-9._-]+, got {name!r}")
        if name in names:
            raise SpecError(f"duplicate arm name {name!r}")
        names.add(name)
        method = raw.get("method")
        if method not in METHODS:
            raise SpecError(f"arm {name}: method must be one of {sorted(METHODS)}")
        env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
        if method == "command":
            command = raw.get("command")
            if not command or not isinstance(command, list):
                raise SpecError(f"arm {name}: a command arm needs a command list")
            gpus = int(raw.get("gpus", 1))
            if not 0 < gpus <= node_gpus:
                raise SpecError(f"arm {name}: needs 1 to {node_gpus} GPUs, given {gpus}")
            arms.append(
                Arm(
                    name=name,
                    index=index,
                    gpus=gpus,
                    served_model="",
                    serve_argv=[str(part) for part in command],
                    bench_argv=[],
                    env=env,
                    metadata={},
                    kind="command",
                )
            )
            continue
        engine = {**common_engine, **(raw.get("engine") or {})}
        tp = int(raw.get("tensor_parallel_size", 1))
        dp = int(engine.get("data_parallel_size") or 1)
        gpus = int(raw.get("gpus", tp * dp))
        if gpus < tp * dp or gpus > node_gpus:
            raise SpecError(f"arm {name}: needs {tp * dp} GPUs, given {gpus} of {node_gpus}")
        served_model = f"{served_base}-{name}"
        speculative = _speculative_config(raw, tp)
        serve_argv = [
            "vllm", "serve",
            "--model", target_model,
            "--served-model-name", served_model,
            "--host", "127.0.0.1",
            "--chat-template-content-format", "string",
            "--tensor-parallel-size", str(tp),
            "--enable-auto-tool-choice",
            "--tool-call-parser", "apertus",
            "--compilation-config.pass_config.fuse_allreduce_rms", "false",
            *_engine_args(engine),
        ]  # fmt: skip
        if speculative is not None:
            serve_argv += ["--speculative-config", json.dumps(speculative, sort_keys=True)]
        serve_argv += [str(arg) for arg in raw.get("vllm_args") or []]

        metadata = {
            "sweep_arm": name,
            "max_num_batched_tokens": str(engine.get("max_num_batched_tokens", "engine-default")),
            "max_num_seqs": str(engine.get("max_num_seqs", "engine-default")),
            "speculative_config": json.dumps(speculative, sort_keys=True) if speculative else "",
            "engine": json.dumps(engine, sort_keys=True),
            "extra_vllm_args": " ".join(str(arg) for arg in raw.get("vllm_args") or []),
            "load_generator": "replica node (same container step)",
        }
        if method == "baseline":
            bench = ["--method", "none"]
        elif method == "dspark":
            bench = ["--method", "dspark", "--num-speculative-tokens", str(raw["k"])]
            metadata["dspark_checkpoint"] = str(raw["drafter"])
        else:
            bench = [
                "--method", "eagle3", "--algorithm", "eagle31",
                "--num-speculative-tokens", str(raw["k"]),
                "--draft-tensor-parallel-size", str(tp),
            ]  # fmt: skip
            if check_heads:
                metadata.update(_eagle_metadata(Path(raw["drafter"]), stage, served_base, tp))
        arms.append(
            Arm(
                name=name,
                index=index,
                gpus=gpus,
                served_model=served_model,
                serve_argv=serve_argv,
                bench_argv=bench,
                env=env,
                metadata=metadata,
            )
        )
    if not arms:
        raise SpecError("spec has no arms")
    return Plan(
        name=spec.get("name", "sweep"),
        arms=arms,
        loadtests=loadtests,
        probe=probe,
        node_gpus=node_gpus,
    )


def bench_command(arm: Arm, test: Loadtest, output: Path, deployment_id: str) -> list[str]:
    base = [
        sys.executable, "-m", "apertus_bench",
        "loadtest" if test.label != "probe" else "run",
        "--base-url", f"http://127.0.0.1:{arm.port}",
        "--model", arm.served_model,
        "--workloads", str(test.workloads_file),
        "--workload", test.workload,
        "--variant", arm.name,
        "--metadata", f"deployment_id={deployment_id}",
    ]  # fmt: skip
    if test.label == "probe":
        base += ["--concurrency", "8", "--requests", str(test.min_requests)]
    else:
        base += ["--concurrencies", *[str(c) for c in test.concurrencies]]
        base += ["--requests", str(test.min_requests)]
        base += ["--requests-per-slot", str(test.requests_per_slot), "--metrics-interval", "1"]
    if test.max_tokens:
        base += ["--max-tokens", str(test.max_tokens)]
    for key, value in test.generation.items():
        if key == "ignore_eos":
            if value:
                base.append("--ignore-eos")
            continue
        base += [f"--{key.replace('_', '-')}", str(value)]
    base += arm.bench_argv
    for key, value in arm.metadata.items():
        if value:  # the client refuses empty values
            base += ["--metadata", f"{key}={value}"]
    base += ["--output", str(output)]
    return base


async def _wait_ready(arm: Arm, process: asyncio.subprocess.Process, timeout: float) -> float:
    started = time.monotonic()
    url = f"http://127.0.0.1:{arm.port}"
    async with httpx.AsyncClient(timeout=30) as client:
        while time.monotonic() - started < timeout:
            if process.returncode is not None:
                raise RuntimeError(f"vllm serve exited with {process.returncode}")
            try:
                response = await client.get(f"{url}/v1/models")
                if response.status_code == 200:
                    probe = await client.post(
                        f"{url}/v1/chat/completions",
                        json={
                            "model": arm.served_model,
                            "messages": [{"role": "user", "content": "Say hi."}],
                            "max_tokens": 4,
                        },
                    )
                    if probe.status_code == 200:
                        return time.monotonic() - started
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError(f"{arm.name} not ready after {timeout:.0f} s")


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    for sig, grace in ((signal.SIGINT, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), grace)
            return
        except TimeoutError:
            continue


def _pythonpath() -> str:
    """The repo's src ahead of the inherited path, which keeps the image's
    /workspace/vllm entry (the patched vLLM) in place."""
    inherited = os.environ.get("PYTHONPATH", "")
    src = str(REPO_ROOT / "src")
    return src if not inherited else f"{src}:{inherited}"


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


async def run_arm(
    arm: Arm, gpu_ids: list[int], plan: Plan, output: Path, deployment_prefix: str
) -> dict[str, Any]:
    arm_dir = output / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    deployment_id = f"{deployment_prefix}-{arm.name}"
    status: dict[str, Any] = {
        "arm": arm.name,
        "gpus": gpu_ids,
        "port": arm.port,
        "serve_argv": arm.serve_argv,
        "env": arm.env,
        "deployment_id": deployment_id,
        "tests": {},
    }
    status_path = arm_dir / "status.json"
    cache = Path(os.environ.get("SWEEP_CACHE_ROOT", output / "vllm-cache")) / arm.name
    cache.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
        "VLLM_CACHE_ROOT": str(cache),
        **arm.env,
    }
    status["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_status(status_path, status)
    if arm.kind == "command":
        argv = [
            part.replace("{arm_dir}", str(arm_dir)).replace("{repo}", str(REPO_ROOT))
            for part in arm.serve_argv
        ]
        with (arm_dir / "command.log").open("wb") as command_log:
            command = await asyncio.create_subprocess_exec(
                *argv,
                stdout=command_log,
                stderr=asyncio.subprocess.STDOUT,
                env={**env, "PYTHONPATH": _pythonpath()},
            )
            code = await command.wait()
        status["exit_code"] = code
        status["status"] = "ok" if code == 0 else "failed"
        status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_status(status_path, status)
        return status
    serve_argv = [*arm.serve_argv, "--port", str(arm.port)]
    log = (arm_dir / "replica.out").open("wb")
    process = await asyncio.create_subprocess_exec(
        *serve_argv, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True
    )
    try:
        status["ready_seconds"] = round(await _wait_ready(arm, process, 2400), 1)
        _write_status(status_path, status)
        tests = ([plan.probe] if plan.probe else []) + plan.loadtests
        for test in tests:
            target = arm_dir / test.label
            command = bench_command(arm, test, target, deployment_id)
            with (arm_dir / f"{test.label}.log").open("wb") as bench_log:
                bench = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=bench_log,
                    stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": _pythonpath()},
                )
                code = await bench.wait()
            status["tests"][test.label] = {"exit_code": code}
            _write_status(status_path, status)
            if process.returncode is not None:
                raise RuntimeError(f"vllm serve exited with {process.returncode} during tests")
        status["status"] = (
            "ok" if all(t["exit_code"] == 0 for t in status["tests"].values()) else "test_failed"
        )
    except Exception as error:  # noqa: BLE001 - recorded, the other arms go on
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        await _stop(process)
        log.close()
        status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        replica = arm_dir / "replica.out"
        if replica.is_file():
            lines = replica.read_text(errors="replace").splitlines()
            excerpt = [line for line in lines if EXCERPT.search(line)]
            (arm_dir / "engine-excerpt.txt").write_text("\n".join(excerpt) + "\n")
        _write_status(status_path, status)
    return status


async def run_plan(plan: Plan, output: Path) -> list[dict[str, Any]]:
    """Start arms in spec order as soon as enough GPUs are free."""
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{plan.name}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    free = list(range(plan.node_gpus))
    released = asyncio.Condition()
    results: list[dict[str, Any]] = []

    async def one(arm: Arm) -> None:
        async with released:
            await released.wait_for(lambda: len(free) >= arm.gpus)
            gpu_ids = [free.pop(0) for _ in range(arm.gpus)]
        try:
            results.append(await run_arm(arm, gpu_ids, plan, output, prefix))
        finally:
            async with released:
                free.extend(gpu_ids)
                free.sort()
                released.notify_all()

    # Arms start in order: each waits on the condition until its GPUs are free.
    tasks = []
    for arm in plan.arms:
        tasks.append(asyncio.create_task(one(arm)))
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    summary = {arm["arm"]: arm.get("status") for arm in results}
    (output / "sweep-status.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus_bench.node_sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan", help="print the expanded arms of a spec")
    plan_cmd.add_argument("spec", type=Path)
    plan_cmd.add_argument("--no-check-heads", action="store_true")
    run_cmd = sub.add_parser("run", help="run a spec on this node")
    run_cmd.add_argument("spec", type=Path)
    run_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = yaml.safe_load(args.spec.read_text())
    if args.command == "plan":
        plan = build_plan(spec, check_heads=not args.no_check_heads)
        for arm in plan.arms:
            print(json.dumps({"name": arm.name, "gpus": arm.gpus, "serve": arm.serve_argv}))
        return
    plan = build_plan(spec)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "spec.yaml").write_text(args.spec.read_text())
    results = asyncio.run(run_plan(plan, args.output))
    failed = [r["arm"] for r in results if r.get("status") != "ok"]
    print(json.dumps({"output": str(args.output), "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
