"""Native-effort experiments: immutable inputs, sequential deployments and gated reports.

Run inside the serving container via serving/thinking-effort.sh. Only Phase 1
uses a released head. Phase 2 accepts four matched, validated training outputs;
the training specification is emitted only after a complete Phase 1 proceed gate.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from apertus_bench.client import RequestMeasurement, StreamingChatClient
from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta
from apertus_bench.runner import CellSettings, GenerationSettings, Variant, summarize
from apertus_bench.stats import distribution
from apertus_bench.workloads import Prompt, file_sha256, load_prompts

EFFORTS = ("low", "medium", "high")
ROLES = ("low", "medium", "high", "mixed")
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "experiments/thinking-effort/gpt-oss-20b.yaml"
PREFLIGHT = (
    "What is 17 multiplied by 23? Give the answer briefly.",
    "Write a Python function square(x) that returns x*x.",
    "What is the sum of the integers from 1 to 10? Give the answer briefly.",
    "Which is larger, 0.9 or 0.11? Answer briefly.",
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("expected schema_version: 1 experiment mapping")
    default = yaml.safe_load(SPEC_PATH.read_text())
    unknown = set(spec) - set(default)
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
    spec = {**default, **spec}
    if spec["efforts"] != list(EFFORTS) or spec["repeats"] != 3:
        raise ValueError("the experiment requires low/medium/high and three deployment blocks")
    for key in ("expected_prompts", "max_tokens", "max_model_len", "warmup_requests"):
        if not isinstance(spec[key], int) or spec[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if spec["num_speculative_tokens"] != 7 or spec["max_num_seqs"] != 1:
        raise ValueError("the controlled experiment fixes k=7 and concurrency=1")
    if spec["max_tokens"] >= spec["max_model_len"]:
        raise ValueError("output cap must leave room for the prompt")
    datetime.strptime(spec["fixed_date"], "%Y-%m-%d")
    for key in ("acceptance_change_threshold", "max_cap_fraction", "specialist_latency_threshold"):
        if not 0 < spec[key] < 1:
            raise ValueError(f"{key} must be between zero and one")
    return spec


def prepare_workload(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Keep the versioned probe's task content/IDs, without Apertus framing."""
    source = Path(spec["source_workload"])
    if not source.is_absolute():
        source = REPO / source
    if file_sha256(source) != spec["source_sha256"]:
        raise ValueError("source probe digest differs from the declared manifest")
    prompts = load_prompts(source)
    if len(prompts) != spec["expected_prompts"]:
        raise ValueError("source probe has an unexpected number of prompts")
    output.parent.mkdir(parents=True, exist_ok=True)
    strata: Counter[str] = Counter()
    removed: dict[str, list[dict[str, str]]] = {}
    with output.open("x", encoding="utf-8") as sink:
        for prompt in prompts:
            if prompt.id.startswith("probe-math-"):
                stratum = "math"
            elif prompt.id.startswith("probe-humaneval-"):
                stratum = "code"
            else:
                raise ValueError(f"unrecognized probe stratum: {prompt.id}")
            messages = [m for m in prompt.messages if m["role"] not in {"system", "developer"}]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError(f"probe {prompt.id} must end in an unchanged user task")
            removed[prompt.id] = [m for m in prompt.messages if m not in messages]
            strata[stratum] += 1
            sink.write(
                json.dumps(
                    {
                        "id": prompt.id,
                        "workload": stratum,
                        "messages": messages,
                        "max_tokens": spec["max_tokens"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if strata != {"math": 32, "code": 32}:
        raise ValueError(f"expected the original 32 math / 32 code probe, got {dict(strata)}")
    return {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "sha256": file_sha256(output),
        "rows": len(prompts),
        "strata": dict(strata),
        "removed_framing": removed,
        "prompt_ids": [p.id for p in prompts],
    }


def model_manifest(root: Path) -> dict[str, Any]:
    if not (root / "config.json").is_file():
        raise ValueError(f"missing model config: {root}")
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.suffix
        in {
            ".json",
            ".safetensors",
            ".jinja",
            ".py",
            ".model",
            ".tiktoken",
        }
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"no safetensors weights in {root}")
    manifest = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    # huggingface_hub local_dir downloads retain the original commit per file.
    revisions = set()
    for p in files:
        meta = root / ".cache/huggingface/download" / (p.name + ".metadata")
        if meta.is_file():
            candidate = meta.read_text().splitlines()[0]
            if re.fullmatch(r"[a-f0-9]{40}", candidate):
                revisions.add(candidate)
    return {
        "path": str(root.resolve()),
        "files": manifest,
        "sha256": digest,
        "revision": next(iter(revisions)) if len(revisions) == 1 else f"sha256:{digest}",
    }


def prepare_inputs(spec: dict[str, Any], campaign: Path) -> dict[str, Any]:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _prepare_inputs(spec, campaign)


def source_identity() -> dict[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "serving", "experiments"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"revision": revision, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


def stage_harmony_vocab(spec: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Populate Harmony's Rust cache with the pinned vocabulary, without network."""
    source = Path(spec["harmony_vocab_path"])
    if file_sha256(source) != spec["harmony_vocab_sha256"]:
        raise ValueError("Harmony vocabulary differs from its published hash")
    directory.mkdir(parents=True, exist_ok=False)
    cache_key = hashlib.sha1(spec["harmony_vocab_url"].encode()).hexdigest()
    cached = directory / cache_key
    cached.write_bytes(source.read_bytes())
    return {
        "source": str(source),
        "url": spec["harmony_vocab_url"],
        "cache_path": str(cached),
        "sha256": file_sha256(cached),
    }


def _prepare_inputs(spec: dict[str, Any], campaign: Path) -> dict[str, Any]:
    """Expensive hashing belongs in the allocation, never on a login node."""
    inputs = campaign / "inputs"
    if inputs.exists():
        lock = json.loads((inputs / "lock.json").read_text())
        if lock["spec"] != spec:
            raise ValueError("campaign configuration differs from its immutable input lock")
        if lock["source"] != source_identity():
            raise ValueError("experiment source changed; use a new campaign directory")
        if file_sha256(campaign / "inputs/prompts.jsonl") != lock["workload"]["sha256"]:
            raise ValueError("prepared prompt corpus changed after input locking")
        for key in ("target", "drafter"):
            if model_manifest(Path(lock[key]["path"]))["sha256"] != lock[key]["sha256"]:
                raise ValueError(f"{key} snapshot changed since the campaign was locked")
        if file_sha256(Path(spec["image"])) != lock["image_sha256"]:
            raise ValueError("container changed since the campaign was locked")
        if file_sha256(Path(lock["harmony_vocab"]["cache_path"])) != spec["harmony_vocab_sha256"]:
            raise ValueError("Harmony vocabulary changed since the campaign was locked")
        return lock
    inputs.mkdir(parents=True, exist_ok=False)
    target = model_manifest(Path(spec["target_path"]))
    drafter = model_manifest(Path(spec["drafter_path"]))
    config = json.loads((Path(spec["drafter_path"]) / "config.json").read_text())
    if "dflash" not in json.dumps(config.get("architectures", [])).lower():
        raise ValueError("drafter is not a DFlash checkpoint")
    template_path = Path(spec["target_path"]) / "chat_template.jinja"
    (inputs / "target-chat-template.jinja").write_text(template_path.read_text())
    lock = {
        "schema_version": 1,
        "spec": spec,
        "source": source_identity(),
        "target": target,
        "drafter": drafter,
        "harmony_vocab": stage_harmony_vocab(spec, inputs / "harmony-cache"),
        "image_sha256": file_sha256(Path(spec["image"])),
        "template_sha256": file_sha256(inputs / "target-chat-template.jinja"),
        "rendering": {
            "backend": "vllm-native-harmony",
            "fixed_date": spec["fixed_date"],
            "overlay_sha256": file_sha256(Path(__file__).with_name("effort_server.py")),
        },
        "workload": prepare_workload(spec, inputs / "prompts.jsonl"),
        "channel_token_counts": "reencoded channel text; excludes Harmony boundary tokens",
    }
    write_json(inputs / "drafter-config.json", config)
    write_json(
        inputs / "target-config.json",
        json.loads((Path(spec["target_path"]) / "config.json").read_text()),
    )
    write_json(inputs / "lock.json", lock)
    return lock


def serve_command(spec: dict[str, Any], head: Path | None, *, provenance: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "apertus_bench.effort_server",
        "--fixed-date",
        spec["fixed_date"],
        "--provenance-file",
        str(provenance),
        "--",
        "--model",
        spec["target_path"],
        "--served-model-name",
        "gpt-oss-20b",
        "--host",
        "127.0.0.1",
        "--port",
        str(spec["port"]),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(spec["max_model_len"]),
        "--max-num-seqs",
        str(spec["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(spec["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(spec["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--generation-config",
        "vllm",
        "--middleware",
        "apertus_bench.effort_server.NativeRenderMiddleware",
    ]
    if head is not None:
        command += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dflash",
                    "model": str(head),
                    "num_speculative_tokens": spec["num_speculative_tokens"],
                },
                sort_keys=True,
            ),
        ]
    return command


def schedule(block: int, *, phase: str = "phase1") -> tuple[list[str], list[str]]:
    if block not in range(3):
        raise ValueError("block must be 0, 1 or 2")
    arms = ["plain", "dflash"] if phase == "phase1" else ["plain", *ROLES]
    if phase == "phase1":
        arms = arms if block % 2 == 0 else list(reversed(arms))
    else:
        arms = arms[block:] + arms[:block]
    efforts = list(EFFORTS[block:] + EFFORTS[:block])
    return arms, efforts


async def metrics(client: StreamingChatClient) -> str:
    response = await client.http.get(client.base_url + "/metrics")
    response.raise_for_status()
    return response.text


async def wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            try:
                response = await client.get(base_url + "/v1/models")
                if response.status_code == 200 and any(
                    m["id"] == "gpt-oss-20b" for m in response.json().get("data", [])
                ):
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError("server registration timed out")


async def render_check(
    client: StreamingChatClient,
    tokenizer: Any,
    effort: str,
    prompt: Prompt,
    directory: Path,
    spec: dict[str, Any],
) -> list[int]:
    payload = {
        "model": client.model,
        "messages": prompt.messages,
        "reasoning_effort": effort,
        "add_generation_prompt": True,
    }
    response = await client.http.post(
        client.base_url + "/v1/chat/completions/render",
        json=payload,
    )
    response.raise_for_status()
    tokens = response.json()["token_ids"]
    rendered = tokenizer.decode(tokens, skip_special_tokens=False)
    if f"Reasoning: {effort}\n" not in rendered:
        raise ValueError(f"server did not render native {effort} effort")
    if spec["fixed_date"] not in rendered:
        raise ValueError("server ignored the fixed-date template")
    if len(tokens) + spec["max_tokens"] > spec["max_model_len"]:
        raise ValueError(f"prompt {prompt.id} plus output cap exceeds context")
    write_json(
        directory / f"{effort}-{prompt.id}.json",
        {
            "request": payload,
            "token_ids": tokens,
            "rendered": rendered,
            "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        },
    )
    return tokens


def validity(
    measurements: list[RequestMeasurement],
    expected: int,
    spec_metrics: dict[str, Any],
    *,
    speculative: bool,
    max_cap_fraction: float,
) -> dict[str, Any]:
    reasons = []
    if len(measurements) != expected:
        reasons.append("incomplete request set")
    if any(not m.success for m in measurements):
        reasons.append("request errors or missing token usage")
    if any(not m.final_output or not m.final_output.strip() for m in measurements):
        reasons.append("missing final answer")
    if any(m.finish_reason not in {"stop", "length"} for m in measurements):
        reasons.append("unexpected termination")
    cap_fraction = sum(m.finish_reason == "length" for m in measurements) / max(expected, 1)
    if cap_fraction > max_cap_fraction:
        reasons.append("output-cap fraction exceeds declared limit")
    if speculative and (not spec_metrics.get("enabled") or not spec_metrics.get("drafts", 0)):
        reasons.append("speculative counters missing or no verification rounds")
    return {"valid": not reasons, "reasons": reasons, "cap_fraction": cap_fraction}


async def measure_cell(
    client: StreamingChatClient,
    prompts: list[Prompt],
    output: Path,
    spec: dict[str, Any],
    effort: str,
    block: int,
    metadata: dict[str, Any],
    *,
    speculative: bool,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics").mkdir()
    generation = GenerationSettings(
        temperature=spec["temperature"],
        top_p=spec["top_p"],
        top_k=spec["top_k"],
        min_p=spec["min_p"],
        reasoning_effort=effort,
        seed=block,
        max_tokens=spec["max_tokens"],
    )
    settings = CellSettings("probe", 1, len(prompts), spec["warmup_requests"], block, generation)
    write_json(
        output / "metadata.json",
        {
            **metadata,
            "schema_version": 1,
            "cell": asdict(settings),
            "started_at": datetime.now(UTC).isoformat(),
            "prompt_ids": [p.id for p in prompts],
        },
    )
    write_json(output / "status.json", {"status": "running"})
    measurements: list[RequestMeasurement] = []
    try:
        for index in range(spec["warmup_requests"]):
            warm = Prompt(
                f"warmup-{index}",
                "warmup",
                [
                    {
                        "role": "user",
                        "content": PREFLIGHT[index % len(PREFLIGHT)],
                    }
                ],
                spec["max_tokens"],
            )
            result = await client.request(
                warm,
                index,
                **asdict(generation),
                store_output=True,
            )
            if not result.success or not result.final_output or result.finish_reason != "stop":
                raise RuntimeError(
                    f"warmup failed at {effort}: {result.error or result.finish_reason}"
                )
        raw_before = await metrics(client)
        (output / "metrics_before.prom").write_text(raw_before)
        first = parse_speculative_snapshot(raw_before, model=client.model)
        previous = first
        active_seconds = 0.0
        with (output / "requests.jsonl").open("x", encoding="utf-8") as journal:
            for index, prompt in enumerate(prompts):
                raw = await metrics(client)
                before = parse_speculative_snapshot(raw, model=client.model)
                (output / "metrics" / f"{index:03d}-before.prom").write_text(raw)
                # Also detect a reset in the gap between consecutive requests.
                speculative_delta(previous, before)
                kwargs = asdict(generation)
                kwargs["seed"] = block + index
                result = await client.request(
                    prompt,
                    index,
                    **kwargs,
                    store_output=True,
                    store_stream=True,
                )
                measurements.append(result)
                active_seconds += result.e2e_ms / 1000
                # Preserve the response even if the following scrape fails.
                record = result.to_dict()
                try:
                    raw = await metrics(client)
                    after = parse_speculative_snapshot(raw, model=client.model)
                    (output / "metrics" / f"{index:03d}-after.prom").write_text(raw)
                    record["speculative_decoding"] = speculative_delta(before, after)
                finally:
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
                previous = after
                if not result.success:
                    raise RuntimeError(f"request {prompt.id}: {result.error}")
        raw_after = await metrics(client)
        (output / "metrics_after.prom").write_text(raw_after)
        totals = speculative_delta(first, parse_speculative_snapshot(raw_after, model=client.model))
        summary = summarize(measurements, active_seconds, totals)
        summary["timing_basis"] = "sum of request durations; excludes counter scrapes"
        summary["channel_tokens"] = {
            "reasoning": distribution(
                m.reasoning_tokens for m in measurements if m.reasoning_tokens is not None
            ),
            "final": distribution(
                m.final_tokens for m in measurements if m.final_tokens is not None
            ),
            "source": "tokenizer_reencoding; Harmony boundaries are excluded",
        }
        summary["validity"] = validity(
            measurements,
            len(prompts),
            totals,
            speculative=speculative,
            max_cap_fraction=spec["max_cap_fraction"],
        )
        write_json(output / "summary.json", summary)
        write_json(
            output / "status.json",
            {
                "status": "complete" if summary["validity"]["valid"] else "invalid",
                "validity": summary["validity"],
            },
        )
        if not summary["validity"]["valid"]:
            raise ValueError(f"invalid {effort} cell: {summary['validity']['reasons']}")
        return summary
    except BaseException as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "recorded_requests": len(measurements),
            },
        )
        raise


def runtime_provenance() -> dict[str, Any]:
    packages = {}
    for name in ("vllm", "torch", "transformers", "httpx", "huggingface-hub", "openai-harmony"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "harness_git_revision": revision,
        "packages": packages,
        "python": platform.python_version(),
        "node": platform.node(),
        "gpus": gpu,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "gpu_index": "0",
        "slurm_node_id": os.getenv("SLURMD_NODENAME", platform.node()),
    }


def check_training_heads(path: Path, target_sha256: str) -> dict[str, dict[str, Any]]:
    """Require matched training evidence, not four arbitrarily named checkpoints."""
    heads = json.loads(path.read_text())
    if set(heads) != set(ROLES):
        raise ValueError("Phase 2 requires low, medium, high and mixed training outputs")
    matched = []
    for role, head in heads.items():
        provenance = json.loads(Path(head["provenance"]).read_text())
        if provenance["target_sha256"] != target_sha256:
            raise ValueError(f"{role} was trained against a different target")
        if provenance["effort"] != role:
            raise ValueError(f"{role} training-data effort does not match its role")
        fields = (
            "initialization_sha256",
            "architecture_sha256",
            "optimizer_schedule_sha256",
            "supervised_tokens",
            "training_seed",
            "train_prompt_ids_sha256",
            "development_prompt_ids_sha256",
            "evaluation_prompt_ids_sha256",
        )
        matched.append(tuple(provenance[f] for f in fields))
        manifest = model_manifest(Path(head["path"]))
        if manifest["sha256"] != provenance["export_sha256"]:
            raise ValueError(f"{role} exported checkpoint differs from training provenance")
        head["manifest"] = manifest
        head["training_provenance"] = provenance
    if len(set(matched)) != 1:
        raise ValueError("Phase 2 heads do not have matched training budgets and splits")
    if len({h["manifest"]["sha256"] for h in heads.values()}) != 4:
        raise ValueError("Phase 2 requires four distinct trained checkpoints")
    return heads


async def run_block(
    spec: dict[str, Any],
    campaign: Path,
    block: int,
    *,
    phase: str = "phase1",
    preflight_only: bool = False,
    heads_path: Path | None = None,
    phase1_reference: Path | None = None,
) -> None:
    # Transformers is needed only in the serving container, not by report/plan.
    from transformers import AutoTokenizer

    lock = prepare_inputs(spec, campaign)
    phase_dir = campaign / phase
    phase_dir.mkdir(exist_ok=True)
    if phase == "phase2":
        from apertus_bench.effort_report import build_report

        parent = phase1_reference or campaign
        if parent != campaign:
            from apertus_bench.effort_training import validate_parent

            validate_parent(campaign, parent)
        phase1 = build_report(parent, phase="phase1")
        if phase1["decision"] != "proceed":
            raise ValueError("Phase 2 is disabled without a complete Phase 1 proceed decision")
        if heads_path is None:
            raise ValueError("Phase 2 requires --heads with matched training provenance")
        heads = check_training_heads(heads_path, lock["target"]["sha256"])
        for role, head in heads.items():
            evidence = head["training_provenance"]
            if evidence["initialization_sha256"] != lock["drafter"]["sha256"]:
                raise ValueError(f"{role} did not start from the Phase 1 released head")
            if (
                evidence["evaluation_prompt_ids_sha256"]
                != hashlib.sha256(
                    json.dumps(lock["workload"]["prompt_ids"], sort_keys=True).encode()
                ).hexdigest()
            ):
                raise ValueError(f"{role} training exclusion did not use these evaluation prompts")
    else:
        heads = {"dflash": {"path": spec["drafter_path"], "manifest": lock["drafter"]}}
    block_dir = phase_dir / f"block-{block}"
    block_dir.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["target_path"], local_files_only=True)
    prompts = load_prompts(campaign / "inputs/prompts.jsonl")
    runtime = runtime_provenance()
    write_json(block_dir / "runtime.json", runtime)
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "serving", "experiments"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (block_dir / "source.patch").write_text(patch)
    arm_names, efforts = schedule(block, phase=phase)
    write_json(block_dir / "schedule.json", {"arms": arm_names, "efforts": efforts, "seed": block})
    try:
        for name in arm_names:
            deployment = block_dir / name
            deployment.mkdir(exist_ok=False)
            deployment_id = f"{campaign.name}-{phase}-b{block}-{name}"
            head = None if name == "plain" else Path(heads[name]["path"])
            command = serve_command(
                spec,
                head,
                provenance=deployment / "harmony-overlay.json",
            )
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HF_HUB_OFFLINE": "1",
                    "VLLM_CACHE_ROOT": str(deployment / "compile-cache"),
                    "VLLM_LOGGING_LEVEL": "INFO",
                    "TIKTOKEN_RS_CACHE_DIR": str(campaign / "inputs/harmony-cache"),
                }
            )
            provenance = {
                **runtime,
                "phase": phase,
                "block": block,
                "variant": name,
                "deployment_id": deployment_id,
                "launch_argv": command,
                "environment": {
                    k: env[k]
                    for k in (
                        "CUDA_VISIBLE_DEVICES",
                        "HF_HUB_OFFLINE",
                        "VLLM_CACHE_ROOT",
                        "TIKTOKEN_RS_CACHE_DIR",
                    )
                },
                "image": spec["image"],
                "image_sha256": lock["image_sha256"],
                "input_lock": str(campaign / "inputs/lock.json"),
                "target_revision": lock["target"]["revision"],
                "target_sha256": lock["target"]["sha256"],
                "drafter": heads.get(name),
                "source_patch_sha256": file_sha256(block_dir / "source.patch"),
            }
            write_json(deployment / "provenance.json", provenance)
            process = None
            log = (deployment / "launch.log").open("x")
            try:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_json(deployment / "process.json", {"pid": process.pid})
                base = f"http://127.0.0.1:{spec['port']}"
                await wait_ready(base, process, spec["startup_timeout_seconds"])
                async with StreamingChatClient(
                    base,
                    "gpt-oss-20b",
                    None,
                    timeout_seconds=spec["request_timeout_seconds"],
                    token_counter=lambda text: len(
                        tokenizer.encode(text, add_special_tokens=False)
                    ),
                ) as client:
                    rendering = deployment / "rendered-prompts"
                    rendering.mkdir()
                    # These are the exact payloads and tokenizer route used by
                    # the live server, not just a local template demonstration.
                    for effort in efforts:
                        for prompt in prompts:
                            await render_check(client, tokenizer, effort, prompt, rendering, spec)
                    rendered_hash = hashlib.sha256(
                        "".join(file_sha256(p) for p in sorted(rendering.glob("*.json"))).encode()
                    ).hexdigest()
                    preflight_results = []
                    for effort in efforts:
                        for index, text in enumerate(PREFLIGHT):
                            result = await client.request(
                                Prompt(
                                    f"preflight-{index}",
                                    "preflight",
                                    [
                                        {
                                            "role": "user",
                                            "content": text,
                                        }
                                    ],
                                    spec["max_tokens"],
                                ),
                                index,
                                temperature=spec["temperature"],
                                top_p=spec["top_p"],
                                top_k=spec["top_k"],
                                min_p=spec["min_p"],
                                reasoning_effort=effort,
                                seed=block + index,
                                max_tokens=spec["max_tokens"],
                                ignore_eos=False,
                                store_output=True,
                                store_stream=True,
                            )
                            preflight_results.append({"effort": effort, **result.to_dict()})
                            write_json(deployment / "preflight.json", preflight_results)
                            if (
                                not result.success
                                or not result.final_output
                                or result.finish_reason != "stop"
                            ):
                                raise ValueError("preflight did not produce a natural final answer")
                    if not any(r["reasoning_output"] for r in preflight_results):
                        raise ValueError("no analysis output parsed in any preflight request")
                    check = parse_speculative_snapshot(await metrics(client), model=client.model)
                    if head is not None and (not check.enabled or not check.drafts):
                        raise ValueError("no DFlash verification rounds observed in preflight")
                    if not preflight_only:
                        for effort in efforts:
                            manifest = lock["target"]["files"]
                            metadata = {
                                "model": client.model,
                                "variant": asdict(
                                    Variant(
                                        name,
                                        "none" if head is None else "dflash",
                                        num_speculative_tokens=None if head is None else 7,
                                    )
                                ),
                                "workload_file_sha256": lock["workload"]["sha256"],
                                "extra": {
                                    "deployment_id": deployment_id,
                                    "block_id": str(block),
                                    "target_model": spec["target_model"],
                                    "target_revision": lock["target"]["revision"],
                                    "target_sha256": lock["target"]["sha256"],
                                    "target_tensor_parallel_size": "1",
                                    "tokenizer_sha256": manifest["tokenizer.json"]["sha256"],
                                    "template_sha256": lock["template_sha256"],
                                    "harmony_overlay_sha256": lock["rendering"]["overlay_sha256"],
                                    "harmony_vocab_sha256": lock["harmony_vocab"]["sha256"],
                                    "rendered_prompts_sha256": rendered_hash,
                                    "checkpoint_sha256": None
                                    if head is None
                                    else heads[name]["manifest"]["sha256"],
                                    "container_image": spec["image"],
                                    "container_sha256": lock["image_sha256"],
                                    "vllm_version": runtime["packages"]["vllm"],
                                    "slurm_job_id": runtime["slurm_job_id"],
                                    "slurm_node_id": runtime["slurm_node_id"],
                                    "gpu_identity": runtime["gpus"],
                                    "phase": phase,
                                    "source_patch_sha256": provenance["source_patch_sha256"],
                                    "harness_git_revision": runtime["harness_git_revision"],
                                },
                            }
                            await measure_cell(
                                client,
                                prompts,
                                deployment / "cells" / effort,
                                spec,
                                effort,
                                block,
                                metadata,
                                speculative=head is not None,
                            )
                write_json(
                    deployment / "status.json",
                    {
                        "status": "preflight_complete" if preflight_only else "complete",
                    },
                )
            except BaseException as error:
                write_json(
                    deployment / "status.json",
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise
            finally:
                if process is not None:
                    await stop_server(process)
                log.close()
        write_json(
            block_dir / "status.json",
            {
                "status": "preflight_complete" if preflight_only else "complete",
            },
        )
    except BaseException as error:
        write_json(
            block_dir / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


async def stop_server(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.to_thread(process.wait, timeout=60)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await asyncio.to_thread(process.wait, timeout=30)


def training_plan(
    campaign: Path,
    output: Path,
    train: Path,
    development: Path,
    supervised_tokens: int,
    optimizer: Path,
) -> dict[str, Any]:
    """Prepare the conditional training handoff without inventing a trainer API.

    The trainer must support DFlash + GPT-OSS and export the provenance schema used
    by check_training_heads. These files are inputs to an existing compatible recipe.
    """
    from apertus_bench.effort_report import build_report

    report = build_report(campaign)
    if report["decision"] != "proceed":
        raise ValueError("training is disabled without a complete Phase 1 proceed decision")
    if supervised_tokens <= 0:
        raise ValueError("an explicit positive per-head supervised-token budget is required")
    lock = json.loads((campaign / "inputs/lock.json").read_text())
    evaluation = load_prompts(campaign / "inputs/prompts.jsonl")
    train_prompts = load_prompts(train)
    dev_prompts = load_prompts(development)

    def keys(prompts: list[Prompt]) -> tuple[set[str], set[str]]:
        ids = {p.id for p in prompts}
        texts = {
            json.dumps(
                [
                    {"role": m["role"], "content": " ".join(m["content"].split())}
                    for m in p.messages
                    if m["role"] not in {"system", "developer"}
                ],
                sort_keys=True,
            )
            for p in prompts
        }
        if len(texts) != len(prompts):
            raise ValueError("duplicate task text in a training/development prompt set")
        return ids, texts

    keyed = [keys(ps) for ps in (evaluation, train_prompts, dev_prompts)]
    for left in range(3):
        for right in range(left + 1, 3):
            if keyed[left][0] & keyed[right][0] or keyed[left][1] & keyed[right][1]:
                raise ValueError("training, development and evaluation prompts overlap")
    schedule_config = json.loads(optimizer.read_text())
    plan = {
        "schema_version": 1,
        "phase1_decision": report,
        "status": "requires_compatible_training_recipe",
        "target": lock["target"],
        "initialization": lock["drafter"],
        "train_manifest": {"path": str(train.resolve()), "sha256": file_sha256(train)},
        "development_manifest": {
            "path": str(development.resolve()),
            "sha256": file_sha256(development),
        },
        "evaluation_manifest": lock["workload"],
        "optimizer_schedule": schedule_config,
        "optimizer_schedule_sha256": file_sha256(optimizer),
        "training_seed": 0,
        "heads": {
            role: {
                "efforts": list(EFFORTS) if role == "mixed" else [role],
                "supervised_tokens": supervised_tokens,
                "mixture_basis": "equal supervised tokens" if role == "mixed" else "single effort",
                "initialization_sha256": lock["drafter"]["sha256"],
            }
            for role in ROLES
        },
        "required_export_provenance": [
            "target_sha256",
            "effort",
            "initialization_sha256",
            "architecture_sha256",
            "optimizer_schedule_sha256",
            "supervised_tokens",
            "training_seed",
            "train_prompt_ids_sha256",
            "development_prompt_ids_sha256",
            "evaluation_prompt_ids_sha256",
            "export_sha256",
        ],
        "data_generation": {
            "target_frozen": True,
            "efforts": list(EFFORTS),
            "same_task_ids_per_effort": True,
            "preserve_analysis_and_final": True,
            "fixed_date": lock["spec"]["fixed_date"],
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training-plan.json", plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("plan", help="print the fixed schedule without running anything")
    run = commands.add_parser("run", help="run a block inside a Slurm GPU allocation")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--block", type=int, choices=range(3), required=True)
    run.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    run.add_argument("--heads", type=Path)
    run.add_argument("--phase1-reference", type=Path)
    run.add_argument("--preflight-only", action="store_true")
    report_parser = commands.add_parser("report", help="write a new immutable paired comparison")
    report_parser.add_argument("campaign", type=Path)
    report_parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    report_parser.add_argument("--output", type=Path)
    training = commands.add_parser("training-plan", help="gated, budgeted training handoff")
    training.add_argument("campaign", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--train-prompts", type=Path, required=True)
    training.add_argument("--development-prompts", type=Path, required=True)
    training.add_argument("--supervised-tokens", type=int, required=True)
    training.add_argument("--optimizer", type=Path, required=True)
    args = parser.parse_args()
    spec = read_spec(args.spec)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "spec": spec,
                    "phase1": {
                        "deployments": 6,
                        "cells": 18,
                        "requests": 18 * spec["expected_prompts"],
                        "blocks": [
                            dict(zip(("arms", "efforts"), schedule(b), strict=True))
                            for b in range(3)
                        ],
                    },
                    "phase2_conditional": {
                        "training_runs": 4,
                        "deployments": 15,
                        "cells": 45,
                        "requests": 45 * spec["expected_prompts"],
                    },
                },
                indent=2,
            )
        )
    elif args.action == "run":

        async def run_with_signals() -> None:
            task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signum, task.cancel)
            await run_block(
                spec,
                args.output.resolve(),
                args.block,
                phase=args.phase,
                preflight_only=args.preflight_only,
                heads_path=args.heads,
                phase1_reference=args.phase1_reference.resolve() if args.phase1_reference else None,
            )

        asyncio.run(run_with_signals())
    elif args.action == "report":
        from apertus_bench.effort_report import build_report, markdown_report

        report = build_report(args.campaign, phase=args.phase)
        output = args.output or args.campaign / "reports" / (
            args.phase + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "comparison.json", report)
        (output / "comparison.md").write_text(markdown_report(report))
        print(json.dumps({"decision": report["decision"], "output": str(output)}))
    elif args.action == "training-plan":
        training_plan(
            args.campaign,
            args.output,
            args.train_prompts,
            args.development_prompts,
            args.supervised_tokens,
            args.optimizer,
        )
        print(args.output / "training-plan.json")


if __name__ == "__main__":
    main()
