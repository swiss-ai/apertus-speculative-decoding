"""Matched effort fine-tuning using the existing Speculators DFlash trainer.

This module owns data, launch, export and provenance glue. The upstream trainer
owns the model, loss, backward pass, optimizer, validation and checkpoint selection.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from apertus_bench.effort import (
    EFFORTS,
    REPO,
    ROLES,
    model_manifest,
    prepare_inputs,
    read_spec,
    runtime_provenance,
    serve_command,
    source_identity,
    stop_server,
    training_plan,
    wait_ready,
    write_json,
)
from apertus_bench.effort_report import build_report
from apertus_bench.workloads import file_sha256, load_prompts

TRAINING_CONFIG = REPO / "experiments/thinking-effort/phase2.yaml"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("unsupported Phase 2 configuration")
    if config["supervised_tokens_per_head"] % (3 * config["epochs"]):
        raise ValueError("per-head budget must divide exactly into three efforts and epochs")
    if config["checkpoint_selection"] != "lowest_development_loss":
        raise ValueError("the frozen checkpoint rule is lowest development loss")
    return config


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalized_task(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        [
            {"role": m["role"], "content": " ".join(m["content"].split())}
            for m in messages
            if m["role"] not in {"system", "developer"}
        ],
        sort_keys=True,
    )


def select_prompts(
    domains: dict[str, list[dict[str, Any]]],
    evaluation: list,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split task IDs before generation, excluding normalized evaluation duplicates."""
    forbidden = {normalized_task(p.messages) for p in evaluation}
    forbidden_ids = {p.id for p in evaluation}
    seen = set(forbidden)
    train, development = [], []
    for domain in ("math", "code"):
        candidates = sorted(domains[domain], key=lambda p: sha_json([config["seed"], p["id"]]))
        selected = []
        total = config["train_prompts_per_domain"] + config["development_prompts_per_domain"]
        for prompt in candidates:
            text = normalized_task(prompt["messages"])
            if prompt["id"] in forbidden_ids or text in seen:
                continue
            seen.add(text)
            selected.append(prompt)
            if len(selected) == total:
                break
        if len(selected) != total:
            raise ValueError(f"insufficient disjoint {domain} tasks")
        boundary = config["train_prompts_per_domain"]
        train.extend(selected[:boundary])
        development.extend(selected[boundary:])
    return train, development


def budget_examples(
    samples: list[dict[str, Any]],
    role: str,
    per_epoch: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Set an exact assistant-token budget; retain full causal contexts.

    Whole sequences may recur when a bounded corpus is smaller than the budget.
    Only the final example's loss mask is shortened; its token sequence is intact.
    """
    if role not in ROLES or per_epoch <= 0 or (role == "mixed" and per_epoch % 3):
        raise ValueError("invalid role or token budget")
    efforts = EFFORTS if role == "mixed" else (role,)
    remaining = dict.fromkeys(efforts, per_epoch // len(efforts))
    pools = {}
    for effort in efforts:
        pool = sorted([s for s in samples if s["effort"] == effort], key=lambda s: s["id"])
        random.Random(seed).shuffle(pool)
        if not pool or any(len(s["token_ids"]) <= s["prompt_tokens"] for s in pool):
            raise ValueError(f"missing assistant tokens for {effort}")
        pools[effort] = pool
    offsets = dict.fromkeys(efforts, 0)
    rows = []
    while any(remaining.values()):
        for effort in efforts:
            if not remaining[effort]:
                continue
            sample = pools[effort][offsets[effort] % len(pools[effort])]
            offsets[effort] += 1
            start = sample["prompt_tokens"]
            available = len(sample["token_ids"]) - start
            supervised = min(available, remaining[effort])
            rows.append(
                {
                    "source_id": sample["id"],
                    "effort": effort,
                    "input_ids": sample["token_ids"],
                    "seq_len": len(sample["token_ids"]),
                    "loss_mask": [0] * start + [1] * supervised + [0] * (available - supervised),
                    "hidden_states_path": sample["hidden_states_path"],
                }
            )
            remaining[effort] -= supervised
    return rows


def validate_parent(campaign: Path, parent: Path) -> dict[str, Any]:
    report = build_report(parent, phase="phase1")
    if report["decision"] != "proceed":
        raise ValueError("Phase 2 requires a complete valid Phase 1 proceed decision")
    before = json.loads((parent / "inputs/lock.json").read_text())
    after = json.loads((campaign / "inputs/lock.json").read_text())
    for key in ("target", "drafter"):
        if before[key]["sha256"] != after[key]["sha256"]:
            raise ValueError(f"Phase 2 {key} differs from Phase 1")
    if before["workload"]["sha256"] != after["workload"]["sha256"]:
        raise ValueError("Phase 2 evaluation prompts differ from Phase 1")
    for key in before["spec"]:
        if key == "source_workload":
            continue  # The same hashed source can have a different absolute path.
        if before["spec"][key] != after["spec"][key]:
            raise ValueError(f"Phase 2 serving setting differs from Phase 1: {key}")
    return before


def upstream_identity(upstream: Path, config: dict[str, Any]) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(["git", "-C", str(upstream), "diff", "--binary", "HEAD"])
    if revision != config["upstream_revision"] or patch:
        raise ValueError("Speculators must match the pinned, unmodified training source")
    return {"revision": revision, "patch_sha256": hashlib.sha256(patch).hexdigest()}


def prepare_prompts(campaign: Path, config: dict[str, Any]) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = campaign / "training-data"
    output.mkdir(exist_ok=False)
    api = HfApi()
    domains, sources = {}, {}
    for domain, repo, subset in (
        ("math", "openai/gsm8k", "main"),
        ("code", "google-research-datasets/mbpp", "full"),
    ):
        revision = config["dataset_revisions"][repo]
        api.dataset_info(
            repo, revision=revision
        )  # Fail before generation if the pin is unavailable.
        dataset = load_dataset(repo, subset, split="train", revision=revision)
        prompts = []
        for index, item in enumerate(dataset):
            if domain == "math":
                text = item["question"]
            else:
                text = (
                    "Write a Python function to solve this task:\n"
                    + item["text"]
                    + "\nThe implementation must satisfy these tests:\n"
                    + "\n".join(item["test_list"])
                )
            prompts.append(
                {
                    "id": f"phase2-{domain}-{index}",
                    "workload": domain,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": 32768,
                    "source": {
                        "dataset": repo,
                        "revision": revision,
                        "split": "train",
                        "index": index,
                    },
                }
            )
        domains[domain] = prompts
        sources[domain] = {
            "dataset": repo,
            "revision": revision,
            "subset": subset,
            "split": "train",
        }
    evaluation = load_prompts(campaign / "inputs/prompts.jsonl")
    train, development = select_prompts(domains, evaluation, config)
    for name, prompts in (("train", train), ("development", development)):
        with (output / f"{name}-prompts.jsonl").open("x") as sink:
            for prompt in prompts:
                sink.write(json.dumps(prompt) + "\n")
    write_json(
        output / "prompt-manifest.json",
        {
            "sources": sources,
            "train_ids": [p["id"] for p in train],
            "development_ids": [p["id"] for p in development],
            "evaluation_ids": [p.id for p in evaluation],
            "train_sha256": file_sha256(output / "train-prompts.jsonl"),
            "development_sha256": file_sha256(output / "development-prompts.jsonl"),
        },
    )


@asynccontextmanager
async def servers(
    campaign: Path,
    spec: dict[str, Any],
    config: dict[str, Any],
    upstream: Path,
    *,
    extraction: bool,
    label: str | None = None,
):
    """Independent deployments per GPU, with launch/source/container provenance."""
    root = campaign / "data-deployments" / (label or ("features" if extraction else "responses"))
    root.mkdir(parents=True, exist_ok=False)
    processes, logs, urls = [], [], []
    try:
        for gpu in range(config["generation_workers"]):
            directory = root / f"gpu-{gpu}"
            directory.mkdir()
            port = spec["port"] + gpu
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TIKTOKEN_RS_CACHE_DIR": str(campaign / "inputs/harmony-cache"),
                    "VLLM_CACHE_ROOT": str(directory / "compile-cache"),
                }
            )
            if extraction:
                layers = json.loads((campaign / "inputs/drafter-config.json").read_text())[
                    "dflash_config"
                ]["target_layer_ids"]
                command = [
                    sys.executable,
                    str(upstream / "scripts/launch_vllm.py"),
                    "train",
                    spec["target_path"],
                    "--target-layer-ids",
                    *[str(i + 1) for i in layers],
                    "--hidden-states-path",
                    str(directory / "hidden-state-transfer"),
                    "--provenance-dir",
                    str(directory),
                    "--",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--served-model-name",
                    "gpt-oss-20b",
                    "--tensor-parallel-size",
                    "1",
                    "--max-model-len",
                    str(spec["max_model_len"]),
                    "--max-num-seqs",
                    str(config["generation_concurrency_per_worker"]),
                    "--gpu-memory-utilization",
                    str(spec["gpu_memory_utilization"]),
                    "--no-enable-prefix-caching",
                    "--no-async-scheduling",
                    "--api-server-count",
                    "1",
                    "--renderer-num-workers",
                    "1",
                ]
            else:
                settings = {
                    **spec,
                    "port": port,
                    "max_num_seqs": config["generation_concurrency_per_worker"],
                }
                command = serve_command(
                    settings, None, provenance=directory / "harmony-overlay.json"
                )
            write_json(
                directory / "provenance.json",
                {
                    "launch_argv": command,
                    "environment": {
                        key: env[key]
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "HF_HUB_OFFLINE",
                            "TIKTOKEN_RS_CACHE_DIR",
                            "VLLM_CACHE_ROOT",
                        )
                    },
                    "source": source_identity(),
                    "upstream": upstream_identity(upstream, config),
                    "runtime": runtime_provenance(),
                    "input_lock": str(campaign / "inputs/lock.json"),
                },
            )
            log = (directory / "launch.log").open("x")
            logs.append(log)
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(process)
            urls.append(f"http://127.0.0.1:{port}")
        await asyncio.gather(
            *[
                wait_ready(url, process, spec["startup_timeout_seconds"])
                for url, process in zip(urls, processes, strict=True)
            ]
        )
        yield urls
    finally:
        await asyncio.gather(*[stop_server(p) for p in processes])
        for log in logs:
            log.close()


async def generate_responses(
    campaign: Path, spec: dict[str, Any], config: dict[str, Any], urls: list[str]
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["target_path"], local_files_only=True)
    data = campaign / "training-data"
    raw = data / "responses"
    raw.mkdir(exist_ok=False)
    tasks = []
    for split in ("train", "development"):
        prompts = load_prompts(data / f"{split}-prompts.jsonl")
        for index, prompt in enumerate(prompts):
            for effort in EFFORTS:
                tasks.append((split, index, prompt, effort))
    semaphores = [asyncio.Semaphore(config["generation_concurrency_per_worker"]) for _ in urls]
    async with httpx.AsyncClient(
        timeout=spec["request_timeout_seconds"], limits=httpx.Limits(max_connections=None)
    ) as client:

        async def one(index, item):
            split, prompt_index, prompt, effort = item
            worker = index % len(urls)
            payload = {
                "model": "gpt-oss-20b",
                "messages": prompt.messages,
                "reasoning_effort": effort,
                "temperature": spec["temperature"],
                "top_p": spec["top_p"],
                "top_k": spec["top_k"],
                "min_p": spec["min_p"],
                "seed": config["seed"] + prompt_index,
                "max_tokens": spec["max_tokens"],
                "return_token_ids": True,
            }
            async with semaphores[worker]:
                response = await client.post(urls[worker] + "/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
            native = tokenizer.decode(result["prompt_token_ids"], skip_special_tokens=False)
            if f"Reasoning: {effort}\n" not in native or spec["fixed_date"] not in native:
                raise ValueError("training response did not use the frozen native effort/date")
            record = {
                "id": prompt.id,
                "effort": effort,
                "split": split,
                "request": payload,
                "response": result,
            }
            write_json(raw / f"{prompt.id}-{effort}.json", record)
            return record

        return await asyncio.gather(*[one(index, task) for index, task in enumerate(tasks)])


def retain_responses(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retain a task at all efforts together, using natural complete outputs."""
    samples, bad = [], {}
    for record in records:
        response = record["response"]
        choice = response["choices"][0]
        message = choice["message"]
        prompt_ids = response.get("prompt_token_ids")
        completion_ids = choice.get("token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
            raise ValueError("vLLM did not return native prompt and completion token IDs")
        tokens = prompt_ids + completion_ids
        valid = (
            choice["finish_reason"] == "stop"
            and (message.get("content") or "").strip()
            and completion_ids
            and len(tokens) <= config["maximum_training_sequence"]
        )
        if not valid:
            bad[record["id"]] = "incomplete output or context above frozen training limit"
        samples.append(
            {
                "id": record["id"],
                "effort": record["effort"],
                "split": record["split"],
                "prompt_tokens": len(prompt_ids),
                "token_ids": tokens,
                "raw_response": f"responses/{record['id']}-{record['effort']}.json",
            }
        )
    samples = [sample for sample in samples if sample["id"] not in bad]
    for split, key in (
        ("train", "minimum_retained_train_prompts"),
        ("development", "minimum_retained_development_prompts"),
    ):
        ids = {s["id"] for s in samples if s["split"] == split}
        if len(ids) < config[key]:
            raise ValueError(f"insufficient complete shared {split} tasks")
        for identifier in ids:
            if {s["effort"] for s in samples if s["id"] == identifier} != set(EFFORTS):
                raise ValueError("an effort is missing for a retained training task")
    return samples, bad


async def extract_features(
    campaign: Path, config: dict[str, Any], urls: list[str], samples: list[dict[str, Any]]
) -> None:
    import torch
    from openai import AsyncOpenAI
    from safetensors.torch import load_file
    from speculators.data_generation.offline import check_hidden_states
    from speculators.data_generation.vllm_client import generate_hidden_states_async

    output = campaign / "training-data/features"
    output.mkdir(exist_ok=False)
    clients = [AsyncOpenAI(base_url=url + "/v1", api_key="EMPTY", max_retries=0) for url in urls]
    semaphores = [asyncio.Semaphore(config["generation_concurrency_per_worker"]) for _ in urls]

    async def one(index, sample):
        worker = index % len(urls)
        async with semaphores[worker]:
            handle = await generate_hidden_states_async(
                clients[worker],
                "gpt-oss-20b",
                {"input_ids": sample["token_ids"]},
                timeout=1800,
                max_retries=0,
            )
            tensors = await asyncio.to_thread(load_file, handle)
            await asyncio.to_thread(check_hidden_states, tensors, sample["token_ids"])
            if tuple(tensors["hidden_states"].shape[1:]) != (6, 2880):
                raise ValueError(
                    "GPT-OSS hidden feature layer order/shape differs from the pinned head"
                )
            if tensors["hidden_states"].dtype != torch.bfloat16:
                raise ValueError("unexpected teacher feature dtype")
            path = output / f"{sample['id']}-{sample['effort']}.safetensors"
            shutil.move(handle, path)
            sample["hidden_states_path"] = str(path)
            sample["hidden_states_sha256"] = await asyncio.to_thread(file_sha256, path)

    try:
        await asyncio.gather(*[one(index, sample) for index, sample in enumerate(samples)])
    finally:
        await asyncio.gather(*[client.close() for client in clients])


def package_datasets(campaign: Path, config: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    from datasets import Dataset

    data = campaign / "training-data"
    training = [sample for sample in samples if sample["split"] == "train"]
    development = [sample for sample in samples if sample["split"] == "development"]
    manifests = {}
    for role in ROLES:
        root = data / "roles" / role
        root.mkdir(parents=True, exist_ok=False)
        train_rows = budget_examples(
            training, role, config["supervised_tokens_per_head"] // config["epochs"], config["seed"]
        )
        allowed = EFFORTS if role == "mixed" else (role,)
        dev_rows = [
            {
                "source_id": s["id"],
                "effort": s["effort"],
                "input_ids": s["token_ids"],
                "seq_len": len(s["token_ids"]),
                "hidden_states_path": s["hidden_states_path"],
                "loss_mask": [0] * s["prompt_tokens"]
                + [1] * (len(s["token_ids"]) - s["prompt_tokens"]),
            }
            for s in development
            if s["effort"] in allowed
        ]
        rows = train_rows + dev_rows
        (root / "hidden_states").mkdir()
        for index, row in enumerate(rows):
            (root / "hidden_states" / f"hs_{index}.safetensors").symlink_to(
                row["hidden_states_path"]
            )
        Dataset.from_list(
            [{key: row[key] for key in ("input_ids", "loss_mask", "seq_len")} for row in rows]
        ).with_format("torch").save_to_disk(str(root))
        counts = {
            effort: sum(sum(r["loss_mask"]) for r in train_rows if r["effort"] == effort)
            for effort in EFFORTS
        }
        manifests[role] = {
            "train_rows": len(train_rows),
            "development_rows": len(dev_rows),
            "train_data_ratio": (len(train_rows) + 0.5) / len(rows),
            "supervised_tokens_per_epoch": sum(counts.values()),
            "training_tokens_by_effort": counts,
            "dataset_files": {
                p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                for p in sorted(root.iterdir())
                if p.is_file()
            },
            "source_rows": [
                {k: r[k] for k in ("source_id", "effort", "hidden_states_path")} for r in rows
            ],
        }
        write_json(root / "training-manifest.json", manifests[role])
    write_json(data / "dataset-manifests.json", manifests)


async def feature_preflight(campaign, parent, spec, config, upstream):
    """Check real GPT-OSS feature extraction and reconstructed teacher logits."""
    import torch
    from openai import AsyncOpenAI
    from safetensors.torch import load_file
    from speculators.data_generation.offline import check_hidden_states
    from speculators.data_generation.vllm_client import extract_output
    from speculators.utils.loading import load_model_layers

    output = campaign / "training-data/feature-preflight"
    output.mkdir()
    settings = {**config, "generation_workers": 1, "generation_concurrency_per_worker": 1}
    weights = load_model_layers(["model.norm.weight", "lm_head.weight"], spec["target_path"])
    norm = weights["model.norm.weight"].to("cuda", dtype=torch.bfloat16)
    head = weights["lm_head.weight"].to("cuda", dtype=torch.bfloat16)
    epsilon = json.loads((Path(spec["target_path"]) / "config.json").read_text())["rms_norm_eps"]
    prompts = sorted((parent / "phase1/block-0/plain/rendered-prompts").glob("low-*.json"))[:3]
    if len(prompts) != 3:
        raise ValueError("parent native prompt-rendering artifacts are missing")
    results = []
    async with (
        servers(
            campaign, spec, settings, upstream, extraction=True, label="feature-preflight"
        ) as urls,
        AsyncOpenAI(base_url=urls[0] + "/v1", api_key="EMPTY", max_retries=0) as client,
    ):
        for index, path in enumerate(prompts):
            tokens = json.loads(path.read_text())["token_ids"]
            response = await client.completions.create(
                model="gpt-oss-20b",
                prompt=tokens,
                max_tokens=1,
                temperature=0,
                extra_body={"return_token_ids": True},
            )
            handle = extract_output(response, tokens)
            features = load_file(handle)
            check_hidden_states(features, tokens)
            if tuple(features["hidden_states"].shape) != (len(tokens), 6, 2880):
                raise ValueError(
                    "teacher features must contain five draft layers and the final layer"
                )
            last = features["hidden_states"][-1, -1].to("cuda")
            normalized = (last.float() * torch.rsqrt(last.float().square().mean() + epsilon)).to(
                last.dtype
            ) * norm
            predicted = int(torch.nn.functional.linear(normalized, head).argmax().item())
            actual = response.choices[0].token_ids[0]
            record = {
                "rendered_prompt": str(path),
                "rendered_prompt_sha256": file_sha256(path),
                "reconstructed_greedy_token": predicted,
                "server_greedy_token": actual,
                "shape": list(features["hidden_states"].shape),
                "response": response.model_dump(),
            }
            shutil.move(handle, output / f"{index}.safetensors")
            results.append(record)
            write_json(output / "report.json", results)
            if actual != predicted:
                raise ValueError("extracted features do not reconstruct GPT-OSS teacher logits")
    del weights, norm, head, features, last, normalized
    torch.cuda.empty_cache()


async def data_stage(
    campaign: Path, parent: Path, spec: dict[str, Any], config: dict[str, Any], upstream: Path
) -> None:
    lock = prepare_inputs(spec, campaign)
    validate_parent(campaign, parent)
    upstream_identity(upstream, config)
    write_json(
        campaign / "phase1-reference.json",
        {
            "campaign": str(parent),
            "report": build_report(parent),
            "target_sha256": lock["target"]["sha256"],
        },
    )
    write_json(campaign / "data-runtime.json", runtime_provenance())
    prepare_prompts(campaign, config)
    write_json(
        campaign / "optimizer.json",
        {
            "optimizer": "AdamW",
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "scheduler": "constant",
            "epochs": config["epochs"],
            "seed": config["seed"],
            "loss": {
                k: config[k]
                for k in (
                    "loss_fn",
                    "loss_implementation",
                    "position_loss_weight",
                    "position_loss_gamma",
                    "max_anchors",
                    "noise_std",
                )
            },
            "token_basis": config["supervised_token_basis"],
            "checkpoint_selection": config["checkpoint_selection"],
        },
    )
    training_plan(
        parent,
        campaign / "training-handoff",
        campaign / "training-data/train-prompts.jsonl",
        campaign / "training-data/development-prompts.jsonl",
        config["supervised_tokens_per_head"],
        campaign / "optimizer.json",
    )
    await feature_preflight(campaign, parent, spec, config, upstream)
    async with servers(campaign, spec, config, upstream, extraction=False) as urls:
        records = await generate_responses(campaign, spec, config, urls)
    samples, excluded = retain_responses(records, config)
    write_json(campaign / "training-data/excluded-tasks.json", excluded)
    async with servers(campaign, spec, config, upstream, extraction=True) as urls:
        await extract_features(campaign, config, urls, samples)
    write_json(campaign / "training-data/feature-manifest.json", samples)
    package_datasets(campaign, config, samples)
    write_json(
        campaign / "data-status.json",
        {
            "status": "complete",
            "retained_samples": len(samples),
            "excluded_task_ids": sorted(excluded),
        },
    )


class HistoryHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record):
        if isinstance(record.msg, Mapping) and not getattr(record, "hparams", False):
            with self.path.open("a") as sink:
                sink.write(
                    json.dumps({"timestamp": datetime.now(UTC).isoformat(), **record.msg}) + "\n"
                )


def train_role(campaign: Path, config: dict[str, Any], upstream: Path, role: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    from speculators.convert.dflash.converter import DFlashConverter
    from speculators.models.dflash import DFlashDraftModel
    from speculators.train import cli
    from speculators.train.config import TrainConfig

    lock = json.loads((campaign / "inputs/lock.json").read_text())
    if lock["source"] != source_identity():
        raise ValueError("Phase 2 source changed before training")
    upstream_identity(upstream, config)
    if json.loads((campaign / "data-status.json").read_text())["status"] != "complete":
        raise ValueError("training requires complete data/feature generation")
    root = campaign / "training" / role
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "runtime.json",
        {**runtime_provenance(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    )
    source = Path(lock["spec"]["drafter_path"])
    converted = root / "initialization"
    DFlashConverter().convert(source, converted, base_model=lock["spec"]["target_path"])
    dataset = campaign / "training-data/roles" / role
    manifest = json.loads((dataset / "training-manifest.json").read_text())
    for name, evidence in manifest["dataset_files"].items():
        if file_sha256(dataset / name) != evidence["sha256"]:
            raise ValueError(f"prepared training corpus changed: {role}/{name}")
    assert (
        manifest["supervised_tokens_per_epoch"] * config["epochs"]
        == config["supervised_tokens_per_head"]
    )
    checkpoint = root / "checkpoints"
    argv = [
        "--verifier-name-or-path",
        lock["spec"]["target_path"],
        "--from-pretrained",
        str(converted),
        "--speculator-type",
        "dflash",
        "--data-path",
        str(dataset),
        "--save-path",
        str(checkpoint),
        "--draft-vocab-size",
        "201088",
        "--epochs",
        str(config["epochs"]),
        "--lr",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--scheduler-type",
        "none",
        "--total-seq-len",
        str(config["maximum_training_sequence"]),
        "--train-data-ratio",
        str(manifest["train_data_ratio"]),
        "--loss-fn",
        config["loss_fn"],
        "--loss-implementation",
        config["loss_implementation"],
        "--per-position-loss-weight",
        config["position_loss_weight"],
        "--dflash-decay-gamma",
        str(config["position_loss_gamma"]),
        "--max-anchors",
        str(config["max_anchors"]),
        "--noise-std",
        str(config["noise_std"]),
        "--hidden-states-dtype",
        "bfloat16",
        "--seed",
        str(config["seed"]),
        "--num-workers",
        "2",
        "--on-missing",
        "raise",
        "--no-resume-from-checkpoint",
        "--save-best",
        "--log-freq",
        "1",
        "--checkpoint-freq",
        "1",
    ]
    setup = cli.setup_metric_logger
    build_model = cli.build_draft_model
    observed = {"supervised_tokens": 0, "training_batches": 0}

    def count_training_tokens(model, positional, kwargs):
        if model.training:
            observed["supervised_tokens"] += int(kwargs["loss_mask"].sum().item())
            observed["training_batches"] += 1

    def build_with_budget_counter(*args, **kwargs):
        model = build_model(*args, **kwargs).float()
        model.register_forward_pre_hook(count_training_tokens, with_kwargs=True)
        return model

    def setup_with_history(**kwargs):
        setup(**kwargs)
        logging.getLogger("speculators.metrics").addHandler(HistoryHandler(root / "history.jsonl"))

    cli.setup_metric_logger = setup_with_history
    cli.build_draft_model = build_with_budget_counter
    write_json(root / "train-argv.json", argv)
    os.chdir(upstream)  # Speculators provenance must describe its own checkout.
    cli.main(TrainConfig.resolve(argv))
    history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
    validations = [row for row in history if "val" in row]
    if len(validations) != config["epochs"]:
        raise ValueError("training did not finish every budgeted epoch and development pass")
    if observed["supervised_tokens"] != config["supervised_tokens_per_head"]:
        raise ValueError(f"upstream loader changed the supervised-token budget: {observed}")
    if any(not math.isfinite(row["val"]["loss_epoch"]) for row in validations):
        raise ValueError("non-finite development loss")
    best = checkpoint / "checkpoint_best"
    if not best.exists():
        raise ValueError("upstream trainer produced no development-selected checkpoint")
    model = DFlashDraftModel.from_pretrained(str(best.resolve()), dtype=torch.float32)
    original = load_file(str(source / "model.safetensors"))
    state = model.state_dict()
    export = root / "export"
    export.mkdir()
    save_file(
        {
            key: state[key].detach().cpu().to(tensor.dtype).contiguous()
            for key, tensor in original.items()
        },
        str(export / "model.safetensors"),
    )
    shutil.copyfile(source / "config.json", export / "config.json")
    for path in source.glob("*.py"):
        shutil.copyfile(path, export / path.name)
    exported = model_manifest(export)
    prompt_manifest = json.loads((campaign / "training-data/prompt-manifest.json").read_text())
    provenance = {
        "target_sha256": lock["target"]["sha256"],
        "effort": role,
        "initialization_sha256": lock["drafter"]["sha256"],
        "architecture_sha256": file_sha256(source / "config.json"),
        "optimizer_schedule_sha256": file_sha256(campaign / "optimizer.json"),
        "supervised_tokens": config["supervised_tokens_per_head"],
        "training_seed": config["seed"],
        **{
            f"{split}_prompt_ids_sha256": sha_json(prompt_manifest[f"{key}_ids"])
            for split, key in (
                ("train", "train"),
                ("development", "development"),
                ("evaluation", "evaluation"),
            )
        },
        "export_sha256": exported["sha256"],
        "export_weight_bytes": sum(t.numel() * t.element_size() for t in original.values()),
        "stored_checkpoint_bytes": sum(p["bytes"] for p in exported["files"].values()),
        "selected_checkpoint": str(best.resolve()),
        "upstream": upstream_identity(upstream, config),
        "source": source_identity(),
        "training_data_manifest": file_sha256(dataset / "training-manifest.json"),
        "target_contract": lock["target"],
        "epochs_completed": len(validations),
        "observed_training_budget": observed,
        "token_basis": config["supervised_token_basis"],
    }
    write_json(root / "provenance.json", provenance)
    write_json(root / "history.json", history)
    write_json(root / "evals.json", validations)
    write_json(
        root / "export-report.json",
        {
            "manifest": exported,
            "config_identical_to_initialization": file_sha256(export / "config.json")
            == file_sha256(source / "config.json"),
        },
    )
    write_json(
        root / "train-summary.json",
        {
            "status": "complete",
            "provenance": provenance,
            "best_validation_loss": min(r["val"]["loss_epoch"] for r in validations),
        },
    )


async def training_stage(campaign: Path, config: dict[str, Any], upstream: Path) -> None:
    prepare_inputs(read_spec(campaign / "experiment.yaml"), campaign)
    samples = json.loads((campaign / "training-data/feature-manifest.json").read_text())
    for sample in samples:
        if file_sha256(Path(sample["hidden_states_path"])) != sample["hidden_states_sha256"]:
            raise ValueError(f"teacher features changed before training: {sample['id']}")
    processes, logs = [], []
    try:
        for gpu, role in enumerate(ROLES):
            log_path = campaign / f"train-{role}.log"
            log = log_path.open("x")
            logs.append(log)
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "1",
                }
            )
            for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
                env.pop(key, None)
            command = [
                sys.executable,
                "-m",
                "apertus_bench.effort_training",
                "train-role",
                "--campaign",
                str(campaign),
                "--config",
                str(campaign / "phase2.yaml"),
                "--upstream",
                str(upstream),
                "--role",
                role,
            ]
            processes.append(
                subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            )
        codes = await asyncio.gather(*[asyncio.to_thread(p.wait) for p in processes])
        if any(codes):
            raise RuntimeError(f"training child failed: {dict(zip(ROLES, codes, strict=True))}")
        heads = {
            role: {
                "path": str(campaign / "training" / role / "export"),
                "provenance": str(campaign / "training" / role / "provenance.json"),
            }
            for role in ROLES
        }
        write_json(campaign / "heads.json", heads)
        write_json(campaign / "training-status.json", {"status": "complete", "heads": heads})
    finally:
        await asyncio.gather(*[stop_server(p) for p in processes if p.poll() is None])
        for log in logs:
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "train", "train-role"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase1-reference", type=Path)
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES)
    args = parser.parse_args()
    config = load_config(args.config)
    campaign = args.campaign.resolve()
    if args.action == "data":
        if args.phase1_reference is None:
            parser.error("data generation requires --phase1-reference")
        spec = read_spec(campaign / "experiment.yaml")
        coroutine = data_stage(
            campaign, args.phase1_reference.resolve(), spec, config, args.upstream
        )
    elif args.action == "train":
        coroutine = training_stage(campaign, config, args.upstream)
    else:
        if args.role is None:
            parser.error("train-role requires --role")
        train_role(campaign, config, args.upstream, args.role)
        return

    async def run_with_signals():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, task.cancel)
        await coroutine

    asyncio.run(run_with_signals())


if __name__ == "__main__":
    main()
