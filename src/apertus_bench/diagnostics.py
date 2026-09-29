"""Diagnostic configurations, fixed-output corpus, and break-even arithmetic."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apertus_bench.workloads import Prompt

TARGET_70B = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B"
DRAFT_8B = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B"
MECHANISTIC_WORKLOAD = "mechanistic_fixed256"
FIXED_OUTPUT_TOKENS = 256
PROMPT_COUNT = 32
TARGET_WORDS = 520

TOPICS = (
    "speculative round cost on a four-GPU GH200 node",
    "n-gram prompt lookup inside long source documents",
    "asynchronous scheduling overlap with token streaming",
    "scheduled-token budget after speculative lookahead reservation",
    "CUDA graph capture versus eager draft-model execution",
    "tensor-parallel collectives inside an embedded 8B drafter",
    "KV-cache capacity after loading a second set of weights",
    "verification cost when every draft token is rejected",
    "verification cost when every draft token is accepted",
    "prefix-cache hits that the mechanistic block must disable",
    "standalone 8B decode latency without a 70B neighbour",
    "depth-1 speculation as a setup-cost versus slope probe",
    "depth-5 speculation when acceptance survives past position 3",
    "extractive summarization that copies spans from the prompt",
    "abstractive summarization that paraphrases the same source",
    "code completion that continues a supplied file",
    "code generation from a short natural-language instruction",
    "grounded entity extraction versus open-ended chat",
    "translation of a summary instead of copying the source language",
    "target-aligned distillation of an 8B drafter",
    "a smaller compatible drafter without target-specific training",
    "draft-only quantization with the 70B precision held fixed",
    "Nsight timeline attribution across all four ranks",
    "host synchronization and device copies on the draft path",
    "padded batch shapes that inflate draft attention and logits",
    "cache rollback after an immediate first-token rejection",
    "EOS and output-limit behaviour at the end of a speculative round",
    "mixed prefill and decode batches under the same token budget",
    "cold versus warm KV-cache serving for the same prompts",
    "Poisson arrival capacity versus closed-loop concurrency cells",
    "workload routing that uses only information available at request start",
    "stopping when no supported layout beats the measured break-even cost",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    id: str
    launcher: str
    method: str
    async_scheduling: bool
    max_num_batched_tokens: int
    enable_prefix_caching: bool
    baseline_role: str | None = None
    num_speculative_tokens: int | None = None
    draft_tensor_parallel_size: int | None = None
    prompt_lookup_min: int | None = None
    prompt_lookup_max: int | None = None
    target_model: str = TARGET_70B
    draft_model: str | None = None
    allow_depth_1: bool = False
    apply_image_token_patch: bool = False
    notes: str = ""


CONFIGS: dict[str, DiagnosticConfig] = {
    "B0": DiagnosticConfig(
        id="B0",
        launcher="baseline.sh",
        method="none",
        async_scheduling=True,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        baseline_role="operational",
        notes="Operational reference: async on, batched 8192, prefix cache off.",
    ),
    "B1": DiagnosticConfig(
        id="B1",
        launcher="baseline.sh",
        method="none",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        baseline_role="matched",
        notes="B1 vs B0: scheduling and output delivery with async off.",
    ),
    "B2": DiagnosticConfig(
        id="B2",
        launcher="baseline.sh",
        method="none",
        async_scheduling=False,
        max_num_batched_tokens=7168,
        enable_prefix_caching=False,
        baseline_role="matched",
        notes="B2 vs B1: reduced token budget at the actual batch shape.",
    ),
    "N3": DiagnosticConfig(
        id="N3",
        launcher="ngram.sh",
        method="ngram",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        num_speculative_tokens=3,
        prompt_lookup_min=1,
        prompt_lookup_max=4,
        notes="N-gram depth 3, lookup 1-4, historical effective budget 8192.",
    ),
    "N3m": DiagnosticConfig(
        id="N3m",
        launcher="ngram.sh",
        method="ngram",
        async_scheduling=False,
        max_num_batched_tokens=7168,
        enable_prefix_caching=False,
        num_speculative_tokens=3,
        prompt_lookup_min=1,
        prompt_lookup_max=4,
        notes="N3m vs N3: budget sensitivity; N3m vs B2: n-gram path cost.",
    ),
    "D3": DiagnosticConfig(
        id="D3",
        launcher="draft-model.sh",
        method="draft_model",
        async_scheduling=False,
        # Historical draft_model passes 8192 and the engine reserves lookahead,
        # logging max_num_scheduled_tokens=7168. Passing 7168 here would shrink
        # further. Record the resolved scheduled value from the replica log.
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        num_speculative_tokens=3,
        draft_tensor_parallel_size=4,
        draft_model=DRAFT_8B,
        apply_image_token_patch=True,
        notes="8B draft depth 3 TP=4; compare with B2 after matching controls.",
    ),
    "D1": DiagnosticConfig(
        id="D1",
        launcher="draft-model.sh",
        method="draft_model",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        num_speculative_tokens=1,
        draft_tensor_parallel_size=4,
        draft_model=DRAFT_8B,
        allow_depth_1=True,
        apply_image_token_patch=True,
        notes="Depth 1 diagnostic; enable only after the engine accepts it.",
    ),
    "D5": DiagnosticConfig(
        id="D5",
        launcher="draft-model.sh",
        method="draft_model",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        num_speculative_tokens=5,
        draft_tensor_parallel_size=4,
        draft_model=DRAFT_8B,
        apply_image_token_patch=True,
        notes="Depth 5 on the same prefixes as D3.",
    ),
    "D8": DiagnosticConfig(
        id="D8",
        launcher="draft-model.sh",
        method="draft_model",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        num_speculative_tokens=8,
        draft_tensor_parallel_size=4,
        draft_model=DRAFT_8B,
        apply_image_token_patch=True,
        notes="Depth 8 on the same prefixes as D3.",
    ),
    "P2": DiagnosticConfig(
        id="P2",
        launcher="standalone-8b.sh",
        method="none",
        async_scheduling=False,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        target_model=DRAFT_8B,
        baseline_role="standalone",
        notes="Standalone 8B TP=4 bound on embedded draft-step latency.",
    ),
}

SCREENING_IDS = ("B0", "B1", "B2", "N3", "N3m", "D3")
# Predetermined orders: B0 first in each block; remaining shuffled with seed 20260919.
BLOCK_ORDERS = {
    1: ("B0", "D3", "B1", "N3m", "B2", "N3"),
    2: ("B0", "N3", "B2", "D3", "N3m", "B1"),
}


def get_config(config_id: str) -> DiagnosticConfig:
    try:
        return CONFIGS[config_id]
    except KeyError as error:
        known = ", ".join(CONFIGS)
        raise ValueError(f"unknown diagnostic config {config_id!r}; known: {known}") from error


def env_exports(config_id: str) -> dict[str, str]:
    config = get_config(config_id)
    env = {
        "DIAG_CONFIG": config.id,
        "ASYNC_SCHEDULING": "1" if config.async_scheduling else "0",
        "ENABLE_PREFIX_CACHING": "1" if config.enable_prefix_caching else "0",
        "MAX_NUM_BATCHED_TOKENS": str(config.max_num_batched_tokens),
        "TARGET_MODEL": config.target_model,
        "ALLOW_DEPTH_1": "1" if config.allow_depth_1 else "0",
    }
    if config.num_speculative_tokens is not None:
        env["NUM_SPECULATIVE_TOKENS"] = str(config.num_speculative_tokens)
    if config.draft_tensor_parallel_size is not None:
        env["DRAFT_TP"] = str(config.draft_tensor_parallel_size)
    if config.draft_model:
        env["DRAFT_MODEL"] = config.draft_model
    if config.prompt_lookup_min is not None:
        env["PROMPT_LOOKUP_MIN"] = str(config.prompt_lookup_min)
    if config.prompt_lookup_max is not None:
        env["PROMPT_LOOKUP_MAX"] = str(config.prompt_lookup_max)
    if config.baseline_role:
        env["BASELINE_ROLE"] = config.baseline_role
    env["APPLY_IMAGE_TOKEN_PATCH"] = "1" if config.apply_image_token_patch else "0"
    env["LAUNCHER"] = config.launcher
    env["DIAG_METHOD"] = config.method
    return env


def format_env_exports(config_id: str) -> str:
    lines = [f"export {key}={shlex.quote(value)}" for key, value in env_exports(config_id).items()]
    return "\n".join(lines) + "\n"


def _paragraph(topic: str, index: int, para: int) -> str:
    measure = 1100 + index * 41 + para * 13
    latency = 8 + (index % 7) + (para % 5)
    tokens = 512 + index * 9 + para * 6
    templates = (
        (
            f"Section {para + 1} of the briefing on {topic} reports that run "
            f"{index:02d}-{para:02d} produced {measure} committed output tokens at a "
            f"client-observed TPOT of {latency}.{para} milliseconds. The input used "
            f"{tokens} Apertus tokens, prefix caching was disabled, and greedy decoding "
            f"with ignore_eos held the completion length fixed. The open question is "
            f"whether that latency is draft arithmetic, verification, or exposed CPU time."
        ),
        (
            f"The same {topic} note then records a second window, labelled "
            f"{index:02d}x{para:02d}, in which the scheduler advertised a token budget of "
            f"{7168 if para % 2 else 8192} while the observed active batch never exceeded "
            f"{1 + para} sequences. Memory accounting listed {measure * 3} KV tokens on "
            f"the tightest rank. Operators must not treat the advertised budget as proof "
            f"that the budget bound the measured cell."
        ),
        (
            f"A correctness probe for {topic} compared two independently launched "
            f"deployments on prompts {index:02d}a and {index:02d}b. They shared a prefix of "
            f"{180 + para * 11} characters and then diverged, with completion-token counts "
            f"differing by {para % 4}. That divergence is a numerical-environment effect "
            f"until a same-configuration control shows otherwise, and it is unique to this "
            f"source document's identifiers {index}-{para}-{measure}."
        ),
        (
            f"Finally, the {topic} appendix lists unresolved risks: graph breaks during "
            f"draft steps, NCCL waits on rank {(index + para) % 4}, padded logits of width "
            f"{256 * (1 + para % 3)}, and a break-even round cost of {40 + para} to "
            f"{50 + para} milliseconds at the observed acceptance. Item "
            f"R{index:02d}{para:02d} is not reused in any other briefing in this corpus."
        ),
    )
    return templates[para % 4]


def build_user_content(index: int, topic: str) -> str:
    instruction = (
        "Continue the following technical briefing with a careful, detailed expansion "
        "of the unresolved measurement questions. Write continuously without a closing "
        "summary and without repeating the identifiers already present."
    )
    paragraphs = [_paragraph(topic, index, para) for para in range(8)]
    body = "\n\n".join(paragraphs)
    words = body.split()
    # Keep the body inside the 512-1024 token bucket under a ~1.3 tokens/word prior.
    if len(words) > TARGET_WORDS:
        body = " ".join(words[:TARGET_WORDS])
    return f"{instruction}\n\n{body}"


def generate_prompts() -> list[Prompt]:
    if len(TOPICS) != PROMPT_COUNT:
        raise ValueError(f"expected {PROMPT_COUNT} topics, got {len(TOPICS)}")
    prompts: list[Prompt] = []
    for index, topic in enumerate(TOPICS):
        prompts.append(
            Prompt(
                id=f"fixed256-{index:02d}",
                workload=MECHANISTIC_WORKLOAD,
                messages=[{"role": "user", "content": build_user_content(index, topic)}],
                max_tokens=FIXED_OUTPUT_TOKENS,
            )
        )
    return prompts


def write_corpus(path: Path) -> dict[str, Any]:
    prompts = generate_prompts()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            record = {
                "id": prompt.id,
                "workload": prompt.workload,
                "messages": prompt.messages,
                "max_tokens": prompt.max_tokens,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    word_counts = [len(prompt.messages[0]["content"].split()) for prompt in prompts]
    manifest = {
        "workload": MECHANISTIC_WORKLOAD,
        "prompts": len(prompts),
        "max_tokens": FIXED_OUTPUT_TOKENS,
        "ignore_eos": True,
        "target_input_token_bucket": [512, 1024],
        "word_count_min": min(word_counts),
        "word_count_max": max(word_counts),
        "word_count_mean": sum(word_counts) / len(word_counts),
        "notes": (
            "Word counts are a planning prior. Fill Apertus token counts from the "
            "tokenizer or from the first B0 /v1/chat/completions usage.prompt_tokens."
        ),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"corpus": str(path), "manifest": str(manifest_path), **manifest}


_SCHEDULED_SET_TO = re.compile(r"max_num_scheduled_tokens is set to (\d+)")
_BATCHED_ASSIGNED = re.compile(r"max_num_batched_tokens[=:](\d+)")
_TOKEN_BUDGET = re.compile(r"(\d{4,})")


def extract_effective_scheduled_tokens(
    replica_log: str,
    *,
    requested_batched_tokens: int | str | None = None,
) -> str:
    """Return the engine's resolved scheduled-token budget from a replica log.

    Speculative decoding logs ``max_num_scheduled_tokens is set to N`` after
    lookahead reservation. Plain target serving usually logs only
    ``max_num_batched_tokens``. The matched B2/D3 comparison needs the numeric
    value, not the raw log line.
    """
    matches = _SCHEDULED_SET_TO.findall(replica_log)
    if matches:
        return matches[-1]
    matches = _BATCHED_ASSIGNED.findall(replica_log)
    if matches:
        return matches[-1]
    if requested_batched_tokens not in (None, ""):
        return str(requested_batched_tokens)
    return "unknown"


def normalize_scheduled_tokens(value: Any) -> Any:
    """Coerce a log line or metadata string to the integer budget when present."""
    if value is None:
        return None
    text = str(value).strip()
    scheduled = _SCHEDULED_SET_TO.findall(text)
    if scheduled:
        return scheduled[-1]
    digits = _TOKEN_BUDGET.search(text)
    if digits:
        return digits.group(1)
    return text


def inferred_round_metrics(
    completion_tokens: float | None,
    accepted_tokens: float | None,
    tpot_ms: float | None,
    drafts: float | None = None,
) -> dict[str, float | None]:
    """Infer round cost from cell aggregates. Not a kernel measurement."""
    if completion_tokens is None or tpot_ms is None or completion_tokens <= 0:
        return {
            "inferred_steps": None,
            "inferred_g": None,
            "inferred_t_round_ms": None,
            "prometheus_g": None,
        }
    accepted = float(accepted_tokens or 0.0)
    steps = float(completion_tokens) - accepted
    prometheus_g = None
    if drafts:
        prometheus_g = 1.0 + accepted / float(drafts)
    if steps <= 0:
        return {
            "inferred_steps": steps,
            "inferred_g": None,
            "inferred_t_round_ms": None,
            "prometheus_g": prometheus_g,
        }
    return {
        "inferred_steps": steps,
        "inferred_g": float(completion_tokens) / steps,
        "inferred_t_round_ms": float(tpot_ms) * float(completion_tokens) / steps,
        "prometheus_g": prometheus_g,
    }


def break_even_report(
    rows: list[dict[str, Any]],
    *,
    t0_variant: str,
    spec_variant: str,
    workload: str | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    def select(variant: str) -> list[dict[str, Any]]:
        chosen = [row for row in rows if row.get("variant") == variant]
        if workload is not None:
            chosen = [row for row in chosen if row.get("workload") == workload]
        if concurrency is not None:
            chosen = [row for row in chosen if row.get("concurrency") == concurrency]
        return chosen

    t0_rows = select(t0_variant)
    spec_rows = select(spec_variant)
    if not t0_rows:
        raise ValueError(f"no cells for t0 variant {t0_variant!r}")
    if not spec_rows:
        raise ValueError(f"no cells for speculative variant {spec_variant!r}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    t0_values = [float(row["tpot_p50_ms"]) for row in t0_rows if row.get("tpot_p50_ms") is not None]
    if not t0_values:
        raise ValueError(f"{t0_variant} has no tpot_p50_ms")
    t0 = mean(t0_values)

    comparisons = []
    for row in spec_rows:
        metrics = inferred_round_metrics(
            row.get("completion_tokens"),
            row.get("accepted_tokens"),
            row.get("tpot_p50_ms"),
            row.get("drafts"),
        )
        g = metrics["inferred_g"]
        t_round = metrics["inferred_t_round_ms"]
        depth = row.get("num_speculative_tokens")
        perfect_g = (int(depth) + 1) if isinstance(depth, int) else None
        predicted = None
        ceiling = None
        break_even = None
        if g is not None and t_round is not None and t_round > 0:
            predicted = g * t0 / t_round
            break_even = g * t0
        if perfect_g is not None and t_round is not None and t_round > 0:
            ceiling = perfect_g * t0 / t_round
        comparisons.append(
            {
                "path": row.get("path"),
                "workload": row.get("workload"),
                "concurrency": row.get("concurrency"),
                "repeat": row.get("repeat"),
                "t0_ms": t0,
                **metrics,
                "break_even_t_round_ms": break_even,
                "predicted_speedup": predicted,
                "perfect_g": perfect_g,
                "perfect_speedup_at_same_cost": ceiling,
                "measured_speedup_vs_baseline": row.get("speedup_vs_baseline"),
                "measured_speedup_vs_matched_baseline": row.get("speedup_vs_matched_baseline"),
            }
        )
    return {
        "t0_variant": t0_variant,
        "spec_variant": spec_variant,
        "t0_ms": t0,
        "t0_cells": len(t0_values),
        "comparisons": comparisons,
    }


def append_ledger(path: Path, record: dict[str, Any]) -> None:
    payload = {"recorded_at": datetime.now(UTC).isoformat(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def config_public_dict(config_id: str) -> dict[str, Any]:
    return asdict(get_config(config_id))
