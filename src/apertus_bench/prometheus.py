from __future__ import annotations

from dataclasses import asdict, dataclass, field

from prometheus_client.parser import text_string_to_metric_families

SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
POSITION_COUNTER = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


@dataclass(frozen=True)
class SpeculativeSnapshot:
    drafts: float = 0.0
    draft_tokens: float = 0.0
    accepted_tokens: float = 0.0
    accepted_tokens_per_position: dict[int, float] = field(default_factory=dict)
    enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _matches_model(labels: dict[str, str], model: str | None) -> bool:
    if model is None:
        return True
    for label_name in ("model_name", "served_model_name", "model"):
        if label_name in labels:
            return labels[label_name] == model
    return True


def parse_speculative_snapshot(text: str, model: str | None = None) -> SpeculativeSnapshot:
    totals = dict.fromkeys(SPEC_COUNTERS, 0.0)
    positions: dict[int, float] = {}
    found = False
    name_to_key = {metric_name: key for key, metric_name in SPEC_COUNTERS.items()}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not _matches_model(sample.labels, model):
                continue
            if sample.name in name_to_key:
                found = True
                totals[name_to_key[sample.name]] += float(sample.value)
            elif sample.name == POSITION_COUNTER:
                found = True
                position = int(sample.labels["position"])
                positions[position] = positions.get(position, 0.0) + float(sample.value)

    return SpeculativeSnapshot(
        drafts=totals["drafts"],
        draft_tokens=totals["draft_tokens"],
        accepted_tokens=totals["accepted_tokens"],
        accepted_tokens_per_position=positions,
        enabled=found,
    )


def speculative_delta(before: SpeculativeSnapshot, after: SpeculativeSnapshot) -> dict[str, object]:
    if not before.enabled and not after.enabled:
        return {"enabled": False}

    deltas = {
        "drafts": after.drafts - before.drafts,
        "draft_tokens": after.draft_tokens - before.draft_tokens,
        "accepted_tokens": after.accepted_tokens - before.accepted_tokens,
    }
    all_positions = (
        before.accepted_tokens_per_position.keys() | after.accepted_tokens_per_position.keys()
    )
    per_position = {
        position: after.accepted_tokens_per_position.get(position, 0.0)
        - before.accepted_tokens_per_position.get(position, 0.0)
        for position in sorted(all_positions)
    }
    if any(value < 0 for value in [*deltas.values(), *per_position.values()]):
        raise ValueError(
            "speculative counters decreased; the server likely restarted during the run"
        )

    draft_tokens = deltas["draft_tokens"]
    drafts = deltas["drafts"]
    return {
        "enabled": True,
        **deltas,
        "acceptance_rate": deltas["accepted_tokens"] / draft_tokens if draft_tokens else None,
        "mean_acceptance_length": 1 + deltas["accepted_tokens"] / drafts if drafts else None,
        "acceptance_rate_per_position": {
            str(position): accepted / drafts if drafts else None
            for position, accepted in per_position.items()
        },
    }
