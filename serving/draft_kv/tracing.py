"""Host-only scheduling for drafter KV trace samples."""

from __future__ import annotations

import os


def trace_due(step):
    return bool(os.environ.get("VLLM_DRAFT_KV_TRACE")) and (step == 1 or step % 64 == 0)


def first_completed_prompt_requests(speculator, batch):
    """Trace completed prefills once per resident request, independent of step cadence.

    Host-side prefill progress is already available; no GPU work is needed to
    decide whether to sample. Request removal clears the IDs for reuse/resume.
    """
    if not os.environ.get("VLLM_DRAFT_KV_TRACE"):
        return []
    completed = getattr(speculator, "_apertus_trace_completed_requests", None)
    if completed is None:
        completed = speculator._apertus_trace_completed_requests = set()
    first = []
    for req_id, computed, scheduled, prompt_length in zip(
        batch.req_ids,
        batch.num_computed_prefill_tokens_np,
        batch.num_scheduled_tokens,
        batch.prefill_len_np,
        strict=True,
    ):
        if int(computed) + int(scheduled) >= int(prompt_length) and req_id not in completed:
            first.append(req_id)
            completed.add(req_id)
    return first
