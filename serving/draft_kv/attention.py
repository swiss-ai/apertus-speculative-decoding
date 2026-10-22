"""Drafter-only attention over a bounded native paged KV allocation.

Context keys are already normalized and rotated at their original token
positions. ``block_tags`` stores those original page starts; it is never a
compact-cache position. Query K/V live only in the current forward pass.

The optional importance update is the exact softmax mass assigned by the
request's anchor query to each retained context page, summed over query heads.
Its denominator includes current query K/V and the virtual attention sink.
There is no full-history backing allocation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class SparseAttentionWorkspace:
    """Allocate once per graph shape and reuse across all drafter layers."""

    page_lse: torch.Tensor
    anchor_lse: torch.Tensor

    @classmethod
    def allocate(
        cls, num_requests: int, num_heads: int, num_slots: int, device: torch.device
    ) -> SparseAttentionWorkspace:
        return cls(
            torch.empty(
                (num_requests, num_heads, num_slots),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty((num_requests, num_heads), dtype=torch.float32, device=device),
        )


@triton.jit
def _sparse_attention_kernel(
    Q,
    K_QUERY,
    V_QUERY,
    CACHE_K,
    CACHE_V,
    BLOCK_IDS,
    BLOCK_TAGS,
    CONTEXT_ENDS,
    QUERY_STARTS,
    POSITIONS,
    ANCHORS,
    SINKS,
    OUTPUT,
    PAGE_LSE,
    ANCHOR_LSE,
    q_s0: tl.constexpr,
    q_s1: tl.constexpr,
    q_s2: tl.constexpr,
    kq_s0: tl.constexpr,
    kq_s1: tl.constexpr,
    kq_s2: tl.constexpr,
    vq_s0: tl.constexpr,
    vq_s1: tl.constexpr,
    vq_s2: tl.constexpr,
    ck_s0: tl.constexpr,
    ck_s1: tl.constexpr,
    ck_s2: tl.constexpr,
    ck_s3: tl.constexpr,
    cv_s0: tl.constexpr,
    cv_s1: tl.constexpr,
    cv_s2: tl.constexpr,
    cv_s3: tl.constexpr,
    bid_s0: tl.constexpr,
    bid_s1: tl.constexpr,
    tag_s0: tl.constexpr,
    tag_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // HEADS_PER_KV
    query_start = tl.load(QUERY_STARTS + request)
    query_end = tl.load(QUERY_STARTS + request + 1)
    query_len = query_end - query_start
    context_end = tl.load(CONTEXT_ENDS + request)
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    tokens = tl.arange(0, BLOCK_N)
    active_rows = rows < query_len
    query_rows = query_start + rows
    query_pos = tl.load(POSITIONS + query_rows, active_rows, other=0)
    q = tl.load(
        Q + query_rows[:, None] * q_s0 + head * q_s1 + dims[None, :] * q_s2,
        active_rows[:, None] & (dims[None, :] < HEAD_DIM),
        other=0,
    )
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    if HAS_SINK:
        # The sink is one extra softmax logit with a zero value vector.
        sink = tl.load(SINKS + head).to(tl.float32)
        m = tl.full((BLOCK_M,), 0, tl.float32) + sink
        normalizer = tl.full((BLOCK_M,), 1, tl.float32)
    if SCORE:
        anchor = tl.load(ANCHORS + request) - query_start
        anchor_valid = (anchor >= 0) & (anchor < query_len)

    for slot in range(NUM_SLOTS):
        # Physical IDs index the global pool, whose per-layer span can exceed
        # 2**31 elements. Promote before multiplying by the cache page stride;
        # promoting the resulting pointer would leave an overflowed int32 offset.
        physical_page = tl.load(BLOCK_IDS + request * bid_s0 + slot * bid_s1).to(tl.int64)
        page_start = tl.load(BLOCK_TAGS + request * tag_s0 + slot * tag_s1)
        valid_page = (physical_page >= 0) & (page_start >= 0)
        context_pos = page_start + tokens
        valid_tokens = valid_page & (tokens < PAGE_SIZE) & (context_pos < context_end)
        keys = tl.load(
            CACHE_K
            + physical_page * ck_s0
            + tokens[None, :] * ck_s1
            + kv_head * ck_s2
            + dims[:, None] * ck_s3,
            valid_tokens[None, :] & (dims[:, None] < HEAD_DIM),
            other=0,
        )
        values = tl.load(
            CACHE_V
            + physical_page * cv_s0
            + tokens[:, None] * cv_s1
            + kv_head * cv_s2
            + dims[None, :] * cv_s3,
            valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        )
        logits = tl.dot(q, keys, input_precision="ieee") * SCALE
        visible = active_rows[:, None] & valid_tokens[None, :]
        if WINDOW > 0:
            visible = visible & (context_pos[None, :] >= query_pos[:, None] - WINDOW + 1)
            visible = visible & (context_pos[None, :] <= query_pos[:, None] + WINDOW - 1)
        if CAUSAL:
            visible = visible & (context_pos[None, :] <= query_pos[:, None])
        logits = tl.where(visible, logits, -float("inf"))
        if SCORE:
            anchor_logits = tl.sum(tl.where(rows[:, None] == anchor, logits, 0.0), axis=0)
            page_max = tl.max(anchor_logits, axis=0)
            safe_page_max = tl.where(page_max == -float("inf"), 0.0, page_max)
            page_sum = tl.sum(tl.exp(anchor_logits - safe_page_max), axis=0)
            page_logsumexp = page_max + tl.log(page_sum)
            tl.store(
                PAGE_LSE + (request * NUM_HEADS + head) * NUM_SLOTS + slot,
                tl.where(anchor_valid, page_logsumexp, -float("inf")),
            )
        next_m = tl.maximum(m, tl.max(logits, axis=1))
        safe_m = tl.where(next_m == -float("inf"), 0.0, next_m)
        correction = tl.where(normalizer > 0, tl.exp(m - safe_m), 0.0)
        weights = tl.exp(logits - safe_m[:, None])
        normalizer = normalizer * correction + tl.sum(weights, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            weights.to(values.dtype), values, input_precision="ieee"
        )
        m = next_m

    # Query K/V are separate from the context cache and never survive this call.
    # BLOCK_M >= 16, satisfying the tensor-core dot tile's minimum N dimension.
    query_tokens = tl.arange(0, BLOCK_M)
    key_query_rows = query_start + query_tokens
    key_query_pos = tl.load(POSITIONS + key_query_rows, query_tokens < query_len, other=0)
    query_keys = tl.load(
        K_QUERY + key_query_rows[None, :] * kq_s0 + kv_head * kq_s1 + dims[:, None] * kq_s2,
        (query_tokens[None, :] < query_len) & (dims[:, None] < HEAD_DIM),
        other=0,
    )
    query_values = tl.load(
        V_QUERY + key_query_rows[:, None] * vq_s0 + kv_head * vq_s1 + dims[None, :] * vq_s2,
        (query_tokens[:, None] < query_len) & (dims[None, :] < HEAD_DIM),
        other=0,
    )
    logits = tl.dot(q, query_keys, input_precision="ieee") * SCALE
    visible = active_rows[:, None] & (query_tokens[None, :] < query_len)
    if WINDOW > 0:
        visible = visible & (key_query_pos[None, :] >= query_pos[:, None] - WINDOW + 1)
        visible = visible & (key_query_pos[None, :] <= query_pos[:, None] + WINDOW - 1)
    if CAUSAL:
        visible = visible & (key_query_pos[None, :] <= query_pos[:, None])
    logits = tl.where(visible, logits, -float("inf"))
    next_m = tl.maximum(m, tl.max(logits, axis=1))
    safe_m = tl.where(next_m == -float("inf"), 0.0, next_m)
    correction = tl.where(normalizer > 0, tl.exp(m - safe_m), 0.0)
    weights = tl.exp(logits - safe_m[:, None])
    normalizer = normalizer * correction + tl.sum(weights, axis=1)
    accumulator = accumulator * correction[:, None] + tl.dot(
        weights.to(query_values.dtype), query_values, input_precision="ieee"
    )
    result = accumulator / tl.where(normalizer > 0, normalizer, 1.0)[:, None]
    tl.store(
        OUTPUT + query_rows[:, None] * out_s0 + head * out_s1 + dims[None, :] * out_s2,
        result,
        active_rows[:, None] & (dims[None, :] < HEAD_DIM),
    )
    if SCORE:
        all_lse = next_m + tl.log(normalizer)
        anchor_lse = tl.sum(tl.where(rows == anchor, all_lse, 0.0), axis=0)
        tl.store(
            ANCHOR_LSE + request * NUM_HEADS + head,
            tl.where(anchor_valid, anchor_lse, -float("inf")),
        )


@triton.jit
def _accumulate_page_mass_kernel(
    PAGE_LSE,
    ANCHOR_LSE,
    SCORES,
    score_s0: tl.constexpr,
    score_s1: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    request = tl.program_id(0)
    slot = tl.program_id(1)
    heads = tl.arange(0, BLOCK_H)
    page_lse = tl.load(
        PAGE_LSE + (request * NUM_HEADS + heads) * NUM_SLOTS + slot,
        heads < NUM_HEADS,
        other=-float("inf"),
    )
    denominator_lse = tl.load(
        ANCHOR_LSE + request * NUM_HEADS + heads,
        heads < NUM_HEADS,
        other=-float("inf"),
    )
    valid = (heads < NUM_HEADS) & (denominator_lse != -float("inf"))
    mass = tl.sum(tl.where(valid, tl.exp(page_lse - denominator_lse), 0.0), axis=0)
    score_ptr = SCORES + request * score_s0 + slot * score_s1
    tl.store(score_ptr, tl.load(score_ptr) + mass)


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_tags: torch.Tensor,
    context_ends: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_positions: torch.Tensor,
    anchor_indices: torch.Tensor,
    scores: torch.Tensor | None,
    *,
    window_size: int | None = 2048,
    causal: bool = False,
    softmax_scale: float | None = None,
    sinks: torch.Tensor | None = None,
    max_query_len: int = 16,
    output: torch.Tensor | None = None,
    workspace: SparseAttentionWorkspace | None = None,
) -> torch.Tensor:
    """Attend to retained original-position context and the current query block.

    ``q`` is ``[Q,Hq,D]``; ``k/v`` are ``[Q,Hkv,D]``. ``kv_cache`` uses native
    FlashAttention layout ``[physical_page,2,page_token,Hkv,D]``. Page IDs and
    original page-start token tags are ``[request,slot]`` (negative means empty).
    ``context_ends[r]`` excludes the anchor and any stale speculative cache tail.
    Query starts are a prefix sum, and anchor indices are absolute rows into q.
    ``max_query_len`` must bound each request's query count; it is host metadata,
    never inferred by copying a GPU tensor to the host.

    Original positions govern both RoPE (already applied by the caller) and the
    native sliding window, including the right window for noncausal drafting.
    With ``scores=None``, score scratch and the importance kernel are omitted.
    Supply output/workspace allocated before graph capture to keep addresses
    stable. Scores accumulate across layers; the caller owns clearing/recycling.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q/k/v must have shape [query,head,dimension]")
    num_queries, num_heads, head_dim = q.shape
    if k.shape[0] != num_queries or k.shape[2] != head_dim:
        raise ValueError("q/k/v query counts and head dimensions must agree")
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError("native cache must be [physical_page,2,page_token,Hkv,D]")
    if tuple(kv_cache.shape[3:]) != (num_kv_heads, head_dim):
        raise ValueError("cache and query KV geometry must agree")
    if block_ids.ndim != 2 or block_tags.shape != block_ids.shape:
        raise ValueError("page IDs and tags must have matching [request,slot] shapes")
    num_requests, num_slots = block_ids.shape
    if num_slots < 1 or not 1 <= max_query_len <= 64:
        raise ValueError("at least one slot and max_query_len in [1,64] are required")
    if context_ends.numel() != num_requests or anchor_indices.numel() != num_requests:
        raise ValueError("context ends and anchors must contain one entry per request")
    if query_start_loc.numel() != num_requests + 1 or query_positions.numel() != num_queries:
        raise ValueError("query starts/positions do not match the request/query counts")
    if window_size is not None and window_size < 1:
        raise ValueError("window_size must be positive or None")
    if sinks is not None and sinks.numel() != num_heads:
        raise ValueError("one virtual sink bias is required per query head")
    if head_dim > 256 or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported head dimension <=256 and floating q/k/v are required")
    if k.dtype != q.dtype or v.dtype != q.dtype or kv_cache.dtype != q.dtype:
        raise ValueError("q/k/v/cache must use the same dtype")
    if not q.is_cuda:
        raise ValueError("sparse_attention requires CUDA tensors")
    if output is None:
        output = torch.empty_like(q)
    if output.shape != q.shape or output.dtype != q.dtype:
        raise ValueError("output must match q shape and dtype")
    if scores is not None:
        if scores.shape != block_ids.shape or scores.dtype != torch.float32:
            raise ValueError("scores must be float32 [request,slot]")
        if workspace is None:
            workspace = SparseAttentionWorkspace.allocate(
                num_requests, num_heads, num_slots, q.device
            )
        if workspace.page_lse.shape != (num_requests, num_heads, num_slots):
            raise ValueError("page_lse workspace shape mismatch")
        if workspace.anchor_lse.shape != (num_requests, num_heads):
            raise ValueError("anchor_lse workspace shape mismatch")
        if not workspace.page_lse.is_contiguous() or not workspace.anchor_lse.is_contiguous():
            raise ValueError("workspace buffers must be contiguous")
        if workspace.page_lse.dtype != torch.float32 or workspace.anchor_lse.dtype != torch.float32:
            raise ValueError("workspace buffers must be float32")
    cache_k, cache_v = kv_cache.unbind(1)
    page_size = kv_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(max_query_len))
    block_n = max(16, triton.next_power_of_2(page_size))
    _sparse_attention_kernel[(num_requests, num_heads)](
        q,
        k,
        v,
        cache_k,
        cache_v,
        block_ids,
        block_tags,
        context_ends,
        query_start_loc,
        query_positions,
        anchor_indices,
        sinks if sinks is not None else q,
        output,
        workspace.page_lse if scores is not None else q,
        workspace.anchor_lse if scores is not None else q,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *cache_k.stride(),
        *cache_v.stride(),
        *block_ids.stride(),
        *block_tags.stride(),
        *output.stride(),
        NUM_HEADS=num_heads,
        HEADS_PER_KV=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        NUM_SLOTS=num_slots,
        PAGE_SIZE=page_size,
        WINDOW=window_size or 0,
        CAUSAL=causal,
        HAS_SINK=sinks is not None,
        SCORE=scores is not None,
        SCALE=softmax_scale if softmax_scale is not None else head_dim**-0.5,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        num_warps=4 if block_m <= 32 and head_dim <= 128 else 8,
    )
    if scores is not None:
        _accumulate_page_mass_kernel[(num_requests, num_slots)](
            workspace.page_lse,
            workspace.anchor_lse,
            scores,
            *scores.stride(),
            NUM_HEADS=num_heads,
            NUM_SLOTS=num_slots,
            BLOCK_H=triton.next_power_of_2(num_heads),
        )
    return output
