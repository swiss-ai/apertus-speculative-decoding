"""Pinned vLLM DSpark integration for physically bounded drafter pages.

Only the drafter is intercepted. Target attention and rejection sampling retain
vLLM's implementation. Query graphs are opt-in and use native capture shapes;
context/page bookkeeping remains eager. Target compilation/graphs are unaffected.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import time
import types

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

PAGE = 16
HORIZON = 2048
MAX_PAGES = 129
_CURRENT = None


def settings():
    policy = os.environ.get("VLLM_APERTUS_DRAFT_KV_POLICY", "native")
    if policy == "off":
        policy = "native"
    if policy not in {"native", "full", "recent", "importance"}:
        raise ValueError(f"unsupported drafter KV policy {policy!r}")
    budget = int(os.environ.get("VLLM_APERTUS_DRAFT_KV_BUDGET", "2048"))
    recent = int(os.environ.get("VLLM_APERTUS_DRAFT_KV_RECENT", "128"))
    if budget not in {512, 1024, 2048} or recent != 128:
        raise ValueError("supported drafter budgets: 512/1024/2048; recent must be128")
    return policy, budget, recent


def enabled():
    return settings()[0] != "native"


@dataclasses.dataclass(frozen=True, kw_only=True)
class SparseDraftSpec(SlidingWindowSpec):
    budget_tokens: int = 2048
    recent_tokens: int = 128
    policy: str = "full"

    def max_admission_blocks_per_request(self, max_in_flight_tokens, max_model_len):
        # Context insertion discards positions outside the native horizon even
        # for a very large prefill chunk. Query KV uses separate small scratch.
        return min(MAX_PAGES, math.ceil(max_model_len / self.block_size) + 1)


class SparseDraftManager(SlidingWindowManager):
    """Own physical slots, whose logical tags are maintained on the GPU."""

    def __init__(self, kv_cache_spec, **kwargs):
        super().__init__(kv_cache_spec, **kwargs)
        if self.enable_caching:
            raise ValueError("sparse drafter experiment requires prefix caching off")
        if self.block_size != PAGE:
            raise ValueError("sparse drafter experiment requires16-token pages")
        self.compressed = set()

    def _required(self, request_id, num_tokens):
        cap = MAX_PAGES
        if request_id in self.compressed and self.kv_cache_spec.policy != "full":
            cap = self.kv_cache_spec.budget_tokens // PAGE + 1
        return min(cap, math.ceil(num_tokens / PAGE) + 1)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        if new_computed_blocks:
            raise ValueError("sparse draft prefix cache hits are unsupported")
        return max(
            0,
            self._required(request_id, num_tokens_main_model)
            - len(self.req_to_blocks.get(request_id, ())),
        )

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        blocks = self.req_to_blocks[request_id]
        count = max(0, self._required(request_id, num_tokens_main_model) - len(blocks))
        new = self.block_pool.get_new_blocks(count) if count else []
        blocks.extend(new)
        return new

    def get_num_skipped_tokens(self, num_computed_tokens):
        return 0

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        # Recency does not determine physical ownership for this spec.
        return None

    def cache_blocks(self, request, num_tokens, retention_interval=None):
        return None

    def compress(self, request_id, keep_ids):
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return
        by_id = {block.block_id: block for block in blocks}
        if len(set(keep_ids)) != len(keep_ids) or not set(keep_ids).issubset(by_id):
            raise RuntimeError("invalid sparse drafter bootstrap release")
        released = [block for block in blocks if block.block_id not in set(keep_ids)]
        self.req_to_blocks[request_id] = [by_id[bid] for bid in keep_ids]
        self.compressed.add(request_id)
        self.block_pool.free_blocks(reversed(released))
        record = {
            "event": "bootstrap_release",
            "request_id": request_id,
            "policy": self.kv_cache_spec.policy,
            "budget_tokens": self.kv_cache_spec.budget_tokens,
            "allocated_pages_before": len(blocks),
            "allocated_pages_after": len(keep_ids),
            "released_pages": len(released),
            "page_size": PAGE,
        }
        print("APERTUS_DRAFT_KV_METRICS " + json.dumps(record), flush=True)
        trace_record(record)

    def pop_blocks_for_free(self, request_id):
        self.compressed.discard(request_id)
        return super().pop_blocks_for_free(request_id)


def register_spec():
    # Register built-ins first: registry's lazy guard considers any entry an
    # initialized registry, so registering only our class would skip built-ins.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        SparseDraftSpec, SparseDraftManager, uniform_type_base_spec=SlidingWindowSpec
    )


register_spec()


@triton.jit
def _prepare_pages(
    tags_ptr,
    scores_ptr,
    lookup_ptr,
    ends_ptr,
    positions_ptr,
    starts_ptr,
    rejected_ptr,
    idx_ptr,
    physical_counts_ptr,
    count_stride: tl.constexpr,
    group_id: tl.constexpr,
    batch_ids_ptr,
    batch_tags_ptr,
    batch_scores_ptr,
    table_ptr,
    table_stride: tl.constexpr,
    C: tl.constexpr,
    V: tl.constexpr,
    PAGE: tl.constexpr = 16,
    HORIZON: tl.constexpr = 2048,
    MAX_PAGES: tl.constexpr = 129,
):
    r = tl.program_id(0)
    state = tl.load(idx_ptr + r)
    lo = tl.load(starts_ptr + r)
    hi = tl.load(starts_ptr + r + 1) - tl.load(rejected_ptr + r)
    end = tl.load(positions_ptr + hi - 1) + 1
    first = tl.load(positions_ptr + lo)
    tl.store(ends_ptr + r, end)
    offsets = tl.arange(0, V)
    tags = tl.load(tags_ptr + state * MAX_PAGES + offsets, offsets < MAX_PAGES, other=-1)
    scores = tl.load(scores_ptr + state * MAX_PAGES + offsets, offsets < MAX_PAGES, other=0.0)
    count = tl.load(physical_counts_ptr + group_id * count_stride + state)
    tl.device_assert(count <= MAX_PAGES, "sparse draft allocated page cap exceeded")
    valid = offsets < count
    tags = tl.where(valid & (tags + PAGE > end - (HORIZON - 1)), tags, -1)
    scores = tl.where(tags >= 0, scores, 0.0)
    page = tl.maximum(first, end - (HORIZON - 1)) // PAGE * PAGE
    while page < end:
        exists = tl.sum((tags == page).to(tl.int32), 0) > 0
        if not exists:
            free = tl.min(tl.where(valid & (tags < 0), offsets, V), 0)
            # count is sized for native bootstrap or budget+staging. An absent
            # free slot is an invariant violation detected through error tags.
            tl.device_assert(free < V, "sparse draft page capacity exhausted")
            tags = tl.where(offsets == free, page, tags)
            scores = tl.where(offsets == free, 0.0, scores)
        page += PAGE
    tl.store(tags_ptr + state * MAX_PAGES + offsets, tags, offsets < MAX_PAGES)
    tl.store(scores_ptr + state * MAX_PAGES + offsets, scores, offsets < MAX_PAGES)
    tl.store(lookup_ptr + state * MAX_PAGES + offsets, -1, offsets < MAX_PAGES)
    # modulo129 is injective across the at-most129-page native horizon.
    tl.store(
        lookup_ptr + state * MAX_PAGES + (tags // PAGE) % MAX_PAGES,
        offsets,
        (offsets < MAX_PAGES) & (tags >= 0),
    )
    phys = tl.load(table_ptr + state * table_stride + offsets, offsets < count, other=-1)
    tl.store(batch_ids_ptr + r * C + offsets, phys, offsets < C)
    tl.store(batch_tags_ptr + r * C + offsets, tags, offsets < C)
    tl.store(batch_scores_ptr + r * C + offsets, scores, offsets < C)


@triton.jit
def _context_slots(
    out_ptr,
    context_positions_ptr,
    positions_ptr,
    starts_ptr,
    rejected_ptr,
    idx_ptr,
    ends_ptr,
    lookup_ptr,
    tags_ptr,
    table_ptr,
    table_stride: tl.constexpr,
    B: tl.constexpr,
    PAGE: tl.constexpr = 16,
    HORIZON: tl.constexpr = 2048,
    MAX_PAGES: tl.constexpr = 129,
):
    r, block = tl.program_id(0), tl.program_id(1)
    state = tl.load(idx_ptr + r)
    start = tl.load(starts_ptr + r)
    stop = tl.load(starts_ptr + r + 1)
    valid_stop = stop - tl.load(rejected_ptr + r)
    end = tl.load(ends_ptr + r)
    i = start + block * B + tl.arange(0, B)
    pos = tl.load(positions_ptr + i, i < stop, other=0)
    page = pos // PAGE * PAGE
    slot = tl.load(lookup_ptr + state * MAX_PAGES + (page // PAGE) % MAX_PAGES)
    tag = tl.load(tags_ptr + state * MAX_PAGES + tl.maximum(slot, 0))
    phys = tl.load(table_ptr + state * table_stride + tl.maximum(slot, 0))
    valid = (i < valid_stop) & (pos >= end - (HORIZON - 1)) & (slot >= 0) & (tag == page)
    tl.store(out_ptr + i, tl.where(valid, phys * PAGE + pos % PAGE, -1), i < stop)
    tl.store(context_positions_ptr + i, pos, i < stop)


@triton.jit
def _finish_pages(
    tags_ptr,
    scores_ptr,
    batch_scores_ptr,
    ends_ptr,
    idx_ptr,
    final_ptr,
    C: tl.constexpr,
    V: tl.constexpr,
    BUDGET: tl.constexpr,
    RECENT: tl.constexpr,
    FULL: tl.constexpr,
    IMPORTANCE: tl.constexpr,
    PAGE: tl.constexpr = 16,
    MAX_PAGES: tl.constexpr = 129,
):
    r = tl.program_id(0)
    state = tl.load(idx_ptr + r)
    offsets = tl.arange(0, V)
    tags = tl.load(tags_ptr + state * MAX_PAGES + offsets, offsets < MAX_PAGES, other=-1)
    old_scores = tl.load(scores_ptr + state * MAX_PAGES + offsets, offsets < MAX_PAGES, other=0.0)
    scores = tl.load(batch_scores_ptr + r * C + offsets, offsets < C, other=0.0)
    final = tl.load(final_ptr + r)
    end = tl.load(ends_ptr + r)
    if final and not FULL:
        n = tl.sum((tags >= 0).to(tl.int32), 0)
        while n > BUDGET:
            protected = tags + PAGE > end - RECENT
            eligible = (tags >= 0) & ~protected
            rank = scores if IMPORTANCE else tags.to(tl.float32)
            smallest = tl.min(tl.where(eligible, rank, float("inf")), 0)
            oldest = tl.min(tl.where(eligible & (rank == smallest), tags, 2147483647), 0)
            kill = tl.min(tl.where(eligible & (tags == oldest), offsets, V), 0)
            tags = tl.where(offsets == kill, -1, tags)
            scores = tl.where(offsets == kill, 0.0, scores)
            n -= 1
    scores = tl.where(final, scores, old_scores)
    tl.store(tags_ptr + state * MAX_PAGES + offsets, tags, offsets < MAX_PAGES)
    tl.store(scores_ptr + state * MAX_PAGES + offsets, scores, offsets < MAX_PAGES)


class SparseDraftRuntime:
    def __init__(self, speculator):
        self.owner = speculator
        self.policy, self.budget, self.recent = settings()
        config = speculator.vllm_config
        if config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("sparse drafter experiment currently requires TP1")
        if not speculator.sample_from_anchor:
            raise ValueError("sparse drafter scoring requires anchor-first DSpark")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("sparse drafter experiment requires prefix caching off")
        groups = speculator.draft_kv_cache_group_ids
        if len(groups) != 1:
            raise ValueError("sparse DSpark requires one shared five-layer KV group")
        self.gid = groups[0]
        if speculator.block_tables.kernel_block_sizes[self.gid] != PAGE:
            raise ValueError("sparse DSpark requires16-token kernel pages")
        self.device = speculator.device
        self.tags = torch.full(
            (speculator.max_num_reqs, MAX_PAGES), -1, dtype=torch.int64, device=self.device
        )
        self.scores = torch.zeros_like(self.tags, dtype=torch.float32)
        self.lookup = torch.full_like(self.tags, -1, dtype=torch.int32)
        self.registered = {}
        self.compressed = set()
        self.releases = {}
        self.steps = 0
        self.active = None
        self.workspace = None
        self.query_graphs = None
        self.worker_trace = None

    def remove_request(self, req_id):
        state = self.registered.pop(req_id, None)
        if state is not None:
            self.tags[state].fill_(-1)
            self.scores[state].zero_()
        self.compressed.discard(req_id)
        self.releases.pop(req_id, None)

    def attention(self, layer, q, k, v, positions):
        from .attention import SparseAttentionWorkspace, sparse_attention

        info = self.active
        if info is None:
            return layer.attn(q, k, v)
        q = q.view(-1, layer.num_heads, layer.head_dim)
        k = k.view(-1, layer.num_kv_heads, layer.head_dim)
        v = v.view(-1, layer.num_kv_heads, layer.head_dim)
        if self.workspace is None or self.workspace.page_lse.shape != (
            info["ids"].shape[0],
            layer.num_heads,
            info["ids"].shape[1],
        ):
            self.workspace = SparseAttentionWorkspace.allocate(
                info["ids"].shape[0], layer.num_heads, info["ids"].shape[1], self.device
            )
        out = sparse_attention(
            q,
            k,
            v,
            layer.attn.kv_cache,
            info["ids"],
            info["tags"],
            info["ends"],
            info["query_starts"],
            positions,
            info["anchors"],
            info["scores"] if self.policy == "importance" else None,
            window_size=HORIZON,
            causal=layer.causal,
            softmax_scale=layer.scaling,
            sinks=layer.attention_sink_bias,
            max_query_len=self.owner.num_query_per_req,
            workspace=self.workspace,
        )
        return out.reshape(q.shape[0], -1)

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        global _CURRENT
        s = self.owner
        self.worker_trace = None
        n = input_batch.num_reqs
        tokens = input_batch.num_tokens
        for req_id, state in zip(input_batch.req_ids, input_batch.idx_mapping_np, strict=True):
            state = int(state)
            if req_id not in self.registered:
                self.tags[state].fill_(-1)
                self.scores[state].zero_()
                self.registered[req_id] = state
        if aux_hidden_states:
            hidden = s.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        else:
            hidden = last_hidden_states
        s.hidden_states[:tokens].copy_(hidden[:tokens])
        s._copy_request_inputs(n, input_batch.idx_mapping, temperature, seeds)
        bt = s.block_tables
        table = bt.block_tables[self.gid].gpu
        max_pages = (
            MAX_PAGES
            if (
                self.policy == "full"
                or any(rid not in self.compressed for rid in input_batch.req_ids)
            )
            else self.budget // PAGE + 1
        )
        ids = torch.empty((n, max_pages), dtype=torch.int32, device=self.device)
        tags = torch.empty((n, max_pages), dtype=torch.int64, device=self.device)
        scores = torch.empty((n, max_pages), dtype=torch.float32, device=self.device)
        ends = torch.empty(n, dtype=torch.int64, device=self.device)
        timing = None
        if trace_due(self.steps + 1):
            timing = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            timing[0].record()
        _prepare_pages[(n,)](
            self.tags,
            self.scores,
            self.lookup,
            ends,
            input_batch.positions,
            input_batch.query_start_loc,
            num_rejected,
            input_batch.idx_mapping,
            bt.num_blocks.gpu,
            bt.num_blocks.gpu.stride(0),
            self.gid,
            ids,
            tags,
            scores,
            table,
            table.stride(0),
            C=max_pages,
            V=256,
            PAGE=PAGE,
            HORIZON=HORIZON,
            MAX_PAGES=MAX_PAGES,
            debug=True,
        )
        # Populate the original query/sample buffers, but use dummy slot output:
        # query K/V are consumed directly by sparse_attention, never persisted.
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

        prepare_dflash_inputs(
            s.input_buffers,
            bt.slot_mappings[self.gid],
            s.context_positions,
            s._context_slot_mappings[0],
            s.sample_indices,
            s.sample_pos,
            s.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            table[input_batch.idx_mapping],
            PAGE,
            s.parallel_drafting_token_id,
            s.num_query_per_req,
            s.num_speculative_steps,
            s.max_num_reqs,
            s.max_num_tokens,
            s.max_model_len,
            s.sample_from_anchor,
        )
        # Native preparation writes absolute-index slot maps; replace context
        # slots with the physical retained-page mapping before any cache write.
        _context_slots[(n, triton.cdiv(int(input_batch.num_scheduled_tokens.max()), 256))](
            s._context_slot_mappings[0],
            s.context_positions,
            input_batch.positions,
            input_batch.query_start_loc,
            num_rejected,
            input_batch.idx_mapping,
            ends,
            self.lookup,
            self.tags,
            table,
            table.stride(0),
            B=256,
            PAGE=PAGE,
            HORIZON=HORIZON,
            MAX_PAGES=MAX_PAGES,
        )
        if timing is not None:
            timing[1].record()
        s.model.precompute_and_store_context_kv(
            s.hidden_states[:tokens],
            s.context_positions[:tokens],
            s._context_slot_mappings[0, :tokens],
        )
        query_count = n * s.num_query_per_req
        s.input_buffers.is_padding[:query_count].fill_(False)
        query_starts = s.input_buffers.query_start_loc[: n + 1]
        anchors = torch.arange(n, device=self.device, dtype=torch.int64) * s.num_query_per_req
        self.active = {
            "ids": ids,
            "tags": tags,
            "scores": scores,
            "ends": ends,
            "query_starts": query_starts,
            "anchors": anchors,
        }
        _CURRENT = self
        try:
            s._prepare_eplb_forward(query_count)
            from vllm.config.compilation import CUDAGraphMode

            replayed = self.query_graphs is not None and self.query_graphs.run(self.active, n)
            if not replayed:
                s._generate_draft(
                    n,
                    query_count,
                    None,
                    None,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )
        finally:
            _CURRENT = None
            self.active = None
        is_final = (
            input_batch.num_computed_prefill_tokens_np + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        final = torch.as_tensor(is_final, device=self.device, dtype=torch.int32)
        if timing is not None:
            timing[2].record()
        _finish_pages[(n,)](
            self.tags,
            self.scores,
            scores,
            ends,
            input_batch.idx_mapping,
            final,
            C=max_pages,
            V=256,
            BUDGET=self.budget // PAGE,
            RECENT=self.recent,
            FULL=self.policy == "full",
            IMPORTANCE=self.policy == "importance",
            PAGE=PAGE,
            MAX_PAGES=MAX_PAGES,
        )
        if timing is not None:
            timing[3].record()
        for i, (req_id, state) in enumerate(
            zip(input_batch.req_ids, input_batch.idx_mapping_np, strict=True)
        ):
            if req_id in self.compressed or not is_final[i]:
                continue
            if self.policy == "full":
                self.compressed.add(req_id)
                continue
            state = int(state)
            count = int(bt.num_blocks.np[self.gid, state])
            row_tags = self.tags[state, :count].cpu().tolist()
            keep = [j for j, tag in enumerate(row_tags) if tag >= 0]
            spare = next((j for j, tag in enumerate(row_tags) if tag < 0), None)
            if spare is not None:
                keep.append(spare)
            selected = torch.tensor(keep, dtype=torch.long, device=self.device)
            selected_ids = table[state, :count].index_select(0, selected)
            keep_ids = selected_ids.cpu().tolist()
            selected_tags = self.tags[state, :count].index_select(0, selected)
            selected_scores = self.scores[state, :count].index_select(0, selected)
            self.tags[state].fill_(-1)
            self.scores[state].zero_()
            self.tags[state, : len(keep)].copy_(selected_tags)
            self.scores[state, : len(keep)].copy_(selected_scores)
            table[state, : len(keep)].copy_(selected_ids)
            bt.num_blocks.np[self.gid, state] = len(keep)
            self.releases[req_id] = keep_ids
            self.compressed.add(req_id)
        bt.num_blocks.copy_to_uva()
        self.steps += 1
        if self.steps == 1:
            print(
                "APERTUS_DRAFT_KV_METRICS "
                + json.dumps(
                    {
                        "event": "runtime",
                        "policy": self.policy,
                        "budget_tokens": self.budget,
                        "horizon_tokens": HORIZON,
                        "recent_tokens": self.recent,
                        "drafter_cuda_graphs": (
                            self.query_graphs is not None and bool(self.query_graphs.manager.graphs)
                        ),
                        "target_attention_unchanged": True,
                    }
                ),
                flush=True,
            )
        state_selection_ms = None
        if timing is not None:
            timing[3].synchronize()
            state_selection_ms = timing[0].elapsed_time(timing[1]) + timing[2].elapsed_time(
                timing[3]
            )
        self.trace(input_batch, ends, state_selection_ms)
        return s.draft_tokens[:n]

    def trace(self, input_batch, ends, state_selection_ms):
        if not trace_due(self.steps):
            return
        indices = input_batch.idx_mapping_np.tolist()
        row_tags = self.tags[indices].cpu().tolist()
        counts = [(sum(tag >= 0 for tag in row)) for row in row_tags]
        owned = [int(self.owner.block_tables.num_blocks.np[self.gid, i]) for i in indices]
        metadata = self.tags.numel() * 16
        if self.workspace is not None:
            metadata += (self.workspace.page_lse.numel() + self.workspace.anchor_lse.numel()) * 4
        record = {
            "event": "step",
            "step": self.steps,
            "request_ids": input_batch.req_ids,
            "policy": self.policy,
            "budget_tokens": self.budget,
            "native_sequence_lengths": ends.cpu().tolist(),
            "draft_context_pages": counts,
            "draft_allocated_pages": owned,
            "draft_metadata_bytes": metadata,
            "draft_context_live_bytes": sum(counts) * PAGE * 2 * 8 * 128 * 2 * 5,
            "draft_context_projection_scratch_bytes": (
                input_batch.num_tokens * 2 * 8 * 128 * 2 * 5
            ),
            "draft_query_scratch_bytes": len(indices)
            * self.owner.num_query_per_req
            * 2
            * 8
            * 128
            * 2
            * 5,
            "bootstrap_active_requests": sum(r not in self.compressed for r in input_batch.req_ids),
            "bootstrap_peak_pages_per_request": MAX_PAGES,
            "selection_ms": None,
            "selection_state_ms": state_selection_ms,
            "selection_ms_note": (
                "state mapping/pruning measured; "
                "importance score kernel included in drafter forward"
            ),
            "drafter_cuda_graphs": False,
            "target_attention_unchanged": True,
        }
        if self.query_graphs is not None:
            record.update(
                {
                    "drafter_cuda_graphs": self.query_graphs.last_replayed,
                    "drafter_graphs_captured": len(self.query_graphs.manager.graphs),
                    "drafter_graph_replays": self.query_graphs.replay_count,
                    "drafter_eager_fallbacks": self.query_graphs.fallback_count,
                    "drafter_graph_padded_requests": self.query_graphs.last_padded_requests,
                    "draft_graph_metadata_bytes": self.query_graphs.metadata_bytes,
                }
            )
            record["draft_metadata_bytes"] += self.query_graphs.metadata_bytes
        trace_record(record)
        self.worker_trace = record


def sparse_layer_attention(layer, q, k, v, positions):
    if _CURRENT is None:
        return layer.attn(q, k, v)
    return _CURRENT.attention(layer, q, k, v, positions)


def install_model_hooks(namespace):
    cls = namespace["DFlashQwen3Attention"]
    original = cls.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not enabled():
            return
        old_spec = self.attn.get_kv_cache_spec

        def get_spec(attention, config):
            spec = old_spec(config)
            if not isinstance(spec, SlidingWindowSpec) or spec.sliding_window != HORIZON:
                raise ValueError("DSpark sparse KV requires native sliding_window2048")
            policy, budget, recent = settings()
            fields = {field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)}
            return SparseDraftSpec(
                **fields, policy=policy, budget_tokens=budget, recent_tokens=recent
            )

        self.attn.get_kv_cache_spec = types.MethodType(get_spec, self.attn)

    cls.__init__ = initialize


def install_speculator_hooks(namespace):
    cls = namespace["DFlashSpeculator"]
    original = cls.propose
    original_graphs = cls.init_cudagraph_manager
    original_capture = cls.capture

    @functools.wraps(original)
    def propose(self, *args, **kwargs):
        if not enabled() or kwargs.get("dummy_run", False):
            result = original(self, *args, **kwargs)
            if not kwargs.get("dummy_run", False):
                trace_native(self, args[0], args[6])
            return result
        if not hasattr(self, "_apertus_sparse_runtime"):
            self._apertus_sparse_runtime = SparseDraftRuntime(self)
        return self._apertus_sparse_runtime.propose(*args, **kwargs)

    @functools.wraps(original_graphs)
    def init_graphs(self, mode):
        if enabled():
            from vllm.config.compilation import CUDAGraphMode

            mode = CUDAGraphMode.NONE
        return original_graphs(self, mode)

    @functools.wraps(original_capture)
    def capture(self, attn_states=None):
        if not enabled() or os.environ.get("VLLM_APERTUS_DRAFT_KV_GRAPHS", "0") != "1":
            return original_capture(self, attn_states)
        from .graphs import SparseQueryGraphs

        if not hasattr(self, "_apertus_sparse_runtime"):
            self._apertus_sparse_runtime = SparseDraftRuntime(self)
        runtime = self._apertus_sparse_runtime
        runtime.query_graphs = SparseQueryGraphs(runtime)
        runtime.query_graphs.capture()
        return None

    cls.propose = propose
    cls.init_cudagraph_manager = init_graphs
    cls.capture = capture


def notify_request_removed(speculator, req_id):
    runtime = getattr(speculator, "_apertus_sparse_runtime", None)
    if runtime is not None:
        runtime.remove_request(req_id)


def attach_release_events(speculator, output):
    runtime = getattr(speculator, "_apertus_sparse_runtime", None)
    if runtime is not None and runtime.releases:
        output.apertus_draft_kv_releases = runtime.releases
        runtime.releases = {}
    worker_trace = (
        runtime.worker_trace
        if runtime is not None
        else getattr(speculator, "_apertus_worker_trace", None)
    )
    if worker_trace is not None:
        output.apertus_draft_kv_trace = worker_trace
    if runtime is not None:
        runtime.worker_trace = None
    else:
        speculator._apertus_worker_trace = None


def apply_release_events(kv_cache_manager, output):
    releases = getattr(output, "apertus_draft_kv_releases", None)
    for manager in kv_cache_manager.coordinator.single_type_managers:
        if releases and isinstance(manager, SparseDraftManager):
            for req_id, keep_ids in releases.items():
                manager.compress(req_id, keep_ids)
    trace_allocations(kv_cache_manager, output)


def trace_due(step):
    return bool(os.environ.get("VLLM_DRAFT_KV_TRACE")) and (step == 1 or step % 64 == 0)


def trace_record(record):
    path = os.environ.get("VLLM_DRAFT_KV_TRACE")
    if not path:
        return
    record["timestamp_ns"] = time.time_ns()
    payload = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def trace_native(speculator, batch, rejected):
    step = getattr(speculator, "_apertus_trace_step", 0) + 1
    speculator._apertus_trace_step = step
    speculator._apertus_worker_trace = None
    if not trace_due(step):
        return
    rejection = rejected.cpu().tolist()
    positions = batch.positions.cpu().tolist()
    ends = [
        positions[int(batch.query_start_loc_np[i + 1]) - rejection[i] - 1] + 1
        for i in range(batch.num_reqs)
    ]
    manager = speculator.query_cudagraph_manager
    graph_mode = False
    if manager is not None:
        from vllm.config.compilation import CUDAGraphMode

        desc = manager.dispatch(
            batch.num_reqs,
            batch.num_reqs * speculator.num_query_per_req,
            speculator.num_query_per_req,
            0,
        )
        graph_mode = desc.cg_mode == CUDAGraphMode.FULL
    record = {
        "event": "step",
        "step": step,
        "request_ids": batch.req_ids,
        "policy": "off",
        "budget_tokens": HORIZON,
        "native_sequence_lengths": ends,
        "draft_context_projection_scratch_bytes": batch.num_tokens * 2 * 8 * 128 * 2 * 5,
        "draft_query_scratch_bytes": (
            batch.num_reqs * speculator.num_query_per_req * 2 * 8 * 128 * 2 * 5
        ),
        "draft_metadata_bytes": 0,
        "bootstrap_active_requests": 0,
        "selection_ms": 0.0,
        "drafter_cuda_graphs": graph_mode,
        "drafter_graphs_captured": len(manager.graphs) if manager is not None else 0,
        "target_attention_unchanged": True,
    }
    trace_record(record)
    speculator._apertus_worker_trace = record


def trace_allocations(cache_manager, output):
    step = getattr(cache_manager, "_apertus_trace_step", 0) + 1
    cache_manager._apertus_trace_step = step
    worker_snapshot = getattr(output, "apertus_draft_kv_trace", None)
    if not trace_due(step) and worker_snapshot is None:
        return
    groups = cache_manager.coordinator.kv_cache_config.kv_cache_groups
    managers = cache_manager.coordinator.single_type_managers
    all_requests = sorted({rid for manager in managers for rid in manager.req_to_blocks})
    draft_pages = target_pages = draft_bytes = target_bytes = 0
    details = []
    for group, manager in zip(groups, managers, strict=True):
        spec = group.kv_cache_spec
        # In the fixed8B DSpark target, only drafter layers are SWA.
        draft = isinstance(spec, SlidingWindowSpec)
        counts = [
            sum(not block.is_null for block in manager.req_to_blocks.get(rid, ()))
            for rid in all_requests
        ]
        pages = sum(counts)
        size = spec.page_size_bytes * len(group.layer_names)
        if draft:
            draft_pages += pages
            draft_bytes += pages * size
        else:
            target_pages += pages
            target_bytes += pages * size
        details.append(
            {
                "group_id": manager.kv_cache_group_id,
                "drafter": draft,
                "pages_by_request": counts,
                "bytes_per_logical_page": size,
            }
        )
    config = cache_manager.coordinator.kv_cache_config
    physical_bytes_per_pool_block = (
        sum(t.size for t in config.kv_cache_tensors) // config.num_blocks
    )
    trace_record(
        {
            "event": "allocation",
            "step": step,
            "request_ids": all_requests,
            "scheduled_request_ids": output.req_ids,
            "worker_snapshot": worker_snapshot,
            "policy": "off" if settings()[0] == "native" else settings()[0],
            "budget_tokens": settings()[1],
            "draft_allocated_pages_total": draft_pages,
            "target_allocated_pages_total": target_pages,
            "draft_allocated_bytes": draft_pages * physical_bytes_per_pool_block,
            "target_allocated_bytes": target_pages * physical_bytes_per_pool_block,
            "draft_live_bytes": draft_bytes,
            "target_live_bytes": target_bytes,
            "physical_bytes_per_pool_block": physical_bytes_per_pool_block,
            "shared_pool_reserved_bytes": sum(t.size for t in config.kv_cache_tensors),
            "groups": details,
        }
    )


def sparse_draft_config(config):
    """Disable compilation for this draft model alone, preserving target config."""
    if not enabled():
        return config
    from vllm.config import replace
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    original = config.compilation_config
    draft_compilation = replace(
        original, mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
    )
    # vLLM's replace intentionally reconstructs only init=True fields. The
    # static forward registry is init=False, so it otherwise becomes an empty
    # dictionary. Attention layers registered while loading the drafter must
    # remain visible to the owner's profiling and forward-context lookup.
    # Share these registries by identity, as native DSpark loading does.
    draft_compilation.static_forward_context = original.static_forward_context
    draft_compilation.static_all_moe_layers = original.static_all_moe_layers
    draft_config = replace(config, compilation_config=draft_compilation)
    if (
        draft_config.compilation_config.static_forward_context
        is not original.static_forward_context
    ):
        raise RuntimeError("sparse draft config lost the shared attention registry")
    return draft_config
