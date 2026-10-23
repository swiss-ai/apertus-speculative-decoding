"""CUDA correctness checks for bounded drafter attention (no model deployment).

Run inside the serving image: ``python -m draft_kv.check_gpu`` with ``serving``
on PYTHONPATH, or execute this file directly. Fails if CUDA/Triton are absent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from draft_kv.attention import SparseAttentionWorkspace, sparse_attention
else:
    from .attention import SparseAttentionWorkspace, sparse_attention


def reference_attention(
    q,
    k,
    v,
    kv_cache,
    block_ids,
    block_tags,
    context_ends,
    query_start_loc,
    query_positions,
    anchor_indices,
    *,
    window_size,
    causal,
    softmax_scale=None,
    sinks=None,
):
    """Independent dense fp32 definition; only the test copies metadata to CPU."""
    num_queries, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    num_requests, num_slots = block_ids.shape
    page_size = kv_cache.shape[2]
    output = torch.zeros((num_queries, num_heads, head_dim), device=q.device)
    mass = torch.zeros((num_requests, num_slots), device=q.device)
    ids = block_ids.cpu().tolist()
    tags = block_tags.cpu().tolist()
    ends = context_ends.cpu().tolist()
    starts = query_start_loc.cpu().tolist()
    anchors = anchor_indices.cpu().tolist()
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    for request in range(num_requests):
        begin, end = starts[request : request + 2]
        if begin == end:
            continue
        context_k, context_v, positions, slots = [], [], [], []
        for slot, (physical, position) in enumerate(zip(ids[request], tags[request], strict=True)):
            if physical < 0 or position < 0:
                continue
            context_k.append(kv_cache[physical, 0].float())
            context_v.append(kv_cache[physical, 1].float())
            positions.extend(range(position, position + page_size))
            slots.extend([slot] * page_size)
        n_context = len(positions)
        key_tensors = context_k + [k[begin:end].float()]
        value_tensors = context_v + [v[begin:end].float()]
        all_k = torch.cat(key_tensors).repeat_interleave(num_heads // num_kv_heads, dim=1)
        all_v = torch.cat(value_tensors).repeat_interleave(num_heads // num_kv_heads, dim=1)
        context_positions = torch.tensor(positions, device=q.device, dtype=torch.int64)
        all_positions = torch.cat((context_positions, query_positions[begin:end]))
        query_pos = query_positions[begin:end]
        visible = torch.cat(
            (
                context_positions < ends[request],
                torch.ones(end - begin, dtype=torch.bool, device=q.device),
            )
        ).expand(end - begin, -1)
        if window_size is not None:
            visible = visible & (all_positions[None, :] >= query_pos[:, None] - window_size + 1)
            visible = visible & (all_positions[None, :] <= query_pos[:, None] + window_size - 1)
        if causal:
            visible = visible & (all_positions[None, :] <= query_pos[:, None])
        logits = torch.einsum("mhd,nhd->hmn", q[begin:end].float(), all_k) * scale
        logits.masked_fill_(~visible[None, :, :], -torch.inf)
        if sinks is not None:
            sink_logits = sinks.float()[:, None, None].expand(-1, end - begin, 1)
            logits = torch.cat((logits, sink_logits), dim=-1)
            all_v = torch.cat((all_v, torch.zeros((1, num_heads, head_dim), device=q.device)))
        probabilities = logits.softmax(-1)
        output[begin:end] = torch.einsum("hmn,nhd->mhd", probabilities, all_v)
        anchor = anchors[request] - begin
        if 0 <= anchor < end - begin and n_context:
            page_mass = probabilities[:, anchor, :n_context].sum(0)
            mass[request].index_add_(
                0, torch.tensor(slots, device=q.device, dtype=torch.int64), page_mass
            )
    return output, mass


def make_case(dtype, num_slots=32, page_size=16, head_dim=128):
    torch.manual_seed(80917)
    device = torch.device("cuda")
    num_requests, num_heads, num_kv_heads = 3, 8, 2
    starts_host = [0, 8, 13, 13]  # Ragged requests plus one graph-padding request.
    ends_host = [4099, 3077, 10000]
    total_queries = starts_host[-1]
    # Slice a shared projection to exercise non-contiguous q/k/v strides.
    projected = torch.randn(
        (total_queries, num_heads + 2 * num_kv_heads, head_dim), device=device, dtype=dtype
    )
    q = projected[:, :num_heads]
    k = projected[:, num_heads : num_heads + num_kv_heads]
    v = projected[:, num_heads + num_kv_heads :]
    physical_pages = num_requests * num_slots + 7
    # Also exercise noncontiguous cache dimensions, including the unbound K/V.
    storage = torch.randn(
        (physical_pages, 2, page_size, num_kv_heads, 2 * head_dim),
        device=device,
        dtype=dtype,
    )
    cache = storage[..., ::2]
    physical_ids = torch.randperm(physical_pages)[: num_requests * num_slots].reshape(
        num_requests, num_slots
    )
    tags_host = [
        [end // page_size * page_size - slot * page_size for slot in range(num_slots)]
        for end in ends_host
    ]
    for request in range(num_requests):
        # One wholly expired page, and holes that must never read physical -1.
        tags_host[request][-1] = (ends_host[request] - 2048) // page_size * page_size - page_size
        for slot in range(5, num_slots - 1, 7):
            physical_ids[request, slot] = -1
            tags_host[request][slot] = -1
    positions_host = [
        ends_host[request] + offset
        for request in range(num_requests)
        for offset in range(starts_host[request + 1] - starts_host[request])
    ]
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=cache,
        block_ids=physical_ids.to(device=device, dtype=torch.int32),
        block_tags=torch.tensor(tags_host, device=device, dtype=torch.int64),
        context_ends=torch.tensor(ends_host, device=device, dtype=torch.int64),
        query_start_loc=torch.tensor(starts_host, device=device, dtype=torch.int32),
        query_positions=torch.tensor(positions_host, device=device, dtype=torch.int64),
        anchor_indices=torch.tensor(starts_host[:-1], device=device, dtype=torch.int32),
    )


def compare_case(dtype, num_slots, causal, with_sink, window_size, page_size=16):
    case = make_case(dtype, num_slots, page_size)
    sinks = torch.linspace(-1.5, 3.0, case["q"].shape[1], device="cuda") if with_sink else None
    options = dict(causal=causal, window_size=window_size, sinks=sinks)
    expected, expected_mass = reference_attention(**case, **options)
    scores = torch.zeros_like(expected_mass)
    workspace = SparseAttentionWorkspace.allocate(
        case["block_ids"].shape[0], case["q"].shape[1], num_slots, case["q"].device
    )
    output = torch.empty_like(case["q"])
    actual = sparse_attention(
        **case,
        scores=scores,
        output=output,
        workspace=workspace,
        max_query_len=8,
        **options,
    )
    atol = 0.025 if dtype == torch.bfloat16 else 0.006
    torch.testing.assert_close(actual.float(), expected, rtol=0.015, atol=atol)
    torch.testing.assert_close(scores, expected_mass, rtol=0.003, atol=0.001)
    assert torch.isfinite(actual).all() and torch.isfinite(scores).all()
    assert torch.count_nonzero(scores[-1]) == 0, "padding request acquired importance"
    assert (scores.sum(-1) <= case["q"].shape[1] + 0.001).all(), (
        "mass omitted query/sink denominator"
    )
    # Accumulation is across five drafter layers, never an overwrite per layer.
    for _ in range(4):
        sparse_attention(
            **case,
            scores=scores,
            output=output,
            workspace=workspace,
            max_query_len=8,
            **options,
        )
    torch.testing.assert_close(scores, 5 * expected_mass, rtol=0.003, atol=0.003)
    # The inexpensive policy may omit score collection entirely.
    sparse_attention(**case, scores=None, output=output, max_query_len=8, **options)
    torch.testing.assert_close(output.float(), expected, rtol=0.015, atol=atol)
    return {
        "dtype": str(dtype),
        "slots": num_slots,
        "page_size": page_size,
        "causal": causal,
        "sinks": with_sink,
        "window": window_size,
        "max_output_error": (actual.float() - expected).abs().max().item(),
        "max_mass_error": (scores - 5 * expected_mass).abs().max().item(),
    }


def check_graph_replay():
    case = make_case(torch.bfloat16, num_slots=32)
    q = case["q"]
    scores = torch.zeros(case["block_ids"].shape, device="cuda", dtype=torch.float32)
    output = torch.empty_like(q)
    workspace = SparseAttentionWorkspace.allocate(3, q.shape[1], 32, q.device)
    options = dict(causal=False, window_size=2048, max_query_len=8)
    for _ in range(2):
        sparse_attention(**case, scores=scores, output=output, workspace=workspace, **options)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scores.zero_()
        for _ in range(5):
            sparse_attention(**case, scores=scores, output=output, workspace=workspace, **options)
    # Replay with changed values and page mapping, retaining every buffer address.
    q.mul_(0.75)
    case["block_ids"][:, 0].copy_(case["block_ids"][:, 1])
    case["block_tags"][:, 0].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    expected, expected_mass = reference_attention(**case, causal=False, window_size=2048)
    torch.testing.assert_close(output.float(), expected, rtol=0.015, atol=0.025)
    torch.testing.assert_close(scores, 5 * expected_mass, rtol=0.003, atol=0.003)
    assert torch.count_nonzero(scores[:, 0]) == 0
    return {"cuda_graph_replay": "passed", "layers": 5}


def check_large_cache_offsets():
    """Reject int32 address overflow with realistic global-pool physical IDs.

    A native bf16 page occupies32768 elements. ID65536 therefore crosses the
    signed32-bit element-offset boundary. Allocate the4.3GB address span without
    initializing the backing history; only five selected pages are touched.
    """
    device = torch.device("cuda")
    torch.manual_seed(22091)
    cache = torch.empty((65538, 2, 16, 8, 128), device=device, dtype=torch.bfloat16)
    initialized_ids = torch.tensor([0, 32768, 65535, 65536, 65537], device=device)
    payload = torch.randn((5, 2, 16, 8, 128), device=device, dtype=cache.dtype)
    cache.index_copy_(0, initialized_ids, payload)
    q = torch.randn((8, 32, 128), device=device, dtype=cache.dtype)
    k = torch.randn((8, 8, 128), device=device, dtype=cache.dtype)
    v = torch.randn_like(k)
    starts = torch.tensor([0, 8], device=device, dtype=torch.int32)
    ends = torch.tensor([4053], device=device, dtype=torch.int64)
    positions = torch.arange(4053, 4061, device=device, dtype=torch.int64)
    anchors = torch.tensor([0], device=device, dtype=torch.int32)
    sinks = torch.linspace(-1.0, 2.0, 32, device=device)
    # The second view models layouts whose page stride includes large padding
    # or layer interleaving. It uses the same actual allocation and populated
    # physical memory, so neither case fabricates an out-of-bounds tensor view.
    strided = cache.as_strided((3, 2, 16, 8, 128), (2**30, 16384, 1024, 128, 1))
    reports = []
    for name, cache_view, ids_host in (
        ("native_high_ids", cache, [0, 65535, 65536, 65537]),
        ("large_page_stride", strided, [0, 1, 2]),
    ):
        ids = torch.tensor([ids_host], device=device, dtype=torch.int32)
        tags = torch.arange(4000, 4000 + 16 * len(ids_host), 16, device=device).view(1, -1)
        case = dict(
            q=q,
            k=k,
            v=v,
            kv_cache=cache_view,
            block_ids=ids,
            block_tags=tags,
            context_ends=ends,
            query_start_loc=starts,
            query_positions=positions,
            anchor_indices=anchors,
        )
        expected, expected_mass = reference_attention(
            **case, window_size=2048, causal=False, sinks=sinks
        )
        scores = torch.zeros_like(expected_mass)
        actual = sparse_attention(
            **case,
            scores=scores,
            window_size=2048,
            causal=False,
            sinks=sinks,
            max_query_len=8,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected, rtol=0.015, atol=0.025)
        torch.testing.assert_close(scores, expected_mass, rtol=0.003, atol=0.001)
        maximum_offset = max(ids_host) * cache_view.stride(0)
        assert maximum_offset >= 2**31, "regression did not cross the int32 boundary"
        reports.append(
            {
                "layout": name,
                "max_element_offset": maximum_offset,
                "max_output_error": (actual.float() - expected).abs().max().item(),
            }
        )
    return {"large_cache_offsets": "passed", "allocated_bytes": cache.numel() * 2, "cases": reports}


def check_draft_config():
    """Keep shared layer registries while disabling compilation only for draft."""
    integration = importlib.import_module("vllm.apertus_draft_kv.integration")
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

    @dataclass
    class ConfigHolder:
        compilation_config: object = None

    target = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        # The server resolves these delayed defaults before loading a drafter.
        # In particular, replacing a bare config with capture_sizes=None would
        # revalidate that non-optional list and fail before testing registries.
        cudagraph_capture_sizes=[8, 16, 32],
        max_cudagraph_capture_size=32,
        use_inductor_graph_partition=False,
        ir_enable_torch_wrap=True,
        backend="inductor",
    )
    target_layer, draft_layer = object(), object()
    forward_registry = {"model.layers.0.self_attn.attn": target_layer}
    moe_registry = ["target.moe"]
    target.static_forward_context = forward_registry
    target.static_all_moe_layers = moe_registry
    config = ConfigHolder(target)
    with patch.dict(
        "os.environ",
        {
            "VLLM_APERTUS_DRAFT_KV_POLICY": "importance",
            "VLLM_APERTUS_DRAFT_KV_BUDGET": "512",
            "VLLM_APERTUS_DRAFT_KV_RECENT": "128",
        },
    ):
        draft_config = integration.sparse_draft_config(config)
    draft = draft_config.compilation_config
    assert draft_config is not config and draft is not target
    assert draft.mode == CompilationMode.NONE and draft.cudagraph_mode == CUDAGraphMode.NONE
    assert target.mode == CompilationMode.VLLM_COMPILE
    assert target.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert target.cudagraph_capture_sizes == [8, 16, 32]
    assert target.max_cudagraph_capture_size == 32
    assert draft.static_forward_context is forward_registry
    assert draft.static_all_moe_layers is moe_registry
    assert draft.static_forward_context["model.layers.0.self_attn.attn"] is target_layer
    # Registrations happen after the config copy, while loading the drafter.
    draft.static_forward_context["model.layers.32.self_attn.attn"] = draft_layer
    draft.static_all_moe_layers.append("draft.moe")
    assert target.static_forward_context["model.layers.32.self_attn.attn"] is draft_layer
    assert target.static_all_moe_layers == ["target.moe", "draft.moe"]
    with patch.dict(
        "os.environ",
        {
            "VLLM_APERTUS_DRAFT_KV_POLICY": "native",
            "VLLM_APERTUS_DRAFT_KV_BUDGET": "2048",
        },
    ):
        assert integration.sparse_draft_config(config) is config
    return {"draft_config_registry": "passed", "target_compilation_unchanged": True}


def check_state_kernels(policy):
    """Compare installed GPU tag/recycle/selection kernels with the CPU policy."""
    integration = importlib.import_module("vllm.apertus_draft_kv.integration")
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

    from draft_kv.policy import PolicyConfig, select_pages

    device = torch.device("cuda")
    page_size, max_pages, horizon = 16, 129, 2048
    states_host, group_id = [2, 0], 1  # Batch order differs from request-state order.
    states = torch.tensor(states_host, device=device, dtype=torch.int32)
    tags = torch.full((3, max_pages), -1, device=device, dtype=torch.int64)
    scores = torch.zeros((3, max_pages), device=device, dtype=torch.float32)
    lookup = torch.full((3, max_pages), -1, device=device, dtype=torch.int32)
    table = torch.arange(3 * 256, device=device, dtype=torch.int32).reshape(3, 256) + 700
    counts = UvaBackedTensor((2, 3), dtype=torch.int32)
    counts.np[group_id, states_host] = max_pages
    counts.copy_to_uva()
    config = PolicyConfig(policy="off" if policy == "full" else policy, budget=512)

    def prepare(rows, rejected_host, capacity):
        starts_host = [0, len(rows[0]), len(rows[0]) + len(rows[1])]
        positions = torch.tensor(rows[0] + rows[1], device=device, dtype=torch.int64)
        starts = torch.tensor(starts_host, device=device, dtype=torch.int32)
        rejected = torch.tensor(rejected_host, device=device, dtype=torch.int32)
        ends = torch.empty(2, device=device, dtype=torch.int64)
        batch_ids = torch.empty((2, capacity), device=device, dtype=torch.int32)
        batch_tags = torch.empty((2, capacity), device=device, dtype=torch.int64)
        batch_scores = torch.empty((2, capacity), device=device, dtype=torch.float32)
        integration._prepare_pages[(2,)](
            tags,
            scores,
            lookup,
            ends,
            positions,
            starts,
            rejected,
            states,
            counts.gpu,
            counts.gpu.stride(0),
            group_id,
            batch_ids,
            batch_tags,
            batch_scores,
            table,
            table.stride(0),
            C=capacity,
            V=256,
            PAGE=16,
            HORIZON=2048,
            MAX_PAGES=129,
        )
        context_slots, context_positions = torch.empty_like(positions), torch.empty_like(positions)
        integration._context_slots[(2, (max(map(len, rows)) + 255) // 256)](
            context_slots,
            context_positions,
            positions,
            starts,
            rejected,
            states,
            ends,
            lookup,
            tags,
            table,
            table.stride(0),
            B=256,
            PAGE=16,
            HORIZON=2048,
            MAX_PAGES=129,
        )
        torch.testing.assert_close(context_positions, positions)
        ends_host = ends.cpu().tolist()
        actual_slots = context_slots.cpu().tolist()
        ids_host, tags_host = batch_ids.cpu().tolist(), batch_tags.cpu().tolist()
        for request, (row, rejected_count) in enumerate(zip(rows, rejected_host, strict=True)):
            assert ends_host[request] == row[len(row) - rejected_count - 1] + 1
            page_ids = {
                tag: physical
                for tag, physical in zip(tags_host[request], ids_host[request], strict=True)
                if tag >= 0
            }
            assert len(page_ids) == sum(tag >= 0 for tag in tags_host[request])
            for offset, position in enumerate(row):
                valid = (
                    offset < len(row) - rejected_count
                    and position >= ends_host[request] - horizon + 1
                )
                expected = -1
                if valid:
                    tag = position // page_size * page_size
                    assert tag in page_ids, "accepted context page was lost"
                    expected = page_ids[tag] * page_size + position % page_size
                assert actual_slots[starts_host[request] + offset] == expected
        return ends, batch_scores

    def finish(ends, batch_scores, capacity, is_final):
        state_tags, incoming = tags.cpu().tolist(), batch_scores.cpu().tolist()
        expected_sets = []
        for request, state in enumerate(states_host):
            end = int(ends[request])
            candidates = {
                tag // page_size: incoming[request][slot]
                for slot, tag in enumerate(state_tags[state])
                if tag >= 0
            }
            expected_sets.append(
                set(
                    select_pages(
                        candidates,
                        config=config,
                        newest_position=end - 1,
                        oldest_visible_position=max(0, end - horizon + 1),
                    )
                )
                if is_final
                else set(candidates)
            )
        final = torch.full((2,), int(is_final), device=device, dtype=torch.int32)
        integration._finish_pages[(2,)](
            tags,
            scores,
            batch_scores,
            ends,
            states,
            final,
            C=capacity,
            V=256,
            BUDGET=32,
            RECENT=128,
            FULL=policy == "full",
            IMPORTANCE=policy == "importance",
            PAGE=16,
            MAX_PAGES=129,
        )
        state_tags = tags.cpu().tolist()
        for request, state in enumerate(states_host):
            actual = {tag // page_size for tag in state_tags[state] if tag >= 0}
            assert actual == expected_sets[request], (
                policy,
                request,
                actual,
                expected_sets[request],
            )
        assert torch.isfinite(scores).all(), "nonfinite cumulative page scores"

    # Unaligned native bootstrap uses129 pages. Nonfinal prefill cannot prune
    # context or commit scores; final prefill applies the requested budget.
    ends_host = [2049, 2053]
    ends, batch_scores = prepare([list(range(end)) for end in ends_host], [0, 0], max_pages)
    assert (tags >= 0).sum().item() == 2 * max_pages
    batch_scores.copy_(torch.arange(max_pages, 0, -1, device=device).expand(2, -1))
    finish(ends, batch_scores, max_pages, False)
    assert torch.count_nonzero(scores) == 0, "nonfinal prefill scores leaked"
    finish(ends, batch_scores, max_pages, True)
    capacity = max_pages
    if policy != "full":
        # Runtime compacts selected page metadata and physical block ownership,
        # retaining exactly one free staging page and updating the UVA count.
        for state in states_host:
            live = (tags[state] >= 0).nonzero().flatten()
            spare = (tags[state] < 0).nonzero().flatten()[:1]
            retained = torch.cat((live, spare))
            selected_tags = tags[state].index_select(0, retained)
            selected_scores = scores[state].index_select(0, retained)
            selected_ids = table[state, :max_pages].index_select(0, retained)
            assert live.numel() == 32 and retained.numel() == 33
            tags[state].fill_(-1)
            scores[state].zero_()
            tags[state, :33].copy_(selected_tags)
            scores[state, :33].copy_(selected_scores)
            table[state, :33].copy_(selected_ids)
            counts.np[group_id, state] = 33
        counts.copy_to_uva()
        capacity = 33
    # Accepted8-token rounds cross a page boundary. The last round accepts one
    # position then rejects16 suffix positions, including a future page.
    for accepted, rejected_count in ((8, 4), (8, 4), (1, 16)):
        rows = [list(range(end, end + accepted + rejected_count)) for end in ends_host]
        ends, batch_scores = prepare(rows, [rejected_count] * 2, capacity)
        batch_scores.add_(0.125)
        finish(ends, batch_scores, capacity, True)
        ends_host = [end + accepted for end in ends_host]
    assert (tags[1] == -1).all(), "unused request state was modified"
    return {"state_kernels": "passed", "policy": policy, "capacity": capacity}


def check_graph_helper():
    """Test actual graph-helper metadata and sampling without model weights.

    A bounded two-shape capture manager replaces distributed setup. The real
    graph helper, runtime adapter, attention kernels and Gumbel sampler remain
    unchanged, including five-layer importance accumulation and request padding.
    """
    graphs_module = importlib.import_module("vllm.apertus_draft_kv.graphs")
    integration = importlib.import_module("vllm.apertus_draft_kv.integration")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    class BoundedManager:
        def __init__(self, config, device, mode, decode_query_len):
            self.graphs, self.query_len = {}, decode_query_len

        def needs_capture(self):
            return True

        def capture(self, create_forward, description):
            for requests in (4, 2):
                desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.FULL,
                    num_tokens=requests * self.query_len,
                    num_reqs=requests,
                    uniform_token_count=self.query_len,
                )
                warmup, _ = create_forward(desc, True)
                warmup(CUDAGraphMode.NONE)
                torch.cuda.synchronize()
                forward, _ = create_forward(desc, False)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward(CUDAGraphMode.NONE)
                self.graphs[desc] = graph

        def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras):
            return next(
                desc
                for desc in sorted(self.graphs, key=lambda d: d.num_reqs)
                if desc.num_reqs >= num_reqs
            )

        def run_fullgraph(self, desc):
            self.graphs[desc].replay()

    torch.manual_seed(8171)
    device = torch.device("cuda")
    query_len, heads, kv_heads, dim, slots = 8, 8, 2, 64, 33
    q = torch.randn((32, heads, dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((32, kv_heads, dim), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache = torch.randn((128, 2, 16, kv_heads, dim), device=device, dtype=q.dtype)
    sink = torch.linspace(-0.5, 1.5, heads, device=device)
    layers = [
        SimpleNamespace(
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            scaling=dim**-0.5,
            causal=False,
            attention_sink_bias=sink,
            attn=SimpleNamespace(kv_cache=cache),
        )
        for _ in range(5)
    ]
    compilation = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    owner = SimpleNamespace(
        max_num_reqs=4,
        num_query_per_req=query_len,
        vllm_config=SimpleNamespace(compilation_config=compilation),
        model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=layer) for layer in layers])
        ),
        input_buffers=SimpleNamespace(
            input_ids=torch.empty(32, device=device, dtype=torch.int64),
            positions=torch.empty(32, device=device, dtype=torch.int64),
            query_start_loc=torch.empty(5, device=device, dtype=torch.int32),
        ),
        sample_indices=torch.empty(32, device=device, dtype=torch.int64),
        sample_pos=torch.empty(32, device=device, dtype=torch.int64),
        sample_idx_mapping=torch.empty(32, device=device, dtype=torch.int32),
        temperature=torch.zeros(4, device=device),
        seeds=torch.full((4,), 719, device=device, dtype=torch.int64),
    )
    outputs = torch.empty((5, 32, heads, dim), device=device, dtype=q.dtype)
    processed = torch.full((4, query_len, 32), 31337.0, device=device)
    sampled = torch.empty((4, query_len), device=device, dtype=torch.int64)
    columns = torch.arange(32, device=device, dtype=torch.int32) % query_len
    held_active = {"sentinel": True}
    runtime = SimpleNamespace(
        owner=owner,
        device=device,
        policy="importance",
        budget=512,
        active=held_active,
        workspace=None,
        steps=17,
        registered={"held": 2},
        compressed={"held"},
        releases={"held": [7]},
    )
    runtime.attention = MethodType(integration.SparseDraftRuntime.attention, runtime)

    def generate(self, requests, tokens, *args, **kwargs):
        for layer_index, layer in enumerate(layers):
            result = integration.sparse_layer_attention(
                layer,
                q[:tokens],
                k[:tokens],
                v[:tokens],
                owner.input_buffers.positions[:tokens],
            ).view(tokens, heads, dim)
            outputs[layer_index, :tokens].copy_(result)
        count = requests * query_len
        logits = result[owner.sample_indices[:count], 0, :32].float()
        sampled[:requests].copy_(
            gumbel_sample(
                logits,
                owner.sample_idx_mapping[:count],
                owner.temperature,
                owner.seeds,
                owner.sample_pos[:count],
                apply_temperature=True,
                output_processed_logits=processed,
                output_processed_logits_col=columns[:count],
            ).view(requests, query_len)
        )

    owner._generate_draft = MethodType(generate, owner)
    old_current = integration._CURRENT
    with patch.object(graphs_module, "CudaGraphManager", BoundedManager):
        graph_helper = graphs_module.SparseQueryGraphs(runtime)
        graph_helper.capture()
    assert integration._CURRENT is old_current
    assert runtime.active is held_active and runtime.workspace is None
    assert runtime.steps == 17 and runtime.registered == {"held": 2}
    assert runtime.compressed == {"held"} and runtime.releases == {"held": [7]}
    assert compilation.cudagraph_mode == CUDAGraphMode.FULL
    assert (processed == 31337).all(), "capture wrote through dummy state mappings"
    tracked = (
        graph_helper.ids,
        graph_helper.tags,
        graph_helper.scores,
        graph_helper.ends,
        graph_helper.workspace.page_lse,
        graph_helper.workspace.anchor_lse,
    )
    pointers = tuple(t.data_ptr() for t in tracked)
    ids = torch.arange(3 * slots, device=device, dtype=torch.int32).view(3, slots)
    ends = torch.tensor([4101, 6203, 10007], device=device, dtype=torch.int64)
    tags = ends[:, None] // 16 * 16 - torch.arange(slots, device=device)[None, :] * 16
    scores = torch.empty((3, slots), device=device)
    request_states = [2, 0, 3]
    for iteration, requests in enumerate((3, 1, 3)):
        if iteration == 2:
            tags[:, 0].fill_(-1)
            ids[:, 1].copy_(ids[:, 2])
            q.mul_(0.8)
        count = requests * query_len
        positions = (ends[:requests, None] + torch.arange(query_len, device=device)).flatten()
        starts = torch.arange(requests + 1, device=device, dtype=torch.int32) * query_len
        owner.input_buffers.positions[:count].copy_(positions)
        owner.input_buffers.input_ids[:count].fill_(1)
        owner.input_buffers.query_start_loc[: requests + 1].copy_(starts)
        owner.sample_indices.zero_()
        owner.sample_indices[:count].copy_(torch.arange(count, device=device))
        owner.sample_pos.zero_()
        owner.sample_pos[:count].copy_(positions + 1)
        owner.sample_idx_mapping.fill_(-1)
        owner.sample_idx_mapping[:count].copy_(
            torch.tensor(
                request_states[:requests], device=device, dtype=torch.int32
            ).repeat_interleave(query_len)
        )
        owner.temperature.fill_(1.0)
        processed.fill_(31337)
        scores.copy_(torch.arange(3 * slots, device=device).view(3, slots) * 0.001)
        initial_scores = scores[:requests].clone()
        active = {
            "ids": ids[:requests],
            "tags": tags[:requests],
            "scores": scores[:requests],
            "ends": ends[:requests],
            "query_starts": starts,
            "anchors": torch.arange(requests, device=device) * query_len,
        }
        assert graph_helper.run(active, requests), "eligible batch did not replay"
        torch.cuda.synchronize()
        expected, mass = reference_attention(
            q[:count],
            k[:count],
            v[:count],
            cache,
            ids[:requests],
            tags[:requests],
            ends[:requests],
            starts,
            positions,
            active["anchors"],
            window_size=2048,
            causal=False,
            sinks=sink,
        )
        for layer_output in outputs[:, :count]:
            torch.testing.assert_close(layer_output.float(), expected, rtol=0.015, atol=0.025)
        torch.testing.assert_close(
            scores[:requests], initial_scores + 5 * mass, rtol=0.003, atol=0.003
        )
        padded = graph_helper.last_padded_requests
        assert padded == (4 if requests == 3 else 2)
        assert torch.isfinite(outputs[:, : padded * query_len]).all(), "padding became nonfinite"
        assert torch.count_nonzero(graph_helper.scores[requests:padded]) == 0
        for request, state in enumerate(request_states[:requests]):
            expected_logits = expected[request * query_len : (request + 1) * query_len, 0, :32]
            torch.testing.assert_close(processed[state], expected_logits, rtol=0.015, atol=0.025)
        for state in set(range(4)) - set(request_states[:requests]):
            assert (processed[state] == 31337).all(), "padding wrote inactive processed logits"
    assert len(graph_helper.manager.graphs) == 2 and graph_helper.replay_count == 3
    oversized = dict(active, ids=torch.empty((3, 129), device=device, dtype=torch.int32))
    assert not graph_helper.run(oversized, 3), "bootstrap unexpectedly created another graph"
    assert graph_helper.fallback_count == 1 and len(graph_helper.manager.graphs) == 2
    assert pointers == tuple(
        t.data_ptr()
        for t in (
            graph_helper.ids,
            graph_helper.tags,
            graph_helper.scores,
            graph_helper.ends,
            graph_helper.workspace.page_lse,
            graph_helper.workspace.anchor_lse,
        )
    )
    return {
        "graph_helper": "passed",
        "captured_shapes": [2, 4],
        "replays": 3,
        "bootstrap_fallbacks": 1,
        "layers": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one bf16 case plus graph replay")
    parser.add_argument(
        "--state-check",
        action="store_true",
        help="also test installed vLLM overlay state kernels against CPU policy",
    )
    parser.add_argument(
        "--graph-helper-check",
        action="store_true",
        help="test installed graph helper with synthetic five-layer attention and sampling",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU skip is counted as validation")
    config_report = check_draft_config() if args.state_check else None
    if config_report:
        print(json.dumps(config_report), flush=True)
    offset_report = check_large_cache_offsets()
    print(json.dumps(offset_report), flush=True)
    reports = []
    configurations = [(torch.bfloat16, 32, False, True, 2048, 16)]
    if not args.smoke:
        configurations = [
            (dtype, slots, causal, sink, 2048, 16)
            for dtype in (torch.float16, torch.bfloat16)
            for slots in (32, 64)
            for causal, sink in ((False, False), (False, True), (True, True))
        ]
        configurations += [
            (torch.bfloat16, 32, False, True, None, 16),
            (torch.bfloat16, 32, True, False, 1, 16),
            (torch.float16, 16, False, True, 2048, 32),
        ]
    for config in configurations:
        report = compare_case(*config)
        reports.append(report)
        print(json.dumps(report), flush=True)
    reports.append(check_graph_replay())
    graph_report = reports[-1]
    state_reports = [config_report] if config_report else []
    if args.state_check:
        for policy in ("importance", "recent", "full"):
            report = check_state_kernels(policy)
            state_reports.append(report)
            print(json.dumps(report), flush=True)
    graph_helper_report = check_graph_helper() if args.graph_helper_check else None
    if graph_helper_report:
        print(json.dumps(graph_helper_report), flush=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "cases": len(configurations),
                "graph": graph_report,
                "large_cache_offsets": offset_report,
                "state": state_reports,
                "graph_helper": graph_helper_report,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
