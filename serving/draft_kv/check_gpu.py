"""CUDA correctness checks for bounded drafter attention (no model deployment).

Run inside the serving image: ``python -m draft_kv.check_gpu`` with ``serving``
on PYTHONPATH, or execute this file directly. Fails if CUDA/Triton are absent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

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


def check_state_kernels(policy):
    """Compare installed GPU tag/recycle/selection kernels with the CPU policy."""
    integration = importlib.import_module("vllm.apertus_draft_kv.integration")
    from draft_kv.policy import PolicyConfig, select_pages
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one bf16 case plus graph replay")
    parser.add_argument(
        "--state-check",
        action="store_true",
        help="also test installed vLLM overlay state kernels against CPU policy",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU skip is counted as validation")
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
    state_reports = []
    if args.state_check:
        for policy in ("importance", "recent", "full"):
            report = check_state_kernels(policy)
            state_reports.append(report)
            print(json.dumps(report), flush=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "cases": len(configurations),
                "graph": graph_report,
                "state": state_reports,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
