"""CUDA correctness checks for bounded drafter attention (no model deployment).

Run inside the serving image: ``python -m draft_kv.check_gpu`` with ``serving``
on PYTHONPATH, or execute this file directly. Fails if CUDA/Triton are absent.
"""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one bf16 case plus graph replay")
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
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "cases": len(configurations),
                "graph": reports[-1],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
