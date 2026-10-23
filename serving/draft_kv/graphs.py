"""Bounded CUDA graphs for the DSpark sparse query forward.

Context projection, retained-page insertion, and eviction remain outside the
captured graph. Request padding and capture shapes come from vLLM's own graph
manager. Each deployment captures one slot-count family; bootstrap batches that
need more pages use eager dispatch. There is no cache keyed by arbitrary runtime
batch sizes, context lengths, or retained-page sets.
"""

from __future__ import annotations

import json

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import AttentionState, CudaGraphManager

from .attention import SparseAttentionWorkspace


class SparseQueryGraphs:
    """Capture backbone plus Markov sampling with stable attention metadata."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.owner = runtime.owner
        self.slots = 129 if runtime.policy == "full" else runtime.budget // 16 + 1
        self.max_requests = self.owner.max_num_reqs
        self.query_len = self.owner.num_query_per_req
        self.device = runtime.device
        config = self.owner.vllm_config.compilation_config
        requested_mode = config.cudagraph_mode
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if requested_mode and requested_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.manager = CudaGraphManager(
            self.owner.vllm_config, self.device, mode, decode_query_len=self.query_len
        )
        self.ids = torch.full(
            (self.max_requests, self.slots), -1, dtype=torch.int32, device=self.device
        )
        self.tags = torch.full_like(self.ids, -1, dtype=torch.int64)
        self.scores = torch.zeros_like(self.ids, dtype=torch.float32)
        self.ends = torch.zeros(self.max_requests, dtype=torch.int64, device=self.device)
        self.anchors = (
            torch.arange(self.max_requests, device=self.device, dtype=torch.int64) * self.query_len
        )
        layers = self.owner.model.model.layers
        heads = layers[0].self_attn.num_heads
        if any(layer.self_attn.num_heads != heads for layer in layers):
            raise ValueError("sparse query graphs require equal query-head counts across layers")
        self.workspace = SparseAttentionWorkspace.allocate(
            self.max_requests, heads, self.slots, self.device
        )
        self.last_replayed = False
        self.last_padded_requests = 0
        self.replay_count = 0
        self.fallback_count = 0

    @property
    def metadata_bytes(self):
        tensors = (
            self.ids,
            self.tags,
            self.scores,
            self.ends,
            self.anchors,
            self.workspace.page_lse,
            self.workspace.anchor_lse,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def _active(self, requests):
        return {
            "ids": self.ids[:requests],
            "tags": self.tags[:requests],
            "scores": self.scores[:requests],
            "ends": self.ends[:requests],
            "anchors": self.anchors[:requests],
            "query_starts": self.owner.input_buffers.query_start_loc[: requests + 1],
        }

    def _workspace_view(self, requests):
        return SparseAttentionWorkspace(
            self.workspace.page_lse[:requests], self.workspace.anchor_lse[:requests]
        )

    @torch.inference_mode()
    def capture(self):
        if not self.manager.needs_capture():
            return
        from . import integration

        owner = self.owner
        old_current = integration._CURRENT
        old_active = self.runtime.active
        old_workspace = self.runtime.workspace
        # All capture rows are valid dummy query blocks; no context page is
        # dereferenced because every ID/tag is -1. Markov sampling reads valid
        # row0. Negative dummy sample mappings are explicitly masked by the
        # native Gumbel kernels, avoiding concurrent state0 logit writes.
        owner.input_buffers.input_ids.zero_()
        owner.input_buffers.positions.zero_()
        owner.sample_indices.zero_()
        owner.sample_pos.zero_()
        owner.sample_idx_mapping.fill_(-1)
        owner.temperature.zero_()
        self.scores.zero_()
        allocated_before = torch.cuda.memory_allocated(self.device)

        def create_forward(desc, warmup):
            requests = desc.num_reqs
            if requests is None:
                raise RuntimeError("sparse query graphs require uniform padded request counts")
            tokens = desc.num_tokens
            owner.input_buffers.query_start_loc[: requests + 1].copy_(
                torch.arange(requests + 1, dtype=torch.int32, device=self.device) * self.query_len
            )
            self.runtime.active = self._active(requests)
            self.runtime.workspace = self._workspace_view(requests)

            def forward(mode):
                owner._generate_draft(
                    requests,
                    tokens,
                    None,
                    None,
                    None,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )

            return forward, AttentionState(None, {})

        try:
            integration._CURRENT = self.runtime
            self.manager.capture(create_forward, "Capturing sparse DSpark query graphs")
        finally:
            integration._CURRENT = old_current
            self.runtime.active = old_active
            self.runtime.workspace = old_workspace
        # Warmup/capture accumulates synthetic scores. Every real replay
        # overwrites active rows, and padded rows must start from zero.
        self.scores.zero_()
        print(
            "APERTUS_DRAFT_KV_METRICS "
            + json.dumps(
                {
                    "event": "query_graph_capture",
                    "slots": self.slots,
                    "graphs": len(self.manager.graphs),
                    "request_shapes": sorted({d.num_reqs for d in self.manager.graphs}),
                    "metadata_bytes": self.metadata_bytes,
                    "target_graphs_unchanged": True,
                    "allocated_bytes_before_capture": allocated_before,
                    "allocated_bytes_after_capture": torch.cuda.memory_allocated(self.device),
                    "reserved_bytes_after_capture": torch.cuda.memory_reserved(self.device),
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def run(self, active, requests):
        """Return True if replayed; otherwise caller runs the eager forward."""
        self.last_replayed = False
        self.last_padded_requests = requests
        if active["ids"].shape[1] != self.slots:
            self.fallback_count += 1
            return False
        desc = self.manager.dispatch(
            num_reqs=requests,
            num_tokens=requests * self.query_len,
            uniform_token_count=self.query_len,
            num_active_loras=0,
        )
        if desc.cg_mode != CUDAGraphMode.FULL:
            self.fallback_count += 1
            return False
        padded = desc.num_reqs
        if padded is None:
            raise RuntimeError("a FULL sparse graph lacks a padded request count")
        self.ids[:requests].copy_(active["ids"])
        self.tags[:requests].copy_(active["tags"])
        self.scores[:requests].copy_(active["scores"])
        self.ends[:requests].copy_(active["ends"])
        if padded > requests:
            self.ids[requests:padded].fill_(-1)
            self.tags[requests:padded].fill_(-1)
            self.scores[requests:padded].zero_()
            self.ends[requests:padded].zero_()
            lo, hi = requests * self.query_len, padded * self.query_len
            self.owner.input_buffers.input_ids[lo:hi].zero_()
            self.owner.input_buffers.positions[lo:hi].zero_()
            # Private drafter padding uses finite uniform dummy query blocks.
            # Zero-length blocks would leave padded attention output unwritten,
            # permitting stale NaNs to enter padded MLP rows during replay.
            self.owner.input_buffers.query_start_loc[requests + 1 : padded + 1].copy_(
                torch.arange(requests + 1, padded + 1, dtype=torch.int32, device=self.device)
                * self.query_len
            )
        self.manager.run_fullgraph(desc)
        active["scores"].copy_(self.scores[:requests])
        self.last_replayed = True
        self.last_padded_requests = padded
        self.replay_count += 1
        return True
