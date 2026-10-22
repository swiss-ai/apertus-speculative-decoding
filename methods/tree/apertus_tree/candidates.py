"""Draft candidates of DSpark e10, EAGLE e10 and n-gram lookup at every decode step.

Input: greedy target continuations captured by plain vLLM
(``apertus_eagle.verify_offline capture --mode plain``) and their prompts.
For each sequence (prompt + greedy continuation) this script

1. runs the target once (Hugging Face, teacher forcing) and keeps the auxiliary
   hidden states both drafters read: EAGLE e10 the residual stream after HF
   decoder layers 1/15/28 (vLLM ids 2/16/29), DSpark e10 after 0/7/14/21/28
   (``--dspark-hf-layers``; vLLM ids 1/8/15/22/29);
2. at every decode step (anchor ``a`` = the newest token; the target has hidden
   states for positions < a) builds
   - the DSpark tree: one block forward gives per-slot logits; the vanilla Markov
     head adds ``W2 W1[parent token]``, so the children of any node are exact
     and cheap; expanded EAGLE-2 style (top ``--width`` frontier per depth, top
     ``--topk`` children each, depth 7), plus DSpark's greedy chain;
   - the EAGLE tree: the same dynamic expansion with one EAGLE-3 forward per
     depth over the frontier, each node attending to the context keys and to its
     ancestors' keys (TorchSpec's training-time layout), plus the greedy chain;
   - n-gram / prompt-lookup branches (n = 4..1, most recent match first);
3. keeps each neural tree's ``--keep`` best nodes by its own path probability,
   merges everything into one trie per anchor and scores *every* node under
   both drafters (EAGLE forwards the trie's internal nodes again), so the
   simulator (``apertus_bench.tree_sim``) can rank any union by any drafter.

Records go to a gzip JSONL shard; they contain token ids of the restricted
workloads and stay on the cluster. Only aggregates leave it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from apertus_bench import tree_sim as ts
from apertus_tree.trie import NgramIndex, Trie

BF16 = torch.bfloat16
NEG = float("-inf")


# --------------------------------------------------------------------------- shared math


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def llama3_inv_freq(dim: int, rope: dict[str, Any]) -> torch.Tensor:
    theta = float(rope["rope_theta"])
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    factor = float(rope["factor"])
    low, high = float(rope["low_freq_factor"]), float(rope["high_freq_factor"])
    old = float(rope["original_max_position_embeddings"])
    wavelen = 2 * math.pi / inv
    out = torch.where(wavelen > old / low, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = (wavelen >= old / high) & (wavelen <= old / low)
    return torch.where(medium, smoothed, out).float()


class Rope:
    def __init__(self, dim: int, rope: dict[str, Any], device):
        self.inv = llama3_inv_freq(dim, rope).to(device)

    def __call__(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [..., heads, dim], pos broadcastable to x.shape[:-2]."""
        freqs = pos.float()[..., None] * self.inv
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos()[..., None, :], emb.sin()[..., None, :]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rot = torch.cat([-x2, x1], -1)
        return (x.float() * cos + rot.float() * sin).to(x.dtype)


def mlp(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def attend(q, ctx_k, ctx_v, ctx_valid, own_k, own_v, own_valid, scale):
    """Grouped-query attention over a shared context plus per-query extra keys.

    q [n, H, D]; ctx_k/ctx_v [T, G, D]; ctx_valid [n, T] bool;
    own_k/own_v [n, J, G, D]; own_valid [n, J] bool. Returns [n, H*D].
    """
    n, heads, dim = q.shape
    groups = ctx_k.shape[1]
    qg = q.view(n, groups, heads // groups, dim).float()
    s_ctx = torch.einsum("ngrd,tgd->ngrt", qg, ctx_k.float()) * scale
    s_ctx.masked_fill_(~ctx_valid[:, None, None, :], NEG)
    s_own = torch.einsum("ngrd,njgd->ngrj", qg, own_k.float()) * scale
    s_own.masked_fill_(~own_valid[:, None, None, :], NEG)
    p = torch.softmax(torch.cat([s_ctx, s_own], -1), -1)
    t = ctx_k.shape[0]
    out = torch.einsum("ngrt,tgd->ngrd", p[..., :t], ctx_v.float())
    out = out + torch.einsum("ngrj,njgd->ngrd", p[..., t:], own_v.float())
    return out.reshape(n, heads * dim).to(q.dtype)


# --------------------------------------------------------------------------- EAGLE-3 head


class Eagle:
    def __init__(self, head: Path, embed: torch.Tensor, device):
        cfg = json.loads((head / "config.json").read_text())
        w = load_file(str(head / "model.safetensors"), device=str(device))
        self.cfg = cfg
        self.heads, self.groups = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.dim = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.vocab = cfg["draft_vocab_size"]
        self.naux = cfg["num_aux_hidden_states"]
        self.embed = embed[: cfg["vocab_size"]]
        self.fc = w["fc.weight"]
        self.fc_norm = [w[f"fc_norm.{i}.weight"] for i in range(self.naux)]
        p = "layers.0."
        self.q, self.k, self.v, self.o = (w[p + f"self_attn.{n}_proj.weight"] for n in "qkvo")
        self.gate, self.up, self.down = (
            w[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up", "down")
        )
        self.in_ln, self.hid_ln = w[p + "input_layernorm.weight"], w[p + "hidden_norm.weight"]
        self.post_ln = w[p + "post_attention_layernorm.weight"]
        self.norm, self.lm_head = w["norm.weight"], w["lm_head.weight"]
        rope = dict(cfg["rope_parameters"])
        rope.setdefault("rope_theta", cfg.get("rope_theta"))
        self.rope = Rope(self.dim, rope, device)
        self.scale = self.dim**-0.5
        if not (cfg.get("fc_norm") and cfg.get("norm_output")):
            raise SystemExit("this simulator implements the fc_norm + norm_output EAGLE head")

    def project(self, aux: torch.Tensor) -> torch.Tensor:
        chunks = aux.chunk(self.naux, -1)
        normed = torch.cat(
            [rms(c, n, self.eps) for c, n in zip(chunks, self.fc_norm, strict=True)], -1
        )
        return F.linear(normed.float(), self.fc.float()).to(BF16)

    def _qkv(self, tokens, hidden, pos):
        emb = self.embed[tokens.clamp(max=self.embed.shape[0] - 1)]
        x = torch.cat([rms(emb, self.in_ln, self.eps), rms(hidden, self.hid_ln, self.eps)], -1)
        n = x.shape[:-1]
        q = F.linear(x, self.q).view(*n, self.heads, self.dim)
        k = F.linear(x, self.k).view(*n, self.groups, self.dim)
        v = F.linear(x, self.v).view(*n, self.groups, self.dim)
        return self.rope(q, pos), self.rope(k, pos), v

    def _finish(self, attn, residual):
        h = residual + F.linear(attn, self.o)
        return h + mlp(rms(h, self.post_ln, self.eps), self.gate, self.up, self.down)

    def logprobs(self, h_raw: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(
            F.linear(rms(h_raw, self.norm, self.eps), self.lm_head).float(), -1
        )

    def context(self, ids: torch.Tensor, aux: torch.Tensor):
        """Step 0 over the whole sequence: pair j = (aux_j, token_{j+1}) predicts j+2."""
        hidden = self.project(aux[:-1])
        tokens = ids[1:]
        pos = torch.arange(tokens.shape[0], device=ids.device)
        q, k, v = self._qkv(tokens, hidden, pos)
        rep = self.heads // self.groups
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.repeat_interleave(rep, 1).transpose(0, 1)[None],
            v.repeat_interleave(rep, 1).transpose(0, 1)[None],
            is_causal=True,
        )[0].transpose(0, 1)
        h = self._finish(out.reshape(tokens.shape[0], -1), hidden)
        return {"k": k, "v": v, "h": h, "next": rms(h, self.norm, self.eps)}

    def nodes(self, ctx, anchors, depth, tokens, parent_hidden, anc_k, anc_v):
        """One tree step for nodes at ``depth`` (>= 1) of anchors ``anchors``.

        A depth-d node sits at TTT step d of the pair at position a-1: RoPE
        position a-1+d, keys = context pairs 0..a-1 plus its ancestors and itself.
        """
        pos = anchors - 1 + depth
        q, k, v = self._qkv(tokens, parent_hidden, pos)
        own_k = torch.cat([anc_k, k[:, None]], 1)
        own_v = torch.cat([anc_v, v[:, None]], 1)
        t = ctx["k"].shape[0]
        ctx_valid = torch.arange(t, device=q.device)[None] < anchors[:, None]
        own_valid = torch.ones(own_k.shape[:2], dtype=torch.bool, device=q.device)
        out = attend(q, ctx["k"], ctx["v"], ctx_valid, own_k, own_v, own_valid, self.scale)
        h = self._finish(out, parent_hidden)
        return k, v, h


# --------------------------------------------------------------------------- DSpark


class DSpark:
    def __init__(self, path: Path, embed: torch.Tensor, lm_head: torch.Tensor, device):
        cfg = json.loads((path / "config.json").read_text())
        tl = cfg["transformer_layer_config"]
        w = load_file(str(path / "model.safetensors"), device=str(device))
        if cfg.get("markov_head_type") != "vanilla" or not cfg.get("sample_from_anchor", True):
            raise SystemExit(
                "this simulator implements the vanilla Markov head, sample_from_anchor"
            )
        if cfg.get("sliding_window_non_causal"):
            raise SystemExit("expected a causal in-block mask (sliding_window_non_causal false)")
        self.cfg = cfg
        self.block = cfg["block_size"]
        self.mask_id = cfg["mask_token_id"]
        self.heads, self.groups, self.dim = (
            tl["num_attention_heads"],
            tl["num_key_value_heads"],
            tl["head_dim"],
        )
        self.eps = tl["rms_norm_eps"]
        self.window = tl["sliding_window"] if tl.get("use_sliding_window") else None
        self.embed, self.lm_head = embed, lm_head
        self.fc, self.hidden_norm, self.norm = (
            w["fc.weight"],
            w["hidden_norm.weight"],
            w["norm.weight"],
        )
        self.w1 = w["markov_head.markov_w1.weight"]
        self.w2 = w["markov_head.markov_w2.weight"]
        self.layers = []
        for i in range(tl["num_hidden_layers"]):
            p = f"layers.{i}."
            self.layers.append(
                {
                    n: w[p + key]
                    for n, key in {
                        "q": "self_attn.q_proj.weight",
                        "k": "self_attn.k_proj.weight",
                        "v": "self_attn.v_proj.weight",
                        "o": "self_attn.o_proj.weight",
                        "qn": "self_attn.q_norm.weight",
                        "kn": "self_attn.k_norm.weight",
                        "in": "input_layernorm.weight",
                        "post": "post_attention_layernorm.weight",
                        "gate": "mlp.gate_proj.weight",
                        "up": "mlp.up_proj.weight",
                        "down": "mlp.down_proj.weight",
                    }.items()
                }
            )
        self.rope = Rope(self.dim, tl["rope_parameters"], device)
        self.scale = self.dim**-0.5

    def context(self, aux: torch.Tensor):
        """Per-layer context keys/values from the target's hidden states."""
        target = rms(F.linear(aux, self.fc), self.hidden_norm, self.eps)
        pos = torch.arange(aux.shape[0], device=aux.device)
        out = []
        for layer in self.layers:
            k = F.linear(target, layer["k"]).view(-1, self.groups, self.dim)
            k = self.rope(rms(k, layer["kn"], self.eps), pos)
            v = F.linear(target, layer["v"]).view(-1, self.groups, self.dim)
            out.append((k, v))
        return out

    def block_logits(self, ctx, ids: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Base logits [A, block, V] of the blocks anchored at ``anchors``."""
        a_n, b = anchors.shape[0], self.block
        tokens = torch.full((a_n, b), self.mask_id, device=ids.device, dtype=torch.long)
        tokens[:, 0] = ids[anchors]
        h = self.embed[tokens]
        pos = anchors[:, None] + torch.arange(b, device=ids.device)[None]
        t = ctx[0][0].shape[0]
        j = torch.arange(t, device=ids.device)[None]
        valid = j < anchors[:, None]
        if self.window is not None:
            valid &= j >= anchors[:, None] - self.window
        ctx_valid = valid[:, None, :].expand(a_n, b, t).reshape(a_n * b, t)
        causal = torch.tril(torch.ones(b, b, dtype=torch.bool, device=ids.device))
        own_valid = causal[None].expand(a_n, b, b).reshape(a_n * b, b)
        for layer, (ck, cv) in zip(self.layers, ctx, strict=True):
            x = rms(h, layer["in"], self.eps)
            q = F.linear(x, layer["q"]).view(a_n, b, self.heads, self.dim)
            q = self.rope(rms(q, layer["qn"], self.eps), pos)
            k = F.linear(x, layer["k"]).view(a_n, b, self.groups, self.dim)
            k = self.rope(rms(k, layer["kn"], self.eps), pos)
            v = F.linear(x, layer["v"]).view(a_n, b, self.groups, self.dim)
            own_k = (
                k[:, None]
                .expand(a_n, b, b, self.groups, self.dim)
                .reshape(a_n * b, b, self.groups, self.dim)
            )
            own_v = (
                v[:, None]
                .expand(a_n, b, b, self.groups, self.dim)
                .reshape(a_n * b, b, self.groups, self.dim)
            )
            out = attend(
                q.reshape(a_n * b, self.heads, self.dim),
                ck,
                cv,
                ctx_valid,
                own_k,
                own_v,
                own_valid,
                self.scale,
            )
            h = h + F.linear(out.view(a_n, b, -1), layer["o"])
            h = h + mlp(rms(h, layer["post"], self.eps), layer["gate"], layer["up"], layer["down"])
        return F.linear(rms(h, self.norm, self.eps), self.lm_head)

    def logprobs(self, base: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """log-softmax of base slot logits [n, V] plus the Markov bias of ``prev``."""
        bias = F.linear(self.w1[prev], self.w2)
        return torch.log_softmax(base.float() + bias.float(), -1)


# --------------------------------------------------------------------------- trees


def expand(anchor_count, root_children, children, width, topk, depth_limit):
    """EAGLE-2-style dynamic expansion shared by both drafters.

    ``root_children()`` -> (tokens [A, k], logprobs [A, k]) for depth 1.
    ``children(items)`` with items [(anchor i, node)] -> (tokens [n, k], lps [n, k]);
    it sees the generated trees through the returned ``trees`` list.
    Returns per anchor: parents, tokens, cumulative scores, greedy chain nodes.
    """
    trees = [{"p": [], "t": [], "c": [], "d": [], "chain": []} for _ in range(anchor_count)]

    def add(i, parent, toks, lps):
        tree = trees[i]
        base = 0.0 if parent < 0 else tree["c"][parent]
        d = 1 if parent < 0 else tree["d"][parent] + 1
        made = []
        for tok, lp in zip(toks, lps, strict=True):
            tree["p"].append(parent)
            tree["t"].append(int(tok))
            tree["c"].append(base + float(lp))
            tree["d"].append(d)
            made.append(len(tree["t"]) - 1)
        return made

    toks, lps = root_children()
    for i in range(anchor_count):
        made = add(i, -1, toks[i].tolist(), lps[i].tolist())
        trees[i]["chain"].append(made[0])
    for depth in range(1, depth_limit):
        items = []
        for i, tree in enumerate(trees):
            level = [n for n, d in enumerate(tree["d"]) if d == depth]
            level.sort(key=lambda n: -tree["c"][n])
            frontier = level[:width]
            chain = tree["chain"][-1]
            if chain not in frontier:
                frontier.append(chain)
            items.extend((i, n) for n in frontier)
        toks, lps = children(items, trees)
        for (i, node), tk, lp in zip(items, toks.tolist(), lps.tolist(), strict=True):
            made = add(i, node, tk[:topk], lp[:topk])
            if node == trees[i]["chain"][-1]:
                trees[i]["chain"].append(made[0])
    return trees


def keep_best(tree, keep):
    sel = set(ts.select_top(tree["p"], tree["c"], keep))
    sel.update(tree["chain"])
    return sel


# --------------------------------------------------------------------------- per sequence


class Builder:
    def __init__(self, args, eagle: Eagle, dspark: DSpark, device):
        self.args, self.eagle, self.dspark, self.device = args, eagle, dspark, device

    def sequence(self, ids: list[int], prompt_len: int, aux_e, aux_d):
        args, eagle, dspark = self.args, self.eagle, self.dspark
        dev = self.device
        full = torch.tensor(ids, device=dev)
        ectx = eagle.context(full, aux_e)
        dctx = dspark.context(aux_d)
        ngram = NgramIndex(ids)
        anchors = list(range(max(prompt_len, 1), len(ids) - 1))
        if args.max_anchors:
            anchors = anchors[: args.max_anchors]
        for start in range(0, len(anchors), args.anchor_chunk):
            chunk = anchors[start : start + args.anchor_chunk]
            yield from self.chunk(ids, full, chunk, ectx, dctx, ngram)

    def chunk(self, ids, full, chunk, ectx, dctx, ngram):
        args, eagle, dspark, dev = self.args, self.eagle, self.dspark, self.device
        a_t = torch.tensor(chunk, device=dev)
        n_a = len(chunk)
        depth_limit = min(ts.MAX_DEPTH, dspark.block - 1)
        base = dspark.block_logits(dctx, full, a_t)[:, :depth_limit]  # [A, 7, V]
        k = args.topk

        # DSpark tree: children of a node = Markov-biased slot logits.
        def d_root():
            lp = dspark.logprobs(base[:, 0], full[a_t])
            top = lp.topk(k, -1)
            return top.indices, top.values

        def d_children(items, trees):
            rows = torch.stack([base[i, trees[i]["d"][n]] for i, n in items])
            prev = torch.tensor([trees[i]["t"][n] for i, n in items], device=dev)
            top = dspark.logprobs(rows, prev).topk(k, -1)
            return top.indices, top.values

        d_trees = expand(n_a, d_root, d_children, args.width, k, depth_limit)

        # EAGLE tree: one forward per depth over the frontier.
        cache: dict[tuple[int, int], tuple] = {}

        def e_root():
            top = eagle.logprobs(ectx["h"][a_t - 1]).topk(k, -1)
            return top.indices, top.values

        def e_children(items, trees):
            return self._eagle_children(items, trees, chunk, ectx, cache, k)

        e_trees = expand(n_a, e_root, e_children, args.width, k, depth_limit)
        cache.clear()

        tries = []
        nms = []
        for i, a in enumerate(chunk):
            trie = Trie()
            for tree, flag, chain_flag in (
                (e_trees[i], ts.E_TREE, ts.E_CHAIN),
                (d_trees[i], ts.D_TREE, ts.D_CHAIN),
            ):
                kept = sorted(keep_best(tree, args.keep), key=lambda n: tree["d"][n])
                chain = set(tree["chain"])
                for n in kept:
                    path = self._path(tree, n)
                    trie.add_path(path, flag | (chain_flag if n in chain else 0))
            branches, longest = ngram.branches(a, limit=args.ngram_branches, depth=depth_limit)
            for rank, branch in enumerate(branches):
                trie.add_path(branch, ts.N_TREE | (ts.N_CHAIN if rank == 0 else 0), ng=rank)
            tries.append(trie)
            nms.append(longest)

        ld = self._score_dspark(tries, chunk, base, full)
        le = self._score_eagle(tries, chunk, ectx)
        for i, a in enumerate(chunk):
            trie = tries[i]
            yield {
                "a": a,
                "ctx": a + 1,
                "truth": ids[a + 1 : a + 1 + depth_limit],
                "p": trie.parent,
                "t": trie.token,
                "f": trie.flags,
                "le": [round(x, 4) for x in le[i]],
                "ld": [round(x, 4) for x in ld[i]],
                "ng": trie.ng,
                "nm": nms[i],
            }

    @staticmethod
    def _path(tree, node):
        out = []
        while node >= 0:
            out.append(tree["t"][node])
            node = tree["p"][node]
        return out[::-1]

    def _eagle_forward(self, entries, ectx, depth):
        """entries: [(key, anchor, token, parent_key)] all at ``depth``; ``self._kv``
        maps key -> (k, v, next_hidden, ancestor keys)."""
        dev = self.device
        anchors = torch.tensor([e[1] for e in entries], device=dev)
        tokens = torch.tensor([e[2] for e in entries], device=dev)
        hidden, anc_k, anc_v = [], [], []
        for _key, anchor, _tok, parent in entries:
            if parent is None:
                hidden.append(ectx["next"][anchor - 1])
                anc_k.append(ectx["k"][:0])
                anc_v.append(ectx["v"][:0])
            else:
                pk, pv, ph, chain_k, chain_v = self._kv[parent]
                hidden.append(ph)
                anc_k.append(torch.cat([chain_k, pk[None]], 0))
                anc_v.append(torch.cat([chain_v, pv[None]], 0))
        hidden = torch.stack(hidden)
        anc_k, anc_v = torch.stack(anc_k), torch.stack(anc_v)
        out = []
        for s in range(0, len(entries), self.args.node_chunk):
            sl = slice(s, s + self.args.node_chunk)
            k, v, h = self.eagle.nodes(
                ectx, anchors[sl], depth, tokens[sl], hidden[sl], anc_k[sl], anc_v[sl]
            )
            lp = self.eagle.logprobs(h)
            nxt = rms(h, self.eagle.norm, self.eagle.eps)
            for j, e in enumerate(entries[sl]):
                self._kv[e[0]] = (k[j], v[j], nxt[j], anc_k[sl][j], anc_v[sl][j])
            out.append(lp)
        return torch.cat(out)

    def _eagle_children(self, items, trees, chunk, ectx, cache, k):
        self._kv = cache
        depth = trees[items[0][0]]["d"][items[0][1]]
        entries = []
        for i, n in items:
            parent = trees[i]["p"][n]
            entries.append(
                ((i, n), chunk[i], trees[i]["t"][n], None if parent < 0 else (i, parent))
            )
        top = self._eagle_forward(entries, ectx, depth).topk(k, -1)
        return top.indices, top.values

    def _gather(self, lp: torch.Tensor, rows: list[int], tokens: list[int]) -> list[float]:
        dev = self.device
        idx = torch.tensor(rows, device=dev, dtype=torch.long)
        tok = torch.tensor(tokens, device=dev, dtype=torch.long)
        return lp[idx, tok].tolist()

    def _score_dspark(self, tries, chunk, base, full):
        """DSpark log-probability of every trie edge (slot logits + Markov bias)."""
        dev = self.device
        row_of: dict[tuple[int, int], int] = {}
        slots, prev = [], []
        for i, trie in enumerate(tries):
            for p in sorted(set(trie.parent)):
                row_of[(i, p)] = len(slots)
                slots.append((i, 0 if p < 0 else trie.depth[p]))
                prev.append(int(full[chunk[i]]) if p < 0 else trie.token[p])
        nodes = [
            (i, n, row_of[(i, p)], tok)
            for i, trie in enumerate(tries)
            for n, (p, tok) in enumerate(zip(trie.parent, trie.token, strict=True))
        ]
        nodes.sort(key=lambda x: x[2])
        out = [[0.0] * len(t.token) for t in tries]
        step = self.args.node_chunk
        cursor = 0
        for s in range(0, len(slots), step):
            rows = torch.stack([base[i, d] for i, d in slots[s : s + step]])
            lp = self.dspark.logprobs(rows, torch.tensor(prev[s : s + step], device=dev))
            group = []
            while cursor < len(nodes) and nodes[cursor][2] < s + step:
                group.append(nodes[cursor])
                cursor += 1
            values = self._gather(lp, [g[2] - s for g in group], [g[3] for g in group])
            for (i, n, _r, _t), value in zip(group, values, strict=True):
                out[i][n] = value
        return out

    def _score_eagle(self, tries, chunk, ectx):
        """EAGLE log-probability of every trie edge: forward all internal nodes."""
        self._kv = {}
        out = [[0.0] * len(t.token) for t in tries]
        root = self.eagle.logprobs(ectx["h"][torch.tensor(chunk, device=self.device) - 1])
        group = [
            (i, n, tok)
            for i, t in enumerate(tries)
            for n, (p, tok) in enumerate(zip(t.parent, t.token, strict=True))
            if p < 0
        ]
        for (i, n, _t), value in zip(
            group, self._gather(root, [g[0] for g in group], [g[2] for g in group]), strict=True
        ):
            out[i][n] = value
        del root
        has_child = [set(p for p in t.parent if p >= 0) for t in tries]
        for depth in range(1, ts.MAX_DEPTH):
            entries = []
            for i, trie in enumerate(tries):
                for n in sorted(has_child[i]):
                    if trie.depth[n] == depth:
                        p = trie.parent[n]
                        entries.append(((i, n), chunk[i], trie.token[n], None if p < 0 else (i, p)))
            if not entries:
                continue
            lp = self._eagle_forward(entries, ectx, depth)
            rowmap = {e[0]: j for j, e in enumerate(entries)}
            group = [
                (i, n, rowmap[(i, p)], tok)
                for i, trie in enumerate(tries)
                for n, (p, tok) in enumerate(zip(trie.parent, trie.token, strict=True))
                if p >= 0 and trie.depth[p] == depth
            ]
            values = self._gather(lp, [g[2] for g in group], [g[3] for g in group])
            for (i, n, _r, _t), value in zip(group, values, strict=True):
                out[i][n] = value
            del lp
        self._kv = {}
        return out


# --------------------------------------------------------------------------- driver


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-tree-candidates")
    parser.add_argument("--capture", type=Path, required=True, help="verify_offline plain capture")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--eagle-head", type=Path, required=True)
    parser.add_argument("--dspark", type=Path, required=True)
    parser.add_argument("--dspark-hf-layers", default="0,7,14,21,28")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--keep", type=int, default=48)
    parser.add_argument("--ngram-branches", type=int, default=8)
    parser.add_argument("--anchor-chunk", type=int, default=32)
    parser.add_argument("--node-chunk", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    from apertus_eagle.contract import (
        aux_layer_ids,
        load_contract,
        num_decoder_layers,
        target_identity,
    )
    from apertus_eagle.features import FeatureCapture, load_teacher
    from apertus_eagle.generate_targets import prompt_messages, render_prompt
    from transformers import AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    started = time.time()
    contract = load_contract(args.contract)
    target_path = contract["source"]["authorized_checkpoint"]
    device = torch.device("cuda")
    model, _loader = load_teacher(target_path)
    model.eval().requires_grad_(False)
    eagle_layers = aux_layer_ids(contract, "hf")
    dspark_layers = [int(x) for x in args.dspark_hf_layers.split(",")]
    capture = FeatureCapture(
        model, sorted(set(eagle_layers) | set(dspark_layers)), num_decoder_layers(contract)
    )
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    lm_head = (head if head is not None else model.lm_head).weight
    eagle = Eagle(args.eagle_head, embed, device)
    dspark = DSpark(args.dspark, embed, lm_head, device)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    builder = Builder(args, eagle, dspark, device)

    report = json.loads(args.capture.read_text())
    cases = {c["id"]: c for c in report["cases"] if "@cap" not in c["id"]}
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if str(r["id"]) in cases]
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]

    stats: dict[str, Any] = {"sequences": [], "anchors": 0}
    walk_totals = {name: ts.Totals() for name in ("chain-dspark", "chain-eagle", "chain-ngram")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), gzip.open(args.output, "wt") as out:
        for row in rows:
            t0 = time.time()
            case = cases[str(row["id"])]
            prompt = list(row.get("prompt_ids") or render_prompt(tokenizer, prompt_messages(row)))
            if sha256_ids(prompt) != case["prompt_ids_sha256"]:
                raise SystemExit(f"prompt ids differ from the capture for {row['id']}")
            ids = prompt + list(case["token_ids"])
            hidden, last, _ = capture.run(torch.tensor([ids], device=device))
            del hidden
            aux_e = torch.cat([capture.captured[i][0] for i in eagle_layers], -1)
            aux_d = torch.cat([capture.captured[i][0] for i in dspark_layers], -1)
            # Does the HF target agree with the vLLM greedy continuation?
            p = len(prompt)
            logits = F.linear(last[p - 1 : -1], lm_head)
            agree = (logits.argmax(-1).cpu() == torch.tensor(ids[p:])).float().mean().item()
            anchors = {}
            for rec in builder.sequence(ids, p, aux_e, aux_d):
                rec = {"seq": str(row["id"]), "wl": args.workload, **rec}
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                anchors[rec["a"]] = rec
            for name in walk_totals:
                walk_totals[name].add(ts.walk(anchors, ts.POLICIES[name], 7))
            stats["anchors"] += len(anchors)
            stats["sequences"].append(
                {
                    "id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()[:16],
                    "prompt_tokens": p,
                    "generated_tokens": len(case["token_ids"]),
                    "hf_argmax_agreement": round(agree, 5),
                    "mean_nodes": round(
                        sum(len(r["t"]) for r in anchors.values()) / max(len(anchors), 1), 1
                    ),
                    "seconds": round(time.time() - t0, 2),
                }
            )
            print(
                json.dumps({"seq": len(stats["sequences"]), **stats["sequences"][-1]}), flush=True
            )

    summary = {
        "workload": args.workload,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "args": {k: str(v) for k, v in vars(args).items()},
        "target": target_identity(contract),
        "eagle_head": str(args.eagle_head),
        "eagle_head_sha256": file_sha256(args.eagle_head / "model.safetensors"),
        "dspark": str(args.dspark),
        "dspark_sha256": file_sha256(args.dspark / "model.safetensors"),
        "eagle_hf_layers": eagle_layers,
        "dspark_hf_layers": dspark_layers,
        "capture": str(args.capture),
        "capture_sha256": file_sha256(args.capture),
        "prompts_sha256": file_sha256(args.prompts),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": round(time.time() - started, 1),
        "anchors": stats["anchors"],
        "hf_argmax_agreement": (
            sum(s["hf_argmax_agreement"] * s["generated_tokens"] for s in stats["sequences"])
            / max(sum(s["generated_tokens"] for s in stats["sequences"]), 1)
        ),
        "chain_k7_accepted_per_step": {
            name: t.summary()["accepted_per_step"] for name, t in walk_totals.items()
        },
        "sequences": stats["sequences"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sequences"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
