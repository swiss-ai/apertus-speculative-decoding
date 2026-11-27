"""Speculators-format drafters (Red Hat EAGLE-3 and DFlash) for the shape replay.

Both classes expose the interface of ``apertus_tree.candidates.Eagle`` /
``DSpark`` that ``apertus_tree.shape.Collector`` uses, so the Apertus heads and
these heads run through the same teacher-forced replay.

Spec (what each class reproduces):

- ``Eagle3Speculators``: ``Eagle3Speculator`` of speculators 0.2.0.dev11 (the
  ``eagle3.py`` shipped in ``RedHatAI/Qwen3-8B-speculator.eagle3``), wired as
  vLLM 0.30 serves it (``llama_eagle3.py``): ``fc`` over the three concatenated
  aux hidden states without a norm, ``input_layernorm`` on the token embedding,
  ``hidden_norm`` on the fused hidden state, which is also the residual when
  ``norm_before_residual`` is set; default RoPE (theta from the config) at the
  drafter position; the next draft step reads the *pre-norm* output (vLLM
  ``norm_output`` false). Draft logits cover ``draft_vocab_size`` tokens and map to
  target ids ``i + d2t[i]``; every other target id gets probability 0.
  (The HF ``forward`` in that ``eagle3.py`` never passes position embeddings to
  its layer, so it runs without RoPE; vLLM applies RoPE, and so does this class.)
- ``DFlash``: ``DFlashDraftModel`` of speculators (``models/dflash``, the head
  was trained with 0.7.0.dev120): context K/V from ``hidden_norm(fc(aux))``
  with ``k_norm`` and RoPE at the context positions, a block of ``block_size``
  query slots at positions ``a .. a+block-1`` (slot 0 = the anchor token, the
  rest the mask token), sliding-window context ``[a - window, a)`` and a causal
  in-block mask (``sliding_window_non_causal`` false), no Markov head.
  ``sample_from_anchor`` false: slot ``d`` predicts position ``a + d``, so depth
  ``d`` reads slot ``d`` (``slot_offset = 1``; DSpark reads slot ``d - 1``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from apertus_tree.candidates import DSpark, attend, mlp, rms

NEG = float("-inf")


def default_inv_freq(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


class DefaultRope:
    """Unscaled (``rope_type: default``) rotary embedding, neox layout."""

    def __init__(self, dim: int, theta: float, device):
        self.inv = default_inv_freq(dim, theta).float().to(device)

    def __call__(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        freqs = pos.float()[..., None] * self.inv
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos()[..., None, :], emb.sin()[..., None, :]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rot = torch.cat([-x2, x1], -1)
        return (x.float() * cos + rot.float() * sin).to(x.dtype)


def _rope_theta(tl: dict[str, Any]) -> float:
    rope = tl.get("rope_parameters") or {}
    scaling = tl.get("rope_scaling") or {}
    kind = rope.get("rope_type") or scaling.get("rope_type") or "default"
    if kind != "default":
        raise SystemExit(f"expected default RoPE, got {kind!r}")
    return float(rope.get("rope_theta") or tl.get("rope_theta"))


class Eagle3Speculators:
    """Red Hat speculators EAGLE-3 head (one llama layer, reduced draft vocabulary)."""

    carries_prenorm = True

    def __init__(self, head: Path, embed: torch.Tensor, device, target_vocab: int | None = None):
        cfg = json.loads((head / "config.json").read_text())
        if cfg.get("speculators_model_type") != "eagle3":
            raise SystemExit(f"{head}: not a speculators eagle3 head")
        tl = cfg["transformer_layer_config"]
        if tl.get("num_hidden_layers", 1) != 1:
            raise SystemExit("this replay implements a one-layer EAGLE-3 head")
        w = load_file(str(head / "model.safetensors"), device=str(device))
        self.cfg = cfg
        self.heads, self.groups = tl["num_attention_heads"], tl["num_key_value_heads"]
        self.dim = tl.get("head_dim") or tl["hidden_size"] // self.heads
        self.eps = tl["rms_norm_eps"]
        self.norm_before_residual = bool(cfg.get("norm_before_residual", False))
        self.vocab = int(target_vocab or tl["vocab_size"])
        self.embed = w.get("embed_tokens.weight", embed)
        self.fc = w["fc.weight"]
        p = "layers.0."
        self.q, self.k, self.v, self.o = (w[p + f"self_attn.{n}_proj.weight"] for n in "qkvo")
        self.gate, self.up, self.down = (
            w[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up", "down")
        )
        self.in_ln, self.hid_ln = w[p + "input_layernorm.weight"], w[p + "hidden_norm.weight"]
        self.post_ln = w[p + "post_attention_layernorm.weight"]
        self.norm, self.lm_head = w["norm.weight"], w["lm_head.weight"]
        draft = self.lm_head.shape[0]
        if "d2t" in w:
            self.draft_to_target = torch.arange(draft, device=device) + w["d2t"].long()
        else:
            self.draft_to_target = torch.arange(draft, device=device)
        if int(self.draft_to_target.max()) >= self.vocab:
            raise SystemExit("d2t maps outside the target vocabulary")
        self.rope = DefaultRope(self.dim, _rope_theta(tl), device)
        self.scale = self.dim**-0.5

    def project(self, aux: torch.Tensor) -> torch.Tensor:
        return F.linear(aux.to(self.fc.dtype), self.fc)

    def _qkv(self, tokens, hidden, pos):
        emb = self.embed[tokens.clamp(max=self.embed.shape[0] - 1)]
        h_n = rms(hidden, self.hid_ln, self.eps)
        residual = h_n if self.norm_before_residual else hidden
        x = torch.cat([rms(emb, self.in_ln, self.eps), h_n], -1)
        n = x.shape[:-1]
        q = F.linear(x, self.q).view(*n, self.heads, self.dim)
        k = F.linear(x, self.k).view(*n, self.groups, self.dim)
        v = F.linear(x, self.v).view(*n, self.groups, self.dim)
        return self.rope(q, pos), self.rope(k, pos), v, residual

    def _finish(self, attn, residual):
        h = residual + F.linear(attn, self.o)
        return h + mlp(rms(h, self.post_ln, self.eps), self.gate, self.up, self.down)

    def draft_logits(self, h_raw: torch.Tensor) -> torch.Tensor:
        return F.linear(rms(h_raw, self.norm, self.eps), self.lm_head)

    def logprobs(self, h_raw: torch.Tensor) -> torch.Tensor:
        """Log-probs over the full target vocabulary (-inf outside the draft vocabulary)."""
        draft = torch.log_softmax(self.draft_logits(h_raw).float(), -1)
        out = draft.new_full((*draft.shape[:-1], self.vocab), NEG)
        out[..., self.draft_to_target] = draft
        return out

    def carry(self, h_raw: torch.Tensor) -> torch.Tensor:
        """Hidden state fed to the next draft step: the pre-norm output."""
        return h_raw

    def context(self, ids: torch.Tensor, aux: torch.Tensor):
        """Step 0 over the whole sequence: pair j = (aux_j, token_{j+1}) at position j."""
        hidden = self.project(aux[:-1])
        tokens = ids[1:]
        pos = torch.arange(tokens.shape[0], device=ids.device)
        q, k, v, residual = self._qkv(tokens, hidden, pos)
        rep = self.heads // self.groups
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.repeat_interleave(rep, 1).transpose(0, 1)[None],
            v.repeat_interleave(rep, 1).transpose(0, 1)[None],
            is_causal=True,
        )[0].transpose(0, 1)
        h = self._finish(out.reshape(tokens.shape[0], -1), residual)
        return {"k": k, "v": v, "h": h, "next": h}

    def nodes(self, ctx, anchors, depth, tokens, parent_hidden, anc_k, anc_v):
        """TTT step ``depth`` (>= 1) of the pair at a-1: position a-1+depth, keys =
        context pairs 0..a-1 plus the ancestors and itself (as in vLLM's draft KV)."""
        pos = anchors - 1 + depth
        q, k, v, residual = self._qkv(tokens, parent_hidden, pos)
        own_k = torch.cat([anc_k, k[:, None]], 1)
        own_v = torch.cat([anc_v, v[:, None]], 1)
        t = ctx["k"].shape[0]
        ctx_valid = torch.arange(t, device=q.device)[None] < anchors[:, None]
        own_valid = torch.ones(own_k.shape[:2], dtype=torch.bool, device=q.device)
        out = attend(q, ctx["k"], ctx["v"], ctx_valid, own_k, own_v, own_valid, self.scale)
        h = self._finish(out, residual)
        return k, v, h


class DFlash(DSpark):
    """Speculators DFlash block drafter (full vocabulary, no Markov head)."""

    def __init__(self, path: Path, embed: torch.Tensor, lm_head: torch.Tensor, device):
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("speculators_model_type") != "dflash":
            raise SystemExit(f"{path}: not a speculators dflash drafter")
        if cfg.get("sample_from_anchor"):
            raise SystemExit("this replay implements sample_from_anchor false")
        if cfg.get("sliding_window_non_causal"):
            raise SystemExit("expected a causal in-block mask (sliding_window_non_causal false)")
        tl = cfg["transformer_layer_config"]
        types = set(tl.get("layer_types") or [])
        if types - {"sliding_attention"} or not tl.get("use_sliding_window"):
            raise SystemExit("this replay implements all-sliding-window DFlash layers")
        w = load_file(str(path / "model.safetensors"), device=str(device))
        self.cfg = cfg
        self.block = cfg["block_size"]
        self.mask_id = cfg["mask_token_id"]
        self.aux_layer_ids = list(cfg["aux_hidden_state_layer_ids"])
        self.heads, self.groups = tl["num_attention_heads"], tl["num_key_value_heads"]
        self.dim = tl.get("head_dim") or tl["hidden_size"] // self.heads
        self.eps = tl["rms_norm_eps"]
        self.window = tl["sliding_window"]
        # Per-layer attention as DSpark.block_logits reads it: all sliding, causal block.
        self.windows = [self.window] * tl["num_hidden_layers"]
        self.causal = [True] * tl["num_hidden_layers"]
        self.embed = w.get("embed_tokens.weight", embed)
        self.lm_head = w.get("lm_head.weight", lm_head)
        self.fc, self.hidden_norm, self.norm = (
            w["fc.weight"],
            w["hidden_norm.weight"],
            w["norm.weight"],
        )
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
        self.rope = DefaultRope(self.dim, _rope_theta(tl), device)
        self.scale = self.dim**-0.5

    # Depth d (the d-th drafted token, position a + d) reads slot d.
    slot_offset = 1

    @property
    def max_depth(self) -> int:
        return self.block - 1

    def logprobs(self, base: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        """log-softmax of a slot's logits; DFlash has no Markov bias (``prev`` unused)."""
        return torch.log_softmax(base.float(), -1)


def hf_layers(vllm_aux_ids: list[int]) -> list[int]:
    """vLLM aux id k is the residual stream entering decoder layer k, i.e. the output
    of HF decoder layer k - 1 (``Qwen2Model.forward`` records idx + 1 after layer idx)."""
    return [int(i) - 1 for i in vllm_aux_ids]
