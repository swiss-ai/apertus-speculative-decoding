# Reference layers of the Red Hat speculators drafters, vendored for tests/test_shape.py.
#
# Apache-2.0, (c) the speculators authors (https://github.com/vllm-project/speculators).
# - Eagle3Attention / Eagle3DecoderLayer: eagle3.py of RedHatAI/Qwen3-8B-speculator.eagle3,
#   revision 08610ffa01dd9f16731fe8f627b85905b6aa51c4 (speculators 0.2.0.dev11),
#   sha256 51a594450e6b67065e7902fbf50f550e294c9958703bead20d864f19612430b9.
# - Qwen3DFlashAttention / Qwen3DFlashDecoderLayer and the anchor-block mask rule:
#   speculators 0.7.0 wheel, models/dflash/model_definitions.py and attention.py
#   (the head was trained with 0.7.0.dev120; the forward math is unchanged to 0.9.0).
#
# Changes: imports trimmed to transformers; ``apply_rotary_pos_emb`` is a local copy
# so the call does not depend on the transformers version (eagle3.py passes the
# deprecated ``position_ids`` positionally); training-only arguments dropped.
# The arithmetic of every forward is unchanged.
from __future__ import annotations

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, repeat_kv
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ----------------------------------------------------------------------------- EAGLE-3


class Eagle3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        input_size = 2 * self.hidden_size
        bias = config.attention_bias
        self.q_proj = nn.Linear(input_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(input_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(input_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_embeddings=None):  # fmt: skip
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        past_key_value_out = None
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        if use_cache:
            past_key_value_out = (key_states, value_states)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
            self.head_dim**0.5
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value_out


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, norm_before_residual: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config, layer_idx)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_embeddings=None):  # fmt: skip
        embeds = hidden_states[:, :, : self.hidden_size]
        hidden = hidden_states[:, :, self.hidden_size : 2 * self.hidden_size]
        if self.norm_before_residual:
            hidden = self.hidden_norm(hidden)
            residual = hidden
        else:
            residual = hidden
            hidden = self.hidden_norm(hidden)
        embeds = self.input_layernorm(embeds)
        attn_input = torch.cat([embeds, hidden], dim=-1)
        attn_output, past = self.self_attn(
            attn_input, attention_mask=attention_mask, past_key_value=past_key_value,
            use_cache=use_cache, position_embeddings=position_embeddings,
        )  # fmt: skip
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, past


# ----------------------------------------------------------------------------- DFlash


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        self.sliding_window = (
            config.sliding_window
            if layer_types is not None and layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output, _ = eager_attention_forward(
            self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling,
            sliding_window=self.sliding_window,
        )  # fmt: skip
        return self.o_proj(attn_output.reshape(bsz, q_len, -1))


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, target_hidden, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, target_hidden, position_embeddings, attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def anchor_block_mask(total_seq_len, anchor, block_size, sliding_window, non_causal=False):
    """Boolean [block, total + block] mask of one anchor block (``create_anchor_block_mask_mod``
    for a single document): base tokens before the anchor (inside the window), plus
    the own block, causal unless ``non_causal``."""
    kv = torch.arange(total_seq_len + block_size)
    q = torch.arange(block_size)[:, None]
    base = (kv < total_seq_len) & (kv < anchor)
    if sliding_window is not None:
        base &= kv >= anchor - sliding_window
    block = (kv >= total_seq_len)[None, :].expand(block_size, -1)
    if not non_causal:
        block = block & (kv[None, :] <= q + total_seq_len)
    return base[None, :] | block
