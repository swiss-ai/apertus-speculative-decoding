import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
shape = pytest.importorskip("apertus_tree.shape")
agg = pytest.importorskip("apertus_tree.shape_aggregate")


def test_walk_counts_leading_matches_and_jumps():
    nan = np.nan
    match = np.array(
        [
            [1, 1, 0, 1, 1, 1, 1],  # step 0: 2 accepted -> jump 3
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, nan],  # step at 3: k=3 caps at 3 -> jump 4
            [0] * 7,
            [0] * 7,
            [1, nan, nan, nan, nan, nan, nan],
        ],
        float,
    )
    assert agg.walk(match, 3) == [(0, 2), (3, 3)]
    assert agg.walk(match, 7) == [(0, 2), (3, 6)]
    assert agg.walk(match, 1) == [(0, 1), (2, 0), (3, 1), (5, 0), (6, 1)]


def test_sampled_lengths_chain_one_minus_tv():
    nan = np.nan
    tv = np.array([[0.0, 0.5, 1.0, 0.0], [0.2, 0.2, nan, nan]])
    # anchor 0: 1 + 1 + 0.5 + 0 (+0) ; anchor 1: 1 + 0.8 + 0.64, the chain ends at nan
    assert np.allclose(agg.sampled_lengths(tv, 3), [2.5, 2.44])
    assert np.allclose(agg.sampled_lengths(tv, 1), [2.0, 1.8])


def test_dist_metrics_identities():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    lp = torch.log_softmax(logits, -1)
    truth = lp.argmax(-1)
    m = shape.dist_metrics(lp, lp, truth, truth)
    assert torch.allclose(m["tv"], torch.zeros(4), atol=1e-6)
    assert torch.allclose(m["kl"], torch.zeros(4), atol=1e-6)
    assert m["match"].eq(1).all() and m["agree"].eq(1).all()
    uniform = torch.full((1, 50), -math.log(50))
    point = torch.full((1, 50), -1e4)
    point[0, 0] = 0.0
    point = torch.log_softmax(point, -1)
    m = shape.dist_metrics(point, uniform, torch.tensor([0]), torch.tensor([0]))
    assert m["ent"].item() == pytest.approx(math.log(50), rel=1e-5)
    assert m["tv"].item() == pytest.approx(1 - 1 / 50, rel=1e-5)
    assert m["kl"].item() == pytest.approx(math.log(50), rel=1e-4)


# --------------------------------------------------------------------------- speculators heads
# Each head's replay logits against its reference implementation (tests/reference_speculators.py,
# vendored from Red Hat's speculators source) on one random sequence, tiny random weights, fp32.


def _tiny(tmp_path, name, config, tensors):
    from safetensors.torch import save_file

    head = tmp_path / name
    head.mkdir()
    (head / "config.json").write_text(__import__("json").dumps(config))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(head / "model.safetensors"))
    return head


def test_eagle3_speculators_matches_reference(tmp_path):
    transformers = pytest.importorskip("transformers")
    ref = pytest.importorskip("reference_speculators")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

    from apertus_tree.qwen_heads import Eagle3Speculators

    torch.manual_seed(0)
    hidden, heads, kv, hd, vocab, draft, length = 64, 4, 2, 16, 300, 120, 24
    tl = dict(
        hidden_size=hidden, intermediate_size=96, num_attention_heads=heads,
        num_key_value_heads=kv, head_dim=hd, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        vocab_size=vocab, num_hidden_layers=1, attention_bias=False, mlp_bias=False,
        hidden_act="silu", model_type="llama", max_position_embeddings=4096,
    )  # fmt: skip
    config = transformers.LlamaConfig(**tl)
    layer = ref.Eagle3DecoderLayer(config, 0, norm_before_residual=True)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.2)
    embed = torch.randn(vocab, hidden)
    fc = torch.randn(hidden, 3 * hidden) * 0.05
    norm = LlamaRMSNorm(hidden, eps=1e-6)
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
    lm_head = torch.randn(draft, hidden) * 0.2
    target_ids = torch.randperm(vocab)[:draft].sort().values
    d2t = target_ids - torch.arange(draft)
    tensors = {f"layers.0.{k}": v.detach() for k, v in layer.state_dict().items()}
    tensors |= {"embed_tokens.weight": embed, "fc.weight": fc, "norm.weight": norm.weight.detach(),
                "lm_head.weight": lm_head, "d2t": d2t,
                "t2d": torch.zeros(vocab, dtype=torch.bool).index_fill(0, target_ids, True)}  # fmt: skip
    cfg = {"speculators_model_type": "eagle3", "draft_vocab_size": draft,
           "norm_before_residual": True, "transformer_layer_config": tl}  # fmt: skip
    head = _tiny(tmp_path, "eagle3", cfg, tensors)
    mine = Eagle3Speculators(head, embed, "cpu", target_vocab=vocab)

    ids = torch.randint(0, vocab, (length,))
    aux = torch.randn(length, 3 * hidden)
    ctx = mine.context(ids, aux)

    # Reference step 0: pairs (aux_j, ids_{j+1}) at positions j, causal, RoPE as served.
    x = torch.cat([embed[ids[1:]], aux[:-1] @ fc.T], -1)[None]
    pos = torch.arange(length - 1)[None]
    rope = LlamaRotaryEmbedding(config)
    mask = torch.full((length - 1, length - 1), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        h_ref, (k_ref, v_ref) = layer(x, attention_mask=mask, use_cache=True,
                                      position_embeddings=rope(x, pos))  # fmt: skip

    def full(h):
        lp = torch.log_softmax(lm_head @ norm(h).T, 0).T
        out = torch.full((h.shape[0], vocab), float("-inf"))
        out[:, target_ids] = lp
        return out

    lp_ref = full(h_ref[0])
    lp = mine.logprobs(ctx["h"])
    finite = torch.isfinite(lp_ref)
    assert torch.equal(finite, torch.isfinite(lp))
    assert torch.allclose(lp[finite], lp_ref[finite], atol=1e-3)

    # TTT step 1 for anchor a: (ids[a+1], pre-norm h of pair a-1) at position a,
    # attending to context pairs 0..a-1 and itself.
    a = 15
    with torch.no_grad():
        x2 = torch.cat([embed[ids[a + 1]][None], h_ref[0, a - 1][None]], -1)[None]
        p2 = torch.tensor([[a]])
        h2_ref, _ = layer(x2, past_key_value=(k_ref[:, :, :a], v_ref[:, :, :a]),
                          position_embeddings=rope(x2, p2))  # fmt: skip
    empty = ctx["k"].new_zeros((1, 0, *ctx["k"].shape[1:]))
    _k, _v, h2 = mine.nodes(ctx, torch.tensor([a]), 1, ids[a + 1][None],
                            mine.carry(ctx["next"][a - 1][None]), empty, empty)  # fmt: skip
    lp2, lp2_ref = mine.logprobs(h2), full(h2_ref[0])
    finite = torch.isfinite(lp2_ref)
    assert torch.allclose(lp2[finite], lp2_ref[finite], atol=1e-3)


def test_dflash_matches_reference(tmp_path):
    transformers = pytest.importorskip("transformers")
    ref = pytest.importorskip("reference_speculators")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

    from apertus_tree.qwen_heads import DFlash

    torch.manual_seed(1)
    hidden, heads, kv, hd, vocab, layers, block, window = 64, 4, 2, 16, 200, 2, 6, 5
    length, naux, mask_id = 20, 3, 199
    tl = dict(
        hidden_size=hidden, intermediate_size=96, num_attention_heads=heads,
        num_key_value_heads=kv, head_dim=hd, rms_norm_eps=1e-6, vocab_size=vocab,
        num_hidden_layers=layers, attention_bias=False, attention_dropout=0.0,
        hidden_act="silu", model_type="qwen3", max_position_embeddings=4096,
        layer_types=["sliding_attention"] * layers, use_sliding_window=True,
        sliding_window=window, rope_theta=1_000_000.0,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )  # fmt: skip
    config = transformers.Qwen3Config(**tl)
    config._attn_implementation = "eager"
    blocks = [ref.Qwen3DFlashDecoderLayer(config, i) for i in range(layers)]
    for b in blocks:
        for p in b.parameters():
            torch.nn.init.normal_(p, std=0.2)
    fc = torch.randn(hidden, naux * hidden) * 0.05
    hidden_norm = Qwen3RMSNorm(hidden, eps=1e-6)
    norm = Qwen3RMSNorm(hidden, eps=1e-6)
    torch.nn.init.normal_(hidden_norm.weight, mean=1.0, std=0.1)
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
    embed, lm_head = torch.randn(vocab, hidden), torch.randn(vocab, hidden) * 0.2
    tensors = {f"layers.{i}.{k}": v.detach() for i, b in enumerate(blocks)
               for k, v in b.state_dict().items()}  # fmt: skip
    tensors |= {"fc.weight": fc, "hidden_norm.weight": hidden_norm.weight.detach(),
                "norm.weight": norm.weight.detach(), "embed_tokens.weight": embed,
                "lm_head.weight": lm_head}  # fmt: skip
    cfg = {"speculators_model_type": "dflash", "block_size": block, "mask_token_id": mask_id,
           "aux_hidden_state_layer_ids": [2, 10, 18], "sample_from_anchor": False,
           "sliding_window_non_causal": False, "draft_vocab_size": vocab,
           "transformer_layer_config": tl}  # fmt: skip
    mine = DFlash(_tiny(tmp_path, "dflash", cfg, tensors), embed, lm_head, "cpu")
    assert mine.slot_offset == 1 and mine.max_depth == block - 1

    ids = torch.randint(0, vocab - 1, (length,))
    aux = torch.randn(length, naux * hidden)
    rope = Qwen3RotaryEmbedding(config)
    ctx = mine.context(aux)
    for a in (3, 9, 14):
        got = mine.block_logits(ctx, ids, torch.tensor([a]))[0]
        noise = embed[torch.tensor([ids[a].item()] + [mask_id] * (block - 1))][None]
        target = hidden_norm(aux @ fc.T)[None]
        pos = torch.cat([torch.arange(length), a + torch.arange(block)])[None]
        allowed = ref.anchor_block_mask(length, a, block, window)
        mask = torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf"))[None, None]
        with torch.no_grad():
            h = noise
            for b in blocks:
                h = b(target, h, mask, rope(h, pos))
            want = norm(h)[0] @ lm_head.T
        assert torch.allclose(got, want, atol=1e-3, rtol=1e-4)
        # Depth d reads slot d, the slot at position a + d.
        assert torch.allclose(
            mine.logprobs(got[1:]), torch.log_softmax(want[1:], -1), atol=1e-3
        )
