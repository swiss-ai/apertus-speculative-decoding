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


def test_sampled_acceptance_chains_one_minus_tv():
    nan = np.nan
    tv = np.array(
        [
            [0.0] * 7,  # target == drafter: every depth accepted
            [1.0] * 7,  # disjoint: nothing accepted
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, nan, nan, nan, nan, nan],  # sequence ends after depth 2
        ]
    )
    assert agg.sampled_lengths(tv, 3).tolist() == [4.0, 1.0, 1.875, 3.0]
    assert agg.sampled_lengths(tv, 7)[0] == 8.0
    assert agg.sampled_lengths(tv, 1).tolist() == [2.0, 1.0, 1.5, 2.0]


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


# --------------------------------------------------------------------------- DSpark vs speculators
#
# The reference is speculators' own DSparkDraftModel (the code the Red Hat
# Qwen3-4B head was trained with; pinned checkout passed on PYTHONPATH as
# SPECULATORS_SRC). Both models see the same aux hidden states; the reference's
# anchor sampler is replaced by fixed anchors. Compared: per-slot log-probs of
# base logits + the vanilla Markov bias of the teacher-forced previous token.


def _import_speculators():
    """speculators' package import pulls in ``datasets`` (training data loaders) even
    though the model code never uses it; the serving image lacks it, so a stub that
    refuses every call stands in for it there."""
    import sys
    import types

    try:
        import datasets  # noqa: F401
    except ImportError:
        stub = types.ModuleType("datasets")

        def _unavailable(*_a, **_k):
            raise RuntimeError("datasets stub: not used by the DSpark model code")

        for name in ("load_from_disk", "load_dataset", "concatenate_datasets"):
            setattr(stub, name, _unavailable)
        stub.Dataset = type("Dataset", (), {})
        sys.modules["datasets"] = stub
    return pytest.importorskip("speculators.models.dspark")


def _reference_logprobs(model, aux, ids, anchors, monkeypatch):
    import speculators.models.dflash.core as dcore

    n_anchor, block = len(anchors), model.block_size
    anchor_t = torch.tensor(anchors, dtype=torch.long, device=aux.device)
    monkeypatch.setattr(
        dcore,
        "select_anchors",
        lambda *_a, **_k: (anchor_t, torch.ones(n_anchor, dtype=torch.bool, device=aux.device)),
    )
    seq = ids[None]
    hidden = model.layers[0].self_attn.q_proj.weight.shape[1]
    _h, logits, _t, _m, idx = model._backbone_forward(
        aux[None],
        seq,
        torch.ones_like(seq),
        torch.zeros(1, len(ids), hidden, dtype=aux.dtype, device=aux.device),
        torch.zeros_like(seq),
        max_anchors=n_anchor,
    )
    prev = seq[0, idx].view(n_anchor, block)
    bias = model.markov_head.block_bias(
        prev_token_ids=prev, hidden_states=_h.view(n_anchor, block, -1)
    )
    return torch.log_softmax((logits.view(n_anchor, block, -1) + bias).float(), -1)


def _ours_logprobs(dspark, aux, ids, anchors):
    a_t = torch.tensor(anchors, device=aux.device)
    base = dspark.block_logits(dspark.context(aux), ids, a_t)
    block = base.shape[1]
    prev = ids[a_t[:, None] + torch.arange(block, device=aux.device)[None]]
    out = [dspark.logprobs(base[:, s], prev[:, s]) for s in range(block)]
    return torch.stack(out, 1)


def test_dspark_matches_speculators_reference_tiny(tmp_path, monkeypatch):
    """Random tiny Qwen3-style DSpark with a sliding and a full layer, window crossed."""
    _import_speculators()
    from safetensors.torch import save_file
    from speculators import SpeculatorModelConfig
    from speculators.models.dspark import DSparkDraftModel

    from apertus_tree.candidates import DSpark

    torch.manual_seed(0)
    vocab, hid, block = 97, 64, 4
    layer_cfg = {
        "model_type": "qwen3",
        "hidden_size": hid,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 256,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "sliding_window": 6,
        "use_sliding_window": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "_attn_implementation": "eager",
    }
    cfg = SpeculatorModelConfig.from_dict(
        {
            "speculators_model_type": "dspark",
            "transformer_layer_config": layer_cfg,
            "draft_vocab_size": vocab,
            "block_size": block,
            "aux_hidden_state_layer_ids": [1, 2],
            "mask_token_id": vocab - 1,
            "sample_from_anchor": True,
            "markov_rank": 8,
            "markov_head_type": "vanilla",
            "speculators_config": {
                "algorithm": "dspark",
                "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": block}],
                "default_proposal_method": "greedy",
                "verifier": {"name_or_path": "tiny", "architectures": ["Qwen3ForCausalLM"]},
            },
        }
    )
    cfg.transformer_layer_config._attn_implementation = "eager"
    ref = DSparkDraftModel(cfg).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.normal_(1.0 if name.endswith("norm.weight") else 0.0, 0.1 if "norm" in name else 0.3)
    state = {k: v.detach().contiguous() for k, v in ref.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(cfg.to_json_string())

    ids = torch.randint(0, vocab - 1, (24,))
    aux = torch.randn(len(ids), 2 * hid)
    anchors = [1, 3, 7, 9, 12, 16, 19]  # past the 6-token window; <= len - block
    with torch.no_grad():
        want = _reference_logprobs(ref, aux, ids, anchors, monkeypatch)
        ours = DSpark(tmp_path, state["embed_tokens.weight"], state["lm_head.weight"], "cpu")
        got = _ours_logprobs(ours, aux, ids, anchors)
    assert ours.windows == [6, None, 6] and ours.causal == [True, False, True]
    assert (got - want).abs().max().item() < 1e-3


def test_dspark_checkpoint_matches_speculators_reference(monkeypatch):
    """The real head on one public sequence longer than its 2048-token window (GPU job only).

    Set SHAPE_REF_DSPARK (speculators DSpark checkpoint) and SHAPE_REF_TARGET (its
    verifier); runs in float32. Aux states are taken from the target's
    ``output_hidden_states`` (index k = after decoder layer k-1), independently of
    the forward hooks the shape run uses.
    """
    import json
    import os
    from pathlib import Path

    path, target = os.environ.get("SHAPE_REF_DSPARK"), os.environ.get("SHAPE_REF_TARGET")
    if not path or not target or not torch.cuda.is_available():
        pytest.skip("needs SHAPE_REF_DSPARK, SHAPE_REF_TARGET and a GPU")
    _import_speculators()
    from safetensors.torch import load_file
    from speculators import SpeculatorModelConfig
    from speculators.models.dspark import DSparkDraftModel
    from transformers import AutoModelForCausalLM

    from apertus_tree.candidates import DSpark

    dev = torch.device("cuda")
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16).to(dev).eval()
    # A public, deterministic token sequence: 2,600 ids drawn with a fixed seed
    # from the ordinary-token range, so the window (2048) is crossed.
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, 100000, (2600,), generator=gen).to(dev)
    with torch.no_grad():
        out = model(ids[None], output_hidden_states=True)
        aux = torch.cat(
            [out.hidden_states[k][0] for k in config["aux_hidden_state_layer_ids"]], -1
        ).float()
    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight
    del out, model
    torch.cuda.empty_cache()

    cfg = SpeculatorModelConfig.from_pretrained(str(path))
    cfg.transformer_layer_config._attn_implementation = "eager"
    ref = DSparkDraftModel(cfg)
    weights = load_file(str(path / "model.safetensors"))
    missing, _unexpected = ref.load_state_dict(weights, strict=False)
    assert not [k for k in missing if not k.startswith("verifier_")], missing
    ref = ref.float().to(dev).eval()
    ours = DSpark(path, embed, lm_head, dev, own_vocab=True, dtype=torch.float32)
    anchors = list(range(5, 2590, 41))  # 64 anchors, before and after the window
    with torch.no_grad():
        want = _reference_logprobs(ref, aux, ids, anchors, monkeypatch)
        got = _ours_logprobs(ours, aux, ids, anchors)
    diff = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    print(json.dumps({"max_abs_logprob_diff": diff, "argmax_agreement": agree,
                      "own_vocab": ours.own_vocab, "windows": ours.windows}))  # fmt: skip
    assert diff < 1e-3


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

    from apertus_tree.speculators_heads import Eagle3Speculators

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

    from apertus_tree.speculators_heads import DFlash

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
