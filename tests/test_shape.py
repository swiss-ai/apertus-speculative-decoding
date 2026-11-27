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
    assert agg.sampled_acceptance(tv, 3).tolist() == [4.0, 1.0, 1.875, 3.0]
    assert agg.sampled_acceptance(tv, 7)[0] == 8.0
    assert agg.sampled_acceptance(tv, 1).tolist() == [2.0, 1.0, 1.5, 2.0]


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
