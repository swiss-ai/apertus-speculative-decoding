"""CPU dry run of the candidate builder on tiny random models.

Needs torch and transformers (the serving image has them; locally e.g. a CPU
venv), so it is skipped in the default environment. Besides running main()
end to end it checks the batched EAGLE tree and DSpark block forwards against
naive per-position reimplementations of the training-time layouts.
"""

import gzip
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/tree"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/eagle"))

from apertus_tree import candidates as cand  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

H, HEADS, KV, HD, V, LAYERS = 32, 4, 2, 8, 64, 8
ROPE = {
    "rope_type": "llama3",
    "rope_theta": 10000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 16,
}


def _w(*shape):
    return (torch.randn(*shape) * 0.2).to(torch.bfloat16)


def make_target(root: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=V,
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HD,
        max_position_embeddings=256,
    )
    path = root / "target"
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    vocab = {f"t{i}": i for i in range(V)}
    tok = Tokenizer(WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(path)
    return path


def make_eagle(root: Path) -> Path:
    path = root / "eagle"
    path.mkdir()
    cfg = {
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HD,
        "rms_norm_eps": 1e-5,
        "vocab_size": V,
        "draft_vocab_size": V,
        "num_aux_hidden_states": 3,
        "fc_norm": True,
        "norm_output": True,
        "rope_parameters": ROPE,
    }
    (path / "config.json").write_text(json.dumps(cfg))
    p = "layers.0."
    w = {
        "fc.weight": _w(H, 3 * H),
        **{f"fc_norm.{i}.weight": 1 + _w(H) for i in range(3)},
        p + "self_attn.q_proj.weight": _w(HEADS * HD, 2 * H),
        p + "self_attn.k_proj.weight": _w(KV * HD, 2 * H),
        p + "self_attn.v_proj.weight": _w(KV * HD, 2 * H),
        p + "self_attn.o_proj.weight": _w(H, HEADS * HD),
        p + "mlp.gate_proj.weight": _w(48, H),
        p + "mlp.up_proj.weight": _w(48, H),
        p + "mlp.down_proj.weight": _w(H, 48),
        p + "input_layernorm.weight": 1 + _w(H),
        p + "hidden_norm.weight": 1 + _w(H),
        p + "post_attention_layernorm.weight": 1 + _w(H),
        "norm.weight": 1 + _w(H),
        "lm_head.weight": _w(V, H),
    }
    save_file(w, str(path / "model.safetensors"))
    return path


def make_dspark(root: Path, layers: int = 2) -> Path:
    path = root / "dspark"
    path.mkdir()
    cfg = {
        "block_size": 8,
        "mask_token_id": V - 1,
        "markov_head_type": "vanilla",
        "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "transformer_layer_config": {
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV,
            "head_dim": HD,
            "rms_norm_eps": 1e-5,
            "num_hidden_layers": layers,
            "sliding_window": 6,
            "use_sliding_window": True,
            "rope_parameters": ROPE,
        },
    }
    (path / "config.json").write_text(json.dumps(cfg))
    w = {"fc.weight": _w(H, 5 * H), "hidden_norm.weight": 1 + _w(H), "norm.weight": 1 + _w(H)}
    w["markov_head.markov_w1.weight"] = _w(V, 4)
    w["markov_head.markov_w2.weight"] = _w(V, 4)
    for i in range(layers):
        p = f"layers.{i}."
        w |= {
            p + "self_attn.q_proj.weight": _w(HEADS * HD, H),
            p + "self_attn.k_proj.weight": _w(KV * HD, H),
            p + "self_attn.v_proj.weight": _w(KV * HD, H),
            p + "self_attn.o_proj.weight": _w(H, HEADS * HD),
            p + "self_attn.q_norm.weight": 1 + _w(HD),
            p + "self_attn.k_norm.weight": 1 + _w(HD),
            p + "input_layernorm.weight": 1 + _w(H),
            p + "post_attention_layernorm.weight": 1 + _w(H),
            p + "mlp.gate_proj.weight": _w(48, H),
            p + "mlp.up_proj.weight": _w(48, H),
            p + "mlp.down_proj.weight": _w(H, 48),
        }
    save_file(w, str(path / "model.safetensors"))
    return path


# --------------------------------------------------------------------------- naive references


def rope_one(x, pos, inv):
    """x [heads, d] at integer position pos."""
    f = pos * inv
    cos, sin = torch.cat([f, f]).cos(), torch.cat([f, f]).sin()
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x.float() * cos + rot.float() * sin


def gqa(q, keys, vals):
    """q [heads, d]; keys/vals lists of [kv, d]."""
    k = torch.stack(keys).float()  # [J, kv, d]
    v = torch.stack(vals).float()
    rep = q.shape[0] // k.shape[1]
    out = []
    for h in range(q.shape[0]):
        g = h // rep
        s = (k[:, g] @ q[h].float()) / math.sqrt(q.shape[-1])
        out.append(torch.softmax(s, 0) @ v[:, g])
    return torch.cat(out)


def naive_eagle_chain_logprobs(e, ids, aux, a, chain):
    """EAGLE log-probs of ``chain`` (tokens after anchor a), TTT layout, by loops."""
    inv = e.rope.inv.float()
    rms = cand.rms

    def step(tok, hidden, pos, keys, vals):
        emb = e.embed[tok]
        x = torch.cat([rms(emb, e.in_ln, e.eps), rms(hidden, e.hid_ln, e.eps)])
        q = (x @ e.q.T).view(e.heads, e.dim)
        k = (x @ e.k.T).view(e.groups, e.dim)
        v = (x @ e.v.T).view(e.groups, e.dim)
        q, k = rope_one(q, pos, inv), rope_one(k, pos, inv).to(torch.bfloat16)
        out = gqa(q.to(torch.bfloat16), keys + [k], vals + [v]).to(torch.bfloat16)
        h = hidden + out @ e.o.T
        h = h + cand.mlp(rms(h, e.post_ln, e.eps), e.gate, e.up, e.down)
        return h, k, v

    proj = e.project(aux)
    k0, v0, h0 = [], [], None
    for j in range(a):  # pairs (aux_j, token_{j+1}), causal
        h0, k, v = step(ids[j + 1], proj[j], j, k0, v0)
        k0.append(k)
        v0.append(v)
    lps = []
    lp = e.logprobs(h0[None])[0]
    hidden = rms(h0, e.norm, e.eps)
    anc_k, anc_v = [], []
    for d, tok in enumerate(chain, 1):
        lps.append(float(lp[tok]))
        h, k, v = step(tok, hidden, a - 1 + d, k0 + anc_k, v0 + anc_v)
        anc_k.append(k)
        anc_v.append(v)
        lp = e.logprobs(h[None])[0]
        hidden = rms(h, e.norm, e.eps)
    return lps


def naive_dspark_base(ds, ids, aux, a):
    """Base logits [block, V] of the block anchored at a, by loops."""
    inv = ds.rope.inv.float()
    rms = cand.rms
    target = rms(aux @ ds.fc.T, ds.hidden_norm, ds.eps)
    toks = [ids[a]] + [ds.mask_id] * (ds.block - 1)
    h = ds.embed[torch.tensor(toks)]
    lo = 0 if ds.window is None else max(0, a - ds.window)
    for layer in ds.layers:
        x = rms(h, layer["in"], ds.eps)
        ctx_k, ctx_v = [], []
        for j in range(lo, a):
            k = rms((target[j] @ layer["k"].T).view(ds.groups, ds.dim), layer["kn"], ds.eps)
            ctx_k.append(rope_one(k, j, inv).to(torch.bfloat16))
            ctx_v.append((target[j] @ layer["v"].T).view(ds.groups, ds.dim))
        bk, bv, outs = [], [], []
        for s in range(ds.block):
            k = rms((x[s] @ layer["k"].T).view(ds.groups, ds.dim), layer["kn"], ds.eps)
            bk.append(rope_one(k, a + s, inv).to(torch.bfloat16))
            bv.append((x[s] @ layer["v"].T).view(ds.groups, ds.dim))
        for s in range(ds.block):
            q = rms((x[s] @ layer["q"].T).view(ds.heads, ds.dim), layer["qn"], ds.eps)
            q = rope_one(q, a + s, inv).to(torch.bfloat16)
            outs.append(gqa(q, ctx_k + bk[: s + 1], ctx_v + bv[: s + 1]).to(torch.bfloat16))
        h = h + torch.stack(outs) @ layer["o"].T
        h = h + cand.mlp(rms(h, layer["post"], ds.eps), layer["gate"], layer["up"], layer["down"])
    return rms(h, ds.norm, ds.eps) @ ds.lm_head.T


# --------------------------------------------------------------------------- tests


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    torch.manual_seed(1)
    target = make_target(root)
    eagle = make_eagle(root)
    dspark = make_dspark(root)
    contract = root / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "source": {"authorized_checkpoint": str(target)},
                "target": {
                    "text": {"num_hidden_layers": LAYERS},
                    "eagle": {
                        "default_aux_hidden_state_layer_ids": [2, 4, 5],
                        "hf_decoder_layer_aux_ids": [1, 3, 4],
                    },
                },
            }
        )
    )
    return {"root": root, "target": target, "eagle": eagle, "dspark": dspark, "contract": contract}


def _sequence(n=40, seed=3):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(1, V - 2, (12,), generator=g).tolist()
    return (base * 4)[:n]  # repeats, so prompt lookup finds matches


def test_batched_forwards_match_naive_references(models):
    torch.manual_seed(5)
    embed = _w(V, H)
    lm_head = _w(V, H)
    e = cand.Eagle(models["eagle"], embed, "cpu")
    ds = cand.DSpark(models["dspark"], embed, lm_head, "cpu")
    ids = _sequence()
    aux_e = _w(len(ids), 3 * H)
    aux_d = _w(len(ids), 5 * H)
    args = Namespace(
        topk=4, width=3, keep=12, ngram_branches=3, node_chunk=7, anchor_chunk=5, max_anchors=6
    )
    builder = cand.Builder(args, e, ds, "cpu")
    with torch.inference_mode():
        recs = list(builder.sequence(ids, 20, aux_e, aux_d))
        assert [r["a"] for r in recs] == list(range(20, 26))
        for rec in recs[::2]:
            a = rec["a"]
            # EAGLE: the log-prob of every edge on the EAGLE chain.
            chain = [i for i, f in enumerate(rec["f"]) if f & cand.ts.E_CHAIN]
            toks = [rec["t"][i] for i in chain]
            ref = naive_eagle_chain_logprobs(e, ids, aux_e, a, toks)
            got = [rec["le"][i] for i in chain]
            assert got == pytest.approx(ref, abs=0.05)
            # DSpark: every node's edge log-prob from the naive block.
            base = naive_dspark_base(ds, ids, aux_d, a)
            for n, (p, tok) in enumerate(zip(rec["p"], rec["t"], strict=True)):
                d = cand.ts.depths(rec["p"])[n]
                prev = ids[a] if p < 0 else rec["t"][p]
                lp = torch.log_softmax(base[d - 1].float() + (ds.w1[prev] @ ds.w2.T).float(), -1)
                assert rec["ld"][n] == pytest.approx(float(lp[tok]), abs=0.05)
            # Truth and structure.
            assert rec["truth"] == ids[a + 1 : a + 8]
            assert all(p < i for i, p in enumerate(rec["p"]))
            assert any(f & cand.ts.N_TREE for f in rec["f"])  # repeated text matches


def test_dspark_chain_is_greedy(models):
    torch.manual_seed(6)
    embed, lm_head = _w(V, H), _w(V, H)
    e = cand.Eagle(models["eagle"], embed, "cpu")
    ds = cand.DSpark(models["dspark"], embed, lm_head, "cpu")
    ids = _sequence(seed=4)
    args = Namespace(
        topk=4, width=3, keep=12, ngram_branches=3, node_chunk=64, anchor_chunk=8, max_anchors=3
    )
    with torch.inference_mode():
        recs = list(
            cand.Builder(args, e, ds, "cpu").sequence(
                ids, 25, _w(len(ids), 3 * H), _w(len(ids), 5 * H)
            )
        )
    for rec in recs:
        children = {}
        for n, p in enumerate(rec["p"]):
            children.setdefault(p, []).append(n)
        node = -1
        for _ in range(7):
            kids = [c for c in children.get(node, []) if rec["f"][c] & cand.ts.D_CHAIN]
            assert len(kids) == 1
            best = max(children[node], key=lambda c: rec["ld"][c])
            assert rec["ld"][kids[0]] == pytest.approx(rec["ld"][best], abs=1e-3)
            node = kids[0]


def test_main_end_to_end_and_replay(models, tmp_path, monkeypatch):
    ids = _sequence(n=48, seed=7)
    prompt, cont = ids[:20], ids[20:]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "r0", "prompt_ids": prompt}) + "\n")
    capture = tmp_path / "capture.json"
    digest = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
    capture.write_text(
        json.dumps({"cases": [{"id": "r0", "prompt_ids_sha256": digest, "token_ids": cont}]})
    )
    out = tmp_path / "opb-0.jsonl.gz"
    summary = tmp_path / "opb-0.summary.json"
    cand.main(
        [
            "--capture", str(capture), "--prompts", str(prompts), "--workload", "opb",
            "--eagle-head", str(models["eagle"]), "--dspark", str(models["dspark"]),
            "--dspark-hf-layers", "0,2,3,5,7", "--contract", str(models["contract"]),
            "--max-anchors", "10", "--anchor-chunk", "4",
            "--width", "3", "--topk", "3", "--keep", "10",
            "--output", str(out), "--summary", str(summary),
        ]
    )  # fmt: skip
    rep = json.loads(summary.read_text())
    assert rep["anchors"] == 10
    assert set(rep["chain_k7_accepted_per_step"]) == {"chain-dspark", "chain-eagle", "chain-ngram"}
    assert 0.0 <= rep["hf_argmax_agreement"] <= 1.0
    with gzip.open(out, "rt") as handle:
        assert len(handle.readlines()) == 10

    from apertus_bench import tree_sim

    table = Path(__file__).resolve().parents[1] / "methods/dspark/scheduler/step-time-8b-gh200.json"
    cost = tmp_path / "cost.json"
    t = json.loads(table.read_text())
    cost.write_text(
        json.dumps(
            {
                "verify_tokens": t["tokens"],
                "verify_ms": t["step_ms"],
                "draft_reqs": [1, 256],
                "draft_ms_by_len": {str(k): [1.5, 15.0] for k in range(1, 8)},
                "kv_ms_per_token": 3.6e-5,
            }
        )
    )
    result = tmp_path / "tree-sim.json"
    tree_sim.main(
        [str(out), "--cost-table", str(cost), "--budgets", "1,4,16", "--output", str(result)]
    )
    rows = json.loads(result.read_text())["rows"]
    assert {r["policy"] for r in rows} == set(tree_sim.POLICIES)
