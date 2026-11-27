"""Target and drafter next-token distributions along the target's own greedy trajectory.

For every sequence (prompt + greedy continuation captured by plain vLLM, see
``apertus_eagle.verify_offline capture --mode plain``) this script runs the target
once in Hugging Face with teacher forcing and evaluates DSpark e10 and EAGLE e10
on the same hidden states (the model code is ``apertus_tree.candidates``, which
reproduced served acceptance in the tree-drafter screen).

Definitions (anchor ``a`` = the newest token of a decode step, the token the
target emitted last; the target has hidden states for positions < a; depth
``d`` = 1..7 is the d-th drafted token, which predicts position ``a + d``):

- target distribution at position t: ``p_t = softmax(lm_head(h_{t-1}))``, the
  distribution the verifier compares a drafted token for position t against;
- drafter distribution at (a, d), teacher-forced: the drafter's distribution for
  position a+d given that all of ids[a+1 .. a+d-1] were drafted correctly, which
  is the only case in which depth d is reached at all.
  * DSpark drafts a block in one forward: slot 0 holds the anchor token, slots
    1..7 the mask token; slot ``d-1`` gives base logits for position a+d, and the
    vanilla Markov head adds ``W2 W1[ids[a+d-1]]`` (the previous token in the
    block). The base logits do not depend on the drafted tokens, only the Markov
    bias does, so teacher forcing changes nothing but that bias.
  * EAGLE drafts autoregressively: depth 1 reads the target's hidden state at a-1
    and token a; depth d > 1 feeds ids[a+d-1] and its own previous hidden state,
    attending to the context and to its ancestors (TTT layout).

The same replay runs the Red Hat speculators heads (``apertus_tree.qwen_heads``):
``--eagle-format speculators`` (EAGLE-3 with a reduced draft vocabulary) and
``--block-format dflash`` (DFlash: depth d reads slot d, no Markov head), with
``--depth`` up to the block's draft length and ``--names`` for the drafter keys.

Per (a, d) the shard stores, for both drafters: drafter entropy, top-1 prob,
TV(p, q), KL(p || q), whether the drafter's top-1 equals the trajectory token
(the greedy acceptance test) and the target's top-1, and q(trajectory token).
Per generated position it stores the target's entropy, top-1 prob, p(trajectory
token) and whether the HF argmax equals the vLLM token. The realised acceptance
of a greedy chain at any k is replayed from the match arrays by
``apertus_tree.shape_aggregate``.

The npz shards hold no token ids; they stay on the cluster with the captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from apertus_tree.candidates import DSpark, Eagle, file_sha256, rms, sha256_ids

DEPTH = 7
DRAFTER_FIELDS = ("ent", "top1p", "tv", "kl", "match", "agree", "ptruth")


def dist_metrics(lp_t: torch.Tensor, lp_q: torch.Tensor, truth: torch.Tensor, t_top: torch.Tensor):
    """lp_t, lp_q [n, V] log-probs; truth, t_top [n] token ids. Returns dict of [n]."""
    p, q = lp_t.exp(), lp_q.exp()
    ent = -(q * lp_q.clamp(min=-1e30)).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    kl = (p * (lp_t - lp_q).clamp(min=-1e30, max=1e30)).nan_to_num(0.0).sum(-1)
    top1p, top1 = q.max(-1)
    return {
        "ent": ent,
        "top1p": top1p,
        "tv": tv,
        "kl": kl,
        "match": (top1 == truth).float(),
        "agree": (top1 == t_top).float(),
        "ptruth": q.gather(-1, truth[:, None])[:, 0],
    }


class Collector:
    def __init__(self, args, eagle: Eagle, dspark: DSpark, lm_head: torch.Tensor, device):
        self.args, self.eagle, self.dspark, self.lm_head, self.device = (
            args, eagle, dspark, lm_head, device,
        )  # fmt: skip
        # Drafter keys (block drafter first) and depth; defaults are the Apertus run's.
        self.block_name, self.eagle_name = getattr(args, "names", "dspark,eagle").split(",")
        self.depth = int(getattr(args, "depth", DEPTH))
        self.slot = int(getattr(dspark, "slot_offset", 0))

    def carry(self, h: torch.Tensor) -> torch.Tensor:
        """EAGLE hidden state for the next step (Apertus heads: normed; speculators: raw)."""
        if hasattr(self.eagle, "carry"):
            return self.eagle.carry(h)
        return rms(h, self.eagle.norm, self.eagle.eps)

    def sequence(self, ids: list[int], p: int, last: torch.Tensor, aux_e, aux_d):
        dev = self.device
        full = torch.tensor(ids, device=dev)
        n = len(ids)
        # Target log-probs for every generated position t in [p, n-1]: row t-p.
        t_lp = torch.log_softmax(F.linear(last[p - 1 : n - 1], self.lm_head).float(), -1)
        t_prob = t_lp.exp()
        target = {
            "ent": -(t_prob * t_lp.clamp(min=-1e30)).sum(-1),
            "top1p": t_prob.max(-1).values,
            "ptruth": t_prob.gather(-1, full[p:, None])[:, 0],
        }
        t_top = t_lp.argmax(-1)
        target["hf_agree"] = (t_top == full[p:]).float()
        del t_prob

        ectx = self.eagle.context(full, aux_e)
        dctx = self.dspark.context(aux_d)
        anchors = list(range(p, n - 1))
        depth_max = self.depth
        out = {
            name: {
                f: np.full((len(anchors), depth_max), np.nan, np.float32) for f in DRAFTER_FIELDS
            }
            for name in (self.block_name, self.eagle_name)
        }
        for start in range(0, len(anchors), self.args.anchor_chunk):
            chunk = anchors[start : start + self.args.anchor_chunk]
            a_t = torch.tensor(chunk, device=dev)
            depth = min(depth_max, n - 1 - chunk[0])
            base = self.dspark.block_logits(dctx, full, a_t)[:, self.slot : self.slot + depth_max]
            # EAGLE teacher-forced chain state
            e_hidden = ectx["next"][a_t - 1]
            anc_k = ectx["k"].new_zeros((len(chunk), 0, *ectx["k"].shape[1:]))
            anc_v = ectx["v"].new_zeros((len(chunk), 0, *ectx["v"].shape[1:]))
            e_lp = self.eagle.logprobs(ectx["h"][a_t - 1])
            for d in range(1, depth + 1):
                pos = a_t + d  # predicted position
                valid = pos <= n - 1
                rows = valid.nonzero()[:, 0]
                posv = pos[rows]
                truth = full[posv]
                lp_t = t_lp[posv - p]
                top_t = t_top[posv - p]
                d_lp = self.dspark.logprobs(base[rows, d - 1], full[posv - 1])
                for name, lp_q in ((self.block_name, d_lp), (self.eagle_name, e_lp[rows])):
                    m = dist_metrics(lp_t, lp_q, truth, top_t)
                    idx = (start + rows).cpu().numpy()
                    for f in DRAFTER_FIELDS:
                        out[name][f][idx, d - 1] = m[f].float().cpu().numpy()
                del d_lp
                if d == depth:
                    break
                # EAGLE step d: feed ids[a+d] (all anchors; positions past the end are
                # clamped and their results never read).
                tok = full[pos.clamp(max=n - 1)]
                k, v, h = self.eagle.nodes(ectx, a_t, d, tok, e_hidden, anc_k, anc_v)
                e_lp = self.eagle.logprobs(h)
                e_hidden = self.carry(h)
                anc_k = torch.cat([anc_k, k[:, None]], 1)
                anc_v = torch.cat([anc_v, v[:, None]], 1)
        target = {k: v.float().cpu().numpy().astype(np.float32) for k, v in target.items()}
        return target, out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-tree-shape")
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--eagle-head", type=Path, required=True)
    parser.add_argument("--dspark", type=Path, required=True)
    parser.add_argument("--dspark-hf-layers", default="0,7,14,21,28")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--anchor-chunk", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eagle-format", choices=("apertus", "speculators"), default="apertus")
    parser.add_argument("--block-format", choices=("dspark", "dflash"), default="dspark")
    parser.add_argument("--names", default="dspark,eagle", help="block drafter key, EAGLE key")
    parser.add_argument("--depth", type=int, default=DEPTH, help="drafted depths per anchor")
    args = parser.parse_args(argv)

    from apertus_eagle.contract import (
        aux_layer_ids,
        load_contract,
        num_decoder_layers,
        target_identity,
    )
    from apertus_eagle.features import FeatureCapture, load_teacher
    from apertus_eagle.generate_targets import render_row
    from transformers import AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    started = time.time()
    contract = load_contract(args.contract)
    target_path = contract["source"]["authorized_checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.eagle_format == "speculators":
        from apertus_tree.qwen_heads import Eagle3Speculators

        eagle = Eagle3Speculators(args.eagle_head, embed, device, target_vocab=lm_head.shape[0])
    else:
        eagle = Eagle(args.eagle_head, embed, device)
        if eagle.lm_head.shape[0] != lm_head.shape[0]:
            raise SystemExit("EAGLE draft vocabulary differs from the target's")
    if args.block_format == "dflash":
        from apertus_tree.qwen_heads import DFlash

        dspark = DFlash(args.dspark, embed, lm_head, device)
        if args.depth > dspark.max_depth:
            raise SystemExit(f"DFlash drafts at most {dspark.max_depth} tokens")
    else:
        dspark = DSpark(args.dspark, embed, lm_head, device)
        if args.depth > dspark.block:
            raise SystemExit(f"DSpark drafts at most {dspark.block} tokens")
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    collector = Collector(args, eagle, dspark, lm_head, device)

    cases: dict[str, dict[str, Any]] = {}
    for path in args.capture:
        for c in json.loads(path.read_text())["cases"]:
            if "@cap" not in c["id"]:
                cases[c["id"]] = c
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if str(r["id"]) in cases]
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]

    arrays: dict[str, list[np.ndarray]] = {}
    seqs = []

    def push(key, value):
        arrays.setdefault(key, []).append(value)

    with torch.inference_mode():
        for row in rows:
            t0 = time.time()
            case = cases[str(row["id"])]
            prompt = list(row.get("prompt_ids") or render_row(contract, tokenizer, row))
            if sha256_ids(prompt) != case["prompt_ids_sha256"]:
                raise SystemExit(f"prompt ids differ from the capture for {row['id']}")
            ids = prompt + list(case["token_ids"])
            if len(case["token_ids"]) < 2:
                continue
            _hidden, last, _ = capture.run(torch.tensor([ids], device=device))
            del _hidden
            aux_e = torch.cat([capture.captured[i][0] for i in eagle_layers], -1)
            aux_d = torch.cat([capture.captured[i][0] for i in dspark_layers], -1)
            target, drafted = collector.sequence(ids, len(prompt), last, aux_e, aux_d)
            for f, v in target.items():
                push(f"target_{f}", v)
            for name, fields in drafted.items():
                for f, v in fields.items():
                    push(f"{name}_{f}", v)
            seq = {
                "seq_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()[:16],
                "workload": row["workload"],
                "prompt_tokens": len(prompt),
                "generated_tokens": len(case["token_ids"]),
                "finish_reason": case.get("finish_reason"),
                "hf_argmax_agreement": float(target["hf_agree"].mean()),
                "seconds": round(time.time() - t0, 2),
            }
            seqs.append(seq)
            print(json.dumps(seq), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        seq_workload=np.array([s["workload"] for s in seqs]),
        seq_sha=np.array([s["seq_sha256"] for s in seqs]),
        seq_gen=np.array([s["generated_tokens"] for s in seqs]),
        seq_prompt=np.array([s["prompt_tokens"] for s in seqs]),
        **{k: np.concatenate(v) for k, v in arrays.items()},
    )
    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "args": {k: str(v) for k, v in vars(args).items()},
        "target": target_identity(contract),
        "eagle_head": str(args.eagle_head),
        "eagle_head_sha256": file_sha256(args.eagle_head / "model.safetensors"),
        "dspark": str(args.dspark),
        "dspark_sha256": file_sha256(args.dspark / "model.safetensors"),
        "drafter_names": [collector.block_name, collector.eagle_name],
        "eagle_format": args.eagle_format,
        "block_format": args.block_format,
        "depth": collector.depth,
        "eagle_hf_layers": eagle_layers,
        "dspark_hf_layers": dspark_layers,
        "captures": {str(p): file_sha256(p) for p in args.capture},
        "prompts_sha256": file_sha256(args.prompts),
        "output_sha256": file_sha256(args.output),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": round(time.time() - started, 1),
        "sequences": seqs,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sequences"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
