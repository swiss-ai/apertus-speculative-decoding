"""Check a running extraction server's hidden states before training on them.

Requests the hidden states of a few regenerated rows exactly as the trainer does
(completions with the token ids, max_tokens 1), then checks the shape
[tokens, len(target layers) + 1, hidden], that the values are finite, and that
the last captured layer, through the target's final norm and lm_head, predicts
the next answer token. Misaligned or partial (e.g. tensor-parallel-sharded)
states fail the last test.
  python3 check_hidden_states.py --endpoint http://localhost:8001/v1 \
      --target DIR --rows answers.jsonl --layers 5 [--n 16] [--device cuda:2]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import openai
import torch
from hs_connectors.transfer import FileTransfer
from safetensors import safe_open
from speculators.data_generation.vllm_client import generate_hidden_states
from speculators.train.data import build_client_item

NORM_KEY = "model.language_model.norm.weight"
HEAD_KEY = "lm_head.weight"


def load_tensor(target: str, key: str, device: str) -> torch.Tensor:
    with open(os.path.join(target, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][key]
    with safe_open(os.path.join(target, shard), framework="pt", device=device) as f:
        return f.get_tensor(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--rows", required=True, help="regenerated jsonl (input_ids, loss_mask)")
    ap.add_argument("--layers", type=int, required=True, help="number of target layer ids")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-agreement", type=float, default=0.5)
    args = ap.parse_args()

    with open(os.path.join(args.target, "config.json")) as f:
        config = json.load(f)["text_config"]
    hidden, eps = config["hidden_size"], config["rms_norm_eps"]
    norm = load_tensor(args.target, NORM_KEY, args.device).float()
    head = load_tensor(args.target, HEAD_KEY, args.device)

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = client.models.list().data[0].id
    ok = True
    agree_total = n_total = 0
    with open(args.rows) as f:
        for k, line in enumerate(f):
            if k == args.n:
                break
            row = json.loads(line)
            ids = row["input_ids"][: args.max_len]
            mask = row["loss_mask"][: args.max_len]
            handle = generate_hidden_states(
                client, model, build_client_item({"input_ids": torch.tensor(ids)})
            )
            # The server writes asynchronously behind a .lock file; read it the
            # way the trainer does, which waits for the lock.
            transfer = FileTransfer(Path(handle).parent)
            data = transfer.get_generated(handle)
            if data is None:
                print(f"row {row['id']}: no hidden-state file at {handle}")
                ok = False
                continue
            hs = data["hidden_states"].to(args.device)
            tok = data["token_ids"]
            transfer.delete(handle)
            shape_ok = tuple(hs.shape) == (len(ids), args.layers + 1, hidden)
            same_ids = tok.tolist() == ids
            finite = bool(torch.isfinite(hs).all())
            x = hs[:, -1].float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm
            pred = (x.to(head.dtype) @ head.T).argmax(-1)
            target = torch.tensor(ids[1:], device=args.device)
            sup = torch.tensor(mask[1:], device=args.device).bool()
            agree = int((pred[:-1] == target)[sup].sum())
            n = int(sup.sum())
            agree_total += agree
            n_total += n
            norms = hs.float().norm(dim=-1).mean(0).tolist()
            print(f"row {row['id']}: shape {tuple(hs.shape)} dtype {hs.dtype} ids {same_ids} "
                  f"finite {finite} top1 {agree}/{n} layer norms "
                  + " ".join(f"{v:.0f}" for v in norms), flush=True)
            ok &= shape_ok and same_ids and finite
    rate = agree_total / max(n_total, 1)
    print(f"top-1 agreement on answer tokens {rate:.3f} ({agree_total}/{n_total})")
    ok &= rate >= args.min_agreement
    print("HS_CHECK_OK" if ok else "HS_CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
