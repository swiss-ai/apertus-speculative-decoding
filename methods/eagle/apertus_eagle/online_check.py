"""Check that online target features equal the disk cache they replace.

For the first ``--samples`` rows of a cached split, recompute the record with
``online_features`` from the ``generate_targets`` rows it was built from and
compare: token ids and loss mask must be identical, hidden states agree to the
bf16 run-to-run noise that ``features`` measured (repeat_hidden_max_abs).
Also reports teacher throughput. Runs on one allocated GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from apertus_eagle.contract import load_contract
from apertus_eagle.online_features import OnlineFeatures, Teacher


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-online-check")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="features/<split> directory")
    parser.add_argument("--rows", type=Path, required=True, help="generate_targets <split>.jsonl")
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-tokens", type=int, default=0, help="also check batched target forwards"
    )
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import load_file

    contract = load_contract(args.contract)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    teacher = Teacher(contract)
    online = OnlineFeatures([args.rows], teacher, max_seq_length=args.max_seq_length)
    position = {online.index.read(i)["id"]: i for i in range(len(online))}
    results, tokens, seconds = [], 0, 0.0
    for sample in manifest["samples"][: args.samples]:
        cached = load_file(str(args.cache / sample["file"]))
        torch.cuda.synchronize()
        started = time.time()
        record = online.load(position[sample["id"]])
        torch.cuda.synchronize()
        seconds += time.time() - started
        tokens += int(record["input_ids"].numel())
        results.append(
            {
                "id": sample["id"],
                "tokens": int(record["input_ids"].numel()),
                "ids_equal": bool(
                    torch.equal(record["input_ids"].cpu(), cached["input_ids"].long())
                ),
                "mask_equal": bool(
                    torch.equal(record["loss_mask"].cpu(), cached["loss_mask"].long())
                ),
                "hidden_max_abs": float(
                    (record["hidden_states"].cpu().float() - cached["hidden_states"].float())
                    .abs()
                    .max()
                ),
                "last_max_abs": float(
                    (
                        record["last_hidden_states"].cpu().float()
                        - cached["last_hidden_states"].float()
                    )
                    .abs()
                    .max()
                ),
            }
        )
    batched = None
    if args.batch_tokens:
        online.max_batch_tokens = args.batch_tokens
        chosen = manifest["samples"][: args.samples]
        torch.cuda.synchronize()
        started = time.time()
        records = online.load_many([position[s["id"]] for s in chosen])
        torch.cuda.synchronize()
        batch_seconds = time.time() - started
        model = teacher.model
        lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
        stats = {"hidden_states": [], "last_hidden_states": []}
        worst_abs, equal_ids, agree, total, first_token_rel = 0.0, True, 0, 0, []
        for sample, record in zip(chosen, records, strict=True):
            cached = load_file(str(args.cache / sample["file"]))
            equal_ids &= bool(torch.equal(record["input_ids"].cpu(), cached["input_ids"].long()))
            for key in stats:
                ours = record[key].cpu().float()
                theirs = cached[key].float()
                worst_abs = max(worst_abs, float((ours - theirs).abs().max()))
                # Position 0 carries Apertus' outlier activations; report it separately.
                stats[key].append(
                    float((ours[1:] - theirs[1:]).norm() / theirs[1:].norm().clamp_min(1e-9))
                )
                if key == "hidden_states":
                    first_token_rel.append(
                        float((ours[0] - theirs[0]).norm() / theirs[0].norm().clamp_min(1e-9))
                    )
            # What training uses: the target's next-token distribution from last_hidden_states.
            with torch.inference_mode():
                ours_top = lm_head(record["last_hidden_states"]).argmax(-1)
                theirs_top = lm_head(cached["last_hidden_states"].to(ours_top.device)).argmax(-1)
            agree += int((ours_top == theirs_top).sum())
            total += int(ours_top.numel())
        batched = {
            "batch_tokens": args.batch_tokens,
            "ids_equal": equal_ids,
            "max_abs": worst_abs,
            "relative_l2_excluding_first_token": {k: max(v) for k, v in stats.items()},
            "relative_l2_first_token_max": max(first_token_rel),
            "target_argmax_agreement": agree / max(total, 1),
            "teacher_tokens_per_second": round(tokens / max(batch_seconds, 1e-9), 1),
        }
    report = {
        "cache": str(args.cache),
        "rows": str(args.rows),
        "samples": len(results),
        "all_ids_equal": all(r["ids_equal"] for r in results),
        "all_masks_equal": all(r["mask_equal"] for r in results),
        "hidden_max_abs": max(r["hidden_max_abs"] for r in results),
        "last_max_abs": max(r["last_max_abs"] for r in results),
        "cache_repeat_hidden_max_abs": (manifest.get("hf_parity") or {}).get(
            "repeat_hidden_max_abs_max"
        ),
        "teacher_tokens_per_second": round(tokens / max(seconds, 1e-9), 1),
        "batched": batched,
        "per_sample": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}))
    ok = report["all_ids_equal"] and report["all_masks_equal"]
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
