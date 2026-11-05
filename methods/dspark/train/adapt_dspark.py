"""DSpark repeat of the T5 best adaptation setting (adapt-mult-n8k-r0).

Steps (methods/dspark/train/adapt-job.sh runs them inside one allocation; the
frozen settings are in methods/dspark/configs/8b/adapt-mult-n8k-r0.json):

    rebind          e10 with an accessible verifier path (weights symlinked)
    convert         traces -> speculators rows; OPB evaluation rows
    build-dataset   prepare-data's arrow layout for these rows, without its shuffle
    check-prepared  the prepared dataset holds exactly those rows, in order
    compare         two checkpoints' tensors (lr-0 evaluations, real updates)
    export          a servable head from the final checkpoint, with hashes
    report          gates, development metrics and the unchanged selection rule

Restricted text never leaves the cluster: outputs hold ids, counts and hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import socket
from collections import Counter
from pathlib import Path
from typing import Any

MAX_LEN = 8192
EOS_IDS = {2, 68, 72}
SERVED_FILES = ("config.json", "config.py", "model.safetensors", "training_state.json",
                "val_metrics.json")  # fmt: skip
RUN_FILES = ("run.yaml", "train_command.txt", "speculators.patch")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids: list[int], mask: list[int]) -> str:
    return hashlib.sha256(json.dumps([ids, mask]).encode()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def jsonl(path: Path, limit: int | None = None):
    # Physical lines only: trace text may hold U+0085/U+2028 (see 3790165).
    with path.open("rb") as handle:
        count = 0
        for line in handle:
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                return
            count += 1
            yield json.loads(line)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


# --------------------------------------------------------------------- rebind
def rebound_config(config: dict, contract: dict, plan: dict) -> dict:
    source = contract["source"]
    verifier = config["speculators_config"]["verifier"]
    if Path(verifier["name_or_path"]).name != source["revision"]:
        raise ValueError("initializer verifier snapshot revision differs from the target contract")
    if contract["target"]["architecture"] not in verifier["architectures"]:
        raise ValueError("initializer verifier architecture differs from the target contract")
    init = plan["initializer"]
    if config["transformer_layer_config"]["num_hidden_layers"] != init["num_hidden_layers"]:
        raise ValueError("unexpected initializer depth")
    if config["aux_hidden_state_layer_ids"] != init["aux_hidden_state_layer_ids"]:
        raise ValueError("initializer auxiliary layers changed")
    result = json.loads(json.dumps(config))
    result["speculators_config"]["verifier"]["name_or_path"] = source["authorized_checkpoint"]
    return result


def cmd_rebind(args: argparse.Namespace) -> None:
    plan, contract = read(args.plan), read(args.contract)
    source = Path(plan["initializer"]["path"])
    weights = source / "model.safetensors"
    actual = sha256(weights)
    if actual != plan["initializer"]["model_safetensors_sha256"]:
        raise ValueError(f"e10 weights changed: {actual}")
    target = Path(contract["source"]["authorized_checkpoint"])
    checks = {
        "config.json": contract["source"]["config_sha256"],
        "tokenizer.json": contract["source"]["tokenizer_json_sha256"],
        "chat_template.jinja": contract["source"]["chat_template_sha256"],
    }
    for name, expected in checks.items():
        if sha256(target / name) != expected:
            raise ValueError(f"target {name} differs from the contract")
    config = read(source / "config.json")
    rebound = rebound_config(config, contract, plan)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for original in sorted(source.iterdir()):
        if original.is_file() and original.name != "config.json":
            (out / original.name).symlink_to(original)
    write_new(out / "config.json", rebound)
    write_new(
        out / "rebinding.json",
        {
            "created_at": now(),
            "source": str(source),
            "model_safetensors_sha256": actual,
            "original_config_sha256": sha256(source / "config.json"),
            "rebound_config_sha256": sha256(out / "config.json"),
            "only_semantic_config_change": "speculators_config.verifier.name_or_path",
            "original_verifier": config["speculators_config"]["verifier"],
            "rebound_verifier": rebound["speculators_config"]["verifier"],
            "target_contract_sha256": sha256(args.contract),
            "target_files_verified": checks,
            "source_files": {
                p.name: sha256(p) for p in sorted(source.iterdir()) if p.is_file()
            },
        },
    )


# -------------------------------------------------------------------- convert
def trace_row(row: dict) -> dict:
    prompt, generated = list(row["prompt_ids"]), list(row["generated_ids"])
    if not prompt or not generated:
        raise ValueError(f"{row['id']}: empty prompt or completion")
    # 44 of the 8,000 multilingual traces stop at the 2,048-token generation cap;
    # the EAGLE setting trained on them too, so they stay (counted in the report).
    reason = row.get("finish_reason")
    if reason not in ("stop", "length") or (reason == "stop" and generated[-1] not in EOS_IDS):
        raise ValueError(f"{row['id']}: unexpected completion end ({reason})")
    ids = prompt + generated
    if len(ids) > MAX_LEN:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed {MAX_LEN}")
    return {"id": row["id"], "input_ids": ids, "loss_mask": [0] * len(prompt) + [1] * len(generated)}


def corpus_row(row: dict) -> dict:
    ids, mask = list(row["input_ids"]), list(row["loss_mask"])
    if len(ids) != len(mask) or not any(mask):
        raise ValueError(f"{row['id']}: malformed speculators row")
    return {"id": row["id"], "input_ids": ids, "loss_mask": mask}


def write_rows(path: Path, rows: list[dict]) -> dict:
    with path.open("x") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    lengths = [len(r["input_ids"]) for r in rows]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "tokens": sum(lengths),
        "loss_tokens": sum(sum(r["loss_mask"]) for r in rows),
        "mean_tokens": sum(lengths) / len(lengths),
        "max_tokens": max(lengths),
        "over_max_len": sum(n > MAX_LEN for n in lengths),
        "row_digests_sha256": hashlib.sha256(
            "".join(ids_digest(r["input_ids"], r["loss_mask"]) for r in rows).encode()
        ).hexdigest(),
    }


def cmd_convert(args: argparse.Namespace) -> None:
    plan = read(args.plan)["data"]
    for key in ("train", "dev", "opb"):
        actual = sha256(Path(plan[key]))
        if actual != plan[key + "_sha256"]:
            raise ValueError(f"{key} input changed: {actual}")
    raw_train, raw_dev = list(jsonl(Path(plan["train"]))), list(jsonl(Path(plan["dev"])))
    finish = {
        name: dict(Counter(r.get("finish_reason") for r in rows))
        for name, rows in (("train", raw_train), ("dev", raw_dev))
    }
    train = [trace_row(r) for r in raw_train]
    dev = [trace_row(r) for r in raw_dev]
    del raw_train, raw_dev
    opb = [corpus_row(r) for r in jsonl(Path(plan["opb"]), plan["opb_limit"])]
    if (len(train), len(dev), len(opb)) != (plan["train_rows"], plan["dev_rows"], plan["opb_limit"]):
        raise ValueError("unexpected row counts")
    ids = [r["id"] for r in train + dev]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trace ids across train and dev")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": now(),
        "inputs": {k: {"path": plan[k], "sha256": plan[k + "_sha256"]} for k in ("train", "dev", "opb")},
        "train_file": write_rows(args.output / "train-dev.jsonl", train + dev),
        "opb_file": write_rows(args.output / "opb-eval.jsonl", dev + opb),
        "train_part": {"rows": len(train), "mean_tokens": sum(len(r["input_ids"]) for r in train) / len(train)},
        "split": plan["split"],
        "finish_reasons": finish,
        "opb_rows_over_max_len": sum(len(r["input_ids"]) > MAX_LEN for r in opb),
    }
    write_new(args.output / "conversion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k.endswith("_file")}, indent=1))


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """The arrow dataset `speculators prepare-data` writes for speculator-format
    rows (clip to 8,192 tokens, drop rows without supervised tokens, torch format),
    without its two shuffles: the trainer holds out trailing rows by position, so
    the dev/OPB rows must stay last. No token_freq.pt: like e10, the run uses the
    full verifier vocabulary and never reads it."""
    from datasets import Dataset

    columns: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    dropped = 0
    for row in jsonl(args.input):
        ids, mask = row["input_ids"][:MAX_LEN], row["loss_mask"][:MAX_LEN]
        if len(row["input_ids"]) != len(row["loss_mask"]):
            raise ValueError(f"{row['id']}: input_ids/loss_mask length mismatch")
        if not any(mask):
            dropped += 1
            continue
        columns["input_ids"].append(ids)
        columns["loss_mask"].append(mask)
        columns["seq_len"].append(len(ids))
    if dropped:
        raise ValueError(f"{dropped} rows without supervised tokens; the split would move")
    data = Dataset.from_dict(columns)
    data.set_format(type="torch")
    data.save_to_disk(str(args.output))


def cmd_check_prepared(args: argparse.Namespace) -> None:
    from datasets import load_from_disk

    data = load_from_disk(str(args.data))
    expected = list(jsonl(args.expected))
    problems = []
    if len(data) != len(expected):
        problems.append(f"rows {len(data)} != {len(expected)}")
    truncated = 0
    for i, (row, want) in enumerate(zip(data, expected)):
        ids, mask = row["input_ids"].tolist(), row["loss_mask"].tolist()
        n = min(len(want["input_ids"]), MAX_LEN)
        if len(want["input_ids"]) > MAX_LEN:
            truncated += 1
        if ids != want["input_ids"][:n] or mask != want["loss_mask"][:n]:
            problems.append(f"row {i} ({want['id']}) differs")
            if len(problems) > 5:
                break
    split = int(len(data) * args.ratio)
    report = {
        "data": str(args.data),
        "expected": str(args.expected),
        "rows": len(data),
        "train_data_ratio": args.ratio,
        "train_rows": split,
        "held_out_rows": len(data) - split,
        "truncated_rows": truncated,
        "held_out_first_id": expected[split]["id"] if split < len(expected) else None,
        "problems": problems,
        "passed": not problems and split == args.expected_train_rows,
    }
    write_new(args.output, report)
    if not report["passed"]:
        raise SystemExit(f"prepared dataset check failed: {report}")


# -------------------------------------------------------------------- compare
def tensors(directory: Path) -> dict:
    from safetensors.torch import load_file

    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors in {directory}")
    out = {}
    for path in files:
        out.update(load_file(str(path)))
    return out


def compare(left: Path, right: Path) -> dict:
    import torch

    a, b = tensors(left), tensors(right)
    if set(a) != set(b):
        return {"same_names": False, "only_left": sorted(set(a) - set(b)),
                "only_right": sorted(set(b) - set(a))}  # fmt: skip
    changed, finite, shape_ok = [], True, True
    max_abs = 0.0
    for name in sorted(a):
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            shape_ok = False
            continue
        if not torch.equal(x, y):
            changed.append(name)
            diff = (x.float() - y.float()).abs().max().item()
            max_abs = max(max_abs, diff)
        finite &= bool(torch.isfinite(y.float()).all())
    return {
        "same_names": True,
        "same_shapes_and_dtypes": shape_ok,
        "tensors": len(a),
        "changed_tensors": len(changed),
        "changed": changed,
        "max_abs_difference": max_abs,
        "right_all_finite": finite,
        "bitwise_equal": shape_ok and not changed,
    }


def cmd_compare(args: argparse.Namespace) -> None:
    report = {"left": str(args.left), "right": str(args.right), **compare(args.left, args.right)}
    write_new(args.output, report)
    if args.require_equal and not report.get("bitwise_equal"):
        raise SystemExit(f"weights changed: {args.left} vs {args.right}")


# --------------------------------------------------------------------- export
def cmd_export(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in SERVED_FILES:
        shutil.copy2(checkpoint / name, out / name)
    for name in RUN_FILES:
        if (checkpoint.parent / name).is_file():
            shutil.copy2(checkpoint.parent / name, out / name)
    base = read(args.initializer / "config.json")
    config = read(out / "config.json")
    differing = sorted(k for k in set(base) | set(config) if base.get(k) != config.get(k))
    nested = {}
    if "speculators_config" in differing:
        b, c = base["speculators_config"], config["speculators_config"]
        nested = {k: [b.get(k), c.get(k)] for k in set(b) | set(c) if b.get(k) != c.get(k)}
    allowed = {"speculators_config"}
    config_ok = set(differing) <= allowed and set(nested) <= {"verifier"} and (
        not nested or nested["verifier"][0]["architectures"] == nested["verifier"][1]["architectures"]
    )
    reload = compare(checkpoint, out)
    versus_e10 = compare(args.initializer, out)
    report = {
        "created_at": now(),
        "checkpoint": str(checkpoint),
        "initializer": str(args.initializer),
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()},  # fmt: skip
        "config_keys_differing_from_e10": differing,
        "speculators_config_differences": nested,
        "config_matches_e10_apart_from_verifier_path": config_ok,
        "reload": {"pass": reload.get("bitwise_equal") is True,
                   "exported_tensors_bitwise_equal": reload.get("bitwise_equal") is True,
                   "tensors": reload.get("tensors")},  # fmt: skip
        "versus_e10": {k: v for k, v in versus_e10.items() if k != "changed"},
        "training_state": read(out / "training_state.json"),
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_new(out / "export-report.json", report)
    if not (config_ok and report["reload"]["pass"] and versus_e10.get("same_shapes_and_dtypes")):
        raise SystemExit("export check failed; see export-report.json")


# --------------------------------------------------------------------- report
LOSS = re.compile(r"train/loss=([^\s,]+)")


def train_losses(log: Path) -> list[float]:
    values = []
    for match in LOSS.finditer(log.read_text(errors="replace")):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            values.append(float("nan"))
    return values


def val(run: Path, epoch: str) -> dict:
    return read(run / "checkpoints" / epoch / "val_metrics.json")


def cmd_report(args: argparse.Namespace) -> None:
    plan, root = read(args.plan), args.root
    runs = root / "runs"
    initial_dev, initial_opb = val(runs / "eval-e10-dev", "0"), val(runs / "eval-e10-opb", "0")
    final_dev, final_opb = val(runs / "eval-final-dev", "0"), val(runs / "eval-final-opb", "0")
    trained_val = {e: val(runs / "train", e) for e in ("0", "1")}
    state = read(runs / "train" / "checkpoints" / "1" / "training_state.json")
    logs = sorted((runs / "train" / "logs").glob("train-*-node0.log"))
    losses = [v for log in logs for v in train_losses(log)]
    export = read(root / "head" / "export-report.json")
    evals_unchanged = {
        name: read(root / "checks" / f"{name}-unchanged.json")["bitwise_equal"]
        for name in ("eval-e10-dev", "eval-e10-opb", "eval-final-dev", "eval-final-opb")
    }
    metric = "eal_epoch"
    finite = bool(losses) and all(math.isfinite(v) for v in losses) and all(
        math.isfinite(m["loss_epoch"]) for m in (final_dev, final_opb)
    )
    gates = {
        "finite_losses": finite,
        "real_updates": export["versus_e10"]["changed_tensors"] > 0,
        "loss_decreased": final_dev["loss_epoch"] < initial_dev["loss_epoch"],
        "heldout_acceptance_improved": final_dev[metric] > initial_dev[metric],
        "steps": state["global_step"] == plan["training"]["max_steps"],
        "export": export["reload"]["pass"] and export["config_matches_e10_apart_from_verifier_path"],
        "evaluation_unchanged": all(evals_unchanged.values()),
    }
    gain = final_dev[metric] - initial_dev[metric]
    minimum = 0.20
    reasons = [] if all(gates.values()) else ["original_scientific_gate_failed"]
    if gain + 1e-12 < minimum:
        reasons.append("development_acceptance_gain_below_0.20")
    head = root / "head"
    summary = {
        "name": plan["name"],
        "created_at": now(),
        "metric": metric,
        "domain_acceptance_initial": initial_dev[metric],
        "domain_acceptance_final": final_dev[metric],
        "opb_acceptance_initial": initial_opb[metric],
        "opb_acceptance_final": final_opb[metric],
        "domain_gain": gain,
        "opb_gain": final_opb[metric] - initial_opb[metric],
        "opb_ratio": final_opb[metric] / initial_opb[metric],
        "accept_len": {
            "dev": [initial_dev["accept_len_epoch"], final_dev["accept_len_epoch"]],
            "opb": [initial_opb["accept_len_epoch"], final_opb["accept_len_epoch"]],
        },
        "validation": {"initial_dev": initial_dev, "final_dev": final_dev,
                       "initial_opb": initial_opb, "final_opb": final_opb,
                       "training_epoch_end": trained_val},  # fmt: skip
        "steps": state["global_step"],
        "logged_train_losses": len(losses),
        "train_loss_first_last": [losses[0], losses[-1]] if losses else None,
        "scientific_gates": gates,
        "evaluations_bitwise_unchanged": evals_unchanged,
        "head_dir": str(head),
        "head_model_sha256": export["files"]["model.safetensors"]["sha256"],
        "selected": not reasons,
        "exclusion_reasons": reasons,
        "selection_rule": plan["selection"],
    }
    write_new(args.output, summary)
    print(json.dumps({k: summary[k] for k in ("domain_acceptance_initial", "domain_acceptance_final",
          "opb_acceptance_initial", "opb_acceptance_final", "scientific_gates", "selected")}, indent=1))  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("rebind")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build-dataset")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("check-prepared")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--expected-train-rows", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-equal", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initializer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"rebind": cmd_rebind, "convert": cmd_convert, "check-prepared": cmd_check_prepared,
     "build-dataset": cmd_build_dataset,
     "compare": cmd_compare, "export": cmd_export, "report": cmd_report}[args.action](args)  # fmt: skip


if __name__ == "__main__":
    main()
