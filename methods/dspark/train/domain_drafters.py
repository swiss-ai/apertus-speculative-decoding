"""Per-domain DSpark drafters (canvas A3): data, checkpoint choice and the
transfer matrix. Runs inside the container (methods/dspark/train/domain-job.sh);
reuses the adaptation tool (adapt_dspark.py) for rebind, build, check, compare
and export.

    lock-data       hash the domain traces once; later jobs verify the lock
    convert-head    one head's training file: 8,000 train rows of its domain
                    (mixed: an equal share of every domain) + 256 dev rows last
    convert-evals   one evaluation file per domain dev set and for OPB held-out
    pick-checkpoint the first checkpoint whose global_step equals max_steps
    matrix          head x evaluation-set EAL table from every lr-0 evaluation

Every file keeps its held-out rows last; ``ratio_for`` gives the
train_data_ratio whose ``int(n * ratio)`` is exactly the number of rows in front.
Restricted text never leaves the cluster: outputs hold ids, counts and hashes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_dspark import (  # noqa: E402
    MAX_LEN,
    corpus_row,
    jsonl,
    now,
    read,
    sha256,
    trace_row,
    write_new,
    write_rows,
)

OPB = "opb"


def ratio_for(front: int, total: int) -> float:
    """train_data_ratio r with int(total * r) == front (speculators' split)."""
    if not 0 < front < total:
        raise ValueError(f"need 0 < front ({front}) < total ({total})")
    ratio = (front + 0.5) / total
    if int(total * ratio) != front:
        raise ValueError("no exact ratio")
    return ratio


def domain_files(plan: dict, domain: str) -> dict[str, Path]:
    root = Path(plan["data"]["traces_root"])
    return {"train": root / domain / "train.jsonl", "dev": root / domain / "dev.jsonl"}


def eval_domains(plan: dict) -> list[str]:
    return plan.get("eval_domains") or plan["domains"]


def cmd_lock_data(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    files = {}
    for domain in plan["domains"]:
        template = plan["data"].get("ready_marker")
        marker = (
            Path(template.format(traces_root=plan["data"]["traces_root"], domain=domain))
            if template
            else None
        )
        if marker is not None and not marker.exists():
            raise SystemExit(f"data not ready: {marker}")
        for split, path in domain_files(plan, domain).items():
            files[f"{domain}/{split}"] = {"path": str(path), "sha256": sha256(path)}
    opb = plan["data"]["opb"]
    lock = {"created_at": now(), "plan_sha256": sha256(args.plan), "files": files,
            "opb_source": {"corpus": opb["corpus"], "val_ids": opb["val_ids"],
                           "val_ids_sha256": sha256(Path(opb["val_ids"]))}}  # fmt: skip
    write_new(args.output, lock)


def verify_lock(plan: dict, lock_path: Path) -> None:
    lock = read(lock_path)
    for key, entry in lock["files"].items():
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise SystemExit(f"domain traces changed since the lock: {key}")


def load_trace_rows(path: Path, expected: int | None) -> list[dict]:
    rows = [trace_row(r) for r in jsonl(path)]
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path}: {len(rows)} rows, expected {expected}")
    return rows


def equal_shares(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def mixed_rows(per_domain: dict[str, list[dict]], total: int, seed: int) -> list[dict]:
    """Equal share of every domain (seeded sample), interleaved by a seeded shuffle."""
    domains = sorted(per_domain)
    picked = []
    for domain, share in zip(domains, equal_shares(total, len(domains)), strict=True):
        rows = per_domain[domain]
        if share > len(rows):
            raise ValueError(f"{domain}: {len(rows)} rows < share {share}")
        picked += random.Random(f"{seed}-{domain}").sample(rows, share)
    random.Random(seed).shuffle(picked)
    return picked


def cmd_convert_head(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    verify_lock(plan, args.lock)
    data = plan["data"]
    n_train, n_dev = data.get("train_rows"), data.get("dev_rows")
    domains = plan["domains"] if args.domain == "mixed" else [args.domain]
    train = {d: load_trace_rows(domain_files(plan, d)["train"], n_train) for d in domains}
    dev = {d: load_trace_rows(domain_files(plan, d)["dev"], n_dev) for d in domains}
    if args.domain == "mixed":
        size = n_train or min(len(r) for r in train.values())
        dev_size = n_dev or min(len(r) for r in dev.values())
        train_rows = mixed_rows(train, size, plan["training"]["seed"])
        dev_rows = mixed_rows(dev, dev_size, plan["training"]["seed"])
    else:
        train_rows, dev_rows = train[args.domain], dev[args.domain]
    ids = [r["id"] for r in train_rows + dev_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trace ids across train and dev")
    args.output.mkdir(parents=True, exist_ok=False)
    ratio = ratio_for(len(train_rows), len(train_rows) + len(dev_rows))
    report = {
        "created_at": now(),
        "head": args.domain,
        "train_file": write_rows(args.output / "train-dev.jsonl", train_rows + dev_rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_data_ratio": ratio,
        "train_rows_by_domain": {
            d: len({x["id"] for x in train[d]} & {r["id"] for r in train_rows}) for d in domains
        },  # fmt: skip
        "train_loss_tokens": sum(sum(r["loss_mask"]) for r in train_rows),
        "train_tokens": sum(len(r["input_ids"]) for r in train_rows),
    }
    write_new(args.output / "conversion.json", report)
    print(json.dumps({k: report[k] for k in ("head", "train_rows", "dev_rows", "train_data_ratio",
                                             "train_tokens", "train_loss_tokens")}))  # fmt: skip


def opb_heldout(plan: dict, limit: int) -> list[dict]:
    """The first ``limit`` rows of Yu's held-out split (val_ids), in corpus order:
    the rows the EAGLE/DSpark held-out evaluations used (heldout.jsonl)."""
    opb = plan["data"]["opb"]
    ids = {line.strip() for line in Path(opb["val_ids"]).read_text().splitlines() if line.strip()}
    rows = []
    for row in jsonl(Path(opb["corpus"])):
        if str(row["id"]) in ids:
            rows.append(corpus_row(row))
            if len(rows) == limit:
                return rows
    raise ValueError(f"only {len(rows)} held-out rows found")


def cmd_convert_evals(args: argparse.Namespace) -> None:
    """One file per evaluation set: a placeholder front part (stepped once at lr 0,
    never trained on) and the evaluated rows last."""
    plan = read(args.plan)
    verify_lock(plan, args.lock)
    data = plan["data"]
    front = data["eval_placeholder_rows"]
    args.output.mkdir(parents=True, exist_ok=False)
    sets = {}
    for domain in eval_domains(plan):
        files = domain_files(plan, domain)
        dev = load_trace_rows(files["dev"], data.get("dev_rows"))
        placeholder = [trace_row(r) for r in jsonl(files["train"], front)]
        sets[domain] = (placeholder, dev)
    opb_rows = opb_heldout(plan, data["opb"]["limit"] + front)
    sets[OPB] = (opb_rows[data["opb"]["limit"] :], opb_rows[: data["opb"]["limit"]])
    manifest = {"created_at": now(), "sets": {}}
    for name, (placeholder, rows) in sets.items():
        path = args.output / f"{name}.jsonl"
        info = write_rows(path, placeholder + rows)
        ratio = ratio_for(len(placeholder), len(placeholder) + len(rows))
        manifest["sets"][name] = {
            **info,
            "front_rows": len(placeholder),
            "evaluated_rows": len(rows),
            "train_data_ratio": ratio,
        }
    write_new(args.output / "eval-sets.json", manifest)
    print(json.dumps({k: v["evaluated_rows"] for k, v in manifest["sets"].items()}))


def cmd_pick_checkpoint(args: argparse.Namespace) -> None:
    found = []

    def order(path: Path) -> int:
        return int(path.parent.name) if path.parent.name.isdigit() else 1 << 30

    for ckpt in sorted(args.run.glob("checkpoints/*/training_state.json"), key=order):
        if read(ckpt)["global_step"] == args.max_steps:
            found.append(ckpt.parent)
    if not found:
        raise SystemExit(f"no checkpoint at global_step {args.max_steps} under {args.run}")
    print(found[0])


def cmd_matrix(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    root = args.root
    sets = eval_domains(plan) + [OPB]
    heads = ["e10"] + [h for h in plan["domains"] + ["mixed"] if (root / "heads" / h).exists()]
    table: dict[str, dict[str, Any]] = {}
    for head in heads:
        row = {}
        for name in sets:
            metrics = (
                root / "runs" / f"eval-{head}-{name}" / "checkpoints" / "0" / "val_metrics.json"
            )
            row[name] = read(metrics) if metrics.exists() else None
        table[head] = row
    metric = "eal_epoch"
    eal = {h: {s: (m[metric] if m else None) for s, m in r.items()} for h, r in table.items()}
    base = eal["e10"]

    def delta(value: float | None, ref: float | None) -> float | None:
        return value - ref if value is not None and ref is not None else None

    gain = {h: {s: delta(v, base[s]) for s, v in r.items()} for h, r in eal.items() if h != "e10"}
    report = {"created_at": now(), "metric": metric, "eal": eal, "gain_vs_e10": gain,
              "accept_len": {h: {s: (m["accept_len_epoch"] if m else None) for s, m in r.items()}
                             for h, r in table.items()},
              "complete": all(v is not None for r in eal.values() for v in r.values())}  # fmt: skip
    write_new(args.output, report)
    width = max(len(s) for s in sets) + 2
    print("head".ljust(10) + "".join(s.rjust(width) for s in sets))
    for h, r in eal.items():
        print(
            h.ljust(10)
            + "".join((f"{v:.3f}" if v is not None else "-").rjust(width) for v in r.values())
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("lock-data")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("convert-head")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("convert-evals")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("pick-checkpoint")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p = sub.add_parser("matrix")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        getattr(args, "domain", None) not in (None, "mixed")
        and args.domain not in read(args.plan)["domains"]
    ):
        raise SystemExit(f"unknown domain {args.domain}")
    actions = {
        "lock-data": cmd_lock_data,
        "convert-head": cmd_convert_head,
        "convert-evals": cmd_convert_evals,
        "pick-checkpoint": cmd_pick_checkpoint,
        "matrix": cmd_matrix,
    }
    actions[args.action](args)


if __name__ == "__main__":
    assert MAX_LEN == 8192
    main()
