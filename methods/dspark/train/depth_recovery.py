"""Prepare and judge the bounded two-checkpoint shallow-drafter recovery.

Run preflight on an allocated node before the generated serving specs. Checkpoint
weights remain on the cluster. Decisions require two complete, counterbalanced
sweeps; simulator ceilings and training EAL are not serving measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

LEVELS = (1, 8, 32, 64, 128, 256)
K = range(1, 8)
DEFAULT_E10 = (
    "/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/"
    "dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final"
)
DEFAULT_EAGLE = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/heads/e31-opb-thinkoff-stage1-se"
)
DEFAULT_WORKLOAD = "/users/faruk_zahiragic/apertus-spec-decoding/workloads/8b/opb-heldout.jsonl"
DEFAULT_IMAGE = (
    "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh"
)


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def specs(candidate: str, control: str, e10: str, eagle: str, workload: str) -> tuple[dict, dict]:
    arms = [{"name": "plain", "method": "baseline"}]
    for name, checkpoint in [("dspark-e10", e10), ("depth-l1", candidate), ("depth-l5", control)]:
        arms.extend(
            {"name": f"{name}-k{k}", "method": "dspark", "drafter": checkpoint, "k": k} for k in K
        )
    # Existing EAGLE e10 is a reference, not another training experiment.
    arms.extend(
        {"name": f"eagle-e10-k{k}", "method": "eagle", "drafter": eagle, "k": k} for k in (3, 7)
    )
    common = {
        "stage": "8b",
        "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256},
        "probe": False,
        "loadtests": [
            {"workload": "opb", "workloads_file": workload, "concurrencies": list(LEVELS)}
        ],
    }
    preflight = {
        "name": "preflight",
        "method": "command",
        "gpus": 4,
        "command": [
            "python3",
            "{repo}/methods/dspark/train/depth_recovery.py",
            "preflight",
            "--candidate",
            candidate,
            "--control",
            control,
            "--e10",
            e10,
            "--eagle",
            eagle,
            "--workload",
            workload,
            "--output",
            "{arm_dir}/manifest.json",
        ],
    }
    return (
        {**common, "name": "depth-recovery-a", "arms": [preflight, *arms]},
        {**common, "name": "depth-recovery-b", "arms": [preflight, *reversed(arms)]},
    )


def checkpoint_evidence(root: Path, expected_depth: int) -> dict:
    state = json.loads((root / "training_state.json").read_text())
    if state["global_step"] != 6200:
        raise ValueError(f"{root}: expected exactly 6200 completed steps, got {state}")
    config = json.loads((root / "config.json").read_text())
    if config["transformer_layer_config"]["num_hidden_layers"] != expected_depth:
        raise ValueError(f"{root}: wrong trained depth")
    if config.get("aux_hidden_state_layer_ids") != [1, 8, 15, 22, 29]:
        raise ValueError(f"{root}: wrong target auxiliary layers")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"{root}: no exported serving weights")
    files = [
        root / "config.json",
        root / "training_state.json",
        root / "val_metrics.json",
        *weights,
    ]
    files.extend(
        p
        for p in (
            root.parent / name for name in ("run.yaml", "speculators.patch", "train_command.txt")
        )
        if p.is_file()
    )
    return {
        "checkpoint": str(root),
        "depth": expected_depth,
        "training_state": state,
        "validation": json.loads((root / "val_metrics.json").read_text()),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }


def read_sweep(root: Path) -> dict:
    status = json.loads((root / "sweep-status.json").read_text())
    expected = {"preflight", "plain", "eagle-e10-k3", "eagle-e10-k7"} | {
        f"{name}-k{k}" for name in ("dspark-e10", "depth-l1", "depth-l5") for k in K
    }
    if set(status) != expected or any(value != "ok" for value in status.values()):
        raise ValueError(f"{root}: incomplete or failed arms")
    cells = {}
    for arm in sorted(expected - {"preflight"}):
        arm_status = json.loads((root / arm / "status.json").read_text())
        if arm_status.get("status") != "ok":
            raise ValueError(f"{root}/{arm}: status is not ok")
        for c in LEVELS:
            path = root / arm / "opb" / "opb" / f"c{c}" / "summary.json"
            summary = json.loads(path.read_text())
            if summary["requests"]["success_rate"] < 0.99:
                raise ValueError(f"{path}: below 99% success")
            speed = summary["tokens"]["output_tokens_per_second"]
            if speed <= 0:
                raise ValueError(f"{path}: invalid throughput")
            cells[arm, c] = {
                "throughput": speed,
                "gpu_seconds_per_million_tokens": 1e6 / speed,
                "acceptance": summary.get("speculative_decoding", {}).get("mean_acceptance_length"),
                "source": str(path),
                "sha256": sha256(path),
                "gpu_ids": arm_status.get("gpus"),
            }
    return cells


def decision(a: Path, b: Path) -> dict:
    sweeps = [read_sweep(a), read_sweep(b)]
    rows = []
    for c in LEVELS:
        for name in ("depth-l1", "depth-l5"):
            repeat = []
            for cells in sweeps:
                best_reference = max(K, key=lambda k: cells[f"dspark-e10-k{k}", c]["throughput"])
                best_candidate = max(K, key=lambda k: cells[f"{name}-k{k}", c]["throughput"])
                ref = cells[f"dspark-e10-k{best_reference}", c]
                candidate = cells[f"{name}-k{best_candidate}", c]
                repeat.append(
                    {
                        "reference_k": best_reference,
                        "candidate_k": best_candidate,
                        "ratio_vs_best_dspark_e10": candidate["throughput"] / ref["throughput"],
                        "reference": ref,
                        "candidate": candidate,
                    }
                )
            rows.append({"depth": name, "concurrency": c, "repeats": repeat})
    eligible = [r for r in rows if r["depth"] == "depth-l1" and r["concurrency"] >= 128]
    passed = any(all(x["ratio_vs_best_dspark_e10"] >= 1.05 for x in r["repeats"]) for r in eligible)
    return {
        "gate_g3_repeatable_throughput_passed": passed,
        "gate": (
            "one-layer best k >=1.05x same-job best DSpark e10 k at C128 or C256 in both repeats"
        ),
        "correctness_confirmation_required_before_recommendation": True,
        "claim_limit": (
            "closed-loop serving throughput; no capacity, training-quality, or losslessness claim"
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--candidate", required=True)
    plan.add_argument("--control", required=True)
    plan.add_argument("--e10", default=DEFAULT_E10)
    plan.add_argument("--eagle", default=DEFAULT_EAGLE)
    plan.add_argument("--workload", default=DEFAULT_WORKLOAD)
    plan.add_argument("--output", type=Path, required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--candidate", type=Path, required=True)
    pre.add_argument("--control", type=Path, required=True)
    pre.add_argument("--e10", type=Path, default=Path(DEFAULT_E10))
    pre.add_argument("--eagle", type=Path, default=Path(DEFAULT_EAGLE))
    pre.add_argument("--workload", type=Path, default=Path(DEFAULT_WORKLOAD))
    pre.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    pre.add_argument("--output", type=Path, required=True)
    judge = sub.add_parser("decide")
    judge.add_argument("a", type=Path)
    judge.add_argument("b", type=Path)
    judge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        for suffix, spec in zip(
            ("a", "b"),
            specs(args.candidate, args.control, args.e10, args.eagle, args.workload),
            strict=True,
        ):
            write_new(args.output / f"depth-recovery-{suffix}.yaml", spec)
    elif args.action == "preflight":
        repo = Path(__file__).resolve().parents[3]
        references = {}
        for name, root in [("dspark_e10", args.e10), ("eagle_e10", args.eagle)]:
            references[name] = {
                str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in [root / "config.json", *sorted(root.glob("*.safetensors"))]
            }
        write_new(
            args.output,
            {
                "candidate": checkpoint_evidence(args.candidate, 1),
                "control": checkpoint_evidence(args.control, 5),
                "reference_checkpoints": references,
                "workload_sha256": sha256(args.workload),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": socket.gethostname(),
                "source_sha256": sha256(Path(__file__)),
                "image": {"path": str(args.image), "sha256": sha256(args.image)},
                "target_contract_sha256": sha256(repo / "targets/8b/contract.json"),
                "patch_sources": {
                    name: sha256(repo / "serving/patches" / name)
                    for name in (
                        "vllm-apertus-image-token.patch",
                        "vllm-apertus-eagle3-aux-layers.patch",
                        "vllm-apertus-eagle-mrv2-inner.patch",
                        "vllm-apertus-dspark-mrv2-inner.patch",
                        "vllm-apertus-dspark-anchor-layout.patch",
                        "vllm-apertus-dspark-num-layers.patch",
                    )
                },
            },
        )
    else:
        write_new(args.output, decision(args.a, args.b))


if __name__ == "__main__":
    main()
