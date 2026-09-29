"""Config-driven A2/A3 pipeline on one allocated node.

Steps (``--steps``, comma-separated, in this order):

- ``generate``: target-regenerated responses for ``dataset.generate_splits``
- ``extract``: bounded feature caches for ``dataset.extract_splits``
- ``parity``: HF-versus-vLLM feature/logit parity on ``dataset.parity_split``
- ``train``: TorchSpec rollout training (``train_rollout``)
- ``export``: vLLM-format export plus reload check (``export_head``)
- ``verify``: A4 offline token-level check (``verify_offline``): plain and EAGLE
  engines, two independent loads each, on ``verify.prompts``

Each step is skipped when its completed output already exists and matches the
contract; nothing is overwritten. Every run and head goes to a new directory
named after ``run_name`` and the Slurm job id.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from apertus_eagle.contract import REPO_ROOT, load_contract

STEPS = (
    "generate",
    "extract",
    "parity",
    "online-check",
    "train",
    "export",
    "verify",
    "sample",
    "repro",
    "diag",
)
# online-check only runs when asked for (it needs dataset.online_check).
DEFAULT_STEPS = tuple(step for step in STEPS if step != "online-check")


def log(message: str, **fields: Any) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps({"t": stamp, "msg": message, **fields}), flush=True)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    log("run", command=" ".join(command))
    started = time.time()
    result = subprocess.run(command, env=os.environ.copy())
    log("done", module=module, rc=result.returncode, seconds=round(time.time() - started, 1))
    if result.returncode != 0:
        raise SystemExit(f"{module} failed with rc={result.returncode}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS))
    parser.add_argument(
        "--run-id", default=os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    steps = [step for step in args.steps.split(",") if step]
    # Multi-node jobs run the pipeline on every node; only node 0 runs the
    # non-training steps and copies results, the others only join training.
    nnodes = int(os.environ.get("EAGLE_NNODES") or 1)
    node_rank = int(os.environ.get("SLURM_NODEID") or 0) if nnodes > 1 else 0
    if node_rank > 0:
        steps = [step for step in steps if step == "train"]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")

    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    contract_path = resolve(cfg["model"]["contract"])
    contract = load_contract(contract_path)
    dataset = cfg["dataset"]
    data_root = Path(dataset["data_root"])
    generated = data_root / "generated"
    features = data_root / "features"
    run_name = f"{cfg['run_name']}-{args.run_id}"
    run_dir = Path(cfg["output_root"]) / run_name
    share_embedding = bool(cfg["model"].get("share_target_embedding", False))
    head_name = run_name + ("-se" if share_embedding else "")
    head_dir = Path(cfg["head_root"]) / head_name
    results = REPO_ROOT / "results" / cfg["stage"] / "eagle"
    resolved = {
        "config": str(config_path),
        "contract": str(contract_path),
        "target": contract["target"]["identity"],
        "algorithm": cfg["model"]["algorithm"],
        "draft_config": str(resolve(cfg["model"]["draft_model_config"])),
        "steps": steps,
        "generated": str(generated),
        "features": str(features),
        "features_mode": dataset.get("features", "cache"),
        "train_cache": dataset.get("train_cache", dataset.get("train_data")),
        "eval_cache": dataset.get("eval_cache", dataset.get("eval_data")),
        "run_dir": str(run_dir),
        "head_dir": str(head_dir),
        "results": str(results),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    log("resolved", **resolved)
    if args.dry_run:
        return
    results.mkdir(parents=True, exist_ok=True)
    common = ["--contract", str(contract_path)]
    gates_failed = False
    max_len = str(dataset.get("max_seq_length", 4096))

    if "generate" in steps:
        summary = generated / "generation-summary.json"
        wanted = dataset["generate_splits"]
        done = json.loads(summary.read_text())["splits"] if summary.is_file() else {}
        if all(split in done for split in wanted):
            log("skip generate", summary=str(summary))
        else:
            if summary.is_file():
                raise SystemExit(f"{summary} exists without {wanted}; use a new data_root")
            inputs = [str(resolve(dataset["source_splits"]) / f"{split}.jsonl") for split in wanted]
            run_module(
                "apertus_eagle.generate_targets",
                *common,
                "--input",
                *inputs,
                "--output-dir",
                str(generated),
                "--max-seq-length",
                max_len,
            )
        (results / "data").mkdir(parents=True, exist_ok=True)
        (results / "data/generation-summary.json").write_text(summary.read_text())

    if "extract" in steps:
        for split in dataset["extract_splits"]:
            parity = ["--parity-samples", "8"] if split == dataset.get("parity_split") else []
            run_module(
                "apertus_eagle.features",
                *common,
                "--input",
                str(generated / f"{split}.jsonl"),
                "--output-dir",
                str(features / split),
                "--max-seq-length",
                max_len,
                *parity,
            )
            manifest = json.loads((features / split / "manifest.json").read_text())
            brief = {k: v for k, v in manifest.items() if k != "samples"}
            (results / "data").mkdir(parents=True, exist_ok=True)
            (results / f"data/features-{split}.json").write_text(json.dumps(brief, indent=2) + "\n")

    if "parity" in steps:
        report = results / "preflight/feature-parity.json"
        if report.is_file():
            log("skip parity", report=str(report))
        else:
            split = dataset["parity_split"]
            (results / "preflight").mkdir(parents=True, exist_ok=True)
            hf = features / split / "hf-parity.json"
            if hf.is_file():
                (results / "preflight/hf-parity.json").write_text(hf.read_text())
            run_module(
                "apertus_eagle.parity_vllm",
                *common,
                "--prefixes",
                str(features / split / "parity-prefixes.safetensors"),
                "--output",
                str(report),
            )
        from apertus_eagle.parity_vllm import summarize

        payload = json.loads(report.read_text())
        summary = summarize(payload["samples"])
        if payload.get("summary") != summary:
            payload["summary_as_run"] = payload.get("summary")
            payload["summary"] = summary
            report.write_text(json.dumps(payload, indent=2) + "\n")
        log("parity", **summary)
        if not summary["offset_confirmed"]:
            raise SystemExit(
                "A2 gate: HF/vLLM auxiliary-layer offset not confirmed; blocking training"
            )

    if "online-check" in steps:
        check = dataset["online_check"]
        run_module(
            "apertus_eagle.online_check",
            *common,
            "--cache",
            str(resolve(check["cache"])),
            "--rows",
            str(resolve(check["rows"])),
            "--max-seq-length",
            max_len,
            "--samples",
            str(check.get("samples", 32)),
            "--batch-tokens",
            str(dataset.get("teacher_batch_tokens") or 0),
            "--output",
            str(results / "preflight" / f"online-check-{run_name}.json"),
        )

    trained = (run_dir / "train-summary.json").is_file()
    if "train" in steps and trained:
        log("skip train: run finished in an earlier job", run_dir=str(run_dir))
        gates = json.loads((run_dir / "train-summary.json").read_text())["gates"]
        gates_failed = gates_failed or not all(gates.values())
    elif "train" in steps:
        # training.data_parallel > 1: one torchrun rank per GPU in CUDA_VISIBLE_DEVICES.
        ranks = int(cfg["training"].get("data_parallel") or 1)
        launcher = [sys.executable, "-m"]
        if nnodes > 1:
            # One torchrun per node, joined through a c10d rendezvous on node 0.
            launcher += [
                "torch.distributed.run",
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--nproc-per-node={ranks}",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={os.environ['EAGLE_MASTER_ADDR']}:29500",
                f"--rdzv-id={os.environ.get('SLURM_JOB_ID', args.run_id)}",
                "-m",
            ]
        elif ranks > 1:
            launcher += ["torch.distributed.run", "--standalone", f"--nproc-per-node={ranks}", "-m"]
        rc = subprocess.run(
            [
                *launcher,
                "apertus_eagle.train_rollout",
                "--config",
                str(config_path),
                "--contract",
                str(contract_path),
                "--output",
                str(run_dir),
            ],
            env=os.environ.copy(),
        ).returncode
        if node_rank > 0:
            # Global rank 0 (node 0) writes the outcome files; nothing else to do here.
            raise SystemExit(0 if rc == 0 else f"train_rollout failed rc={rc}")
        (results / "runs").mkdir(parents=True, exist_ok=True)
        for name in ("provenance.json", "train-summary.json", "evals.json", "history.json"):
            source = run_dir / name
            if source.is_file():
                target = results / "runs" / run_name / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
        log("train finished", rc=rc, run_dir=str(run_dir))
        # The outcome comes from files: torchrun turns any worker status into 1.
        if (run_dir / "training-incomplete.json").is_file():
            # Stopped before the job's time limit with checkpoints/resume saved.
            # Later steps need the finished head; a follow-up job with the same
            # --run-id resumes training and then runs them.
            log("training incomplete; resubmit with the same run id", run_id=args.run_id)
            return
        summary_path = run_dir / "train-summary.json"
        if rc not in (0, 3) or not summary_path.is_file():
            raise SystemExit(f"train_rollout failed rc={rc}")
        if not all(json.loads(summary_path.read_text())["gates"].values()):
            gates_failed = True
            log("train gates not all met; exporting for inspection only")

    if "export" in steps:
        run_module(
            "apertus_eagle.export_head",
            *common,
            "--run-dir",
            str(run_dir),
            "--output",
            str(head_dir),
            *(["--drop-embed-tokens"] if share_embedding else []),
        )
        target = results / "heads" / head_name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "export-report.json"):
            (target / name).write_text((head_dir / name).read_text())
        log("exported", head=str(head_dir))

    if "verify" in steps:
        verify = cfg.get("verify") or {}
        prompts = str(resolve(verify.get("prompts", "results/70b/eagle/data/heldout_test.jsonl")))
        out = results / "verify" / head_name
        out.mkdir(parents=True, exist_ok=True)
        depth = str(verify.get("num_speculative_tokens", 3))
        shared = [
            *common,
            "--prompts",
            prompts,
            "--limit",
            str(verify.get("limit", 128)),
            "--max-tokens",
            str(verify.get("max_tokens", 512)),
            "--num-speculative-tokens",
            depth,
        ]
        for mode in ("plain", "eagle"):
            for repeat in (1, 2):
                extra = ["--head", str(head_dir)] if mode == "eagle" else []
                if (out / f"{mode}-r{repeat}.json").is_file():
                    log("skip capture", file=str(out / f"{mode}-r{repeat}.json"))
                    continue
                run_module(
                    "apertus_eagle.verify_offline",
                    "capture",
                    "--mode",
                    mode,
                    *extra,
                    *shared,
                    "--output",
                    str(out / f"{mode}-r{repeat}.json"),
                )
        for left, right in (
            ("plain-r1", "plain-r2"),
            ("eagle-r1", "eagle-r2"),
            ("plain-r1", "eagle-r1"),
            ("plain-r2", "eagle-r2"),
        ):
            run_module(
                "apertus_eagle.verify_offline",
                "compare",
                str(out / f"{left}.json"),
                str(out / f"{right}.json"),
                "--output",
                str(out / f"compare-{left}-vs-{right}.json"),
            )
        treatment = json.loads((out / "compare-plain-r1-vs-eagle-r1.json").read_text())
        if treatment["divergences"]:
            run_module(
                "apertus_eagle.verify_offline",
                "margins",
                str(out / "compare-plain-r1-vs-eagle-r1.json"),
                *common,
                "--prompts",
                prompts,
                "--output",
                str(out / "margins-plain-r1-vs-eagle-r1.json"),
            )
        log("verify", exact_rate=treatment["exact_rate"], divergences=len(treatment["divergences"]))

    if "sample" in steps:
        verify = cfg.get("verify") or {}
        prompts = str(resolve(verify.get("prompts", "results/70b/eagle/data/heldout_test.jsonl")))
        out = results / "verify" / head_name / "sampled"
        out.mkdir(parents=True, exist_ok=True)
        depth = str(verify.get("num_speculative_tokens", 3))
        # Different seeds: the control measures sampling noise, not RNG replay.
        # Seeds far apart: consecutive seeds share per-sample RNG streams in vLLM.
        seeds = (
            ("plain-ref", "plain", 1),
            ("plain-control", "plain", 10_000_019),
            ("eagle", "eagle", 20_000_039),
        )
        for name, mode, seed in seeds:
            if (out / f"{name}.json").is_file():
                continue
            extra = ["--head", str(head_dir)] if mode == "eagle" else []
            run_module(
                "apertus_eagle.verify_offline",
                "sample",
                "--mode",
                mode,
                *extra,
                *common,
                "--prompts",
                prompts,
                "--seed",
                str(seed),
                "--num-speculative-tokens",
                depth,
                "--output",
                str(out / f"{name}.json"),
            )
        run_module(
            "apertus_eagle.verify_offline",
            "dist",
            str(out / "plain-ref.json"),
            str(out / "plain-control.json"),
            str(out / "eagle.json"),
            "--output",
            str(out / "dist.json"),
        )

    if "repro" in steps:
        # A5 profile crash (job 3512315): CUDA device-side assert with EAGLE,
        # greedy, ignore_eos=True, max_tokens=256. Reproduce offline, synchronously.
        out = results / "verify" / head_name / "repro-ignore-eos"
        out.mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        prompts = str(resolve("workloads/8b/eagle-8b-profile.jsonl"))
        for mode in ("plain", "eagle"):
            extra = ["--head", str(head_dir)] if mode == "eagle" else []
            try:
                run_module(
                    "apertus_eagle.verify_offline",
                    "capture",
                    "--mode",
                    mode,
                    *extra,
                    *common,
                    "--prompts",
                    prompts,
                    "--limit",
                    "32",
                    "--max-tokens",
                    "256",
                    "--cap-sweep",
                    "0",
                    "--ignore-eos",
                    "--output",
                    str(out / f"{mode}.json"),
                )
            except SystemExit as error:
                log("repro failed", mode=mode, error=str(error))

    if "diag" in steps:
        # A5 screening acceptance (6-9%, g ~ 1.17 at every depth) disagrees with
        # A4 (50.9%, g 2.5). Separate head from prompts, offline, one engine each.
        out = results / "verify" / "diag-acceptance"
        out.mkdir(parents=True, exist_ok=True)
        heads = {"se": head_dir, "orig": Path(cfg["head_root"]) / run_name}
        prompt_sets = {
            "validation": resolve("workloads/8b/eagle-8b-validation.jsonl"),
            "heldout": resolve("results/70b/eagle/data/heldout_test.jsonl"),
        }
        for head_key, prompt_key in (
            ("se", "validation"),
            ("se", "heldout"),
            ("orig", "validation"),
            ("orig", "heldout"),
        ):
            target = out / f"{head_key}-{prompt_key}.json"
            if target.is_file():
                continue
            try:
                run_module(
                    "apertus_eagle.verify_offline",
                    "capture",
                    "--mode",
                    "eagle",
                    "--head",
                    str(heads[head_key]),
                    *common,
                    "--prompts",
                    str(prompt_sets[prompt_key]),
                    "--limit",
                    "48",
                    "--max-tokens",
                    "256",
                    "--cap-sweep",
                    "0",
                    "--output",
                    str(target),
                )
            except SystemExit as error:
                log("diag capture failed", head=head_key, prompts=prompt_key, error=str(error))

    if gates_failed:
        # Nonzero so a Slurm afterok dependency does not start the next stage.
        raise SystemExit(3)


if __name__ == "__main__":
    main()
