"""Submit the paired serving confirmation of the DSpark adaptation repeat.

Login node only (stdlib). Runs only if the development report selected the
head under the unchanged EAGLE rule. Freezes a sparse source snapshot of the
current checkout revision (as freeze_adaptation does for EAGLE), reuses the
workloads and corpus hashes of EAGLE confirmation 3814370, and submits the
unchanged design through serving/node-sweep.sh on `normal`.

    python3 methods/dspark/train/submit_adapt_confirmation.py --root ADAPT_ROOT
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EAGLE_ROOT = Path(
    "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030-cache"
)
EAGLE_SELECTION = (
    EAGLE_ROOT / "confirmation-configs/repair-20261031c/adapt-mult-n8k-r0-selection.json"
)
POLICY = EAGLE_ROOT / "preregistration.json"
POLICY_SHA256 = "64c9844f6cd526e6c693f85a5f75cc0d5d7d3f605112b14c3ae1c70a969ef44c"
WORKLOADS = Path("/users/faruk_zahiragic/apertus-spec-decoding/workloads")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def snapshot(root: Path) -> Path:
    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked changes in the checkout; commit them first")
    dest = root / "runtime" / f"source-{revision[:12]}"
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", "--no-checkout", str(REPO), str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "--detach", revision], check=True)
        sparse = ["/methods/", "/src/", "/serving/", "/targets/", "/tools/", "/pyproject.toml",
                  "/README.md", "/results/README.md", "/docs/protocol.md"]  # fmt: skip
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *sparse],
                       check=True)  # fmt: skip
        (dest / "workloads").symlink_to(WORKLOADS, target_is_directory=True)
        files = git("ls-files", cwd=dest).splitlines()
        manifest = {
            "revision": revision,
            "task_branch": git("branch", "--show-current"),
            "files": {f: digest(dest / f) for f in files if (dest / f).is_file()},
        }
        write_new(dest / "source-manifest.json", manifest)
    manifest = json.loads((dest / "source-manifest.json").read_text())
    if manifest["revision"] != revision or git("rev-parse", "HEAD", cwd=dest) != revision:
        raise RuntimeError("snapshot revision differs from the checkout")
    for name, sha in manifest["files"].items():
        if digest(dest / name) != sha:
            raise RuntimeError(f"frozen source differs: {name}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--wall-time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / "development-report.json").read_text())
    if not report["selected"]:
        raise SystemExit(f"not selected under the unchanged rule: {report['exclusion_reasons']}")
    if digest(POLICY) != POLICY_SHA256:
        raise RuntimeError("EAGLE preregistration changed")
    plan = json.loads((REPO / "methods/dspark/configs/8b/adapt-mult-n8k-r0.json").read_text())
    eagle = json.loads(EAGLE_SELECTION.read_text())
    for path, sha in eagle["corpus_sha256"].items():
        if digest(Path(path)) != sha:
            raise RuntimeError(f"confirmation corpus changed: {path}")
    head = Path(report["head_dir"])
    if digest(head / "model.safetensors") != report["head_model_sha256"]:
        raise RuntimeError("exported head changed after the development report")
    name = plan["name"]
    source = snapshot(root)
    configs = root / "confirmation-configs" / args.attempt
    selection = {
        "name": name,
        "domain": "mult",
        "method": "dspark",
        "base_drafter": plan["initializer"]["path"],
        "base_model_sha256": plan["initializer"]["model_safetensors_sha256"],
        "head_dir": str(head),
        "adapted_model_sha256": report["head_model_sha256"],
        "run_dir": str(root / "runs/train"),
        "domain_gain": report["domain_gain"],
        "opb_gain": report["opb_gain"],
        "opb_ratio": report["opb_ratio"],
        "development_metric": report["metric"],
        "traces": 8000,
        "replay_fraction": 0.0,
        "learning_rate": plan["training"]["learning_rate"],
        "workloads": eagle["workloads"],
        "corpus_sha256": eagle["corpus_sha256"],
        "eagle_reference_selection": str(EAGLE_SELECTION),
    }
    selection_path = configs / f"{name}-selection.json"
    write_new(selection_path, selection)
    out = root / "confirmations" / args.attempt
    if out.exists():
        raise RuntimeError(f"confirmation output exists; use a fresh attempt: {out}")
    spec = {
        "name": name + "-confirmation",
        "stage": "8b",
        "loadtests": [{"workload": "opb", "concurrencies": [1]}],
        "arms": [
            {
                "name": "paired",
                "method": "command",
                "gpus": 4,
                "pythonpath": [str(source / "methods/eagle")],
                "command": ["python3", "-m", "apertus_bench.adaptation_confirmation",
                            "--selection", str(selection_path), "--policy", str(POLICY),
                            "--environment", str(out / "environment.toml"),
                            "--output", "{arm_dir}/paired"],  # fmt: skip
            }
        ],
    }
    spec_path = configs / f"{name}.yaml"
    write_new(spec_path, spec)  # JSON is a strict subset of YAML.
    short = Path("/iopsstor/scratch/cscs") / os.environ["USER"] / ("da-" + args.attempt[:8])
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TMPDIR", "PYTHONPATH", "VLLM_PATCH_DIR", "EXTRA_MOUNTS", "SWEEP_PATCHES"}
    } | {
        "SWEEP_PARTITION": "normal",
        "SWEEP_TIME": args.wall_time,
        "SWEEP_OUTPUT": str(out),
        "VLLM_PATCH_DIR": str(short),
        "SML_ENV_DIR": str(root / "serving-environments" / args.attempt),
        "EXTRA_MOUNTS": "",
        "SWEEP_PATCHES": "",
        "ENV_SOURCE": str(source / "methods/eagle/configs/train-env.toml"),
    }
    receipt = {
        "submitted_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": str(source),
        "source_revision": git("rev-parse", "HEAD", cwd=source),
        "source_manifest_sha256": digest(source / "source-manifest.json"),
        "selection": str(selection_path),
        "selection_sha256": digest(selection_path),
        "spec": str(spec_path),
        "policy": str(POLICY),
        "policy_sha256": POLICY_SHA256,
        "output": str(out),
        "environment": {k: env[k] for k in ("SWEEP_PARTITION", "SWEEP_TIME", "SWEEP_OUTPUT",
                        "VLLM_PATCH_DIR", "SML_ENV_DIR", "ENV_SOURCE")},  # fmt: skip
    }
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return
    result = subprocess.run(["bash", "serving/node-sweep.sh", str(spec_path)], cwd=source, env=env,
                            capture_output=True, text=True)  # fmt: skip
    receipt |= {"stdout": result.stdout, "stderr_tail": result.stderr[-4000:],
                "returncode": result.returncode}  # fmt: skip
    jobs = re.findall(r"job=(\d+) output=", result.stdout)
    receipt["job_id"] = jobs[-1] if jobs else None
    write_new(root / "receipts" / f"confirmation-{args.attempt}.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("job_id", "output", "returncode")}))
    if result.returncode or not jobs:
        raise SystemExit("submission failed; see the receipt")


if __name__ == "__main__":
    main()
