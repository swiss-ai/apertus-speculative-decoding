"""Launch and operate the bounded DSpark research pilot through SML.

Preparation only renders SML jobs. Full artifact hashing runs in the allocation,
before OpenTela can advertise the engine. Each deployment owns its source copy,
overlays, environment, submission journal and immutable measurement directories.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import fcntl
import getpass
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from apertus_bench.workloads import file_sha256

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO / "experiments/serving-pilot/dspark-e10.yaml"
RESERVED_OWNERS = {"SwissAI-Research", "CSCS-Inference", "RCP-AIaaS"}
TRAINING_FILES = (
    "config.json",
    "config.py",
    "run.yaml",
    "train_command.txt",
    "speculators.patch",
    "training_state.json",
    "val_metrics.json",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as sink:
        json.dump(value, sink, indent=2, sort_keys=True)
        sink.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration_signature(spec: dict[str, Any]) -> str:
    return digest({k: spec[k] for k in ("target", "drafter", "runtime", "engine", "validation")})


def verify_qualification(path: Path, spec: dict[str, Any]) -> dict:
    report = read_json(path)
    if not report.get("passed") or report.get("status") not in {"ready_for_pilot", "stable"}:
        raise ValueError("research launch needs a passing pilot qualification")
    if report.get("configuration_signature") != configuration_signature(spec):
        raise ValueError("qualification covers a different deployment configuration")
    for file, expected in report["evidence_sha256"].items():
        if file_sha256(Path(file)) != expected:
            raise ValueError(f"qualification evidence changed: {file}")
    return report


def normalize_entrypoint(output: Path) -> None:
    """Older SML uses `vllm serve` with --model; this image uses the API module.

    Patch the rendered self-contained master and inspection copy identically,
    and retain the exact diff. The shared SML installation stays untouched.
    """
    changes = []
    api = "python3 -m vllm.entrypoints.openai.api_server"
    for script in sorted((output / "rendered").glob("*.sh")):
        before = script.read_text()
        after = before.replace("vllm serve ", api + " ")
        if script.name == "master.sh":
            after = after.replace(
                "set -euo pipefail\n", 'set -euo pipefail\nmkdir -p "logs/${SLURM_JOB_ID}"\n', 1
            )
        if after != before:
            changes.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile=f"a/rendered/{script.name}",
                    tofile=f"b/rendered/{script.name}",
                )
            )
            script.write_text(after)
    (output / "rendered-entrypoint.patch").write_text("".join(changes))
    if api not in (output / "rendered/head.sh").read_text():
        raise ValueError("SML rendered an unrecognized vLLM entrypoint")


def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, check=True, **kwargs)


def job_environment() -> dict[str, str]:
    """The serving and direct load-test jobs need local artifacts, not API credentials."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(
            word in k.upper()
            for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CSCS_SERVING_API")
        )
    }


def git_info(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "root": str(root),
        "revision": git("rev-parse", "HEAD"),
        "diff": git("diff", "HEAD", "--", "."),
    }


def path_value(value: str, repo: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else ((repo or REPO) / path).resolve()


def load_profile(path: Path, *, platform: str | None = None) -> dict[str, Any]:
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return (
                value.replace("{home}", str(Path.home()))
                .replace("{user}", getpass.getuser())
                .replace("{repo}", str(REPO))
            )
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    spec = expand(yaml.safe_load(path.read_text()))
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported pilot profile schema")
    if platform:
        spec["deployment"]["platform"] = platform
    dep, engine, runtime = spec["deployment"], spec["engine"], spec["runtime"]
    if dep["platform"] not in {"dev", "research"}:
        raise ValueError("platform must be dev or research")
    for key in ("model_id", "fallback_model_id"):
        model = dep[key]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+/[\w.-]+", model, flags=re.ASCII):
            raise ValueError(f"{key} must be <owner>/<org>/<model>")
        if model.split("/")[0] in RESERVED_OWNERS or model.split("/")[0] != spec["owner"]:
            raise ValueError(f"{key} must use the pilot owner's unreserved namespace")
    if dep["model_id"] == dep["fallback_model_id"]:
        raise ValueError("plain and DSpark must have separate model IDs")
    if not 1 <= spec["drafter"]["k"] <= 7:
        raise ValueError("DSpark draft length must be 1..7")
    if engine["target_tensor_parallel_size"] != 1:
        raise ValueError("this pilot uses one TP=1 engine")
    if engine["enable_thinking"] or engine["enable_prefix_caching"]:
        raise ValueError("this pilot requires thinking and prefix caching off")
    if engine["dtype"] != "bfloat16" or not 0 < engine["gpu_memory_utilization"] < 1:
        raise ValueError("expected bf16 and a fractional GPU memory budget")
    if not re.fullmatch(r"[0-9a-f]{64}", runtime["image_sha256"]):
        raise ValueError("runtime image must have an expected SHA256")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime["vllm_revision"]):
        raise ValueError("runtime must have an expected vLLM Git revision")
    if spec["validation"]["max_tokens"] >= engine["max_model_len"]:
        raise ValueError("validation output cap must be below the context limit")
    for name in ("smoke", "probe", "workloads"):
        value = Path(spec["validation"][name])
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("validation corpora must use paths inside the repository")
    for value in (spec["target"]["path"], spec["drafter"]["path"], runtime["image"]):
        # SML's subprocess wrapper is a shell string. Avoid ambiguous quoting.
        if not re.fullmatch(r"[\w./:@+-]+", value, flags=re.ASCII):
            raise ValueError(f"unsupported shell characters in artifact path: {value!r}")
    return spec


def snapshot_sources(output: Path, spec: dict[str, Any]) -> None:
    source = output / "source"
    source.mkdir()
    for folder in (
        "src",
        "serving",
        "methods/dspark/launch",
        "targets/8b",
        "experiments/serving-pilot",
    ):
        shutil.copytree(
            REPO / folder,
            source / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "AGENTS.md"):
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, source / name)
    for key in ("smoke", "probe", "workloads"):
        relative = spec["validation"][key]
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    info = git_info(REPO)
    (output / "repository.patch").write_text(info.pop("diff") or "")
    write_json(output / "repository.json", info)
    launcher_root = path_value(spec["runtime"]["model_launch_root"])
    launcher_source = launcher_root / "src/swiss_ai_model_launch"
    shutil.copytree(
        launcher_source,
        output / "model-launch-source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    info = git_info(launcher_root)
    (output / "model-launch.patch").write_text(info.pop("diff") or "")
    write_json(output / "model-launch.json", info)
    files = {}
    for root in (source, output / "model-launch-source"):
        for item in sorted(root.rglob("*")):
            if item.is_file():
                files[str(item.relative_to(output))] = file_sha256(item)
    write_json(output / "source-manifest.json", files)
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="source")
        archive.add(output / "model-launch-source", arcname="model-launch-source")


def validate_target(spec: dict[str, Any], contract: dict[str, Any], *, full: bool) -> dict:
    target, expected = Path(spec["target"]["path"]), contract["source"]
    if spec["target"]["model"] != expected["hf_id"]:
        raise ValueError("target model does not match the target contract")
    if spec["target"]["revision"] != expected["revision"]:
        raise ValueError("target revision does not match the target contract")
    assets = {
        "config.json": "config_sha256",
        "tokenizer.json": "tokenizer_json_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
        "chat_template.jinja": "chat_template_sha256",
        "generation_config.json": "generation_config_sha256",
        "special_tokens_map.json": "special_tokens_map_sha256",
        "model.safetensors.index.json": "weight_index_sha256",
    }
    hashes = {}
    for name, key in assets.items():
        if not full and name == "tokenizer.json":
            continue
        actual = file_sha256(target / name)
        if actual != expected[key]:
            raise ValueError(f"target {name} differs from the locked contract")
        hashes[name] = actual
    if full:
        for entry in expected["weight_manifest"]["files"]:
            path = target / entry["name"]
            if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"target weight hash mismatch: {path.name}")
            hashes[entry["name"]] = entry["sha256"]
    return hashes


def validate_drafter(spec: dict[str, Any], *, full: bool) -> dict:
    root = Path(spec["drafter"]["path"])
    cfg = read_json(root / "config.json")
    if (
        cfg.get("speculators_model_type") != "dspark"
        or cfg.get("block_size") != 8
        or cfg.get("sample_from_anchor") is not True
        or cfg.get("aux_hidden_state_layer_ids") != [1, 8, 15, 22, 29]
    ):
        raise ValueError("head is not the expected block-8 Apertus DSpark checkpoint")
    verifier = cfg["speculators_config"]["verifier"]["name_or_path"].rstrip("/")
    if verifier.rsplit("/", 1)[-1] != spec["target"]["revision"]:
        raise ValueError("DSpark head verifier revision differs from the target")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("DSpark checkpoint has no safetensors weights")
    files = {"config.json": file_sha256(root / "config.json")}
    if full:
        files.update({path.name: file_sha256(path) for path in weights})
    report = {"config": cfg, "files": files, "complete": full}
    if full:
        report["checkpoint_sha256"] = digest(files)
        expected = spec["drafter"].get("checkpoint_sha256")
        if expected and expected != report["checkpoint_sha256"]:
            raise ValueError("drafter checkpoint manifest changed")
    return report


def prepare(args: argparse.Namespace) -> Path:
    spec = load_profile(args.profile, platform=args.platform)
    for name in ("partition", "time"):
        if getattr(args, name, None):
            spec["deployment"][name] = getattr(args, name)
    if args.model_suffix:
        if not re.fullmatch(r"[\w.-]+", args.model_suffix, flags=re.ASCII):
            raise ValueError(
                "model suffix must contain only letters, numbers, dot, dash, underscore"
            )
        key = "fallback_model_id" if args.variant == "plain" else "model_id"
        spec["deployment"][key] += "-" + args.model_suffix
    spec = load_resolved_profile(spec)
    qualification_path = getattr(args, "qualification", None)
    if spec["deployment"]["platform"] == "research":
        if not qualification_path:
            raise ValueError(
                "research launch requires --qualification from validated development runs"
            )
        verify_qualification(qualification_path, spec)
    model = spec["deployment"]["fallback_model_id" if args.variant == "plain" else "model_id"]
    run_id = f"{spec['name']}-{args.variant}-{stamp()}"
    output = (args.output or REPO / "results/8b/serving-pilot" / run_id).resolve()
    if not re.fullmatch(r"[\w./-]+", str(output), flags=re.ASCII):
        raise ValueError("run directory must not contain shell metacharacters or whitespace")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "profile.json", spec)
    record = {
        "schema_version": 1,
        "deployment_id": run_id,
        "created_at": now(),
        "operator": getpass.getuser(),
        "variant": args.variant,
        "model": model,
        "platform": spec["deployment"]["platform"],
        "gateway": spec["deployment"]["gateway"][spec["deployment"]["platform"]],
        "fallback_model": spec["deployment"]["fallback_model_id"],
        "profile_sha256": digest(spec),
    }
    write_json(output / "deployment.json", record)
    if qualification_path:
        shutil.copy2(qualification_path, output / "qualification.json")
    try:
        contract = read_json(path_value(spec["target"]["contract"]))
        validate_target(spec, contract, full=False)
        if args.variant == "dspark":
            write_json(output / "checkpoint-check.json", validate_drafter(spec, full=False))
        snapshot_sources(output, spec)
        source = output / "source"
        for name in ("smoke", "probe", "workloads"):
            path = path_value(spec["validation"][name])
            if not path.is_file():
                raise FileNotFoundError(f"missing validation corpus: {path}")
        source_template = Path(spec["target"]["path"]) / "chat_template.jinja"
        # A fixed thinking-off endpoint, including requests that send true.
        (output / "chat_template.jinja").write_text(
            "{% set enable_thinking = false %}\n" + source_template.read_text()
        )
        env_source = path_value(spec["runtime"]["environment"])
        environment = tomllib.loads(env_source.read_text())
        if environment["image"].replace("{arch}", "arm64") != spec["runtime"]["image"]:
            raise ValueError("environment image differs from the pilot's pinned image")
        shutil.copy2(env_source, output / "environment-source.toml")
        env = os.environ.copy()
        for key in (
            "EXTRA_ENV",
            "EXTRA_MOUNTS",
            "MAX_NUM_SCHEDULED_TOKENS",
            "ENFORCE_EAGER",
            "ASYNC_SCHEDULING",
            "PROFILER_DIR",
            "SML_RESERVATION",
        ):
            env.pop(key, None)
        env.update(
            {
                "STAGE": "8b",
                "MODEL_LAUNCH_ROOT": spec["runtime"]["model_launch_root"],
                "SML_BIN": spec["runtime"]["sml_bin"],
                "PYTHON": sys.executable,
                "ENV_SOURCE": str(output / "environment-source.toml"),
                "SML_ENV_DIR": str(output / "environments"),
                "SML_ARCH": "arm64",
                "VLLM_PATCH_DIR": str(output / "overlays"),
                "IMAGE": spec["runtime"]["image"],
                "APPLY_VLLM_PATCHES": "0",
                "SERVED_MODEL": model,
                "TARGET_MODEL": spec["target"]["path"],
                "TARGET_CONTRACT": str(source / spec["target"]["contract"]),
                "TARGET_TP": "1",
                "RESULTS_ROOT": str(output),
                "VALIDATE_ONLY": "0",
                "DSPARK_CHECKPOINT": spec["drafter"]["path"],
                "NUM_SPECULATIVE_TOKENS": str(spec["drafter"]["k"]),
                "SML_PARTITION": spec["deployment"]["partition"],
                "SML_TIME": spec["deployment"]["time"],
                "MAX_MODEL_LEN": str(spec["engine"]["max_model_len"]),
                "GPU_MEMORY_UTILIZATION": str(spec["engine"]["gpu_memory_utilization"]),
                "MAX_NUM_BATCHED_TOKENS": str(spec["engine"]["max_num_batched_tokens"]),
                "MAX_NUM_SEQS": str(spec["engine"]["max_num_seqs"]),
                "ENABLE_PREFIX_CACHING": "0",
                "VLLM_DISABLE_LOG_REQUESTS": "1",
                "VLLM_CHAT_TEMPLATE": str(output / "chat_template.jinja"),
                "VLLM_DTYPE": "bfloat16",
            }
        )
        patches = [str(source / name) for name in spec["runtime"]["patches"]]
        patched = command(
            ["bash", str(source / "serving/patch-vllm.sh"), env["IMAGE"], *patches],
            env=env,
            capture_output=True,
        )
        # Keep the output and source readable at the same shared paths in the job.
        mounts = patched.stdout.splitlines() + [f"{output}:{output}"]
        env["EXTRA_MOUNTS"] = ",".join(mounts)
        cache = f"/iopsstor/scratch/cscs/{getpass.getuser()}/apertus-pilot/{run_id}"
        env["EXTRA_ENV"] = (
            f"VLLM_CACHE_ROOT={cache};VLLM_NO_USAGE_STATS=1;HF_HUB_OFFLINE=1;PYTHONNOUSERSITE=1"
        )
        (output / "bin").mkdir()
        shutil.copy2(path_value(spec["runtime"]["opentela_binary"]), output / "bin/otela-arm64")
        prelaunch = (
            f'export PYTHONPATH={shlex.quote(str(source / "src"))}:"${{PYTHONPATH:-}}"\n'
            + f"export OPENTELA_BIN={shlex.quote(str(output / 'bin/otela-arm64'))}\n"
            + shlex.join(
                ["python3", "-m", "apertus_bench.research_pilot", "preflight", str(output)]
            )
        )
        launcher = (
            "serving/baseline.sh" if args.variant == "plain" else "methods/dspark/launch/dspark.sh"
        )
        argv = [
            "bash",
            str(source / launcher),
            "--account",
            spec["deployment"]["account"],
            "--no-tui",
            "--pre-launch-cmds",
            prelaunch,
            "--output-script",
            str(output / "rendered"),
        ]
        if spec["deployment"]["platform"] == "dev":
            argv.append("--dev")
        write_json(
            output / "launch.json",
            {
                "argv": argv,
                "environment": {
                    key: env[key]
                    for key in sorted(env)
                    if key
                    in {
                        "STAGE",
                        "MODEL_LAUNCH_ROOT",
                        "SML_BIN",
                        "ENV_SOURCE",
                        "EXTRA_ENV",
                        "EXTRA_MOUNTS",
                        "TARGET_MODEL",
                        "SERVED_MODEL",
                        "VLLM_PATCH_DIR",
                        "VLLM_CHAT_TEMPLATE",
                        "SML_PARTITION",
                        "SML_TIME",
                    }
                },
            },
        )
        with (output / "render.log").open("x") as log:
            command(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=source)
        normalize_entrypoint(output)
        for script in (output / "rendered").glob("*.sh"):
            command(["bash", "-n", str(script)])
        if not (output / "rendered/master.sh").is_file():
            raise RuntimeError("SML did not render master.sh; inspect render.log")
        # The environment filename is reported by the existing launcher.
        rendered_env = re.search(r"^environment=(.+)$", (output / "render.log").read_text(), re.M)
        if not rendered_env:
            raise RuntimeError("launcher did not report the resolved environment")
        shutil.copy2(rendered_env[1], output / "environment.toml")
        files = {
            str(p.relative_to(output)): file_sha256(p)
            for root in (
                output / "overlays",
                output / "environments",
                output / "rendered",
                output / "bin",
            )
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        for name in ("profile.json", "deployment.json", "chat_template.jinja", "environment.toml"):
            files[name] = file_sha256(output / name)
        if (output / "qualification.json").exists():
            files["qualification.json"] = file_sha256(output / "qualification.json")
        files["rendered-entrypoint.patch"] = file_sha256(output / "rendered-entrypoint.patch")
        write_json(output / "launch-manifest.json", files)
        write_json(output / "prepared.json", {"status": "prepared", "at": now()})
    except Exception as exc:
        write_json(output / "preparation-failed.json", {"at": now(), "error": str(exc)})
        raise
    print(json.dumps({"run": str(output), "model": model, "status": "prepared"}))
    return output


def load_resolved_profile(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate overrides using the same checks as a profile read."""
    # No temporary config containing environment credentials is needed.
    if spec["deployment"]["model_id"] == spec["deployment"]["fallback_model_id"]:
        raise ValueError("plain and DSpark must have separate model IDs")
    if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", spec["deployment"]["time"]):
        raise ValueError("time must be HH:MM:SS")
    return spec


def verify_manifest(run: Path, name: str) -> None:
    for relative, expected in read_json(run / name).items():
        if file_sha256(run / relative) != expected:
            raise ValueError(f"immutable run artifact changed: {relative}")


def preflight(run: Path) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("full preflight must run inside the Slurm allocation")
    spec, dep = read_json(run / "profile.json"), read_json(run / "deployment.json")
    try:
        verify_manifest(run, "source-manifest.json")
        verify_manifest(run, "launch-manifest.json")
        runtime = spec["runtime"]
        image_hash = file_sha256(Path(runtime["image"]))
        if image_hash != runtime["image_sha256"]:
            raise ValueError("runtime image SHA256 differs from the frozen profile")
        revision = command(
            ["git", "-C", "/workspace/vllm", "rev-parse", "HEAD"], capture_output=True
        ).stdout.strip()
        if revision != runtime["vllm_revision"]:
            raise ValueError("loaded vLLM source revision differs from the frozen profile")
        contract = read_json(run / "source" / spec["target"]["contract"])
        target_hashes = validate_target(spec, contract, full=True)
        drafter = validate_drafter(spec, full=True) if dep["variant"] == "dspark" else None
        if (run / "qualification.json").exists():
            qualification = read_json(run / "qualification.json")
            if target_hashes != qualification["target_files"]:
                raise ValueError("research target differs from the qualified target")
            if drafter and drafter["checkpoint_sha256"] != qualification["checkpoint_sha256"]:
                raise ValueError("research drafter differs from the qualified drafter")
        binary = os.environ.get("OPENTELA_BIN", "/opentelabin/otela-arm64")
        version = command([binary, "--version"], capture_output=True).stdout.strip()
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
        minimum = tuple(map(int, runtime["min_opentela_version"].split(".")))
        if not match or tuple(map(int, match.groups())) < minimum:
            raise ValueError(f"OpenTela {version!r} is below {runtime['min_opentela_version']}")
        target_out = run / "target"
        target_out.mkdir()
        for name in target_hashes:
            if not name.endswith(".safetensors"):
                shutil.copy2(Path(spec["target"]["path"]) / name, target_out / name)
        if drafter:
            training = run / "training"
            training.mkdir()
            for name in TRAINING_FILES:
                shutil.copy2(Path(spec["drafter"]["path"]) / name, training / name)
        corpora = {
            key: {
                "path": str(path_value(spec["validation"][key])),
                "sha256": file_sha256(path_value(spec["validation"][key])),
            }
            for key in ("smoke", "probe", "workloads")
        }
        packages = {}
        for name in ("vllm", "torch", "transformers", "triton", "safetensors"):
            packages[name] = importlib.metadata.version(name)
        write_json(
            run / "preflight.json",
            {
                "status": "passed",
                "at": now(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "node": socket.gethostname(),
                "node_list": os.environ.get("SLURM_JOB_NODELIST"),
                "image_sha256": image_hash,
                "vllm_revision": revision,
                "packages": packages,
                "opentela_version": version,
                "opentela_sha256": file_sha256(Path(binary)),
                "target_files": target_hashes,
                "drafter": drafter,
                "corpora": corpora,
                "chat_template_sha256": file_sha256(run / "chat_template.jinja"),
            },
        )
    except Exception as exc:
        write_json(
            run / "preflight-failed.json",
            {
                "at": now(),
                "error": str(exc),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "node": socket.gethostname(),
            },
        )
        raise


def submit(run: Path) -> dict:
    if not (run / "prepared.json").exists():
        raise ValueError("run has not been prepared successfully")
    with (run / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (run / "job.json").exists():
            return read_json(run / "job.json")
        # An interrupted response is ambiguous. Never blindly submit it again.
        if (run / "submission-attempt.json").exists():
            raise ValueError(
                "submission already attempted; inspect Slurm and the submission journal"
            )
        verify_manifest(run, "source-manifest.json")
        verify_manifest(run, "launch-manifest.json")
        argv = [
            "sbatch",
            "--parsable",
            "--chdir",
            str(run),
            "--output",
            str(run / "slurm.out"),
            "--error",
            str(run / "slurm.err"),
            str(run / "rendered/master.sh"),
        ]
        write_json(run / "submission-attempt.json", {"at": now(), "argv": argv})
        result = subprocess.run(
            argv, text=True, capture_output=True, check=False, env=job_environment()
        )
        write_json(
            run / "submission-result.json",
            {
                "at": now(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.returncode:
            raise RuntimeError(f"sbatch rejected the pilot: {result.stderr.strip()}")
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("unrecognized sbatch response; inspect journal before retrying")
        record = {"job_id": job, "submitted_at": now(), "operator": getpass.getuser()}
        write_json(run / "job.json", record)
        return record


def inspect_run(run: Path) -> dict:
    dep = read_json(run / "deployment.json")
    job = read_json(run / "job.json") if (run / "job.json").exists() else {}
    result = {**dep, **job, "run": str(run)}
    if job:
        result["slurm"] = command(
            [
                "sacct",
                "-j",
                job["job_id"],
                "-X",
                "-n",
                "-P",
                "-o",
                "JobID,State,NodeList,Start,End,Elapsed,AllocTRES",
            ],
            capture_output=True,
        ).stdout.strip()
        logs = run / "logs" / job["job_id"]
        result["log_dir"] = str(logs)
        if (run / "slurm.out").is_file():
            text = (run / "slurm.out").read_text(errors="replace")
            ips = re.findall(r"Replica 0 head IP: ([0-9.]+)", text)
            if ips:
                result["base_url"] = f"http://{ips[-1]}:8080"
    for name in ("preflight", "preflight-failed", "preparation-failed"):
        if (run / f"{name}.json").exists():
            result[name] = read_json(run / f"{name}.json")
    return result


def collect(run: Path) -> Path:
    info = inspect_run(run)
    output = run / "collections" / stamp()
    output.mkdir(parents=True)
    write_json(output / "status.json", info)
    for name in ("slurm.out", "slurm.err"):
        if (run / name).is_file():
            shutil.copy2(run / name, output / name)
    if info.get("log_dir"):
        logs = Path(info["log_dir"])
        for name in ("log.out", "replica_0.out", "replica_0.err"):
            if (logs / name).is_file():
                shutil.copy2(logs / name, output / name)
    return output


def stop(run: Path) -> None:
    job = read_json(run / "job.json")
    if job["operator"] != getpass.getuser():
        raise ValueError("refusing to cancel a deployment submitted by another operator")
    actual = (
        command(["sacct", "-j", job["job_id"], "-X", "-n", "-P", "-o", "User"], capture_output=True)
        .stdout.strip()
        .splitlines()
    )
    if not actual or any(row.rstrip("|") != getpass.getuser() for row in actual):
        raise ValueError("Slurm owner does not match the pilot operator")
    collect(run)
    command(["scancel", job["job_id"]])
    write_json(run / f"stopped-{stamp()}.json", {"at": now(), "job_id": job["job_id"]})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="resolve and check a profile without cluster access")
    plan.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    plan.add_argument("--platform", choices=("dev", "research"))
    prep = sub.add_parser("prepare", help="freeze sources and render a job; never submits")
    prep.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    prep.add_argument("--platform", choices=("dev", "research"))
    prep.add_argument("--variant", choices=("dspark", "plain"), default="dspark")
    prep.add_argument("--model-suffix", default="")
    prep.add_argument("--partition")
    prep.add_argument("--time")
    prep.add_argument("--output", type=Path)
    prep.add_argument("--qualification", type=Path, help="passing development qualification report")
    for name in ("submit", "preflight", "status", "collect", "stop"):
        cmd = sub.add_parser(name)
        cmd.add_argument("run", type=Path)
    # Request validation is kept separate from provisioning and uses the existing client.
    from apertus_bench.pilot_validation import add_parsers

    add_parsers(sub)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "run"):
        args.run = args.run.resolve()
    if args.action == "plan":
        print(json.dumps(load_profile(args.profile, platform=args.platform), indent=2))
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "preflight":
        preflight(args.run)
    elif args.action == "submit":
        print(json.dumps(submit(args.run)))
    elif args.action == "status":
        print(json.dumps(inspect_run(args.run), indent=2))
    elif args.action == "collect":
        print(collect(args.run))
    elif args.action == "stop":
        stop(args.run)
    else:
        from apertus_bench.pilot_validation import dispatch

        result = dispatch(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)


if __name__ == "__main__":
    main()
