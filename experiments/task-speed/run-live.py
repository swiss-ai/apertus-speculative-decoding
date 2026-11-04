"""Measure one window of the deployed pod (plain or DSpark mode) through the gateway.

The live campaign toggles the one Kubernetes Deployment between DSpark and
plain mode (toggle.md) and measures each mode in a window over the same corpus as the controlled run
(a window may span several jobs; measured cells are kept): D1, P1 (round 1), then P2, D2 (round 2).

  run-live.py --window D1 --variant dspark --round 1 --corpus CONTROLLED/corpus.jsonl \
      --output results/8b/task-speed/live-<stamp>

Requests go through the authenticated gateway (route "gateway", what users
see) or, with --direct, through a kubectl port-forward straight to the pod
(route "direct", to separate routing latency). Counters are always read
inside the pod with kubectl exec, so the API key never reaches the metrics
path. A cell counts only if the engine was idle when it started and finished
exactly our requests; otherwise it moves to rejected/ and reruns, at most
--retries times.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from apertus_bench import task_speed
from apertus_bench.client import StreamingChatClient
from apertus_bench.runner import CellSettings, GenerationSettings, Variant, run_cell
from apertus_bench.workloads import file_sha256, load_prompts

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = "https://api.swissai.svc.cscs.ch"
APP = "vllm-apertus-v1p5-8b-dspark-e10"


def model_ids() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("prepare", ROOT / "serving/kubernetes/prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL_IDS


class Kube:
    def __init__(self, args):
        self.base = [args.kubectl, "--context", args.context, "-n", args.namespace]

    def run(self, *argv: str, timeout: float = 60) -> str:
        return subprocess.run(
            [*self.base, *argv], check=True, capture_output=True, text=True, timeout=timeout
        ).stdout

    def pod(self) -> dict:
        pods = json.loads(self.run("get", "pods", "-l", f"app={APP}", "-o", "json"))["items"]
        live = [p for p in pods if p["metadata"].get("deletionTimestamp") is None]
        if len(live) != 1:
            raise RuntimeError(f"expected one {APP} pod, found {len(live)}")
        return live[0]


class _Response:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class KubeMetrics:
    """Duck-types StreamingChatClient.http for runner._fetch_metrics: GET -> kubectl exec."""

    SCRIPT = (
        "import sys, urllib.request; "
        "sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:8080/metrics', "
        "timeout=20).read().decode())"
    )

    def __init__(self, kube: Kube, pod: str):
        self.kube, self.pod = kube, pod
        self.http = self

    async def get(self, _url: str) -> _Response:
        text = await asyncio.to_thread(
            self.kube.run, "exec", self.pod, "-c", "vllm", "--", "python3", "-c", self.SCRIPT
        )
        return _Response(text)


def api_key(args) -> str:
    value = os.environ.get(args.api_key_env)
    if value:
        return value
    getter = (
        "from swiss_ai_model_launch.cli.configuration import InitConfig; import sys; "
        "sys.stdout.write(InitConfig.load().get_non_none_value('swissai_research_api_key'))"
    )
    result = subprocess.run(
        [args.sml_python, "-c", getter], capture_output=True, text=True, timeout=30, check=True
    )
    if not result.stdout:
        raise RuntimeError("no research gateway key from SML")
    return result.stdout


def port_forward(kube: Kube, pod: str) -> tuple[subprocess.Popen, str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [*kube.base, "port-forward", f"pod/{pod}", f"{port}:8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line = process.stdout.readline()  # "Forwarding from 127.0.0.1:<port> -> 8080"
    if "Forwarding from" not in line:
        process.kill()
        raise RuntimeError(f"port-forward failed: {line.strip()}")
    return process, f"http://127.0.0.1:{port}"


async def wait_idle(metrics: KubeMetrics, model: str, seconds: float = 900) -> None:
    # Streams the gateway dropped keep running in the engine; wait them out.
    from apertus_bench.load import parse_load_sample

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sample = parse_load_sample((await metrics.get("")).text, model)
        if not (sample.get("running") or 0) + (sample.get("waiting") or 0):
            return
        await asyncio.sleep(2)


async def window(args) -> None:
    suite = task_speed.load_suite(args.suite)
    manifest = json.loads((args.corpus.parent / "corpus-manifest.json").read_text())
    if file_sha256(args.corpus) != manifest["corpus_sha256"]:
        raise ValueError("corpus does not match its manifest")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus.jsonl"
    if not corpus.exists():
        shutil.copy2(args.corpus, corpus)
        (root / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copy2(args.suite, root / "suite.yaml")
    if file_sha256(corpus) != manifest["corpus_sha256"]:
        raise ValueError("this live run already holds a different corpus")
    route = "direct" if args.direct else "gateway"
    name = args.window if route == "gateway" else f"{args.window}-direct"
    out = root / name
    out.mkdir(exist_ok=True)  # resumable: a window may span several jobs
    model = model_ids()[args.variant]
    kube = Kube(args)
    pod = kube.pod()
    forward = None
    base_url = GATEWAY
    if args.direct:
        forward, base_url = port_forward(kube, pod["metadata"]["name"])
    try:
        await measure(args, suite, manifest, root, corpus, route, name, out, model, kube, pod,
                      base_url)  # fmt: skip
    finally:
        if forward is not None:
            forward.terminate()


async def measure(args, suite, manifest, root, corpus, route, name, out, model, kube, pod,
                  base_url) -> None:  # fmt: skip
    annotations = pod["metadata"]["annotations"]
    variant = annotations.get("apertus.swissai/variant", "dspark")
    if variant != args.variant:
        raise RuntimeError(f"the pod serves {variant}, not {args.variant}; toggle first")
    status = pod["status"]
    identity = {
        "window": args.window,
        "variant": args.variant,
        "round": args.round,
        "route": route,
        "base_url": base_url,
        "model": model,
        "pod": pod["metadata"]["name"],
        "pod_uid": pod["metadata"]["uid"],
        "node": pod["spec"]["nodeName"],
        "pod_started": status.get("startTime"),
        "restarts": [c["restartCount"] for c in status.get("containerStatuses", [])],
        "image_ids": [c.get("imageID") for c in status.get("containerStatuses", [])],
        "annotations": annotations,
        "client_host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "harness_branch": os.environ.get("TASK_SPEED_BRANCH"),
        "harness_revision": os.environ.get("TASK_SPEED_REVISION"),
        "started": datetime.now(UTC).isoformat(),
    }
    earlier = sorted(out.glob("identity-*.json"))
    if earlier and json.loads(earlier[0].read_text())["pod_uid"] != identity["pod_uid"]:
        raise RuntimeError("the pod changed since this window started; start a new window")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    (out / f"identity-{run_id}.json").write_text(json.dumps(identity, indent=2) + "\n")
    deadline = time.monotonic() + 60 * args.budget_minutes if args.budget_minutes else None
    metrics = KubeMetrics(kube, identity["pod"])
    extra = {
        "deployment_id": f"k8s-{identity['pod_uid']}",
        "checkpoint_sha256": annotations["apertus.swissai/checkpoint-sha256"],
        "target_model": "swiss-ai/Apertus-v1.5-8B",
        "target_revision": "a411d838600baf0e3635a3daf66fb7c55fc97bb6",
        "target_tensor_parallel_size": "1",
        "source_revision": annotations["apertus.swissai/source-revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "route": route,
        "node": identity["node"],
        "pod_uid": identity["pod_uid"],
        "window": args.window,
        "round": str(args.round),
        "suite": suite["name"],
    }
    generation = GenerationSettings(**suite["generation"])
    variant_record = Variant(
        name=args.variant,
        method="dspark" if args.variant == "dspark" else "none",
        num_speculative_tokens=3 if args.variant == "dspark" else None,
    )
    labels = args.slices or [entry["label"] for entry in suite["slices"]]
    concurrencies = args.concurrencies or suite["concurrencies"]
    (out / "rejected").mkdir(exist_ok=True)
    key = api_key(args) if route == "gateway" else None
    async with StreamingChatClient(
        base_url, model, key, timeout_seconds=args.timeout_seconds
    ) as client:
        # Pre-warm: reach every slice's shapes, and check the gateway passes a
        # 128-request burst without throttling before anything is measured.
        prewarm = load_prompts(corpus, task_speed.PREWARM)
        for concurrency in (1, 128):
            summary = await run_cell(
                client, prewarm, corpus, out / task_speed.PREWARM / run_id / f"c{concurrency}",
                CellSettings(task_speed.PREWARM, concurrency, 2 * len(prewarm) if concurrency > 1
                             else len(prewarm), 0, 1, generation),
                variant_record, None, model, extra,
            )  # fmt: skip
            if summary["requests"]["success_rate"] < 1:
                raise RuntimeError(f"pre-warm failures at C={concurrency}; stop the window")
        for label in labels:
            prompts = load_prompts(corpus, label)
            for concurrency in concurrencies:
                requests = task_speed.requests_for(suite, concurrency)
                cell = out / label / label / f"c{concurrency}"
                if _valid(cell):
                    continue  # measured by an earlier job of this window
                if cell.exists():  # interrupted before its guard, or failed requests
                    cell.rename(out / "rejected" / f"{label}-c{concurrency}-{run_id}-invalid")
                if deadline and time.monotonic() > deadline:
                    print("budget reached; start the window again to continue", flush=True)
                    break
                for attempt in range(1 + args.retries):
                    await wait_idle(metrics, model)
                    # Warm up as a separate cell, then let the engine drain before measuring.
                    await run_cell(
                        client, prompts, corpus,
                        out / "warmup" / label / f"c{concurrency}-{run_id}-{attempt}",
                        CellSettings(label, concurrency, concurrency, 0, 1, generation),
                        variant_record, None, model, extra, tolerate_warmup_failures=True,
                    )  # fmt: skip
                    await wait_idle(metrics, model)
                    summary = await run_cell(
                        client, prompts, corpus, cell,
                        CellSettings(label, concurrency, requests, 0, 1, generation),
                        variant_record, "kubectl-exec://pod/metrics", model, extra,
                        metrics_interval=args.metrics_interval, tolerate_warmup_failures=True,
                        metrics_client=metrics,
                    )  # fmt: skip
                    ok, reason, detail = task_speed.traffic_guard(
                        (cell / "metrics_before.prom").read_text(),
                        (cell / "metrics_after.prom").read_text(),
                        model,
                        requests,
                    )
                    failed = summary["requests"]["attempted"] - summary["requests"]["successful"]
                    if ok and summary["requests"]["success_rate"] < 0.99:
                        # The engine finished them, but the gateway dropped the streams.
                        ok, reason = False, f"{failed} client-side request failures"
                    (cell / "guard.json").write_text(
                        json.dumps({"valid": ok, "reason": reason, **detail}, indent=2) + "\n"
                    )
                    print(f"{name} {label} c{concurrency} attempt {attempt}: "
                          f"{summary['tokens']['output_tokens_per_second']:.0f} tok/s "
                          f"{'ok' if ok else reason}", flush=True)  # fmt: skip
                    if ok:
                        break
                    rejected = out / "rejected" / f"{label}-c{concurrency}-{run_id}-{attempt}"
                    cell.rename(rejected)
    after = kube.pod()
    same = (
        after["metadata"]["uid"] == identity["pod_uid"]
        and [c["restartCount"] for c in after["status"].get("containerStatuses", [])]
        == identity["restarts"]
    )
    cells = [
        {"slice": label, "concurrency": c, "valid": _valid(out / label / label / f"c{c}")}
        for label in labels
        for c in concurrencies
    ]
    record = {
        **{k: identity[k] for k in ("window", "variant", "round", "route", "pod_uid", "node")},
        "name": name,
        "status": "ok" if same and all(c["valid"] for c in cells) else "incomplete",
        "pod_unchanged": same,
        "started": identity["started"],
        "ended": datetime.now(UTC).isoformat(),
        "cells": cells,
    }
    (out / "window.json").write_text(json.dumps(record, indent=2) + "\n")
    index = root / "windows.json"
    windows = json.loads(index.read_text()) if index.exists() else []
    windows = [w for w in windows if w["name"] != name] + [record]
    index.write_text(json.dumps(windows, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("name", "status", "pod_unchanged")}))


def _valid(cell: Path) -> bool:
    guard, summary = cell / "guard.json", cell / "summary.json"
    if not (guard.exists() and json.loads(guard.read_text())["valid"]):
        return False
    return json.loads(summary.read_text())["requests"]["success_rate"] >= 0.99


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("suite.yaml"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", required=True, help="D1, P1, P2, D2, ...")
    parser.add_argument("--variant", choices=("dspark", "plain"), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--direct", action="store_true", help="port-forward to the pod")
    parser.add_argument("--slices", nargs="*")
    parser.add_argument("--concurrencies", nargs="*", type=int)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--budget-minutes", type=float, help="stop starting cells after this")
    parser.add_argument("--metrics-interval", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--context", default="breithorn-oidc")
    parser.add_argument("--namespace", default="rob-poc")
    parser.add_argument("--api-key-env", default="CSCS_SERVING_API")
    parser.add_argument(
        "--sml-python", default=os.path.expanduser("~/venvs/sml-apertus/bin/python")
    )
    asyncio.run(window(parser.parse_args()))


if __name__ == "__main__":
    main()
