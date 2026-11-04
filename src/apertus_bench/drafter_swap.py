"""Live drafter swap under load (T5, drafter updates).

One ``session`` on one GPU, inside the serving container (a node-sweep
``command`` arm):

1. serve the target with EAGLE head A, the drafter-reload worker extension
   (``apertus_bench.drafter_reload``) and the dev ``/collective_rpc`` endpoint;
2. greedy pass ``pre`` over a fixed prompt set (concurrency 1);
3. closed-loop streaming load at a fixed concurrency for ``--duration`` s,
   sampled at Apertus settings; at ``--swap-at`` s the client asks the server to
   load head B in place, and the load keeps running through the swap;
4. greedy pass ``post`` (head B), then swap back to A and greedy pass ``back``
   (the same drafter as ``pre``: the noise floor of the greedy comparison).

Recorded per session: every request's stream event times, /metrics
spec-decode counters every second, the RPC timings, drafter/target parameter
digests before and after each swap, and ``summary.json`` with acceptance
length before and after the swap, the longest stream stall across the swap
against the stalls before it, and greedy exact-match rates (``post`` vs
``pre`` must not differ more than ``back`` vs ``pre``: speculative decoding
is lossless, so the drafter must not change the target's greedy output).

Greedy outputs hold text of the workload's restricted prompts: they stay in
the arm directory on the cluster. Commit ``summary.json``, ``swaps.json`` and
the metric series only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from apertus_bench.prometheus import parse_speculative_snapshot
from apertus_bench.workloads import load_prompts

EXTENSION = "apertus_bench.drafter_reload.DrafterReloadExtension"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def acceptance_between(samples: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    """Mean accepted length (1 + accepted/drafts) from the counters sampled in [start, end]."""
    inside = [s for s in samples if start <= s["t"] <= end]
    if len(inside) < 2:
        return {"mean_acceptance_length": None, "drafts": 0, "window_s": [start, end]}
    first, last = inside[0], inside[-1]
    drafts = last["drafts"] - first["drafts"]
    accepted = last["accepted_tokens"] - first["accepted_tokens"]
    return {
        "mean_acceptance_length": round(1 + accepted / drafts, 4) if drafts else None,
        "drafts": drafts,
        "accepted_tokens": accepted,
        "window_s": [round(first["t"], 2), round(last["t"], 2)],
    }


def stream_gaps(events: list[list[float]]) -> list[tuple[float, float]]:
    """(start, end) of every gap between consecutive stream events of each request."""
    gaps = []
    for times in events:
        gaps += list(zip(times, times[1:], strict=False))
    return gaps


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def swap_stall(
    events: list[list[float]], rpc_start: float, rpc_end: float, baseline_s: float = 60.0
) -> dict[str, Any]:
    """Longest stream gap overlapping the swap RPC, against the gaps just before it."""
    gaps = stream_gaps(events)
    across = [b - a for a, b in gaps if a <= rpc_end and b >= rpc_start]
    before = [b - a for a, b in gaps if rpc_start - baseline_s <= a and b < rpc_start]
    p99 = percentile(before, 0.99)
    longest = max(across) if across else None
    return {
        "rpc_seconds": round(rpc_end - rpc_start, 4),
        "streams_across_swap": len(across),
        "max_gap_across_swap_s": round(longest, 4) if longest is not None else None,
        "baseline_gap_p50_s": percentile(before, 0.5),
        "baseline_gap_p99_s": p99,
        "baseline_gap_max_s": max(before) if before else None,
        "stall_over_p99_s": round(longest - p99, 4) if longest is not None and p99 else None,
    }


def compare_greedy(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in right}
    pairs = [(row, by_id[row["id"]]) for row in left if row["id"] in by_id]
    exact = sum(1 for a, b in pairs if a["sha256"] == b["sha256"])
    divergence = []
    for a, b in pairs:
        if a["sha256"] != b["sha256"]:
            x, y = a["text"], b["text"]
            first = next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q), None)
            divergence.append(min(len(x), len(y)) if first is None else first)
    return {
        "prompts": len(pairs),
        "exact": exact,
        "exact_rate": round(exact / len(pairs), 4) if pairs else None,
        "first_divergence_chars": sorted(divergence),
    }


class Session:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out: Path = args.output
        self.base = f"http://127.0.0.1:{args.port}"
        self.model = f"{args.served_base}-drafter-swap"
        self.t0 = time.perf_counter()
        self.samples: list[dict[str, Any]] = []
        self.swaps: list[dict[str, Any]] = []
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=30),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None,
                                keepalive_expiry=1.0),
        )  # fmt: skip

    def now(self) -> float:
        return time.perf_counter() - self.t0

    async def rpc(self, method: str, *args: str) -> list[Any]:
        response = await self.http.post(
            f"{self.base}/collective_rpc", json={"method": method, "args": list(args)}
        )
        response.raise_for_status()
        return response.json()["results"]

    async def swap(self, head: str, label: str) -> dict[str, Any]:
        started = self.now()
        results = await self.rpc("reload_drafter", head)
        ended = self.now()
        record = {"label": label, "head": head, "rpc_start": started, "rpc_end": ended,
                  "workers": results}  # fmt: skip
        self.swaps.append(record)
        return record

    async def sample_metrics(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                text = (await self.http.get(f"{self.base}/metrics")).text
                snap = parse_speculative_snapshot(text)
                self.samples.append({"t": self.now(), "drafts": snap.drafts,
                                     "draft_tokens": snap.draft_tokens,
                                     "accepted_tokens": snap.accepted_tokens})  # fmt: skip
            except httpx.HTTPError:
                pass
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 1.0)

    async def greedy(self, prompts, label: str) -> list[dict[str, Any]]:
        rows = []
        for prompt in prompts:
            payload = {
                "model": self.model, "messages": prompt.messages, "temperature": 0.0,
                "max_tokens": self.args.greedy_max_tokens, "seed": 0,
            }  # fmt: skip
            if prompt.chat_template_kwargs:
                payload["chat_template_kwargs"] = prompt.chat_template_kwargs
            body = (await self.http.post(f"{self.base}/v1/chat/completions", json=payload)).json()
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage") or {}
            rows.append(
                {
                    "id": prompt.id,
                    "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
        with (self.out / f"greedy-{label}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows

    async def one_request(self, prompt, index: int) -> dict[str, Any]:
        payload = {
            "model": self.model, "messages": prompt.messages, "max_tokens": prompt.max_tokens,
            "seed": index, "stream": True, "stream_options": {"include_usage": True},
            **SAMPLING,
        }  # fmt: skip
        if prompt.chat_template_kwargs:
            payload["chat_template_kwargs"] = prompt.chat_template_kwargs
        started = self.now()
        events: list[float] = []
        tokens = None
        error = None
        try:
            async with self.http.stream(
                "POST", f"{self.base}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    event = json.loads(line[5:])
                    if event.get("usage"):
                        tokens = event["usage"].get("completion_tokens")
                    if event.get("choices") and event["choices"][0].get("delta", {}).get("content"):
                        events.append(self.now())
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {"index": index, "prompt_id": prompt.id, "start": started, "end": self.now(),
                "events": events, "completion_tokens": tokens, "error": error}  # fmt: skip

    async def load(self, prompts) -> list[dict[str, Any]]:
        args = self.args
        start = self.now()
        deadline = start + args.duration
        counter = iter(range(10**9))
        order = list(range(len(prompts)))
        random.Random(1).shuffle(order)
        records: list[dict[str, Any]] = []

        async def worker() -> None:
            while self.now() < deadline:
                index = next(counter)
                records.append(await self.one_request(prompts[order[index % len(order)]], index))

        async def swapper() -> None:
            await asyncio.sleep(max(0.0, start + args.swap_at - self.now()))
            await self.swap(args.new, "under-load")

        await asyncio.gather(*(worker() for _ in range(args.concurrency)), swapper())
        return records

    async def run(self) -> dict[str, Any]:
        args = self.args
        prompts = load_prompts(args.workload, args.workload_name)
        greedy_prompts = load_prompts(args.greedy_prompts, args.greedy_workload)[
            : args.greedy_limit
        ]
        stop = asyncio.Event()
        sampler = asyncio.create_task(self.sample_metrics(stop))
        info = {"start": await self.rpc("drafter_info")}
        pre = await self.greedy(greedy_prompts, "pre")
        load_started = self.now()
        records = await self.load(prompts)
        load_ended = self.now()
        info["after_swap"] = await self.rpc("drafter_info")
        post = await self.greedy(greedy_prompts, "post")
        await self.swap(args.old, "back")
        info["after_swap_back"] = await self.rpc("drafter_info")
        back = await self.greedy(greedy_prompts, "back")
        stop.set()
        await sampler
        swap = self.swaps[0]
        margin = args.settle
        ok = [r for r in records if not r["error"]]
        summary = {
            "old_head": args.old,
            "new_head": args.new,
            "workload": str(args.workload),
            "concurrency": args.concurrency,
            "duration_s": args.duration,
            "swap_at_s": args.swap_at,
            "requests": len(records),
            "errors": len(records) - len(ok),
            "acceptance": {
                "before_swap": acceptance_between(
                    self.samples, load_started + margin, swap["rpc_start"]
                ),
                "after_swap": acceptance_between(
                    self.samples, swap["rpc_end"] + margin, load_ended
                ),
            },
            "stall": swap_stall([r["events"] for r in ok], swap["rpc_start"], swap["rpc_end"]),
            "swap_workers": swap["workers"],
            "digests": {
                "drafter_changed": info["start"][0]["own_sha256"]
                != info["after_swap"][0]["own_sha256"],
                "drafter_restored": info["start"][0]["own_sha256"]
                == info["after_swap_back"][0]["own_sha256"],
                "target_unchanged": len({i[0]["target_sample_sha256"] for i in info.values()}) == 1,
            },  # fmt: skip
            "greedy": {
                "post_vs_pre": compare_greedy(pre, post),
                "back_vs_pre": compare_greedy(pre, back),
                "max_tokens": args.greedy_max_tokens,
            },
        }
        (self.out / "metrics.jsonl").write_text("".join(json.dumps(s) + "\n" for s in self.samples))
        (self.out / "swaps.json").write_text(json.dumps(self.swaps, indent=2) + "\n")
        (self.out / "drafter-info.json").write_text(json.dumps(info, indent=2) + "\n")
        with (self.out / "requests.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        await self.http.aclose()
        return summary


async def session(args: argparse.Namespace) -> int:
    import subprocess

    from apertus_bench.node_sweep import (
        DEFAULT_ENGINE,
        REPO_ROOT,
        STAGES,
        _eagle_metadata,
        _serve_argv,
        _stop,
        _wait_ready,
    )

    stage = STAGES[args.stage]
    args.served_base = stage["served_base"]
    args.output.mkdir(parents=True, exist_ok=True)
    raw = {"name": "drafter-swap", "method": "eagle", "drafter": args.old, "k": args.k,
           "vllm_args": ["--worker-extension-cls", EXTENSION]}  # fmt: skip
    engine = {**DEFAULT_ENGINE, "max_num_batched_tokens": 16384, "max_num_seqs": 256}
    argv, _tp, _dp, speculative = _serve_argv(
        raw, f"{stage['served_base']}-drafter-swap", stage["target_model"], engine
    )
    argv += ["--port", str(args.port)]
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    metadata = {"serve_argv": argv, "speculative_config": speculative, "engine": engine,
                "env": {"VLLM_SERVER_DEV_MODE": "1"},
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "args": {k: str(v) for k, v in vars(args).items()},
                "harness_revision": subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=False,
                ).stdout.strip(),
                "deployment_id": f"drafter-swap-{os.environ.get('SLURM_JOB_ID', 'local')}"
                                 f"-{args.output.name}",
                "heads": {
                    role: _eagle_metadata(Path(head), args.stage, stage["served_base"], 1)
                    for role, head in (("old", args.old), ("new", args.new))
                }}  # fmt: skip
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "replica.out").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            served = argv[argv.index("--served-model-name") + 1]
            ready = await _wait_ready("drafter-swap", args.port, served, [process], 2400)
            metadata["ready_seconds"] = round(ready, 1)
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            summary = await Session(args).run()
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            await _stop(process)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-bench-drafter-swap")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("session")
    run.add_argument("--old", required=True, help="EAGLE head served at start")
    run.add_argument("--new", required=True, help="head swapped in under load")
    run.add_argument("--k", type=int, default=7)
    run.add_argument("--stage", default="8b")
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--workload-name")
    run.add_argument("--concurrency", type=int, default=64)
    run.add_argument("--duration", type=float, default=600.0)
    run.add_argument("--swap-at", type=float, default=300.0)
    run.add_argument("--settle", type=float, default=30.0, help="s excluded after load start/swap")
    run.add_argument("--greedy-prompts", type=Path, required=True)
    run.add_argument("--greedy-workload")
    run.add_argument("--greedy-limit", type=int, default=32)
    run.add_argument("--greedy-max-tokens", type=int, default=256)
    run.add_argument("--port", type=int, default=8190)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raise SystemExit(asyncio.run(session(args)))


if __name__ == "__main__":
    main()
