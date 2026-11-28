"""Measure a fixed public workload under two arrival orders using SGLang's native API."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

OFFERED = 64
LONG_INDEX = 63
PROMPT_LEN = 1536
WAVE_GAP_S = 0.075


def workload(tokenizer, source):
    prompts = []
    marker = "<<REFERENCE>>"
    for i in range(OFFERED):
        messages = [dict(m) for m in source[i % len(source)]]
        problem = messages[-1]["content"]
        messages[-1]["content"] = f"Exercise {i + 1}.\nReference:\n{marker}\nProblem:\n{problem}"
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        example = source[(i + 1) % len(source)][-1]["content"] + "\n"
        filler = tokenizer.encode(example, add_special_tokens=False)
        n = PROMPT_LEN - len(prefix) - len(suffix)
        if n < 1 or not filler:
            raise ValueError(f"cannot construct the fixed length for prompt {i}")
        prompts.append(prefix + (filler * (n // len(filler) + 1))[:n] + suffix)
    if len({tuple(p) for p in prompts}) != OFFERED or any(len(p) != PROMPT_LEN for p in prompts):
        raise ValueError("expected 64 distinct fixed-length prompts")
    return prompts


async def request(client, url, prompts, i, repeat, cell, start, cap):
    prompt_index = (i + 7 * repeat) % OFFERED
    rid = f"{cell}-{i}"
    params = dict(
        temperature=1,
        top_p=1,
        max_new_tokens=cap,
        ignore_eos=True,
        sampling_seed=107 + repeat * OFFERED + i,
    )
    row = dict(i=i, rid=rid, prompt_index=prompt_index, output_cap=cap, sampling_params=params)
    payload = dict(rid=rid, input_ids=prompts[prompt_index], sampling_params=params, stream=False)
    row["dispatch_s"] = time.monotonic() - start
    try:
        async with client.post(url + "/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
            result = await response.json()
        row["finish_s"] = time.monotonic() - start
        meta = row["meta_info"] = result["meta_info"]
        output = row["output_ids"] = result["output_ids"]
        if meta["prompt_tokens"] != PROMPT_LEN or meta["completion_tokens"] != len(output):
            raise ValueError("server token accounting differs from workload")
        if meta["id"] != rid or len(output) != cap:
            raise ValueError("request ID or generated length mismatch")
        for key in (
            "queue_time",
            "forward_entry_time",
            "prefill_finished_time",
            "request_received_ts",
            "request_finished_ts",
            "num_retractions",
            "spec_verify_ct",
            "spec_accepted_drafts",
        ):
            if not isinstance(meta[key], (float, int)):
                raise ValueError(f"invalid {key}")
    except Exception as error:
        row["error"] = repr(error)
    return row


async def metrics(client, url, dest):
    async with client.get(url + "/metrics") as response:
        response.raise_for_status()
        dest.write_text(await response.text())


async def group(client, url, prompts, out, repeat, order, workload_hash):
    cell = out / f"{order}-r{repeat}"
    cell.mkdir()
    metadata = dict(
        arm=out.name,
        order=order,
        repeat=repeat,
        offered=OFFERED,
        long_index=LONG_INDEX,
        background_cap=256,
        long_cap=768,
        wave_gap_s=WAVE_GAP_S,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        prompt_tokens=PROMPT_LEN,
        tokenized_workload_sha256=workload_hash,
    )
    (cell / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    await metrics(client, url, cell / "metrics_before.prom")
    waves = [[LONG_INDEX], list(range(LONG_INDEX))]
    if order == "last":
        waves.reverse()
    dispatch_unix_s = time.time()
    start = time.monotonic()
    tasks = []
    for index, wave in enumerate(waves):
        if index:
            await asyncio.sleep(max(0, WAVE_GAP_S - (time.monotonic() - start)))
        for i in wave:
            tasks.append(
                asyncio.create_task(
                    request(
                        client,
                        url,
                        prompts,
                        i,
                        repeat,
                        cell.name,
                        start,
                        768 if i == LONG_INDEX else 256,
                    )
                )
            )
    rows = sorted(await asyncio.gather(*tasks), key=lambda r: r["i"])
    successes = [r for r in rows if "error" not in r]
    received = sorted(successes, key=lambda r: (r["meta_info"]["request_received_ts"], r["i"]))
    for rank, row in enumerate(received, start=1):
        row["server_receipt_rank"] = rank
    with (cell / "requests.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    long = rows[LONG_INDEX]
    rank = long.get("server_receipt_rank")
    summary = dict(
        successful=len(successes),
        dispatch_unix_s=dispatch_unix_s,
        group_seconds=max((r["finish_s"] for r in successes), default=None),
        output_tokens=sum(len(r["output_ids"]) for r in successes),
        long_server_receipt_rank=rank,
        order_confirmed=len(successes) == OFFERED and rank == (1 if order == "first" else OFFERED),
    )
    (cell / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    await metrics(client, url, cell / "metrics_after.prom")
    if len(successes) != OFFERED:
        raise RuntimeError(
            f"{cell}: {OFFERED - len(successes)} failed requests, see requests.jsonl"
        )
    print(cell.name, summary, flush=True)


async def measure(args):
    source_bytes = args.prompts.read_bytes()
    prompts = workload(
        AutoTokenizer.from_pretrained(args.model, local_files_only=True), json.loads(source_bytes)
    )
    token_file = args.out / "input_ids.json"
    with token_file.open("x") as f:
        json.dump(prompts, f, separators=(",", ":"))
        f.write("\n")
    digest = hashlib.sha256(token_file.read_bytes()).hexdigest()
    (args.out / "workload.json").write_text(
        json.dumps(
            dict(
                description="Synthetic repeated public DAPO math context",
                public_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                input_ids_sha256=digest,
                prompt_count=OFFERED,
                tokens_per_prompt=PROMPT_LEN,
            ),
            indent=2,
        )
        + "\n"
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as client:
        warm = await request(
            client, args.url, prompts, 0, 0, "pressure-warmup", time.monotonic(), 64
        )
        (args.out / "pressure-warmup.json").write_text(json.dumps(warm) + "\n")
        if "error" in warm:
            raise RuntimeError(warm["error"])
        for repeat in range(3):
            for order in ("last", "first") if repeat % 2 == 0 else ("first", "last"):
                await group(client, args.url, prompts, args.out, repeat, order, digest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("prompts", type=Path)
    p.add_argument("out", type=Path)
    asyncio.run(measure(p.parse_args()))
