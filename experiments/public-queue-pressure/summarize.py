"""Validate pressure experiments and compare designated arrivals at matched cache budgets."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

from cache_budget import ledger

GIB = 2**30
BUDGETS = (17, 24)
MODES = ("snapshot", "compact")
ORDERS = ("first", "last")
METRICS = ("long_queue_s", "long_received_to_prefill_end_s", "group_s")


def read(path):
    return json.loads(path.read_text())


def require(condition, context):
    if not condition:
        raise ValueError(str(context))


def fields(record, expected, context):
    for key, value in expected.items():
        require(
            record.get(key) == value, f"{context}: {key}={record.get(key)!r}, expected {value!r}"
        )


def logged_cache(dir_, mode, budget):
    layout = ledger(mode, budget)
    require(read(dir_ / "cache-budget.json") == layout, f"{dir_}: cache budget mismatch")
    info = read(dir_ / "server-info.json")
    cap = layout["max_total_tokens"]
    fields(
        info,
        dict(
            tp_size=1,
            dp_size=1,
            dtype="bfloat16",
            page_size=1,
            mem_fraction_static=0.8,
            context_length=3072,
            max_running_requests=64,
            max_total_tokens=cap,
            max_total_num_tokens=cap,
            disable_radix_cache=True,
            random_seed=107,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_use_rejection_sampling=True,
            enable_linear_replayssm_spec=(mode == "compact"),
            linear_replayssm_cache_len=16,
        ),
        dir_,
    )
    require(Path(info["model_path"]).name == "Qwen3.5-9B", f"{dir_}: wrong target")
    # The snapshot arm inherits FP32 recurrence from the checkpoint. The compact
    # launcher sets it explicitly. The logged temporal bytes independently verify it.
    require(info["mamba_ssm_dtype"] in (None, "float32"), f"{dir_}: unexpected recurrent dtype")
    internal = info["internal_states"][0]
    require(internal["effective_max_running_requests_per_dp"] == 64, f"{dir_}: slot cap changed")
    log = (dir_ / "server.log").read_text()
    kv = re.findall(
        r"KV Cache is allocated\. dtype: torch.bfloat16, #tokens: (\d+), "
        r"K size: ([\d.]+) GB, V size: ([\d.]+) GB",
        log,
    )
    require(len(kv) == 2, f"{dir_}: expected target and MTP pools")
    measured = {}
    rounding = 0.0
    for (n, k, v), name, layers in zip(kv, ("target", "draft"), (8, 1), strict=True):
        require(int(n) == cap, f"{dir_}: {name} KV cap {n} differs from {cap}")
        expected = (cap + 1) * layers * 4 * 256 * 2 / GIB
        for kind, value in (("k", k), ("v", v)):
            require(abs(float(value) - expected) <= 0.005001, f"{dir_}: {name} {kind} bytes")
            measured[f"{name}_{kind}"] = float(value)
            rounding += 0.005
    state = layout["state_bytes"]
    labels = {
        "persistent_conv": "conv_state",
        "persistent_temporal": "ssm_state",
        "intermediate_conv": "intermediate_conv_window_cache",
        "intermediate_temporal": "intermediate_ssm_state_cache",
    }
    state_logs = re.findall(r"Mamba Cache is allocated\. max_mamba_cache_size: 64, ([^\n]+)", log)
    require(len(state_logs) == 1, f"{dir_}: missing or repeated recurrent allocation")
    for component, label in labels.items():
        match = re.search(rf"\b{label} size: ([\d.]+)GB", state_logs[0])
        require(match is not None, f"{dir_}: missing {label}")
        measured[component] = float(match[1])
        require(
            abs(float(match[1]) - state.get(component, 0) / GIB) <= 0.005001,
            f"{dir_}: {component} differs from budget",
        )
        rounding += 0.005
    rings = re.findall(r"GDN ReplaySSM ring buffers allocated[^:\n]*: ([^\n]+)", log)
    require(len(rings) == (1 if mode == "compact" else 0), f"{dir_}: replay ring count")
    if rings:
        parts = dict(re.findall(r"(\w+)=([\d.]+)GB", rings[0]))
        require(set(parts) == {"d", "k", "g", "rawv", "rawk"}, f"{dir_}: replay ring breakdown")
        measured.update({f"ring_{key}": float(value) for key, value in parts.items()})
        ring_total = sum(float(value) for value in parts.values())
        require(
            abs(ring_total - state["replay_records"] / GIB) <= 0.002501,
            f"{dir_}: replay sizes differ from budget",
        )
        rounding += 0.0025
    total_gib = sum(measured.values())
    expected_gib = budget - layout["unallocated_bytes"] / GIB
    require(abs(total_gib - expected_gib) <= rounding + 1e-6, f"{dir_}: actual cache total")
    return dict(
        max_total_tokens=cap,
        modeled_cache_gib=expected_gib,
        logged_cache_gib=total_gib,
        logged_components_gib=measured,
        log_rounding_bound_gib=rounding,
    )


def checked_workload(dir_, source_sha, public_records=False):
    if public_records:
        manifest = read(dir_ / "workload.json")
        proof = read(dir_ / "input-fingerprints.json")
        fields(
            manifest,
            dict(public_source_sha256=source_sha, prompt_count=64, tokens_per_prompt=1536),
            dir_,
        )
        require(len(proof) == 64 and len({row["sha256"] for row in proof}) == 64, dir_)
        for record in proof:
            require(record["count"] == 1536, f"{dir_}: invalid input length")
            require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), f"{dir_}: input hash")
        return manifest["input_ids_sha256"]

    path = dir_ / "input_ids.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fields(
        read(dir_ / "workload.json"),
        dict(
            public_source_sha256=source_sha,
            input_ids_sha256=digest,
            prompt_count=64,
            tokens_per_prompt=1536,
        ),
        dir_,
    )
    prompts = read(path)
    require(
        len(prompts) == 64
        and len(set(map(tuple, prompts))) == 64
        and all(len(p) == 1536 for p in prompts),
        f"{dir_}: invalid prompt set",
    )
    return digest


def cell_outcome(dir_, order, rep, digest, public_records=False):
    path = dir_ / f"{order}-r{rep}"
    fields(
        read(path / "metadata.json"),
        dict(
            arm=dir_.name,
            order=order,
            repeat=rep,
            offered=64,
            long_index=63,
            background_cap=256,
            long_cap=768,
            temperature=1,
            top_p=1,
            ignore_eos=True,
            prompt_tokens=1536,
            tokenized_workload_sha256=digest,
        ),
        path,
    )
    rows = [json.loads(s) for s in (path / "requests.jsonl").read_text().splitlines()]
    require([r["i"] for r in rows] == list(range(64)), f"{path}: incomplete group")
    for i, row in enumerate(rows):
        require("error" not in row, f"{path}: request {i} failed")
        cap = 768 if i == 63 else 256
        rid = f"{path.name}-{i}"
        fields(row, dict(prompt_index=(i + 7 * rep) % 64, output_cap=cap, rid=rid), path)
        fields(
            row["sampling_params"],
            dict(
                max_new_tokens=cap,
                temperature=1,
                top_p=1,
                ignore_eos=True,
                sampling_seed=107 + 64 * rep + i,
            ),
            path,
        )
        m = row["meta_info"]
        fields(m, dict(id=rid, prompt_tokens=1536, completion_tokens=cap), path)
        if public_records:
            require(re.fullmatch(r"[0-9a-f]{64}", row["output_sha256"]), f"{rid}: output hash")
            output_count = row["output_count"]
        else:
            output_count = len(row["output_ids"])
        require(
            output_count == cap and 0 <= row["dispatch_s"] <= row["finish_s"],
            f"{path}: invalid response {i}",
        )
        times = [
            m[key]
            for key in (
                "request_received_ts",
                "forward_entry_time",
                "prefill_finished_time",
                "request_finished_ts",
            )
        ]
        require(times == sorted(times) and all(map(math.isfinite, times)), f"{rid}: timestamps")
    by_receipt = sorted(rows, key=lambda r: (r["meta_info"]["request_received_ts"], r["i"]))
    for rank, row in enumerate(by_receipt, start=1):
        require(row["server_receipt_rank"] == rank, f"{path}: incorrect receipt rank")
    probe = rows[63]
    m = probe["meta_info"]
    rank = probe["server_receipt_rank"]
    order_ok = rank == (1 if order == "first" else 64)
    last = max(rows, key=lambda r: r["finish_s"])
    fields(
        read(path / "summary.json"),
        dict(
            successful=64,
            group_seconds=last["finish_s"],
            output_tokens=63 * 256 + 768,
            long_server_receipt_rank=rank,
            order_confirmed=order_ok,
        ),
        path,
    )
    received_ts = m["request_received_ts"]
    unique_receipt = all(
        r is probe or r["meta_info"]["request_received_ts"] != received_ts for r in rows
    )
    first_finish = min(r["meta_info"]["request_finished_ts"] for r in rows)
    return dict(
        probe_receipt_rank=rank,
        order_confirmed=order_ok and unique_receipt,
        long_queue_s=m["queue_time"],
        long_received_to_forward_s=m["forward_entry_time"] - m["request_received_ts"],
        long_prefill_s=m["prefill_finished_time"] - m["forward_entry_time"],
        long_received_to_prefill_end_s=m["prefill_finished_time"] - m["request_received_ts"],
        long_prefill_end_to_finish_s=m["request_finished_ts"] - m["prefill_finished_time"],
        long_client_finish_s=probe["finish_s"],
        group_s=last["finish_s"],
        last_finisher=last["i"],
        long_retractions=m["num_retractions"],
        group_retractions=sum(r["meta_info"]["num_retractions"] for r in rows),
        started_before_first_finish=sum(
            r["meta_info"]["forward_entry_time"] < first_finish for r in rows
        ),
    )


def distribution(values):
    return (
        dict(n=len(values), median=median(values), min=min(values), max=max(values))
        if values
        else dict(n=0)
    )


def compare(cells):
    """Pair by node, prompt rotation and arrival order. Retain misordered cells in raw output."""
    results = {}
    for budget in BUDGETS:
        pairs = {}
        for order in ORDERS:
            pairs[order] = {}
            for rep in range(3):
                s = cells[f"snapshot-{budget}"][f"{order}-r{rep}"]
                c = cells[f"compact-{budget}"][f"{order}-r{rep}"]
                if s["order_confirmed"] and c["order_confirmed"]:
                    deltas = {m: s[m] - c[m] for m in METRICS}
                    deltas["group_fraction_saved"] = 1.0 - c["group_s"] / s["group_s"]
                    pairs[order][rep] = deltas
        interactions = {
            rep: {m: pairs["last"][rep][m] - pairs["first"][rep][m] for m in METRICS}
            for rep in pairs["last"].keys() & pairs["first"].keys()
        }
        results[str(budget)] = dict(
            snapshot_minus_compact_by_order=pairs,
            late_minus_early_replay_gain=interactions,
            paired_group_fraction_saved={
                order: distribution([v["group_fraction_saved"] for v in reps.values()])
                for order, reps in pairs.items()
            },
        )
    low = results["17"]["late_minus_early_replay_gain"]
    high = results["24"]["late_minus_early_replay_gain"]
    results["low_minus_high_pressure_interaction"] = {
        rep: {m: low[rep][m] - high[rep][m] for m in METRICS} for rep in low.keys() & high.keys()
    }
    return results


def summarize(run, public_records=False):
    if public_records:
        lines = (run / "source.sha256").read_text().splitlines()
        hashes = dict(line.split(maxsplit=1)[::-1] for line in lines)
        source_sha = hashes["prompts.json"]
    else:
        source_sha = hashlib.sha256((run / "prompts.json").read_bytes()).hexdigest()
    pinned = read(
        Path(__file__).resolve().parents[2]
        / "results/public-hybrid"
        / "queue-pressure-20261128/model-and-workload.json"
    )
    require(source_sha == pinned["prompts_sha256"], f"{run}: wrong public source")
    runtime = {}
    for line in (run / "runtime.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        runtime[Path(path).name] = digest
    fields(runtime, pinned["weight_shards_sha256"], run)
    workload = None
    deployments, cells = {}, {}
    for budget in BUDGETS:
        for mode in MODES:
            arm = f"{mode}-{budget}"
            dir_ = run / arm
            deployments[arm] = logged_cache(dir_, mode, budget)
            sha = checked_workload(dir_, source_sha, public_records)
            require(workload is None or workload == sha, f"{dir_}: unmatched workload")
            workload = sha
            cells[arm] = {
                f"{order}-r{rep}": cell_outcome(dir_, order, rep, sha, public_records)
                for order in ORDERS
                for rep in range(3)
            }
    by_arm = {}
    for arm, groups in cells.items():
        by_arm[arm] = {}
        for order in ORDERS:
            valid = [g for k, g in groups.items() if k.startswith(order) and g["order_confirmed"]]
            by_arm[arm][order] = {
                m: distribution([g[m] for g in valid])
                for m in (*METRICS, "long_prefill_s", "started_before_first_finish")
            }
    return dict(
        run=run.name,
        input_ids_sha256=workload,
        runtime_manifest_sha256=hashlib.sha256((run / "runtime.sha256").read_bytes()).hexdigest(),
        deployment=deployments,
        cells=cells,
        by_arm=by_arm,
        paired=compare(cells),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--public-records",
        action="store_true",
        help="read the publication archive with hashed token vectors",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.public_records), indent=2, allow_nan=False))
