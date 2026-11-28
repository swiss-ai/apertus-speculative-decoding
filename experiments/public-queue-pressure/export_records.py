"""Build public timing records from either original, checksummed Qwen queue-pressure job."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from summarize import BUDGETS, MODES, ORDERS, read, require, summarize

RESULTS = Path(__file__).resolve().parents[2] / "results/public-hybrid/queue-pressure-20261128"
SERVER_FIELDS = [
    "tp_size",
    "dp_size",
    "dtype",
    "page_size",
    "mem_fraction_static",
    "context_length",
    "max_running_requests",
    "max_total_tokens",
    "max_total_num_tokens",
    "disable_radix_cache",
    "random_seed",
    "speculative_algorithm",
    "speculative_eagle_topk",
    "speculative_num_steps",
    "speculative_num_draft_tokens",
    "speculative_use_rejection_sampling",
    "enable_linear_replayssm_spec",
    "linear_replayssm_cache_len",
    "mamba_ssm_dtype",
    "schedule_policy",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "cuda_graph_max_bs_decode",
    "version",
]
REQUEST_FIELDS = [
    "i",
    "rid",
    "prompt_index",
    "output_cap",
    "sampling_params",
    "dispatch_s",
    "finish_s",
    "server_receipt_rank",
]
META_FIELDS = [
    "id",
    "prompt_tokens",
    "completion_tokens",
    "num_retractions",
    "forward_entry_time",
    "prefill_finished_time",
    "queue_time",
    "request_received_ts",
    "request_finished_ts",
    "reasoning_tokens",
    "cached_tokens",
    "dp_rank",
    "e2e_latency",
    "spec_accept_rate",
    "spec_accept_length",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
    "spec_accepted_drafts",
    "spec_proposed_drafts",
    "spec_correct_drafts_histogram",
    "spec_accept_histogram",
    "api_server_dispatch_finish_ts",
    "decode_throughput",
    "response_sent_to_client_ts",
]
LOG_LINE = re.compile(
    r"^\[\d{4}-\d\d-\d\d [\d:. ]+\] (?:"
    r"(?:KV|Mamba) Cache is allocated\.|GDN ReplaySSM ring buffers allocated|"
    r"(?:Prefill|Decode) batch,)"
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fingerprint(ids):
    # Compact JSON, decimal integers and no trailing newline define the per-vector hash.
    require(isinstance(ids, list) and all(type(token) is int for token in ids), "invalid token IDs")
    payload = json.dumps(ids, separators=(",", ":")).encode()
    return {"count": len(ids), "sha256": hashlib.sha256(payload).hexdigest()}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_cell(source, dest):
    dest.mkdir()
    for name in ("metadata.json", "summary.json"):
        shutil.copyfile(source / name, dest / name)
    with (source / "requests.jsonl").open() as rows, (dest / "requests.jsonl").open("x") as out:
        for line in rows:
            row = json.loads(line)
            record = {key: row[key] for key in REQUEST_FIELDS}
            record["meta_info"] = {key: row["meta_info"][key] for key in META_FIELDS}
            output = fingerprint(row["output_ids"])
            record.update(output_count=output["count"], output_sha256=output["sha256"])
            out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    for name in ("metrics_before.prom", "metrics_after.prom"):
        data = (source / name).read_text()
        data = re.sub(r'model_name="[^"]+"', 'model_name="Qwen/Qwen3.5-9B"', data)
        pids = sorted(set(re.findall(r'pid="(\d+)"', data)), key=int)
        for i, pid in enumerate(pids):
            data = data.replace(f'pid="{pid}"', f'pid="process-{i}"')
        (dest / name).write_text(data)


def write_arm(source, dest):
    dest.mkdir()
    for name in ("command.sh", "cache-budget.json", "workload.json"):
        shutil.copyfile(source / name, dest / name)
    write_json(
        dest / "input-fingerprints.json",
        [fingerprint(row) for row in read(source / "input_ids.json")],
    )
    server = read(source / "server-info.json")
    info = {key: server[key] for key in SERVER_FIELDS}
    info["model_path"] = Path(server["model_path"]).name
    info["internal_states"] = [
        {key: state[key] for key in ("effective_max_running_requests_per_dp", "memory_usage")}
        for state in server["internal_states"]
    ]
    write_json(dest / "server-info.json", info)
    lines = (source / "server.log").read_text().splitlines()
    (dest / "server.log").write_text("".join(line + "\n" for line in lines if LOG_LINE.match(line)))
    for order in ORDERS:
        for rep in range(3):
            name = f"{order}-r{rep}"
            write_cell(source / name, dest / name)


def write_run(source, dest):
    dest.mkdir()
    for name in ("source.txt", "source.sha256", "runtime.sha256", "gpu-rotation.txt"):
        shutil.copyfile(source / name, dest / name)
    members = sorted(path for path in source.rglob("*") if path.is_file())
    (dest / "original-files.sha256").write_text(
        "".join(f"{digest(path)}  {path.relative_to(source)}\n" for path in members)
    )
    with (source / "memory.csv").open() as samples, (dest / "memory.csv").open("x") as out:
        rows = csv.DictReader(samples, skipinitialspace=True)
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["timestamp", "index", "memory.used [MiB]"])
        for row in rows:
            writer.writerow([row["timestamp"], int(row["index"]), int(row["memory.used [MiB]"])])
    for budget in BUDGETS:
        for mode in MODES:
            name = f"{mode}-{budget}"
            write_arm(source / name, dest / name)


def pack(source, output):
    # Fixed headers and sorted paths keep repeat exports identical.
    with (
        output.open("xb") as f,
        gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(str(path.relative_to(source.parent)))
            info.mode, info.size = 0o644, path.stat().st_size
            with path.open("rb") as member:
                tar.addfile(info, member)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    originals = read(RESULTS / "provenance.json")["original_files_sha256"]
    archive = args.original.name
    if archive not in ("job-4060288.tar.gz", "job-4060290.tar.gz"):
        parser.error("expected job-4060288.tar.gz or job-4060290.tar.gz")
    require(digest(args.original) == originals[archive], f"{archive}: original checksum mismatch")
    job = archive.removeprefix("job-").removesuffix(".tar.gz")
    summary_file = RESULTS / f"job-{job}.json"
    require(digest(summary_file) == originals[summary_file.name], "job summary checksum mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_name = f"run-{job}"
        source = tmp / "original" / run_name
        with tarfile.open(args.original) as tar:
            tar.extractall(source.parent, filter="data")
        expected = summary_file.read_bytes()
        dest = tmp / run_name
        for path, public in ((source, False), (dest, True)):
            if public:
                write_run(source, dest)
            actual = (
                json.dumps(summarize(path, public), indent=2, allow_nan=False) + "\n"
            ).encode()
            require(actual == expected, f"{path}: job summary differs")
        pack(dest, args.output)


if __name__ == "__main__":
    main()
