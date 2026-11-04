#!/usr/bin/env python3
"""Resume an interrupted terminal archive, retaining each finished file receipt."""

import datetime as dt
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
DEST = ROOT / "results/8b/layout-recovery-20261030"
REMOTE = "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261030"
PARTIAL = HERE / "attempt1-terminal-inventory.json"
JOURNAL = HERE / "collection-v2-files.jsonl"
OUTPUT = HERE / "controller-v5-collection.json"


def hash_bytes(path, compressed=False):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") if compressed else path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def reconstruct_attempt1():
    """Failure was at next tar header, after all opened member streams closed."""
    if PARTIAL.exists():
        return json.loads(PARTIAL.read_text())["files"]
    files = []
    for path in sorted((DEST / "controller-v5").rglob("*")):
        if not path.is_file():
            continue
        compressed = path.name.endswith((".jsonl.gz", ".out.gz", ".log.gz", ".csv.gz"))
        source = path.with_name(path.name[:-3]) if compressed else path
        size, source_hash = hash_bytes(path, compressed)
        stored_size, stored_hash = hash_bytes(path)
        files.append(
            {
                "source_path": str(source.relative_to(DEST)),
                "local_path": str(path.relative_to(ROOT)),
                "source_bytes": size,
                "source_sha256": source_hash,
                "source_mtime": None,
                "source_mode": None,
                "stored_bytes": stored_size,
                "stored_sha256": stored_hash,
                "compression": "gzip-lossless" if compressed else None,
                "receipt_origin": "reconstructed completed attempt1 member; failure at next empty header; gzip CRC verified; original tar mode/mtime unavailable",
            }
        )
    PARTIAL.write_text(
        json.dumps(
            {
                "terminal_error": "tarfile.ReadError: empty header at archive next(); no member output open",
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    return files


def main():
    if OUTPUT.exists():
        raise RuntimeError("Final manifest already exists")
    records = reconstruct_attempt1()
    if JOURNAL.exists():
        records += [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    excluded = {r["source_path"] for r in records}
    if len(excluded) != len(records):
        raise RuntimeError("Duplicate collected source path")
    for name in excluded:
        if "\n" in name:
            raise RuntimeError("Unexpected newline in source path")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "clariden1",
        f"tar -C {REMOTE} --no-wildcards -X - -cf - controller-v5",
    ]
    error = None
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.stdin.write(("\n".join(sorted(excluded)) + "\n").encode())
    process.stdin.close()
    try:
        with (
            JOURNAL.open("a") as journal,
            tarfile.open(fileobj=process.stdout, mode="r|") as archive,
        ):
            for member in archive:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.parts[0] != "controller-v5"
                ):
                    raise RuntimeError("Unexpected archive path")
                target = DEST / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.name in excluded:
                    raise RuntimeError("Unexpected source member")
                target.parent.mkdir(parents=True, exist_ok=True)
                compress = target.suffix in {".jsonl", ".out", ".log", ".csv"}
                stored = target.with_name(target.name + ".gz") if compress else target
                partial = stored.with_name(stored.name + ".collection-v2-partial")
                if stored.exists():
                    raise RuntimeError("Refusing to overwrite a collected file")
                original = hashlib.sha256()
                written = 0
                with archive.extractfile(member) as source, partial.open("xb") as raw:
                    stream = (
                        gzip.GzipFile(
                            filename=target.name, mode="wb", fileobj=raw, mtime=0, compresslevel=6
                        )
                        if compress
                        else raw
                    )
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        original.update(chunk)
                        stream.write(chunk)
                    if compress:
                        stream.close()
                if written != member.size:
                    raise RuntimeError("Source member truncated")
                partial.rename(stored)
                stored_bytes, stored_hash = hash_bytes(stored)
                receipt = {
                    "source_path": str(relative),
                    "local_path": str(stored.relative_to(ROOT)),
                    "source_bytes": written,
                    "source_sha256": original.hexdigest(),
                    "source_mtime": member.mtime,
                    "source_mode": member.mode,
                    "stored_bytes": stored_bytes,
                    "stored_sha256": stored_hash,
                    "compression": "gzip-lossless" if compress else None,
                    "receipt_origin": "v2 completed streamed tar member",
                }
                records.append(receipt)
                journal.write(json.dumps(receipt) + "\n")
                journal.flush()
    except Exception as exc:
        error = repr(exc)
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace")
    code = process.wait()
    report = {
        "command": command,
        "source_root": REMOTE,
        "files": records,
        "returncode": code,
        "stream_error": error,
        "stderr": stderr,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
        "original_bytes": sum(r["source_bytes"] for r in records),
        "stored_bytes": sum(r["stored_bytes"] for r in records),
    }
    if code or error:
        failed = HERE / (
            "collection-v2-failure-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        failed.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"Preserved incomplete attempt: {failed}: {error} {stderr}")
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["returncode", "original_bytes", "stored_bytes", "finished_at"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
