#!/usr/bin/env python3
"""Stream terminal layout artifacts, hashing originals and lossless stored bytes."""

import datetime as dt
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "results/8b/layout-recovery-20261030"
OUTPUT = Path(__file__).with_name("controller-v5-collection.json")
REMOTE = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/"
    "results/8b/layout-recovery-20261030"
)


def main():
    if (DEST / "controller-v5").exists() or OUTPUT.exists():
        raise RuntimeError("Collection destination already exists; inspect before a fresh attempt")
    command = ["ssh", "-o", "BatchMode=yes", "clariden1", f"tar -C {REMOTE} -cf - controller-v5"]
    records = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
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
                if not member.isfile():
                    raise RuntimeError("Unexpected non-regular source artifact")
                target.parent.mkdir(parents=True, exist_ok=True)
                compress = target.suffix in {".jsonl", ".out", ".log", ".csv"}
                stored = target.with_name(target.name + ".gz") if compress else target
                original_hash = hashlib.sha256()
                source = archive.extractfile(member)
                with stored.open("xb") as raw:
                    stream = (
                        gzip.GzipFile(
                            filename=target.name, mode="wb", fileobj=raw, mtime=0, compresslevel=6
                        )
                        if compress
                        else raw
                    )
                    while chunk := source.read(1024 * 1024):
                        original_hash.update(chunk)
                        stream.write(chunk)
                    if compress:
                        stream.close()
                records.append(
                    {
                        "source_path": str(relative),
                        "local_path": str(stored.relative_to(ROOT)),
                        "source_bytes": member.size,
                        "source_sha256": original_hash.hexdigest(),
                        "source_mtime": member.mtime,
                        "source_mode": member.mode,
                        "stored_bytes": stored.stat().st_size,
                        "stored_sha256": hashlib.file_digest(
                            stored.open("rb"), "sha256"
                        ).hexdigest(),
                        "compression": "gzip-lossless" if compress else None,
                    }
                )
        stderr = process.stderr.read().decode(errors="replace")
        code = process.wait()
    report = {
        "command": command,
        "source_root": REMOTE,
        "files": records,
        "returncode": code,
        "stderr": stderr,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(),
        "original_bytes": sum(r["source_bytes"] for r in records),
        "stored_bytes": sum(r["stored_bytes"] for r in records),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    if code:
        raise RuntimeError(stderr)
    print(
        json.dumps(
            {k: report[k] for k in ["original_bytes", "stored_bytes", "returncode", "finished_at"]}
        )
    )


if __name__ == "__main__":
    main()
