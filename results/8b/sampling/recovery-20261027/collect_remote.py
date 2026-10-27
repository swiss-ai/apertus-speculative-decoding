#!/usr/bin/env python3
"""Stream original sampling artifacts as an uncompressed tar to stdout.

Run through SSH and compress on the collecting workstation. Source artifacts
are read-only; model weights and source prompt text are excluded.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

REPO = Path("/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc")
RUNS = ["topk-off-20261022T122308Z", "topk-think-20261022T122311Z"]
PREFIX = Path("results/8b/sampling/recovery-20261027/source-recovery")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    manifest = {
        "collected_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_repo": str(REPO),
        "recovery_git_revision": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": [],
        "recovery_files": [],
        "derived_source_files": [],
        "unavailable_recovery_sources": [],
        "claim_limit": (
            "Original bytes are retained; recovery-time references are labelled separately."
        ),
    }
    seen: set[str] = set()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:

        def add(path: Path, destination: Path, kind: str = "source_files") -> None:
            key = str(destination)
            if key in seen:
                return
            seen.add(key)
            if not path.is_file():
                manifest["unavailable_recovery_sources"].append(str(path))
                return
            checksum = digest(path)
            archive.add(path, arcname=key, recursive=False)
            manifest[kind].append(
                {"source": str(path), "path": key, "bytes": path.stat().st_size, "sha256": checksum}
            )

        def add_bytes(data: bytes, destination: Path, identity: dict) -> None:
            key = str(destination)
            if key in seen:
                return
            seen.add(key)
            info = tarfile.TarInfo(key)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            manifest["derived_source_files"].append(
                {
                    "path": key,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    **identity,
                }
            )

        for name in RUNS:
            run = REPO / "results/8b/sweeps" / name
            for path in sorted(run.rglob("*")):
                if path.is_file():
                    add(path, path.relative_to(REPO))
            revision = (run / "git-revision.txt").read_text().strip()
            for relative in [
                "serving/node-sweep.sh",
                "src/apertus_bench/node_sweep.py",
                "src/apertus_bench/cli.py",
                "src/apertus_bench/client.py",
                "src/apertus_bench/runner.py",
                "src/apertus_bench/prometheus.py",
                "src/apertus_bench/workloads.py",
            ]:
                result = subprocess.run(
                    ["git", "-C", str(REPO), "show", f"{revision}:{relative}"], capture_output=True
                )
                if result.returncode:
                    manifest["unavailable_recovery_sources"].append(f"git:{revision}:{relative}")
                else:
                    add_bytes(
                        result.stdout,
                        PREFIX / "launch-revision-source" / revision / relative,
                        {"source_git_revision": revision, "source_relative_path": relative},
                    )
            for line in (run / "patches.txt").read_text().splitlines():
                path = Path(line.strip())
                add(path, PREFIX / "recovery-patches" / name / path.name, "recovery_files")
            environment = tomllib.loads((run / "environment.toml").read_text())
            for mount in environment.get("mounts", []):
                source = Path(mount.split(":", 1)[0])
                if "vllm-patch-sweep-" in str(source) and source.is_file():
                    relative = source.parts[source.parts.index("workspace") :]
                    add(
                        source,
                        PREFIX / "resolved-overlays" / name / Path(*relative),
                        "recovery_files",
                    )
            workloads = {}
            for meta in run.glob("*/*/*/c*/metadata.json"):
                metadata = json.loads(meta.read_text())
                path = Path(metadata["workload_file"])
                workloads[str(path)] = metadata["workload_file_sha256"]
            for number, (filename, expected) in enumerate(sorted(workloads.items())):
                path = Path(filename)
                if not path.is_file():
                    manifest["unavailable_recovery_sources"].append(filename)
                    continue
                actual = digest(path)
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                schedule = {
                    "source": filename,
                    "source_sha256": actual,
                    "metadata_sha256": expected,
                    "matches_recorded_workload": actual == expected,
                    "rows": [
                        {
                            k: r[k]
                            for k in ("id", "workload", "max_tokens", "chat_template_kwargs")
                            if k in r
                        }
                        for r in rows
                    ],
                    "claim_limit": "Sanitized recovery-time schedule; no prompt text copied.",
                }
                add_bytes(
                    (json.dumps(schedule, indent=2) + "\n").encode(),
                    PREFIX / "prompt-schedules" / name / f"{number}-{path.name}.json",
                    {"source": filename, "source_sha256": actual},
                )
        add(
            REPO / "targets/8b/contract.json",
            PREFIX / "recovery-references/target-contract.json",
            "recovery_files",
        )
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo(str(PREFIX / "collection-manifest.json"))
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    main()
