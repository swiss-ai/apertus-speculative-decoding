"""Wait for the bounded campaign, report, and archive on its compute allocation."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def archive(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "corpus.jsonl")
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = root / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    for suffix, subset in [
        ("raw", files),
        (
            "review",
            [
                p
                for p in files
                if p.name
                in {
                    "summary.json",
                    "metadata.json",
                    "REPORT.md",
                    "paired-results.json",
                    "provenance.json",
                    "corpus-manifest.json",
                    "spec.json",
                    "campaign-status.json",
                    "status.json",
                    "engine-excerpt.txt",
                    "preflight.json",
                    "profile.json",
                    "launch-manifest.json",
                }
            ],
        ),
    ]:
        target = root.with_name(root.name + f"-{suffix}.tar.gz")
        with tarfile.open(target, "x:gz") as bundle:
            for path in subset:
                bundle.add(
                    path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False
                )
        target.with_suffix(target.suffix + ".sha256").write_text(
            hashlib.sha256(target.read_bytes()).hexdigest() + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    if args.archive_only:
        archive(args.root)
        return
    deadline = time.monotonic() + 9000
    while time.monotonic() < deadline:
        status_path = args.root / "campaign-status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if any(s.get("status") != "ok" for s in status):
                raise RuntimeError("campaign recorded failed arm; preserve and investigate")
            if len(status) == 6:
                script = Path(__file__).with_name("report-speed-id-ood.py")
                subprocess.run(["python3", str(script), str(args.root)], check=True)
                subprocess.run(
                    [
                        "srun",
                        "--jobid=3791982",
                        "--overlap",
                        "--nodes=1",
                        "--ntasks=1",
                        "python3",
                        str(Path(__file__).resolve()),
                        str(args.root),
                        "--archive-only",
                    ],
                    check=True,
                )
                print("report and archives ready", flush=True)
                return
        time.sleep(30)
    raise TimeoutError("bounded wait expired; raw campaign remains on cluster")


if __name__ == "__main__":
    main()
