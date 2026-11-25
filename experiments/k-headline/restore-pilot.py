"""Restore the serving pilot that the controlled runs load, after a scratch purge.

The task-speed harness reads the image, overlays, chat template and engine
identity of dev-dspark-12h-20261029-a, the pilot the live Kubernetes Deployment
was frozen from. The 2026-11 scratch purge emptied that directory. This rebuilds
it in a new place from two sources and checks every byte against the hashes
the pilot recorded before the purge:

- the pilot's JSON records, as copied into the committed controlled run
  (results/8b/task-speed/controlled-20261104T133618Z);
- the four vLLM overlays and the chat template, from the live Deployment's
  ConfigMap in the GitOps repository (the same bytes the production pod mounts).

environment.toml is the recorded one with the old pilot path replaced by the new
one, and the compile caches moved under it; nothing else changes.

  python3 restore-pilot.py --records RUN --configmap configmap.yaml \\
      --configmap-source REPO@SHA:PATH --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

RECORDS = (
    "preflight.json",
    "profile.json",
    "launch-manifest.json",
    "source-manifest.json",
    "checkpoint-check.json",
)
OLD_PILOT = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/"
    "dev-dspark-12h-20261029-a"
)
OLD_CACHE = (
    "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot/"
    "apertus15-8b-dspark-e10-dspark-20261029T134252610114Z"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--configmap", type=Path, required=True)
    parser.add_argument("--configmap-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in RECORDS:
        shutil.copy2(args.records / name, out / name)
    launch = json.loads((out / "launch-manifest.json").read_text())
    preflight = json.loads((out / "preflight.json").read_text())
    data = yaml.safe_load(args.configmap.read_text())["data"]
    checks = []

    template = data["chat_template.jinja"].encode()
    (out / "chat_template.jinja").write_bytes(template)
    checks.append(
        {
            "file": "chat_template.jinja",
            "expected": preflight["chat_template_sha256"],
            "actual": sha256(template),
        }
    )
    overlays = [k for k in launch if k.startswith("overlays/") and k.endswith(".py")]
    for key in overlays:
        body = data["overlay-" + Path(key).name].encode()
        target = out / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        checks.append({"file": key, "expected": launch[key], "actual": sha256(body)})

    environment = (args.records / "environment.toml").read_text()
    if OLD_PILOT not in environment or OLD_CACHE not in environment:
        raise SystemExit("recorded environment.toml does not name the old pilot")
    restored = environment.replace(OLD_PILOT, str(out)).replace(OLD_CACHE, str(out / "cache"))
    (out / "environment.toml").write_text(restored)

    failed = [c for c in checks if c["expected"] != c["actual"]]
    report = {
        "restored_at": datetime.now(UTC).isoformat(),
        "passed": not failed,
        "records_from": str(args.records),
        "configmap": {
            "source": args.configmap_source,
            "sha256": sha256(args.configmap.read_bytes()),
        },
        "environment_changes": {OLD_PILOT: str(out), OLD_CACHE: str(out / "cache")},
        "environment_sha256": {
            "recorded": sha256(environment.encode()),
            "restored": sha256(restored.encode()),
        },
        "not_restored": [k for k in launch if k.startswith("overlays/") and k not in overlays],
        "checks": checks,
    }
    (out / "restore-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checked": len(checks)}))
    if failed:
        raise SystemExit(f"hash mismatch: {[c['file'] for c in failed]}")


if __name__ == "__main__":
    main()
