"""Collect compact continuation evidence without moving full run archives."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

root = Path(sys.argv[1]).resolve()
base = root / "results/8b/serving-pilot"
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = base / "operations" / ("continuation-audit-" + stamp)
out.mkdir(parents=True)
shutil.copy2(__file__, out / "snapshot-source.py")
snapshot = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository": str(root), "branch": subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
    "revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "runs": [], "files": {}}
files = []
for name in ("dev-dspark-20261028-a", "dev-plain-20261028-b", "dev-plain-control-20261028-a"):
    run = base / name
    job = json.loads((run / "job.json").read_text())["job_id"]
    pid = int((run / "validation-controller.pid").read_text())
    proc = Path("/proc") / str(pid)
    record = {"directory": name, "job_id": job, "controller_pid": pid,
        "controller_alive": proc.exists(), "preflight": (run / "preflight.json").exists(),
        "preflight_failed": (run / "preflight-failed.json").exists(),
        "slurm": subprocess.check_output(["sacct", "-j", job, "-X", "-n", "-P", "-o",
            "JobID,State,NodeList,Start,End,Elapsed,AllocTRES"], text=True).strip()}
    if proc.exists() and proc.stat().st_uid == os.getuid():
        record["controller_argv"] = [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
    for pattern in ("source-amendments/*/amendment.json", "source-amendments/*/intent.json",
                    "source-amendments/*/source.patch", "source-amendments/*/research_pilot.py.*",
                    "source-amendments/*/source-manifest.*.json", "source-amendments/*/amend-source.py",
                    "controller-replacement-*.json", "automation/*/controller.json", "automation/*/report.json"):
        files.extend(p for p in run.glob(pattern) if p.is_file())
    snapshot["runs"].append(record)
for pattern in ("prestart-opentela-version-*/amendments.json", "prestart-opentela-version-*/amend-source.py",
                "extend-waits-20261028T154025Z/*.json", "extend-waits-20261028T154025Z/*.py"):
    files.extend(p for p in (base / "operations").glob(pattern) if p.is_file())
for path in sorted(set(files)):
    snapshot["files"][str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
archive = out / "continuation-audit.tar.gz"
with tarfile.open(archive, "x:gz") as sink:
    for file in (out / "snapshot.json", out / "snapshot-source.py"):
        sink.add(file, arcname=file.name, recursive=False)
    for path in sorted(set(files)):
        sink.add(path, arcname=str(path.relative_to(base)), recursive=False)
print(json.dumps({"archive": str(archive), "audit": out.name, "files": len(set(files)),
    "states": [{k: v for k, v in r.items() if k != "controller_argv"} for r in snapshot["runs"]]}, indent=2))
