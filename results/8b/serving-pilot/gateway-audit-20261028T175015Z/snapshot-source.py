"""Archive non-secret gateway checks and active controller identities."""
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
out = base / "operations" / ("gateway-audit-" + stamp)
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
            "JobID,State,NodeList,Start,End,Elapsed,AllocTRES"], text=True).strip(),
        "source_manifest_sha256": hashlib.sha256((run / "source-manifest.json").read_bytes()).hexdigest()}
    if proc.exists() and proc.stat().st_uid == os.getuid():
        record["controller_argv"] = [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
    controllers = sorted((run / "automation").glob("*/controller.json"))
    if not controllers:
        raise RuntimeError("controller has not saved its identity yet")
    active = controllers[-1].parent
    metadata = json.loads((active / "controller.json").read_text())
    if not metadata["gateway"]:
        raise RuntimeError("active controller does not enable gateway checks")
    record["active_controller"] = str(active.relative_to(base))
    for file in ("controller.json", "research_pilot.py", "pilot_validation.py", "report.json"):
        if (active / file).is_file():
            files.append(active / file)
    files.extend(sorted(run.glob("controller-replacement-*.json"))[-2:])
    for file in ("preflight.json", "preflight-failed.json"):
        if (run / file).is_file():
            files.append(run / file)
    snapshot["runs"].append(record)
for pattern in ("enable-dev-gateway-*/*.json", "enable-dev-gateway-*/*.py",
                "refresh-gateway-validation-*/*.json", "refresh-gateway-validation-*/*.py"):
    files.extend(p for p in (base / "operations").glob(pattern) if p.is_file())
for path in sorted(set(files)):
    snapshot["files"][str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
archive = out / "gateway-audit.tar.gz"
with tarfile.open(archive, "x:gz") as sink:
    for file in (out / "snapshot.json", out / "snapshot-source.py"):
        sink.add(file, arcname=file.name, recursive=False)
    for path in sorted(set(files)):
        sink.add(path, arcname=str(path.relative_to(base)), recursive=False)
print(json.dumps({"archive": str(archive), "audit": out.name, "files": len(set(files)),
    "states": [{k: v for k, v in r.items() if k != "controller_argv"} for r in snapshot["runs"]]}, indent=2))
