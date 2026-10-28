"""Copy compact launch provenance for review; leave raw corpora and weights in place."""
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile

root = pathlib.Path(sys.argv[1]).resolve()
base = root / "results/8b/serving-pilot"
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = base / "operations" / ("launch-audit-" + stamp)
out.mkdir(parents=True)
shutil.copy2(__file__, out / "snapshot-source.py")
names = ["dev-dspark-20261028-a", "dev-plain-20261028-a", "dev-plain-20261028-b",
         "dev-plain-control-20261028-a"]
snapshot = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "task_branch": subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
    "repository_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "repository_path": str(root), "runs": []}
selected = []
files = ["deployment.json", "profile.json", "checkpoint-check.json", "prepared.json", "job.json",
         "launch.json", "launch-manifest.json", "source-manifest.json", "rendered-entrypoint.patch",
         "environment.toml", "repository.json", "repository.patch", "model-launch.json", "model-launch.patch",
         "submission-attempt.json", "submission-result.json", "render.log", "preflight.json", "preflight-failed.json"]
for name in names:
    run = base / name
    dep = json.loads((run / "deployment.json").read_text())
    job = json.loads((run / "job.json").read_text()) if (run / "job.json").exists() else None
    record = {"directory": name, "remote_path": str(run), "deployment": dep, "job": job}
    if job:
        record["slurm"] = subprocess.check_output(["sacct", "-j", job["job_id"], "-X", "-n", "-P",
            "-o", "JobID,State,NodeList,Start,End,Elapsed,AllocTRES"], text=True).strip()
    else:
        record["submission"] = json.loads((run / "submission-result.json").read_text())
    for file in files:
        if (run / file).is_file():
            selected.append((run / file, name + "/" + file))
    for pattern in ("rendered/*.sh", "automation/*/controller.json", "automation/*/report.json", "controller-replacement-*.json"):
        for path in sorted(run.glob(pattern)):
            selected.append((path, name + "/" + str(path.relative_to(run))))
    snapshot["runs"].append(record)
(out / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
archive = out / "launch-audit.tar.gz"
with tarfile.open(archive, "w:gz") as sink:
    sink.add(out / "snapshot.json", arcname="snapshot.json")
    sink.add(out / "snapshot-source.py", arcname="snapshot-source.py")
    for path, name in selected:
        sink.add(path, arcname=name, recursive=False)
print(json.dumps({"archive": str(archive), "audit": out.name, "files": len(selected)}))
