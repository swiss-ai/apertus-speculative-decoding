"""Retain compact recovery provenance without copying weights or corpora."""
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from apertus_bench import research_pilot as h

base = h.REPO / "results/8b/serving-pilot"
out = base / ("recovery-audit-" + h.stamp())
out.mkdir()
shutil.copy2(__file__, out / "snapshot-source.py")
operation = Path(sys.argv[1])
shutil.copytree(operation, out / "operation")
names = ["dev-dspark-20261028-a", "dev-plain-20261028-b", "dev-plain-control-20261028-a", "dev-dspark-20261029-a", "dev-plain-20261029-a", "dev-plain-control-20261029-a"]
snapshot = {"created_at": h.now(), "repository": h.git_info(h.REPO), "runs": []}
snapshot["repository"].pop("diff")
files = ["deployment.json", "job.json", "preflight.json", "submission-attempt.json", "submission-result.json", "repository.json", "profile.json", "source-manifest.json", "launch-manifest.json", "replaces.json"]
for name in names:
    run = base / name
    snapshot["runs"].append(h.inspect_run(run))
    for file in files:
        source = run / file
        if source.is_file():
            target = out / name / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    patterns = ["automation/*/controller.json", "automation/*/report.json"]
    if "20261029" in name:
        patterns += ["rendered/*.sh", "environment.toml", "launch.json", "rendered-entrypoint.patch"]
    elif "dspark" in name:
        patterns += ["logs/*/replica_0.err", "slurm.out", "slurm.err"]
    for pattern in patterns:
        for source in sorted(run.glob(pattern)):
            target = out / name / source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
h.write_json(out / "snapshot.json", snapshot)
(out / "README.md").write_text("# Pilot launch recovery\n\nSnapshot at " + snapshot["created_at"] + ".\n\nThe original DSpark job 3783300 passed runtime preflight on Clariden node `nid007655`, then failed before serving: the pinned vLLM image rejected `--disable-log-requests`. Commit `c18ea9c0` uses its supported `--no-enable-log-requests` flag. All 244 repository tests and Ruff passed.\n\nThe two queued plain jobs contained the same obsolete flag in Slurm's frozen batch scripts. They were cancelled only after replacements had been rendered and checked. Original run artifacts are preserved. New independent run directories use the existing final DSpark e10 head and unchanged target/image/engine settings.\n\nReplacements: DSpark **3790128** (`debug`), baseline **3790129** (`normal`), control **3790130** (`normal`). All were pending for priority at this snapshot. Each requests one exclusive four-GPU node for 90 minutes; one engine uses one GPU. Gateway-enabled validation controllers wait up to 36 hours; the control stops after smoke checks. No live completion, correctness gate, performance comparison or qualification has passed yet. The dev key was previously accepted by the gateway.\n\nCompute and results are on CSCS Clariden under `faruk_zahiragic`, in `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/`. Code is on `feat/dspark-research-serving-pilot`. Controllers run on the login node; GPU serving and benchmark load run through Slurm.\n\nSee `snapshot.json` for Slurm states and paths, and `operation/` for the exact recovery script and replacement journal. This is a historical snapshot; check live Slurm and controller reports before restarting any run.\n")
hashes = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.rglob("*")) if path.is_file()}
h.write_json(out / "audit-manifest.json", hashes)
archive = operation / "recovery-audit.tar.gz"
with tarfile.open(archive, "w:gz") as sink:
    sink.add(out, arcname=out.name)
print(json.dumps({"audit": str(out), "archive": str(archive), "files": len(hashes)}))
