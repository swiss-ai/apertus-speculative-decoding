"""Version a metadata-command correction before the queued jobs' first start."""
import datetime
import difflib
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def execute(argv):
    return subprocess.check_output(argv, text=True).strip()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def queue(job):
    fields = execute(["squeue", "-j", job, "-h", "-o", "%i|%T|%u|%R"]).split("|")
    if len(fields) != 4 or fields[:3] != [job, "PENDING", getpass.getuser()]:
        raise RuntimeError(f"job identity/state changed: {fields!r}")
    return fields


root = Path(sys.argv[1]).resolve()
fix_revision = sys.argv[2]
if execute(["git", "-C", str(root), "rev-parse", "HEAD"]) != fix_revision:
    raise RuntimeError("fix revision has not been installed")
branch = execute(["git", "-C", str(root), "branch", "--show-current"])
if branch != "feat/dspark-research-serving-pilot":
    raise RuntimeError("not on the pilot task branch")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
operation = root / "results/8b/serving-pilot/operations" / ("prestart-opentela-version-" + stamp)
operation.mkdir(parents=True)
shutil.copy2(__file__, operation / "amend-source.py")
records = []
held = []
try:
    for name in ("dev-dspark-20261028-a", "dev-plain-20261028-b", "dev-plain-control-20261028-a"):
        run = root / "results/8b/serving-pilot" / name
        job = json.loads((run / "job.json").read_text())["job_id"]
        before = queue(job)
        if any((run / p).exists() for p in
               ("preflight.json", "preflight-failed.json", "checks", "measurements", "open-loop")):
            raise RuntimeError(f"{name} already has execution artifacts")
        manifest_path = run / "source-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest.items():
            if sha(run / relative) != digest:
                raise RuntimeError(f"source artifact has changed: {relative}")
        module_name = "source/src/apertus_bench/research_pilot.py"
        module = run / module_name
        original = module.read_text()
        old = 'command([binary, "--version"], capture_output=True).stdout.strip()'
        new = 'command([binary, "version"], capture_output=True).stdout.strip()'
        if original.count(old) != 1:
            raise RuntimeError("expected exactly one old metadata-command call")
        corrected = original.replace(old, new)
        backup = run / "source-amendments" / stamp
        backup.mkdir(parents=True)
        shutil.copy2(module, backup / "research_pilot.py.before")
        shutil.copy2(manifest_path, backup / "source-manifest.before.json")
        shutil.copy2(__file__, backup / "amend-source.py")
        patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
            corrected.splitlines(keepends=True), fromfile="a/" + module_name, tofile="b/" + module_name))
        (backup / "source.patch").write_text(patch)
        record = {"run": str(run), "job_id": job, "before": before,
            "slurm_before": execute(["scontrol", "show", "job", "-o", job]),
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "branch": branch, "fix_revision": fix_revision,
            "reason": "Installed OpenTela exposes a version subcommand; --version is invalid.",
            "module": module_name, "original_sha256": sha(module),
            "original_source_manifest_sha256": sha(manifest_path),
            "original_source_archive": "sources.tar.gz",
            "original_source_archive_sha256": sha(run / "sources.tar.gz"),
            "launch_manifest_sha256": sha(run / "launch-manifest.json"),
            "patch_sha256": sha(backup / "source.patch")}
        save(backup / "intent.json", record)
        # Holding and rechecking prevents the first execution racing the amendment.
        execute(["scontrol", "hold", job])
        held.append(job)
        if queue(job)[3] != "(JobHeldUser)" and queue(job)[3] != "JobHeldUser":
            raise RuntimeError("job was not held by its owner")
        if any((run / p).exists() for p in ("preflight.json", "preflight-failed.json")):
            raise RuntimeError("job started before the hold")
        temporary = module.with_name(module.name + ".amended")
        temporary.write_text(corrected)
        os.replace(temporary, module)
        manifest[module_name] = sha(module)
        temporary_manifest = manifest_path.with_suffix(".amended.json")
        save(temporary_manifest, manifest)
        os.replace(temporary_manifest, manifest_path)
        shutil.copy2(module, backup / "research_pilot.py.after")
        shutil.copy2(manifest_path, backup / "source-manifest.after.json")
        with tarfile.open(backup / "sources.after.tar.gz", "x:gz") as archive:
            for base in ("source", "model-launch-source"):
                archive.add(run / base, arcname=base,
                    filter=lambda info: None if "__pycache__" in Path(info.name).parts else info)
        for relative, digest in manifest.items():
            if sha(run / relative) != digest:
                raise RuntimeError(f"amended source verification failed: {relative}")
        record.update(amended_sha256=sha(module),
            amended_source_manifest_sha256=sha(manifest_path),
            amended_source_archive_sha256=sha(backup / "sources.after.tar.gz"),
            launch_manifest_unchanged=sha(run / "launch-manifest.json") == record["launch_manifest_sha256"],
            original_source_archive_unchanged=sha(run / "sources.tar.gz") == record["original_source_archive_sha256"],
            command=["scontrol", "hold", job], status="amended_before_first_start")
        save(backup / "amendment.json", record)
        records.append(record)
finally:
    releases = []
    for job in held:
        try:
            execute(["scontrol", "release", job])
            releases.append({"job_id": job, "released": True,
                "slurm_after": execute(["scontrol", "show", "job", "-o", job])})
        except Exception as exc:
            releases.append({"job_id": job, "released": False, "error": str(exc)})
    save(operation / "amendments.json", {"records": records, "releases": releases})
    if any(not item["released"] for item in releases):
        raise RuntimeError("a job remains held; inspect the operation release journal")
print(json.dumps({"operation": str(operation), "amended_jobs": [r["job_id"] for r in records],
    "same_job_ids": True, "resubmitted": False}, indent=2))
