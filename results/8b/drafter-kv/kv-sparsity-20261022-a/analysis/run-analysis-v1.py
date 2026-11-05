"""Reproduce the immutable screen analysis with the collected cluster artifacts."""

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


analysis = Path(__file__).resolve().parent
campaign = analysis.parent
repo = analysis.parents[4]
env = os.environ.copy()
settings = {
    "PYTHONPATH": str(repo / "src"),
    "APERTUS_DRAFT_KV_SOURCE_ROOT": "/users/faruk_zahiragic/apertus-spec-decoding-kv-sparsity",
    "APERTUS_DRAFT_KV_ARTIFACT_ROOT": str(repo),
}
env.update(settings)
commands = [
    [sys.executable, "-m", "apertus_bench.drafter_kv_evidence", str(campaign),
     "--phase", "screen", "--output", str(analysis / "screen-v1")],
    [sys.executable, "-m", "apertus_bench.drafter_kv_experiment", "analyze", str(campaign),
     "--phase", "screen", "--evidence", str(analysis / "screen-v1/evidence.json"),
     "--output", str(analysis / "screen-gate-v1.json")],
    [sys.executable, str(analysis / "build_screen_report.py")],
]
sources = [campaign / "ledger.json", repo / "experiments/drafter-kv/design.yaml",
           repo / "src/apertus_bench/drafter_kv_evidence.py",
           repo / "src/apertus_bench/drafter_kv_experiment.py",
           analysis / "build_screen_report.py", Path(__file__).resolve()]
provenance = {
    "created_at": dt.datetime.now(dt.UTC).isoformat(), "environment": settings,
    "analysis_git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    "python": sys.version, "pyyaml_version": importlib.metadata.version("PyYAML"),
    "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sources], "commands": [],
}
if (analysis / "pipeline-provenance-v1.json").exists():
    raise FileExistsError("pipeline already recorded; choose new output versions")
try:
    for index, argv in enumerate(commands, 1):
        started = dt.datetime.now(dt.UTC).isoformat()
        log = analysis / f"pipeline-step-{index}.log"
        with log.open("x") as handle:
            result = subprocess.run(argv, cwd=repo, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, check=False)
        provenance["commands"].append({"argv": argv, "cwd": str(repo), "log": str(log),
                                       "started_at": started, "exit_code": result.returncode,
                                       "ended_at": dt.datetime.now(dt.UTC).isoformat()})
        print(f"analysis step {index}: exit={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"analysis failed; inspect {log}")
finally:
    with (analysis / "pipeline-provenance-v1.json").open("x") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
