"""Create an isolated training environment over the pinned serving container."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
python = args.output / "bin/python"
report = args.output / "runtime-provenance.json"
if report.exists():
    print(python)
    raise SystemExit(0)
if args.output.exists():
    raise ValueError("incomplete training environment; use a fresh attempt directory")
subprocess.run(
    [sys.executable, "-m", "venv", "--system-site-packages", str(args.output)], check=True
)
critical = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
constraints = args.output / "container-constraints.txt"
constraints.write_text("".join(f"{name}=={version}\n" for name, version in critical.items()))
commands = [
    [
        str(python),
        "-m",
        "pip",
        "install",
        "--constraint",
        str(constraints),
        "datasets==5.0.1",
        "loguru==0.7.3",
        "pydantic-settings==2.16.1",
        "rich==15.0.0",
        "typer==0.27.3",
    ],
    [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        str(args.upstream / "hs_connectors"),
        str(args.upstream),
    ],
]
with (args.output / "install.log").open("x") as log:
    for command in commands:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], text=True)
(args.output / "resolved-requirements.txt").write_text(frozen)
report.write_text(
    json.dumps(
        {"container_packages": critical, "commands": commands, "python": str(python)}, indent=2
    )
    + "\n"
)
print(python)
