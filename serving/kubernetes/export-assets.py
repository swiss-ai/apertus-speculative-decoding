"""Stream existing Clariden assets to tar; run in a foreground PVC transfer.

No downloads, training, image builds, Slurm submission, or source modifications.
Checksums are the frozen verified values; verify the payload on the staging pod.
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path


def export(spec_path, stream):
    spec = json.loads(spec_path.read_text())
    root = Path("/capstor/store/cscs/swissai/infra01")
    sources = {
        "target": root / "hf_models/models/swiss-ai/Apertus-v1.5-8B",
        "drafter": root
        / (
            "speculative-decoding/DSpark_Apertus_1.5_8B/"
            "dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final"
        ),
        "opentela": root / "opentela-share/prod",
    }
    files = {}
    for relative in spec["asset_hashes"]:
        group, name = relative.split("/")
        source = sources[group] / name
        if group == "opentela":
            source = sources[group] / "otela-arm64-sai-v0.0.7-00933f1"
        files[relative] = source
    for name in (
        "run.yaml",
        "speculators.patch",
        "train_command.txt",
        "training_state.json",
        "val_metrics.json",
    ):
        files["training-provenance/" + name] = sources["drafter"] / name
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    with tarfile.open(fileobj=stream, mode="w|", dereference=True) as archive:
        for relative, source in files.items():
            archive.add(source, arcname=relative, recursive=False)
        checksums = "".join(f"{value}  {name}\n" for name, value in spec["asset_hashes"].items())
        for name, payload in {
            "SHA256SUMS": checksums.encode(),
            "deployment-provenance.json": spec_path.read_bytes(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    export(args.spec, sys.stdout.buffer)
