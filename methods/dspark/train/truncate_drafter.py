"""Write a DSpark drafter that keeps only the first N decoder layers of another.

    python3 truncate_drafter.py SOURCE_DIR OUT_DIR --layers N

Yu's question (Slack, 16 Oct): does a shallower DSpark drafter pay off under
load? The truncated copy keeps fc, hidden_norm, norm, the Markov and
confidence heads and layers 0..N-1; config.json gets num_hidden_layers = N and
the first N layer_types. Served untrained it gives the cost of an N-layer
drafter (the acceptance is the 5-layer one's with the last layers cut); it is
also the warm start for training (speculators --from-pretrained).

Pure Python (no torch): the safetensors header is rewritten and the kept
tensors are copied byte for byte. OUT_DIR/truncation.json records the source,
both sha256 sums and the kept tensors; the source's provenance files are
copied alongside as source-<name>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

WEIGHTS = "model.safetensors"
LAYER = re.compile(r"^layers\.(\d+)\.")
# Kept next to the weights so a served or published copy stays traceable.
PROVENANCE = ("train_command.txt", "speculators.patch", "run.yaml", "val_metrics.json",
              "training_state.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header and the offset where tensor data starts."""
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(size)), 8 + size


def kept(name: str, layers: int) -> bool:
    match = LAYER.match(name)
    return match is None or int(match.group(1)) < layers


def truncate_config(config: dict, layers: int) -> dict:
    config = json.loads(json.dumps(config))
    layer_config = config["transformer_layer_config"]
    have = layer_config["num_hidden_layers"]
    if not 1 <= layers <= have:
        raise ValueError(f"--layers must be 1..{have}, got {layers}")
    layer_config["num_hidden_layers"] = layers
    if "layer_types" in layer_config:
        layer_config["layer_types"] = layer_config["layer_types"][:layers]
    return config


def truncate_weights(source: Path, out: Path, layers: int) -> list[str]:
    header, start = read_header(source)
    metadata = header.pop("__metadata__", None)
    names = sorted(
        (n for n in header if kept(n, layers)), key=lambda n: header[n]["data_offsets"][0]
    )
    new_header: dict = {} if metadata is None else {"__metadata__": metadata}
    offset = 0
    for name in names:
        begin, end = header[name]["data_offsets"]
        new_header[name] = {**header[name], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with source.open("rb") as src, out.open("wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for name in names:
            begin, end = header[name]["data_offsets"]
            src.seek(start + begin)
            left = end - begin
            while left:
                chunk = src.read(min(left, 1 << 24))
                if not chunk:
                    raise ValueError(f"{source} ends inside {name}")
                dst.write(chunk)
                left -= len(chunk)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args(argv)

    config = json.loads((args.source / "config.json").read_text())
    if config.get("speculators_model_type") != "dspark":
        raise SystemExit(f"{args.source} is not a speculators DSpark checkpoint")
    new_config = truncate_config(config, args.layers)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    names = truncate_weights(args.source / WEIGHTS, args.out / WEIGHTS, args.layers)
    (args.out / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")
    if (args.source / "config.py").exists():
        shutil.copy2(args.source / "config.py", args.out / "config.py")
    for name in PROVENANCE:
        if (args.source / name).exists():
            shutil.copy2(args.source / name, args.out / f"source-{name}")
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source / WEIGHTS),
        "source_num_hidden_layers": config["transformer_layer_config"]["num_hidden_layers"],
        "num_hidden_layers": args.layers,
        "kept_tensors": names,
        "sha256": sha256(args.out / WEIGHTS),
    }
    (args.out / "truncation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "kept_tensors"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
