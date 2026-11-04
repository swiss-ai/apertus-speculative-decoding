"""Rebind a DSpark initializer to an identical, accessible verifier snapshot.

prepare copies only config metadata and links original weights/provenance.
verify hashes weights, target and image inside the training allocation before
any extraction server or optimizer starts. Earlier initializers stay unchanged.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import socket
from pathlib import Path

from depth_recovery import DEFAULT_IMAGE, sha256, verify_target, write_new


def rebound_config(config: dict, contract: dict) -> dict:
    source = contract["source"]
    verifier = config["speculators_config"]["verifier"]
    if Path(verifier["name_or_path"]).name != source["revision"]:
        raise ValueError("initializer verifier snapshot revision does not match target contract")
    if contract["target"]["architecture"] not in verifier["architectures"]:
        raise ValueError("initializer verifier architecture differs from target contract")
    if config["transformer_layer_config"]["num_hidden_layers"] != 1:
        raise ValueError("expected the original one-layer initializer")
    if config["aux_hidden_state_layer_ids"] != [1, 8, 15, 22, 29]:
        raise ValueError("initializer auxiliary layers changed")
    result = copy.deepcopy(config)
    result["speculators_config"]["verifier"]["name_or_path"] = source["authorized_checkpoint"]
    return result


def prepare(source: Path, output: Path, contract_path: Path) -> None:
    source = source.resolve()
    contract = json.loads(contract_path.read_text())
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    rebound = rebound_config(config, contract)
    target_config = Path(contract["source"]["authorized_checkpoint"]) / "config.json"
    if sha256(target_config) != contract["source"]["config_sha256"]:
        raise ValueError("accessible target config differs from original target contract")
    if not (source / "model.safetensors").is_file():
        raise ValueError("original initializer weights missing")
    output.mkdir(parents=True, exist_ok=False)
    for original in sorted(source.iterdir()):
        if original.is_file() and original.name != "config.json":
            (output / original.name).symlink_to(original)
    write_new(output / "config.json", rebound)
    (output / "original-config.json").symlink_to(config_path)
    write_new(
        output / "rebinding.json",
        {
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "source": str(source),
            "original_config_sha256": sha256(config_path),
            "rebound_config_sha256": sha256(output / "config.json"),
            "only_semantic_config_change": "speculators_config.verifier.name_or_path",
            "original_verifier": config["speculators_config"]["verifier"],
            "rebound_verifier": rebound["speculators_config"]["verifier"],
            "target_contract_sha256": sha256(contract_path),
            "expected_target_revision": contract["source"]["revision"],
            "expected_target_hashes": contract["source"],
            "verification_state": "full hashes pending allocated pre-training gate",
        },
    )


def verify(initializer: Path, contract_path: Path, image: Path, revision: str) -> dict:
    binding = json.loads((initializer / "rebinding.json").read_text())
    source = Path(binding["source"])
    contract = json.loads(contract_path.read_text())
    if sha256(contract_path) != binding["target_contract_sha256"]:
        raise ValueError("target contract changed after rebinding")
    original = json.loads((source / "config.json").read_text())
    actual = json.loads((initializer / "config.json").read_text())
    if actual != rebound_config(original, contract):
        raise ValueError("initializer changed beyond the single verifier path")
    if sha256(source / "config.json") != binding["original_config_sha256"]:
        raise ValueError("original initializer config changed")
    if sha256(initializer / "config.json") != binding["rebound_config_sha256"]:
        raise ValueError("rebound initializer config changed")
    weights = {}
    originals = sorted(source.glob("*.safetensors"))
    if not originals or {p.name for p in originals} != {
        p.name for p in initializer.glob("*.safetensors")
    }:
        raise ValueError("initializer weight inventory changed")
    for path in originals:
        linked = initializer / path.name
        if not os.path.samefile(path, linked):
            raise ValueError("initializer weights are not the original physical file")
        digest = sha256(linked)
        if (
            path.name == "model.safetensors"
            and digest != json.loads((source / "truncation.json").read_text())["sha256"]
        ):
            raise ValueError("drafter weight hash differs from original truncation")
        weights[path.name] = {
            "sha256": digest,
            "bytes": linked.stat().st_size,
            "same_file_as_source": True,
            "source": str(path),
        }
    return {
        "passed": True,
        "initializer": str(initializer),
        "source": str(source),
        "only_semantic_config_change": binding["only_semantic_config_change"],
        "original_config_sha256": binding["original_config_sha256"],
        "rebound_config_sha256": binding["rebound_config_sha256"],
        "weights": weights,
        "target": verify_target(contract_path),
        "container": {"path": str(image), "sha256": sha256(image)},
        "source_revision": revision,
        "helper_sha256": sha256(Path(__file__)),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--contract", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--initializer", type=Path, required=True)
    check.add_argument("--contract", type=Path, required=True)
    check.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    check.add_argument("--source-revision", required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.output, args.contract)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = verify(args.initializer, args.contract, args.image, args.source_revision)
        write_new(args.output, report)


if __name__ == "__main__":
    main()
