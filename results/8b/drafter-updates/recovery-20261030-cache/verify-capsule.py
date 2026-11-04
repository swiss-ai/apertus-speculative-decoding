"""Verify collected raw T5 parts, provenance and every complete measurement pass."""

import collections
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

folder = Path(sys.argv[1])
manifest = json.loads((folder / "capsule-manifest.json").read_text())
source = "7f0a41dcc0a3c5d433b83dac80075ee0516a694e"
whole = hashlib.sha256()
cells = {}
checkpoint_hashes = collections.defaultdict(set)
deployment_ids = set()
with tempfile.TemporaryFile() as combined:
    for part in manifest["parts"]:
        data = (folder / part["name"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        whole.update(data)
        combined.write(data)
    assert whole.hexdigest() == manifest["archive_sha256"]
    combined.seek(0)
    with tarfile.open(fileobj=combined, mode="r:gz") as archive:
        files = json.load(archive.extractfile("collection-manifest.json"))
        assert len(files) == manifest["files"]
        selected_name = next(n for n in files if n.endswith("/provenance/selection.json"))
        selected = json.load(archive.extractfile(selected_name))
        contract_name = next(n for n in files if n.endswith("/provenance/target-contract.json"))
        target = json.load(archive.extractfile(contract_name))["source"]
        report_name = next(n for n in files if n.endswith("/confirmation-summary.json"))
        report = json.load(archive.extractfile(report_name))
        assert report["correctness_passed"]
        assert len(report["correctness"]) == 4 * len(selected["workloads"])
        assert all(g["passed"] for g in report["correctness"])
        for name, record in files.items():
            data = archive.extractfile(name).read()
            assert len(data) == record["bytes"], name
            assert hashlib.sha256(data).hexdigest() == record["sha256"], name
            parent = str(Path(name).parent)
            if name.endswith("/metadata.json"):
                d = json.loads(data)
                extra = d["extra"]
                assert all(extra.get(k) for k in (
                    "deployment_id", "checkpoint_sha256", "target_model",
                    "target_revision", "target_tensor_parallel_size",
                )), name
                assert d["harness_git_revision"] == source, name
                assert extra["target_model"] == target["hf_id"], name
                assert extra["target_revision"] == target["revision"], name
                assert int(extra["target_tensor_parallel_size"]) == 1, name
                role = d["variant"]["name"].split("-k")[0]
                assert role in ("base", "adapt"), name
                if role == "adapt":
                    assert extra["eagle_head"] == selected["head_dir"], name
                checkpoint_hashes[role].add(extra["checkpoint_sha256"])
                deployment_ids.add(extra["deployment_id"])
                assert d["workload_file_sha256"] == selected["corpus_sha256"][d["workload_file"]]
                assert d["cell"]["generation"] == {
                    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 1,
                    "max_tokens": None, "ignore_eos": False,
                }, name
                assert d["cell"]["concurrency"] in (1, 64, 256)
                cells.setdefault(parent, {})["metadata"] = d
            elif name.endswith("/requests.jsonl"):
                rows = [json.loads(line) for line in data.splitlines()]
                assert all(row["success"] and row["status_code"] == 200 for row in rows), name
                cells.setdefault(parent, {})["counts"] = collections.Counter(
                    row["prompt_id"] for row in rows
                )
            elif name.endswith("/summary.json"):
                cells.setdefault(parent, {})["summary"] = json.loads(data)
        assert len(cells) == 24 * len(selected["workloads"])
        assert len(deployment_ids) == 8
        assert all(len(hashes) == 1 for hashes in checkpoint_hashes.values())
        assert checkpoint_hashes["base"] != checkpoint_hashes["adapt"]
        total_requests = 0
        for path, cell in cells.items():
            metadata, summary = cell["metadata"], cell["summary"]
            requests = metadata["cell"]["requests"]
            prompts = metadata["prompt_ids"]
            assert requests % len(prompts) == 0, path
            assert cell["counts"] == {p: requests // len(prompts) for p in prompts}, path
            assert summary["requests"]["attempted"] == requests, path
            assert summary["requests"]["successful"] == requests, path
            assert summary["requests"]["failed"] == 0, path
            assert summary["warmup"]["failed"] == 0, path
            total_requests += requests
        result = {
            "archive_sha256": whole.hexdigest(), "verified_member_hashes": len(files),
            "measurement_cells": len(cells), "successful_measured_requests": total_requests,
            "full_prompt_passes_verified": True, "warmup_failures": 0,
            "frozen_source_revision": source, "corpus_hashes_verified": True,
            "required_eagle_provenance_verified": True,
            "target_model": target["hf_id"], "target_revision": target["revision"],
            "checkpoint_sha256": {k: sorted(v) for k, v in checkpoint_hashes.items()},
            "deployment_count": len(deployment_ids),
            "calibrated_correctness_gates_passed": len(report["correctness"]),
            "claim": "Integrity audit; original scientific policy and cell verdicts unchanged.",
        }
(folder / "integrity-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
