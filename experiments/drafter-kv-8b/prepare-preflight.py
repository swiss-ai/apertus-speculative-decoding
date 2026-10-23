"""Freeze one installed-runtime preflight spec using previously hashed inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from apertus_bench.drafter_kv_experiment import _name, digest, exclusive_json, now
from apertus_bench.node_sweep import build_plan


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--retry-set", type=_name)
    parser.add_argument("--stress", action="store_true",
                        help="replay the frozen validation chat cell at concurrency256")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    inventory = args.inventory.resolve()
    inputs = json.loads(inventory.read_text())
    if inputs["head"] != "off" or not inputs["weights_hashed"]:
        parser.error("runtime preflight requires the hashed DSpark-off inventory")
    source = inputs["corpora"]["correctness.jsonl"]["path"]
    env = {"VLLM_DRAFT_KV_TRACE": "{arm_dir}/draft-kv.jsonl",
           "VLLM_DRAFT_KV_CORRECTNESS_WORKLOADS": source}
    arms = []
    for name, policy, budget in [
        ("native-a", "native", 2048), ("native-b", "native", 2048),
        ("full-custom", "full", 2048), ("importance512", "importance", 512),
    ]:
        arms.append({
            "name": name, "method": "dspark", "drafter": inputs["head_path"], "k": 7,
            "env": {**env, "VLLM_APERTUS_DRAFT_KV_POLICY": policy,
                    "VLLM_APERTUS_DRAFT_KV_BUDGET": str(budget),
                    "VLLM_APERTUS_DRAFT_KV_GRAPHS": "1" if args.graphs else "0"},
            "metadata": {"kv_policy": name, "kv_phase": "preflight",
                         "checkpoint_sha256": inputs["checkpoint_sha256"],
                         "target_revision": inputs["target_contract"]["target"]["identity"]
                         ["revision"],
                         "inventory_sha256": digest(inventory)},
        })
    load_source = inputs["corpora"]["eagle-8b-validation.jsonl"]["path"] if args.stress else source
    spec = {
        "name": "drafter-kv-runtime-preflight", "stage": "8b", "node_gpus": 4,
        "probe": False,
        "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256,
                   "async_scheduling": False},
        "loadtests": [{"workload": w, "workloads_file": load_source,
                       "concurrencies": [256] if args.stress else [1, 8],
                       "min_requests": 1024 if args.stress else 2,
                       "requests_per_slot": 4 if args.stress else 1,
                       "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 1}}
                      for w in (["chat"] if args.stress else ["chat", "code", "summarization"])],
        "arms": arms,
        "drafter_kv": {"phase": "preflight", "inventory": str(inventory),
                       "inventory_sha256": digest(inventory), "corpora": inputs["corpora"]},
    }
    build_plan(spec, check_heads=False)
    directory = campaign / "specs/preflight"
    if args.graphs:
        directory = directory / "graphs"
    if args.retry_set:
        directory = directory / args.retry_set
    directory.mkdir(parents=True, exist_ok=False)
    # Predeclared identical retry specs preserve failed-run artifacts and commands.
    specs = []
    for attempt in range(1, 4):
        path = directory / f"runtime-{attempt:02d}.yaml"
        with path.open("x") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)
        specs.append({"path": str(path), "sha256": digest(path)})
    exclusive_json(directory / "generation.json", {
        "phase": "preflight", "generated_at": now(), "inventory": str(inventory),
        "inventory_sha256": digest(inventory), "specs": specs,
        "purpose": "installed allocator/state correctness and full-coverage backend control",
        "custom_query_graphs": args.graphs,
        "high_concurrency_regression": args.stress,
    })
    print(json.dumps(specs, indent=2))


if __name__ == "__main__":
    main()
