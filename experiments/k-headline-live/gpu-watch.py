"""Wait, without reserving anything, for a free GH200 GPU on a node nobody else is testing on.

The operator can read only the rob-poc namespace (no node objects), so the check
is an estimate from the namespace's own pods:
- the GPU node pool is the nodes that passed the scheduler's taint and selector
  filters for the e10 Deployment on 2026-11-25 (nid006583/594/597/599) plus any node
  a rob-poc GPU pod runs on later;
- nodes running a protected pod (Rob's fixed-nov23 and Chonk deployments, by
  app label) are excluded and re-read on every check;
- a node is a candidate if the rob-poc pods on it request at most GPUS_PER_NODE - 1 GPUs.
Pods of other namespaces are invisible, so a candidate is confirmed only when the
real pod schedules; if it stays Pending, the caller scales back to zero.

Every check is appended to --log (JSON lines). Exit 0 with the candidate on stdout,
3 when kubectl fails (expired login), 4 when --hours pass without a candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

POOL = ("nid006583", "nid006594", "nid006597", "nid006599")
GPUS_PER_NODE = 4
PROTECTED = re.compile(r"^(vllm-apertus-v1p5-(?!8b-dspark-e10$).+|vllm-apertus2-chonk.*)$")
OURS = {"vllm-apertus-v1p5-8b-dspark-e10", "task-speed-client"}


def gpus(pod: dict) -> int:
    return sum(
        int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
        for c in pod["spec"]["containers"]
    )


def check(pods: list[dict], pool=POOL) -> dict:
    used: dict[str, int] = {}
    protected: dict[str, list[str]] = {}
    nodes = set(pool)
    for pod in pods:
        node = pod["spec"].get("nodeName")
        phase = pod["status"].get("phase")
        if not node or phase in ("Succeeded", "Failed"):
            continue
        app = pod["metadata"].get("labels", {}).get("app", "")
        if PROTECTED.match(app):
            protected.setdefault(node, []).append(pod["metadata"]["name"])
        if gpus(pod):
            nodes.add(node)
            if app not in OURS:
                used[node] = used.get(node, 0) + gpus(pod)
    candidates = sorted(
        n for n in nodes if n not in protected and used.get(n, 0) <= GPUS_PER_NODE - 1
    )
    return {"protected": protected, "gpus_used": used, "candidates": candidates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval-minutes", type=float, default=15)
    parser.add_argument("--hours", type=float, default=48)
    args = parser.parse_args()
    deadline = time.monotonic() + 3600 * args.hours
    while True:
        command = ["kubectl", "--context", "breithorn-oidc", "-n", "rob-poc", "get", "pods"]
        try:
            raw = subprocess.run(
                [*command, "-o", "json"], capture_output=True, text=True, timeout=120, check=True
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"kubectl failed: {getattr(error, 'stderr', error)}", file=sys.stderr)
            return 3
        result = {"time": datetime.now(UTC).isoformat(), **check(json.loads(raw)["items"])}
        with open(args.log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if result["candidates"]:
            print(json.dumps(result))
            return 0
        if time.monotonic() > deadline:
            return 4
        time.sleep(60 * args.interval_minutes)


if __name__ == "__main__":
    sys.exit(main())
