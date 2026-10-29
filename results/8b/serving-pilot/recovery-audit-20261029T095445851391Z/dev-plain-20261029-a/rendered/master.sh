#!/bin/bash
#SBATCH --job-name=sml_faruk_zahiragic_swiss-ai_Apertus-v1.5-8B-plain_IDxgwIhP
#SBATCH --account=infra01
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err
# shellcheck shell=bash

set -euo pipefail
mkdir -p "logs/${SLURM_JOB_ID}"

critical_pids=()
vmagent_pid=""
health_checker_pid=""

# Self-extract rank scripts: this master.sh was submitted standalone
# (no sibling files), so we materialise the rank scripts under HOME
# (shared FS, visible to all compute nodes) at job start time. The
# single-quoted heredoc keeps each body literal.

RANKS_DIR="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/ranks-${SLURM_JOB_ID}"

mkdir -p "$RANKS_DIR"

cat > "$RANKS_DIR/head.sh" <<'__SML_HEAD_EOF__'
#!/bin/bash
# shellcheck disable=SC2046,SC2086
set -ex

export RAY_CGRAPH_get_timeout=1800
export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

# User-supplied pre-launch commands
echo 'Running pre-launch commands...'
export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/source/src:"${PYTHONPATH:-}"
export OPENTELA_BIN=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/bin/otela-arm64
python3 -m apertus_bench.research_pilot preflight /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a

# shellcheck disable=SC2034
replica_head_ip="$1"

$OPENTELA_BIN start \
    --bootstrap.addr "/ip4/148.187.108.177/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9" \
    --service.name llm \
    --service.port 8080 \
    --label launched_by=$USER \
    --label slurm_job_id=$SLURM_JOB_ID \
    --label slurm_partition=${SLURM_JOB_PARTITION:-unknown} \
    --label worker_group_id=$SLURM_JOB_ID \
    --label framework=vllm \
    --label served_model_name=faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain \
    --label 'framework_args=--port 8080 --model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B --served-model-name faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain --chat-template-content-format string --tensor-parallel-size 1 --gpu-memory-utilization 0.8 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser apertus --compilation-config.pass_config.fuse_allreduce_rms false --chat-template /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/chat_template.jinja --no-enable-log-requests --dtype bfloat16 --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 256' \
    --label started_at=$(date -u +%FT%TZ) \
    --label expires_at=$(date -u -d "+5400 seconds" +%FT%TZ) \
    --subprocess "python3 -m vllm.entrypoints.openai.api_server --port 8080 --model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B     --served-model-name faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain     --chat-template-content-format string     --tensor-parallel-size 1     --gpu-memory-utilization 0.8     --max-model-len 32768     --enable-auto-tool-choice     --tool-call-parser apertus     --compilation-config.pass_config.fuse_allreduce_rms false     --chat-template /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/chat_template.jinja --no-enable-log-requests --dtype bfloat16 --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 256"
__SML_HEAD_EOF__

chmod +x "$RANKS_DIR/head.sh"

curl -sf -X POST "https://sml-dev.swissai.svc.cscs.ch/launches" \
    -H "Content-Type: application/json" \
    -d '{"user": "'"${SLURM_JOB_USER}"'", "job_id": "'"${SLURM_JOB_ID}"'", "slurm_nodes": '"${SLURM_NNODES}"', "slurm_job_name": "'"${SLURM_JOB_NAME}"'", "slurm_partition": "'"${SLURM_JOB_PARTITION}"'", "slurm_time": "01:30:00", "slurm_account": "'"${SLURM_JOB_ACCOUNT}"'", "slurm_environment": "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/environments/env_resolved_arm64_yimEQI.toml", "interactive": false, "serving_framework": "vllm", "framework_args": "--port 8080 --model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B     --served-model-name faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain     --chat-template-content-format string     --tensor-parallel-size 1     --gpu-memory-utilization 0.8     --max-model-len 32768     --enable-auto-tool-choice     --tool-call-parser apertus     --compilation-config.pass_config.fuse_allreduce_rms false     --chat-template /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/chat_template.jinja --no-enable-log-requests --dtype bfloat16 --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 256", "pre_launch_cmds": "export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/source/src:"${PYTHONPATH:-}"
export OPENTELA_BIN=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/bin/otela-arm64
python3 -m apertus_bench.research_pilot preflight /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a", "model_name": "faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain", "replicas": 1, "nodes_per_replica": 1, "framework_port": 8080, "use_router": false, "router_environment": "/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/environments/env_resolved_arm64_yimEQI.toml", "router_port": 30000, "router_args": "", "ocf_enabled": true, "ocf_bootstrap_addr": "/ip4/148.187.108.177/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9", "ocf_service_name": "llm", "ocf_service_port": 8080, "model_launch_version": "26.5.0a1"}' || true

unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE

ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Running on ARM64 (aarch64)"
    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
    export OPENTELA_BIN=/opentelabin/dev/otela-arm64
    SML_ARCH=arm64
    metrics_agent_bin="/capstor/store/cscs/swissai/infra01/opentela-share/vmagent-arm64"
    dcgm_exporter_bin="/capstor/store/cscs/swissai/infra01/opentela-share/dcgm-exporter-arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Running on x86_64"
    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/
    export OPENTELA_BIN=/opentelabin/dev/otela-amd64
    SML_ARCH=amd64
    metrics_agent_bin="/capstor/store/cscs/swissai/infra01/opentela-share/vmagent-amd64"
    dcgm_exporter_bin="/capstor/store/cscs/swissai/infra01/opentela-share/dcgm-exporter-amd64"
else
    echo "Unknown architecture: $ARCH" >&2
    exit 1
fi

SML_ENV_FILE="/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-plain-20261029-a/environments/env_resolved_arm64_yimEQI.toml"
if grep -q "{arch}" "$SML_ENV_FILE"; then
    SML_RESOLVED_ENV="${PWD}/env_resolved_${SLURM_JOB_ID}_${SML_ARCH}.toml"
    sed "s|{arch}|${SML_ARCH}|g" "$SML_ENV_FILE" > "$SML_RESOLVED_ENV"
    SML_ENV_FILE="$SML_RESOLVED_ENV"
    echo "Resolved env file for ${SML_ARCH}: $SML_ENV_FILE"
fi

# A missing image is otherwise reported by pyxis as an opaque failure
# deep inside srun; fail here with the path that was actually wanted.
SML_IMAGE=$(sed -n 's|^ *image *= *"\(/[^"]*\)".*|\1|p' "$SML_ENV_FILE" | head -1)
if [[ -n "$SML_IMAGE" && ! -e "$SML_IMAGE" ]]; then
    echo "Container image not found: $SML_IMAGE" >&2
    echo "  (from env file $SML_ENV_FILE, arch ${SML_ARCH})" >&2
    exit 1
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")
TOTAL_NODES=${#nodes[@]}

echo "Total nodes allocated: $TOTAL_NODES"
for i in "${!nodes[@]}"; do
    echo "Node $i: ${nodes[$i]}"
done

# ── replica 0 head IP ─────────────────────────────────────────────
replica_0_head_node=${nodes[0]}
replica_0_head_ip=$(srun --nodes=1 --ntasks=1 -w "$replica_0_head_node" hostname -i)
if [[ -z "$replica_0_head_ip" ]]; then
    echo "Error: Could not retrieve IP for replica 0 host $replica_0_head_node" >&2
    exit 1
fi
echo "Replica 0 head IP: $replica_0_head_ip"

echo "All replica URLs: http://$replica_0_head_ip:8080"  # NOSONAR

# replica 0, rank 0 (head)
srun --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
    --container-writable \
    --container-mounts="$RANKS_DIR:$RANKS_DIR" \
    --environment="${SML_ENV_FILE}" \
    --output="logs/${SLURM_JOB_ID}/replica_0.out" \
    --error="logs/${SLURM_JOB_ID}/replica_0.err" \
    bash "$RANKS_DIR/head.sh" "$replica_0_head_ip" &
critical_pids+=($!)

# vmagent runs on the batch node; pyxis containers share the host network
# namespace so the framework API server is reachable at localhost:8080.
# vmagent is non-critical: disowned so it's not in `wait -n`'s scope, and
# the EXIT trap in the footer kills it when master.sh terminates so the
# allocation can be released as soon as the framework process is gone.
if [[ -x "$metrics_agent_bin" ]]; then
    if [[ -e /dev/nvidia0 && -x "$dcgm_exporter_bin" ]]; then
        "$dcgm_exporter_bin" \
            --address 0.0.0.0:9400 \
            -f /capstor/store/cscs/swissai/infra01/opentela-share/default-counters.csv \
            > "/tmp/dcgm-exporter-${SLURM_JOB_ID}.log" 2>&1 &
        disown $!
    else
        echo "dcgm-exporter: no NVIDIA GPU or binary not found, skipping" >&2
    fi
    "$metrics_agent_bin" \
        -promscrape.config=/capstor/store/cscs/swissai/infra01/opentela-share/vmagent-scrape.yaml \
        -remoteWrite.url="https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write" \
        -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \
        -remoteWrite.label="model=faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-plain" \
        -remoteWrite.label="framework=vllm" \
        -remoteWrite.label="user=${SLURM_JOB_USER}" \
        -remoteWrite.label="node=$(hostname)" \
        "-remoteWrite.tmpDataPath=/tmp/vmagent-data-${SLURM_JOB_ID}" \
        > "/tmp/vmagent-${SLURM_JOB_ID}.log" 2>&1 &
    vmagent_pid=$!
    disown "$vmagent_pid"
else
    echo "metrics: $metrics_agent_bin not found, skipping push" >&2
fi

# ── replica health checker ───────────────────────────────────────────────
# Background loop on the batch node (it shares the job's internal network),
# probing each replica's framework /health directly and writing an atomic
# JSON report the CLI reads. Non-critical: disowned (out of `wait -n`'s
# scope) and killed by the EXIT trap when master.sh ends. mkdir + the
# python3 guard make startup robust and leave a breadcrumb in the job log.
mkdir -p "logs/${SLURM_JOB_ID}"
cat > "$RANKS_DIR/replica_health_checker.py" <<'__SML_HEALTH_CHECKER_EOF__'
"""In-job replica health checker.

Runs on the batch node of the model's *own* SLURM job (its source is embedded
into the rendered ``master.sh``). The batch node shares the job's internal
network, so this checks every replica directly — no OpenTela bootstrap, no p2p
routing, no API key. Each cycle it:

  * GETs ``http://<replica-head-ip>:<framework-port>/health`` for each replica,
  * once a replica is healthy, resolves its OpenTela peer id by running
    ``/v1/self`` *on that node* via a one-off ``srun`` step (the endpoint is
    localhost-only, and the DNT table carries no internal IP to match against),
  * records the last time the replica was seen HEALTHY,

then writes a JSON report **atomically** (temp file + ``os.replace``) to
``SML_HEALTH_REPORT_PATH`` so a concurrent reader (local or via FirecREST
download) never observes a half-written file. Loops every ``SML_HEALTH_INTERVAL``
seconds.

Stdlib only — runs under whatever ``python3`` the node provides, without the SML
package installed. HPC batch hosts can ship a Python as old as 3.6, so this file
must stay 3.6-compatible: NO ``from __future__ import annotations`` (3.7+), NO
PEP 604 ``X | Y`` unions or builtin-generic subscripts (``list[str]``) in
evaluated positions, and NO ``subprocess.run(capture_output=/text=)`` (3.7+).
Use ``typing`` aliases and ``stdout=/stderr=PIPE`` + ``universal_newlines`` instead.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Peer ids are stable, so stop retrying resolution after a few attempts to avoid
# launching a job step every cycle when /v1/self is unreachable.
_MAX_PEER_ATTEMPTS = 5
_SRUN_TIMEOUT_SECONDS = 30.0
_SCANCEL_TIMEOUT_SECONDS = 30.0


def cancel_previous_job(job_id):
    # type: (str) -> bool
    """scancel the predecessor in a consecutive chain. Returns True on success.

    Run from the batch node once this job's replicas are all healthy, so the
    handover releases the old allocation. Best-effort: a failure (already gone,
    scancel missing) is reported and simply retried next cycle.
    """
    cmd = ["scancel", job_id]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=_SCANCEL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write("replica health checker: scancel %s failed: %s\n" % (job_id, exc))
        sys.stderr.flush()
        return False
    if result.returncode != 0:
        sys.stderr.write(
            "replica health checker: scancel %s exited %s: %s\n" % (job_id, result.returncode, result.stderr.strip())
        )
        sys.stderr.flush()
        return False
    return True


def _http_get(url: str, timeout: float) -> Optional[Tuple[int, bytes]]:
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - fixed internal http endpoint
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return (response.getcode() or 0), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def check_health(node_ip: str, framework_port: int, timeout: float) -> str:
    """Probe a replica's framework /health. Returns a ``ModelHealth`` string."""
    result = _http_get(f"http://{node_ip}:{framework_port}/health", timeout)
    if result is None:
        return "ERROR"
    status, _ = result
    return "HEALTHY" if 200 <= status < 300 else "NOT_RESPONDING"


def _self_fetch_code(opentela_port: int) -> str:
    # Executed on the target node: /v1/self is localhost-only and returns the
    # node's own Peer record, whose "id" is the libp2p peer id.
    return (
        "import json, urllib.request; "
        f"print(json.load(urllib.request.urlopen('http://localhost:{opentela_port}"
        "/v1/self', timeout=5)).get('id') or '')"
    )


def resolve_peer_id(host: str, opentela_port: int, timeout: float) -> Optional[str]:
    """Fetch a node's own OpenTela peer id by querying ``/v1/self`` on that node.

    ``/v1/self`` only answers to localhost, so we run a one-off ``srun --overlap``
    step on the node itself (sharing the live allocation). Returns ``None`` on any
    failure (no host, srun unavailable, timeout, OpenTela not up).
    """
    if not host:
        return None
    cmd = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={host}",
        "python3",
        "-c",
        _self_fetch_code(opentela_port),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    peer_id = result.stdout.strip()
    return peer_id or None


def build_report(
    replica_ips: List[str],
    replica_hosts: List[str],
    nodes_per_replica: int,
    framework_port: int,
    opentela_port: int,
    timeout: float,
    peer_ids: Dict[int, str],
    peer_attempts: Dict[int, int],
    last_seen: Dict[int, int],
    now: int,
) -> Dict[str, Any]:
    """Probe every replica once and assemble the report dict.

    The cache dicts are mutated in place across cycles: ``peer_ids`` is resolved
    once a replica is healthy (peer ids are stable), ``last_seen`` freezes at the
    last HEALTHY observation so a downed replica's "ago" keeps growing, and a
    replica that has never been healthy reads as ``NOT_DEPLOYED`` (still starting)
    rather than a failure.
    """
    replicas: List[Dict[str, Any]] = []
    for index, node_ip in enumerate(replica_ips):
        host = replica_hosts[index] if index < len(replica_hosts) else ""
        health = check_health(node_ip, framework_port, timeout)
        if health == "HEALTHY":
            last_seen[index] = now
            if index not in peer_ids and peer_attempts.get(index, 0) < _MAX_PEER_ATTEMPTS:
                peer_attempts[index] = peer_attempts.get(index, 0) + 1
                resolved = resolve_peer_id(host, opentela_port, _SRUN_TIMEOUT_SECONDS)
                if resolved:
                    peer_ids[index] = resolved
        elif index not in last_seen:
            health = "NOT_DEPLOYED"
        replicas.append(
            {
                "node_rank": index * nodes_per_replica,
                "node_ip": node_ip,
                # The replica head's SLURM node name (e.g. "nid001234"). The CLI
                # uses it to open an interactive shell on the node via
                # `srun --overlap --nodelist=<host> --pty bash`.
                "node_host": host or None,
                "peer_id": peer_ids.get(index),
                "health": health,
                "last_seen": last_seen.get(index),
            }
        )
    return {"checked_at": now, "replicas": replicas}


def all_replicas_healthy(report, expected):
    # type: (Dict[str, Any], int) -> bool
    """Whether every expected replica is present and HEALTHY in this report.

    Guards against ``expected == 0`` (an empty/unset replica list): with no
    replicas the ``all(...)`` below is vacuously true, which would otherwise let
    the handover cancel the predecessor immediately and drop the only allocation.
    """
    if expected <= 0:
        return False
    replicas = report.get("replicas", [])
    return len(replicas) >= expected and all(r.get("health") == "HEALTHY" for r in replicas)


def write_report_atomically(report: Dict[str, Any], report_path: str) -> None:
    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{report_path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(report, handle)
    os.replace(tmp_path, report_path)  # atomic on POSIX (same directory)


def main() -> int:
    report_path = os.environ["SML_HEALTH_REPORT_PATH"]
    framework_port = int(os.environ.get("SML_HEALTH_FRAMEWORK_PORT") or "8080")
    opentela_port = int(os.environ.get("SML_HEALTH_OPENTELA_PORT") or "8092")
    interval = float(os.environ.get("SML_HEALTH_INTERVAL") or "30")
    timeout = float(os.environ.get("SML_HEALTH_TIMEOUT") or "10")
    nodes_per_replica = int(os.environ.get("SML_HEALTH_NODES_PER_REPLICA") or "1")
    replica_ips = os.environ.get("SML_HEALTH_REPLICA_IPS", "").split()
    replica_hosts = os.environ.get("SML_HEALTH_REPLICA_HOSTS", "").split()
    # Predecessor in a consecutive chain to cancel once this job is fully healthy.
    previous_job_id = os.environ.get("SML_HEALTH_PREVIOUS_JOB_ID", "").strip()

    peer_ids: Dict[int, str] = {}
    peer_attempts: Dict[int, int] = {}
    last_seen: Dict[int, int] = {}
    handover_done = False
    expected = len(replica_ips)
    while True:
        # Never let a transient failure (e.g. a momentary write error) kill the
        # checker — log it and keep going so the report self-heals next cycle.
        try:
            report = build_report(
                replica_ips,
                replica_hosts,
                nodes_per_replica,
                framework_port,
                opentela_port,
                timeout,
                peer_ids,
                peer_attempts,
                last_seen,
                int(time.time()),
            )
            write_report_atomically(report, report_path)
            # Handover: cancel the predecessor only once every expected replica is
            # serving here, so the chain never drops below a healthy allocation.
            if previous_job_id and not handover_done and all_replicas_healthy(report, expected):
                sys.stdout.write(
                    "replica health checker: all %d replicas healthy; cancelling predecessor job %s\n"
                    % (expected, previous_job_id)
                )
                sys.stdout.flush()
                handover_done = cancel_previous_job(previous_job_id)
        except Exception as exc:  # resilience: keep the loop alive on any error
            sys.stderr.write(f"replica health checker: iteration failed: {exc}\n")
            sys.stderr.flush()
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
__SML_HEALTH_CHECKER_EOF__
if command -v python3 >/dev/null 2>&1; then
    SML_HEALTH_REPORT_PATH="logs/${SLURM_JOB_ID}/replica_health.json" \
        SML_HEALTH_FRAMEWORK_PORT=8080 \
        SML_HEALTH_OPENTELA_PORT=8092 \
        SML_HEALTH_INTERVAL=30 \
        SML_HEALTH_TIMEOUT=10 \
        SML_HEALTH_NODES_PER_REPLICA=1 \
        SML_HEALTH_REPLICA_IPS="$replica_0_head_ip" \
        SML_HEALTH_REPLICA_HOSTS="${nodes[0]}" \
        SML_HEALTH_PREVIOUS_JOB_ID="" \
        python3 "$RANKS_DIR/replica_health_checker.py" > "logs/${SLURM_JOB_ID}/replica_health_checker.log" 2>&1 &
    health_checker_pid=$!
    disown "$health_checker_pid"
    echo "Replica health checker started (pid $health_checker_pid) -> logs/${SLURM_JOB_ID}/replica_health.json"
else
    echo "python3 not found on the batch node; replica health checker disabled." >&2
fi

echo
echo "To connect to the host node:"
echo "srun --jobid $SLURM_JOB_ID -w ${nodes[0]} --overlap --pty bash"

echo
echo "Make sure to cancel the job at the end:"
echo "scancel $SLURM_JOB_ID"

cleanup() {
    if [[ -n "$vmagent_pid" ]]; then
        kill "$vmagent_pid" 2>/dev/null || true
    fi
    if [[ -n "$health_checker_pid" ]]; then
        kill "$health_checker_pid" 2>/dev/null || true
    fi
    if (( ${#critical_pids[@]} > 0 )); then
        kill "${critical_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

rc=0
wait -n || rc=$?
echo "Master finished at $(date) with code $rc"
exit "$rc"
