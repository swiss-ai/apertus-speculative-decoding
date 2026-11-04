"""Render the persistent DSpark deployment for the Breithorn GitOps repository.

This only writes local files. It does not submit Slurm jobs or contact Kubernetes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results/8b/speed/id-ood-20261030-c"
NAME = "vllm-apertus-v1p5-8b-dspark-e10"
IMAGE = (
    "ghcr.io/swiss-ai/vllm_apertus_1.5_release@"
    "sha256:8eb36e284fbefdd44c6f4dc23072eff9c06c1caed5643ce2f99ac7e0915a06ae"
)
BUSYBOX = (
    "docker.io/library/busybox@"
    "sha256:a7dbc319b1438115a207e11930dc4bb3e643dcb3ec0bdd694bb36cd8fad289ec"
)
# Production identity follows the current GitOps manifests. Its handshake still
# needs verification in-cluster; the corrected DEV identity must not be used here.
BOOTSTRAP = "/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"
NO_PROXY = "127.0.0.1,localhost,0.0.0.0,.svc,.cluster.local"
# Plain mode serves the same target from the same Deployment, only without the
# drafter, under its own ID so DSpark users never silently receive plain output.
MODEL_IDS = {
    "dspark": "SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3",
    "plain": "SwissAI-Research/swiss-ai/Apertus-v1.5-8B-plain-matched",
}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def render(
    output: Path,
    replicas: int,
    source_revision: str,
    variant: str = "dspark",
    pin_node: str | None = None,
):
    if variant not in MODEL_IDS:
        raise ValueError(f"unknown variant {variant!r}")
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads((BENCH / "profile.json").read_text())
    preflight = json.loads((BENCH / "preflight.json").read_text())
    asset_hashes = {f"target/{k}": v for k, v in preflight["target_files"].items()}
    asset_hashes.update({f"drafter/{k}": v for k, v in preflight["drafter"]["files"].items()})
    asset_hashes["opentela/otela-arm64"] = preflight["opentela_sha256"]
    asset_hashes["target/processor_config.json"] = (
        "6ac8e5d4654417f54c73ae7d5949371d1269785e34e11ec424682bb652367711"
    )
    asset_hashes["drafter/config.py"] = (
        "2f4381ff2eb9de506c6ba09bc17f83d0aee871f75651f6303b2e4b581e9b4ca1"
    )
    data = {
        "launch.py": (ROOT / "serving/kubernetes/launch.py").read_text(),
        "chat_template.jinja": (BENCH / "chat_template.jinja").read_text(),
    }
    overlays = {}
    for source in sorted((BENCH / "overlays/workspace/vllm").rglob("*.py")):
        relative = source.relative_to(BENCH / "overlays")
        key = "overlay-" + source.name
        data[key] = source.read_text()
        overlays[key] = "/" + str(relative)
    model_id = MODEL_IDS[variant]
    bare_id = model_id.removeprefix("SwissAI-Research/")
    spec = {
        "source_revision": source_revision,
        "benchmark_source_revision": "f586455fbad788f94e60ef5ba74892f112439fd0",
        "image": IMAGE,
        "vllm_revision": profile["runtime"]["vllm_revision"],
        "target_model": profile["target"]["model"],
        "target_revision": profile["target"]["revision"],
        "checkpoint_sha256": profile["drafter"]["checkpoint_sha256"],
        "model_id": model_id,
        "bootstrap_addr": BOOTSTRAP,
        "asset_hashes": asset_hashes,
        "runtime_hashes": {k: sha(v) for k, v in data.items()},
        "overlays": overlays,
        "engine_argv": [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "/assets/target",
            "--served-model-name",
            model_id,
            bare_id,
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-batched-tokens",
            "16384",
            "--max-num-seqs",
            "256",
            "--no-enable-prefix-caching",
            "--no-enable-log-requests",
            "--chat-template",
            "/runtime/chat_template.jinja",
            "--chat-template-content-format",
            "string",
            *speculative_argv(variant),
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ],
    }
    if variant != "dspark":
        spec["variant"] = variant
    data["deployment.json"] = json.dumps(spec, indent=2) + "\n"
    config_name = NAME + "-" + sha(json.dumps(data, sort_keys=True))[:12]

    def metadata(name):
        return {"name": name, "namespace": "rob-poc"}

    def write(name, obj):
        (output / name).write_text(yaml.safe_dump(obj, sort_keys=False, width=110))

    write(
        "configmap.yaml",
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata(config_name),
            "immutable": True,
            "data": data,
        },
    )
    write(
        "pvc.yaml",
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                **metadata(NAME + "-assets"),
                "annotations": {"argocd.argoproj.io/sync-options": "Prune=false"},
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": "ceph-corbo-cephfs",
                "resources": {"requests": {"storage": "80Gi"}},
            },
        },
    )
    write(
        "stage-pod.yaml",
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": metadata(NAME + "-stage"),
            "spec": {
                "restartPolicy": "Always",
                "automountServiceAccountToken": False,
                "containers": [
                    {
                        "name": "stage",
                        "image": BUSYBOX,
                        "command": ["sh", "-c", "sleep infinity"],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"memory": "512Mi"},
                        },
                        "volumeMounts": [{"name": "assets", "mountPath": "/assets"}],
                    }
                ],
                "volumes": [
                    {"name": "assets", "persistentVolumeClaim": {"claimName": NAME + "-assets"}}
                ],
            },
        },
    )
    probe = {"httpGet": {"path": "/health", "port": "http"}, "timeoutSeconds": 10}
    write(
        "deployment.yaml",
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata(NAME),
            "spec": {
                "replicas": replicas,
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": {"app": NAME}},
                "template": {
                    "metadata": {
                        "labels": {"app": NAME},
                        "annotations": {
                            "apertus.swissai/source-revision": source_revision,
                            "apertus.swissai/checkpoint-sha256": spec["checkpoint_sha256"],
                            **({"apertus.swissai/variant": variant} if variant != "dspark" else {}),
                        },
                    },
                    "spec": {
                        "restartPolicy": "Always",
                        "automountServiceAccountToken": False,
                        "terminationGracePeriodSeconds": 120,
                        "nodeSelector": {
                            "kubernetes.io/arch": "arm64",
                            **({"kubernetes.io/hostname": pin_node} if pin_node else {}),
                        },
                        "imagePullSecrets": [{"name": "ghcr-pull-secret"}],
                        "enableServiceLinks": False,
                        "tolerations": [
                            {"key": "hpc", "operator": "Equal", "value": "true", "effect": x}
                            for x in ("NoSchedule", "NoExecute")
                        ],
                        "containers": [
                            {
                                "name": "vllm",
                                "image": IMAGE,
                                "command": ["python3", "/runtime/launch.py"],
                                "ports": [{"name": "http", "containerPort": 8080}],
                                "resources": {
                                    "requests": {
                                        "nvidia.com/gpu": 1,
                                        "cpu": "4",
                                        "memory": "48Gi",
                                        "ephemeral-storage": "20Gi",
                                    },
                                    "limits": {"nvidia.com/gpu": 1, "memory": "96Gi"},
                                },
                                "startupProbe": {
                                    **probe,
                                    "periodSeconds": 10,
                                    "failureThreshold": 180,
                                },
                                "readinessProbe": {
                                    **probe,
                                    "periodSeconds": 10,
                                    "failureThreshold": 3,
                                },
                                "livenessProbe": {
                                    **probe,
                                    "periodSeconds": 30,
                                    "failureThreshold": 5,
                                },
                                "env": [
                                    # Platform-wide CVE-2026-98432 mitigation (rob-poc 8603558):
                                    # stop Go 1.24+ (the OpenTela binary) opening MPTCP listeners.
                                    {"name": "GODEBUG", "value": "multipathtcp=0"},
                                    *[
                                        {
                                            "name": key,
                                            "valueFrom": {"fieldRef": {"fieldPath": value}},
                                        }
                                        for key, value in {
                                            "POD_NAME": "metadata.name",
                                            "POD_UID": "metadata.uid",
                                            "POD_NAMESPACE": "metadata.namespace",
                                            "NODE_NAME": "spec.nodeName",
                                        }.items()
                                    ],
                                    *[
                                        {"name": key, "value": value}
                                        for key, value in {
                                            "HF_HUB_OFFLINE": "1",
                                            "TRANSFORMERS_OFFLINE": "1",
                                            "HF_HOME": "/tmp/hf",
                                            "PYTHONUNBUFFERED": "1",
                                            "NO_PROXY": NO_PROXY,
                                            "no_proxy": NO_PROXY,
                                        }.items()
                                    ],
                                ],
                                "volumeMounts": [
                                    {"name": "assets", "mountPath": "/assets", "readOnly": True},
                                    {"name": "assets", "mountPath": "/logs", "subPath": "runs"},
                                    {"name": "runtime", "mountPath": "/runtime", "readOnly": True},
                                    {"name": "dshm", "mountPath": "/dev/shm"},
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "assets",
                                "persistentVolumeClaim": {"claimName": NAME + "-assets"},
                            },
                            {"name": "runtime", "configMap": {"name": config_name}},
                            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                        ],
                    },
                },
            },
        },
    )
    write(
        "service.yaml",
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata(NAME),
            "spec": {
                "type": "ClusterIP",
                "selector": {"app": NAME},
                "ports": [{"name": "http", "port": 8080, "targetPort": "http"}],
            },
        },
    )
    (output / "deployment-provenance.json").write_text(json.dumps(spec, indent=2) + "\n")


def speculative_argv(variant):
    if variant == "plain":
        return []
    config = {"method": "dspark", "model": "/assets/drafter", "num_speculative_tokens": 3}
    return ["--speculative-config", json.dumps(config, sort_keys=True)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicas", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--variant",
        choices=sorted(MODEL_IDS),
        default="dspark",
        help="plain drops the drafter and advertises its own model ID (speed baseline)",
    )
    parser.add_argument(
        "--pin-node",
        help="add a kubernetes.io/hostname node selector so every toggle runs on one node",
    )
    args = parser.parse_args()
    if (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        == "main"
    ):
        raise SystemExit("use a task branch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    render(args.output, args.replicas, revision, args.variant, args.pin_node)
