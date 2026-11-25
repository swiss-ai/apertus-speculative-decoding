"""Administrator-run setup for reusable, namespace-scoped Breithorn access.

Run only after the administrator approves a long-lived ServiceAccount credential.
Creates access objects, never model workloads or cluster-wide permissions.
Writes a private kubeconfig; never prints credentials or uses tokens in argv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

NAME = "apertus-dspark-agent"
NAMESPACE = "rob-poc"
SERVER = "https://breithorn.svc.cscs.ch:6443"


def kubectl(context, args, payload=None, check=True):
    result = subprocess.run(
        ["kubectl", "--context", context, "--request-timeout=20s", *args],
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        # An exec authentication plugin can emit sensitive arguments in errors.
        # Do not forward raw stderr or an API response containing a Secret.
        raise RuntimeError(
            f"kubectl failed during {' '.join(args[:3])}. "
            "Check your context, login and permissions. No credentials printed."
        )
    return result


def access_objects():
    def metadata(name):
        return {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"app.kubernetes.io/managed-by": "apertus-dspark-access-setup"},
        }

    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": metadata(NAME),
                "automountServiceAccountToken": False,
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": metadata(NAME),
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": [
                            "pods",
                            "services",
                            "endpoints",
                            "configmaps",
                            "persistentvolumeclaims",
                            "events",
                            "resourcequotas",
                            "limitranges",
                        ],
                        "verbs": ["get", "list", "watch"],
                    },
                    {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
                    {
                        "apiGroups": [""],
                        "resources": ["pods/exec"],
                        "verbs": ["create", "get"],
                    },
                    {
                        "apiGroups": ["apps"],
                        "resources": ["deployments", "replicasets", "statefulsets"],
                        "verbs": ["get", "list", "watch"],
                    },
                    {
                        "apiGroups": ["batch"],
                        "resources": ["jobs"],
                        "verbs": ["get", "list", "watch"],
                    },
                    {
                        "apiGroups": ["argoproj.io"],
                        "resources": ["applications"],
                        "verbs": ["get", "list", "watch"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": metadata(NAME),
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": NAME,
                },
                "subjects": [{"kind": "ServiceAccount", "name": NAME, "namespace": NAMESPACE}],
            },
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    **metadata(NAME + "-token"),
                    "annotations": {"kubernetes.io/service-account.name": NAME},
                },
                "type": "kubernetes.io/service-account-token",
            },
        ],
    }


def make_config(cluster, secret):
    import base64

    if cluster.get("insecure-skip-tls-verify"):
        raise ValueError("Use an administrator context with TLS verification enabled")
    ca = cluster.get("certificate-authority-data") or secret["data"].get("ca.crt")
    if not ca:
        raise ValueError("Cluster CA missing; obtain the Breithorn CA from your administrator")
    target_cluster = {"server": SERVER, "certificate-authority-data": ca}
    if cluster.get("tls-server-name"):
        target_cluster["tls-server-name"] = cluster["tls-server-name"]
    token = base64.b64decode(secret["data"]["token"], validate=True).decode()
    if not token.strip():
        raise ValueError("The ServiceAccount token has not been populated")
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": "breithorn", "cluster": target_cluster}],
        "users": [{"name": NAME, "user": {"token": token}}],
        "contexts": [
            {
                "name": "breithorn-oidc",
                "context": {
                    "cluster": "breithorn",
                    "user": NAME,
                    "namespace": NAMESPACE,
                },
            }
        ],
        "current-context": "breithorn-oidc",
    }


def setup(context, output, approved):
    if not approved:
        raise ValueError("Administrator must explicitly approve the long-lived credential")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}; it will not be overwritten")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    # Capture only, never print: the original config can contain admin credentials.
    original = json.loads(
        kubectl(context, ["config", "view", "--minify", "--flatten", "--raw", "-o", "json"]).stdout
    )
    cluster = original["clusters"][0]["cluster"]
    if cluster["server"].rstrip("/") != SERVER:
        raise ValueError(f"Wrong cluster. Select an administrator context for {SERVER}")
    if cluster.get("insecure-skip-tls-verify") or not cluster.get("certificate-authority-data"):
        raise ValueError("The admin context must include the cluster CA and verify TLS")
    print(f"Checking permissions in {context}, namespace {NAMESPACE}...", flush=True)
    for verb, resource in [
        ("create", "serviceaccounts"),
        ("create", "roles.rbac.authorization.k8s.io"),
        ("create", "rolebindings.rbac.authorization.k8s.io"),
        ("create", "secrets"),
        ("get", "secrets"),
    ]:
        result = kubectl(
            context, ["-n", NAMESPACE, "auth", "can-i", verb, resource, "--quiet"], check=False
        )
        if result.returncode:
            raise PermissionError(
                f"Permission check failed: {verb} {resource} in {NAMESPACE}. "
                "No access objects created. Use a Breithorn administrator account."
            )
    print("Creating dedicated ServiceAccount, Role, RoleBinding and token Secret...", flush=True)
    kubectl(context, ["-n", NAMESPACE, "apply", "-f", "-"], access_objects())
    for _ in range(30):
        secret = json.loads(
            kubectl(
                context, ["-n", NAMESPACE, "get", "secret", NAME + "-token", "-o", "json"]
            ).stdout
        )
        if secret.get("data", {}).get("token"):
            config = make_config(cluster, secret)
            fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(config, indent=2) + "\n")
            print(f"Created private kubeconfig: {output}")
            print("File mode: 0600. Credentials have not been printed.")
            return
        time.sleep(1)
    raise TimeoutError("Token controller did not populate the Secret; no kubeconfig written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, help="existing administrator context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-long-lived-token", action="store_true")
    args = parser.parse_args()
    try:
        setup(args.context, args.output.expanduser(), args.approve_long_lived_token)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error)) from None
