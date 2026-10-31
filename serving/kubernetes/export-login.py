"""Export the current Breithorn login privately; creates no Kubernetes resources.

Run on the laptop where kubectl already works. Reads only the selected context,
executes its existing credential plugin, and writes a self-contained kubeconfig.
The resulting session may expire; it is not a permanent machine credential.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = "https://breithorn.svc.cscs.ch:6443"


def run(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, **kwargs)
    if result.returncode:
        # Authentication plugin error messages can include sensitive arguments.
        raise RuntimeError(
            "Could not obtain the login. First run 'kubectl -n rob-poc get pods' "
            "to log in, then retry. Raw authentication output has been suppressed."
        )
    return json.loads(result.stdout)


def credentials(user, cluster, interactive):
    if "exec" in user:
        plugin = user["exec"]
        info = {
            "apiVersion": plugin["apiVersion"],
            "kind": "ExecCredential",
            "spec": {"interactive": interactive},
        }
        if plugin.get("provideClusterInfo"):
            info["spec"]["cluster"] = cluster
        env = dict(os.environ)
        env.update({item["name"]: item["value"] for item in plugin.get("env", [])})
        env["KUBERNETES_EXEC_INFO"] = json.dumps(info)
        result = run([plugin["command"], *plugin.get("args", [])], env=env, timeout=180)
        status = result["status"]
        if status.get("token"):
            auth = {"token": status["token"]}
        elif status.get("clientCertificateData") and status.get("clientKeyData"):
            auth = {
                "client-certificate-data": base64.b64encode(
                    status["clientCertificateData"].encode()
                ).decode(),
                "client-key-data": base64.b64encode(status["clientKeyData"].encode()).decode(),
            }
        else:
            raise ValueError("The existing credential plugin returned no usable credentials")
        return auth, status.get("expirationTimestamp", "not reported by the login plugin")
    if user.get("token"):
        return {"token": user["token"]}, "not reported by the existing config"
    if user.get("client-certificate-data") and user.get("client-key-data"):
        return {key: user[key] for key in ("client-certificate-data", "client-key-data")}, (
            "see the existing client certificate"
        )
    raise ValueError("Unsupported login type; no credential was exported")


def export(context, output):
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory is missing: {output.parent}")
    if output.resolve() == (Path.home() / ".kube/config").resolve():
        raise ValueError("The export must not replace your default kubeconfig")
    if output.is_symlink():
        raise ValueError("The export destination must not be a symbolic link")
    original = run(
        [
            "kubectl",
            "--context",
            context,
            "config",
            "view",
            "--minify",
            "--flatten",
            "--raw",
            "-o",
            "json",
        ],
        timeout=30,
    )
    cluster = original["clusters"][0]["cluster"]
    if cluster["server"].rstrip("/") != SERVER:
        raise ValueError(f"Select the Breithorn context for {SERVER}")
    if cluster.get("insecure-skip-tls-verify") or not cluster.get("certificate-authority-data"):
        raise ValueError("The Breithorn config must contain its CA and verify TLS")
    auth, expiry = credentials(original["users"][0]["user"], cluster, sys.stdin.isatty())
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": "breithorn", "cluster": cluster}],
        "users": [{"name": "breithorn-session", "user": auth}],
        "contexts": [
            {
                "name": "breithorn-oidc",
                "context": {
                    "cluster": "breithorn",
                    "user": "breithorn-session",
                    "namespace": "rob-poc",
                },
            }
        ],
        "current-context": "breithorn-oidc",
    }
    fd, temporary = tempfile.mkstemp(prefix=".breithorn-session-", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(config, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Saved private kubeconfig: {output}")
    print(f"Session expires: {expiry}")
    print("File mode: 0600. No token printed. No Kubernetes resources changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="breithorn-oidc")
    parser.add_argument("--output", type=Path, default=Path.home() / "breithorn-session.kubeconfig")
    args = parser.parse_args()
    try:
        export(args.context, args.output.expanduser())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.TimeoutExpired):
            raise SystemExit(
                "Login timed out. Run kubectl get pods to log in, then retry."
            ) from None
        raise SystemExit(str(error)) from None
