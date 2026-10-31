"""kubectl exec credential helper for an approved Breithorn device login.

Credentials stay in a private file outside Git. Refresh is unattended while the
identity provider permits it. This does not create users, RBAC or workloads.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def expiration(token):
    # This only schedules refresh. The Kubernetes API verifies the JWT signature
    # and authorization; parsing claims here does not establish trust.
    encoded = token.split(".")[1]
    return int(json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))["exp"])


def save(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".breithorn-credential-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def credential(path):
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = json.loads(path.read_text())
        if saved["issuer"] != "https://dex.breithorn.svc.cscs.ch":
            raise ValueError("Unexpected credential issuer")
        if expiration(saved["id_token"]) <= time.time() + 60:
            if not saved.get("refresh_token"):
                raise ValueError("Breithorn browser login must be renewed")
            data = urllib.parse.urlencode(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": saved["refresh_token"],
                    "client_id": saved["client_id"],
                    "client_secret": saved["client_secret"],
                }
            ).encode()
            request = urllib.request.Request(saved["issuer"] + "/token", data=data)
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    refreshed = json.load(response)
            except urllib.error.HTTPError as error:
                raise ValueError(
                    "Breithorn token refresh failed; browser login may need renewal"
                ) from error
            if not refreshed.get("id_token"):
                raise ValueError("Refresh returned no Kubernetes ID token")
            if expiration(refreshed["id_token"]) <= time.time():
                raise ValueError("Refresh returned an expired ID token")
            saved.update(refreshed, received_at=time.time())
            save(path, saved)
        expires = datetime.fromtimestamp(expiration(saved["id_token"]), UTC)
        return {
            "apiVersion": "client.authentication.k8s.io/v1",
            "kind": "ExecCredential",
            "status": {
                "token": saved["id_token"],
                "expirationTimestamp": expires.isoformat().replace("+00:00", "Z"),
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("credentials", type=Path)
    args = parser.parse_args()
    if "KUBERNETES_EXEC_INFO" not in os.environ:
        raise SystemExit("Run through kubectl; credential output is private")
    try:
        result = credential(args.credentials)
    except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
        raise SystemExit(
            "Breithorn authentication unavailable; inspect private login state"
        ) from None
    # Kubernetes consumes this pipe; never invoke this helper through a tool
    # that would expose the credential JSON in chat or logs.
    print(json.dumps(result))
