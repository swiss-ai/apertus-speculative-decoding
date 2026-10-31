import importlib.util
import json
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "login_export", ROOT / "serving/kubernetes/export-login.py"
)
LOGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGIN)


def test_export_exec_login_without_leaking_tokens(monkeypatch, tmp_path, capsys):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "kubectl":
            return {
                "clusters": [
                    {"cluster": {"server": LOGIN.SERVER, "certificate-authority-data": "CA"}}
                ],
                "users": [
                    {
                        "user": {
                            "exec": {
                                "command": "kubelogin",
                                "args": ["get-token"],
                                "apiVersion": "client.authentication.k8s.io/v1beta1",
                                "provideClusterInfo": True,
                            }
                        }
                    }
                ],
            }
        return {
            "status": {"token": "synthetic-token", "expirationTimestamp": "2099-01-01T00:00:00Z"}
        }

    monkeypatch.setattr(LOGIN, "run", run)
    output = tmp_path / "login"
    LOGIN.export("breithorn-oidc", output)
    result = json.loads(output.read_text())
    assert result["users"] == [{"name": "breithorn-session", "user": {"token": "synthetic-token"}}]
    assert "exec" not in result["users"][0]["user"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "synthetic-token" not in capsys.readouterr().out
    assert all("synthetic-token" not in argv for argv, _ in calls)
    info = json.loads(calls[1][1]["env"]["KUBERNETES_EXEC_INFO"])
    assert info["spec"]["cluster"]["server"] == LOGIN.SERVER


def test_other_cluster_rejected_without_executing_plugin(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return {"clusters": [{"cluster": {"server": "https://elsewhere.invalid"}}]}

    monkeypatch.setattr(LOGIN, "run", run)
    with pytest.raises(ValueError, match="Select the Breithorn context"):
        LOGIN.export("wrong", tmp_path / "login")
    assert len(calls) == 1
    assert not (tmp_path / "login").exists()


def test_plugin_failure_does_not_print_sensitive_stderr(monkeypatch):
    class Result:
        returncode = 1
        stdout = "synthetic-sensitive-output"
        stderr = "synthetic-sensitive-error"

    monkeypatch.setattr(LOGIN.subprocess, "run", lambda *a, **kw: Result())
    with pytest.raises(RuntimeError) as error:
        LOGIN.run(["kubelogin"])
    assert "synthetic-sensitive" not in str(error.value)
