import base64
import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "serving/kubernetes/setup-access.py"
SPEC = importlib.util.spec_from_file_location("kubernetes_access_setup", SOURCE)
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


def fake_api(monkeypatch, denied=False, server=SETUP.SERVER):
    calls = []

    def call(context, args, payload=None, check=True):
        calls.append((args, payload))
        if args[0] == "config":
            value = {
                "clusters": [{"cluster": {"server": server, "certificate-authority-data": "CA"}}],
                "users": [{"user": {"token": "administrator-credential-must-stay-private"}}],
            }
        elif "can-i" in args:
            return SimpleNamespace(returncode=1 if denied else 0, stdout="")
        elif "apply" in args:
            value = {}
        elif "secret" in args:
            value = {"data": {"token": base64.b64encode(b"synthetic-service-account").decode()}}
        else:
            raise AssertionError(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps(value))

    monkeypatch.setattr(SETUP, "kubectl", call)
    return calls


def test_denied_permissions_do_not_create_access_objects(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, denied=True)
    output = tmp_path / "config"
    with pytest.raises(PermissionError):
        SETUP.setup("admin", output, True)
    assert not output.exists()
    assert all(payload is None for _, payload in calls)


def test_wrong_cluster_rejected_before_mutation(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch, server="https://another-cluster.invalid")
    with pytest.raises(ValueError, match="Wrong cluster"):
        SETUP.setup("admin", tmp_path / "config", True)
    assert all(payload is None for _, payload in calls)


def test_private_config_contains_only_new_identity(monkeypatch, tmp_path, capsys):
    calls = fake_api(monkeypatch)
    output = tmp_path / "config"
    SETUP.setup("admin", output, True)
    raw = output.read_text()
    config = json.loads(raw)
    assert "administrator-credential" not in raw
    assert config["users"] == [
        {
            "name": SETUP.NAME,
            "user": {"token": "synthetic-service-account"},
        }
    ]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    printed = capsys.readouterr()
    assert "synthetic-service-account" not in printed.out + printed.err
    assert "administrator-credential" not in printed.out + printed.err
    payloads = [payload for _, payload in calls if payload is not None]
    assert len(payloads) == 1
    assert all(x["metadata"]["namespace"] == "rob-poc" for x in payloads[0]["items"])
    role = next(x for x in payloads[0]["items"] if x["kind"] == "Role")
    assert not any("secrets" in rule["resources"] for rule in role["rules"])


def test_existing_config_is_preserved(monkeypatch, tmp_path):
    calls = fake_api(monkeypatch)
    output = tmp_path / "config"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        SETUP.setup("admin", output, True)
    assert output.read_text() == "existing"
    assert calls == []
